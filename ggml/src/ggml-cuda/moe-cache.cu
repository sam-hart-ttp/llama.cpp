#include "moe-cache.cuh"

#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)

extern "C" size_t ggml_moe_cache_trim(int device) {
    (void) device;
    return 0;
}

void ggml_moe_cache_register(const void * owner) {
    (void) owner;
}

#else

#include "common.cuh"
#include "mmvq.cuh"
#include "quantize.cuh"
#include "ggml-backend-impl.h"
#include "ggml-cuda.h"
#include "../ggml-backend-moe-cache.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <climits>
#include <condition_variable>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#define MOE_CACHE_LOG(...) GGML_LOG_INFO(__VA_ARGS__)
#define MOE_CACHE_STATS_LOG(...) GGML_LOG_STATUS(__VA_ARGS__)

enum class moe_cache_slot_state : uint8_t {
    free,
    copying,
    valid,
};

struct moe_cache_key {
    const void * tensor = nullptr;
    int32_t expert = -1;

    bool operator==(const moe_cache_key & other) const {
        return tensor == other.tensor && expert == other.expert;
    }
};

struct moe_cache_key_hash {
    size_t operator()(const moe_cache_key & key) const {
        uint64_t value = (uint64_t)(uintptr_t)key.tensor;
        value ^= value >> 33;
        value *= 0xff51afd7ed558ccdULL;
        value ^= (uint64_t)(uint32_t)key.expert * 0x9e3779b97f4a7c15ULL;
        value ^= value >> 29;
        return (size_t)value;
    }
};

using moe_cache_partition_key = uint64_t;
static constexpr moe_cache_partition_key MOE_CACHE_NO_PARTITION = UINT64_MAX;

struct moe_cache_partition {
    int residents = 0;
    size_t quota = 0;
    int lru_head = -1;
    int lru_tail = -1;
};

struct moe_cache_slot {
    moe_cache_key key;
    moe_cache_partition_key partition = MOE_CACHE_NO_PARTITION;
    uint64_t generation = 0;
    uint64_t last_used = 0;
    int partition_prev = -1;
    int partition_next = -1;
    int readers = 0;
    moe_cache_slot_state state = moe_cache_slot_state::free;
};

struct moe_cache_pool {
    size_t expert_size = 0;
    int wtype = -1;
    char * slab = nullptr;
    int n_slots = 0;

    std::vector<moe_cache_slot> slots;
    std::vector<int> free_slots;
    std::unordered_map<moe_cache_key, int, moe_cache_key_hash> map;
    std::unordered_map<moe_cache_partition_key, moe_cache_partition> partition_residents;
    uint64_t lru_clock = 0;
};

struct moe_cache_shape {
    size_t expert_size = 0;
    int wtype = -1;
    int64_t n_expert = 0;
    int64_t n_tensors = 0;
    int pool = -1;
    bool finished = false;
};

struct moe_cache_seen_tensor {
    size_t bytes = 0;
    size_t expert_size = 0;
    int wtype = -1;
    moe_cache_partition_key partition = MOE_CACHE_NO_PARTITION;
};

struct moe_cache_job {
    int pool = -1;
    int slot = -1;
    uint64_t generation = 0;
    moe_cache_key key;
    const void * source = nullptr;
    size_t bytes = 0;
};

struct moe_cache_demand {
    uint16_t count = 0;
    size_t expert_size = 0;
};

struct moe_cache_config {
    bool enabled = false;
    bool decode_enabled = false;
    bool automatic = false;
    size_t prefetch_mb = 0;
    size_t budget_mb = 0;
    size_t reserve_mb = 512;
    size_t min_expert_bytes = 256u << 10;
    int max_batch = 1;
    int min_slots = 8;
    int inserts_per_plan = 8;
    int admit_after = 2;
    int readmit_after = 8;
    int queue_max = 128;
    size_t queue_mb = 512;
    int stats_interval_ms = -1;
    int legacy_stats_every = 0;
    int max_devices = INT_MAX;
    int min_compute_capability = 700;
    bool serial_fill = true;
    std::string fail_stage;
};

struct moe_cache_session;

struct moe_cache_scratch {
    size_t ids = 0;
    size_t act = 0;
    size_t q8 = 0;
    size_t out = 0;
};

struct moe_cache_device {
    explicit moe_cache_device(int physical) : physical(physical) {}

    int physical;
    std::atomic<bool> dead{false};
    std::mutex dispatch_mu;

    std::vector<std::unique_ptr<moe_cache_pool>> pools;
    std::vector<moe_cache_shape> shapes;
    std::unordered_map<const void *, moe_cache_seen_tensor> seen_tensors;
    std::unordered_map<moe_cache_key, moe_cache_demand, moe_cache_key_hash> demand_count;
    int stable_visits = 0;
    bool saw_repeat = false;
    bool budget_ready = false;
    size_t budget_limit = 0;
    size_t allocated_bytes = 0;
    moe_cache_scratch scratch_reserve;

    std::deque<moe_cache_job> queue;
    size_t queued_bytes = 0;
    bool worker_started = false;
    bool inflight = false;
    const void * inflight_source = nullptr;
    size_t inflight_bytes = 0;
    std::thread worker;

    cudaStream_t compute_stream = nullptr;
    cudaStream_t prefetch_stream = nullptr;
    cudaEvent_t prefetch_ready[2] = {nullptr, nullptr};
    cudaEvent_t prefetch_consumed[2] = {nullptr, nullptr};
    bool prefetch_ready_recorded[2] = {false, false};
    bool prefetch_consumed_recorded[2] = {false, false};
    char * h_prefetch_weights = nullptr;
    char * d_prefetch_weights = nullptr;
    size_t prefetch_expert_size = 0;
    std::atomic<bool> prefetch_disabled{false};
    int32_t * h_ids = nullptr;
    int32_t * d_ids = nullptr;
    size_t h_ids_cap = 0;
    size_t d_ids_cap = 0;
    float * h_act = nullptr;
    float * d_act = nullptr;
    size_t h_act_cap = 0;
    size_t d_act_cap = 0;
    void * d_act_q8 = nullptr;
    size_t act_q8_cap = 0;
    float * d_out = nullptr;
    size_t d_out_cap = 0;
    float * h_out = nullptr;
    size_t h_out_cap = 0;

    long long hits = 0;
    long long misses = 0;
    long long inserts = 0;
    long long fills = 0;
    long long fill_failures = 0;
    long long evictions = 0;
    long long local_evictions = 0;
    long long partition_reclaims = 0;
    long long insert_skips = 0;
    long long admission_skips = 0;
    long long dispatch_failures = 0;
    long long collect_failures = 0;
    long long nodes = 0;
    long long collect_calls = 0;
    int64_t last_stats_us = 0;
    long long prefill_nodes = 0;
    long long prefill_rows = 0;
    long long prefill_failures = 0;
    long long prefill_overlap_opportunities = 0;
    long long prefill_fully_hidden = 0;
    std::atomic<int> error_logs{0};
};

struct moe_cache_session {
    moe_cache_config config;
    std::vector<std::unique_ptr<moe_cache_device>> devices;
    std::unordered_map<int, int> layer_devices;
    std::unordered_map<const void *, int> tensor_devices;

    std::mutex mu;
    std::mutex fill_mu;
    std::condition_variable cv;
    std::condition_variable idle_cv;
    std::atomic<bool> stopping{false};
    std::atomic<bool> dormant{false};
    bool announced = false;
    int active_scopes = 0;
    int active_nodes = 0;
    struct active_source {
        size_t bytes = 0;
        int references = 0;
    };
    std::unordered_map<const void *, active_source> active_sources;
};

struct moe_cache_pin {
    int slot = -1;
};

struct moe_cache_node {
    moe_cache_session * session = nullptr;
    moe_cache_device * device = nullptr;
    moe_cache_pool * pool = nullptr;
    int pool_index = -1;
    const void * host_base = nullptr;
    size_t expert_size = 0;
    int64_t n_in = 0;
    int64_t n_out = 0;
    int64_t n_expert = 0;
    int wtype = -1;
    moe_cache_partition_key partition = MOE_CACHE_NO_PARTITION;
    std::unique_lock<std::mutex> dispatch_lock;
    moe_cache_pin pins[64];
    int n_pins = 0;
    bool planned = false;
    bool dispatched = false;
};

static std::mutex g_registry_mu;
static std::unordered_set<moe_cache_session *> g_sessions;
static std::atomic<int> g_session_count{0};
struct moe_cache_scope_frame {
    moe_cache_session * requested = nullptr;
    moe_cache_session * active = nullptr;
};
static thread_local std::vector<moe_cache_scope_frame> g_session_stack;
static thread_local int g_session_suppressed = 0;

static size_t moe_cache_trim_session(
        moe_cache_session & session, int physical_device);

static bool moe_cache_env_i64(
        const char * name, int64_t min_value, int64_t max_value, int64_t & value) {
    const char * text = getenv(name);
    if (!text || !text[0]) {
        return false;
    }

    char * end = nullptr;
    errno = 0;
    const long long parsed = strtoll(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || parsed < min_value || parsed > max_value) {
        MOE_CACHE_LOG("[moe-cache] ignoring invalid %s=%s\n", name, text);
        return false;
    }

    value = parsed;
    return true;
}

static moe_cache_config moe_cache_read_config() {
    moe_cache_config config;
    int64_t value = 0;
    bool mode_off = false;
    bool mode_valid = false;

    if (const char * mode = getenv("GGML_CUDA_MOE_CACHE_MODE")) {
        if (strcmp(mode, "auto") == 0) {
            config.decode_enabled = true;
            config.automatic = true;
            mode_valid = true;
        } else if (strcmp(mode, "on") == 0) {
            config.decode_enabled = true;
            config.automatic = false;
            mode_valid = true;
        } else if (strcmp(mode, "off") == 0) {
            config.decode_enabled = false;
            mode_off = true;
            mode_valid = true;
        } else {
            MOE_CACHE_LOG("[moe-cache] ignoring invalid GGML_CUDA_MOE_CACHE_MODE=%s\n", mode);
        }
    }
    if (moe_cache_env_i64("GGML_CUDA_MOE_CACHE", 0, 1, value)) {
        config.decode_enabled = value != 0 && !mode_off;
    }
    if (moe_cache_env_i64("GGML_CUDA_MOE_CACHE_BUDGET_MB", 1, 1024 * 1024, value)) {
        config.budget_mb = (size_t)value;
        config.decode_enabled = !mode_off;
        if (!mode_valid) {
            config.automatic = false;
        }
    }
    if (moe_cache_env_i64("GGML_CUDA_MOE_PREFETCH_BUDGET_MB", 1, 1024 * 1024, value)) {
        config.prefetch_mb = (size_t)value;
    }
    if (moe_cache_env_i64("GGML_CUDA_MOE_CACHE_RESERVE_MB", 0, 1024 * 1024, value)) {
        config.reserve_mb = (size_t)value;
    }
    if (moe_cache_env_i64("GGML_CUDA_MOE_CACHE_MIN_EXPERT_KB", 1, 1024 * 1024, value)) {
        config.min_expert_bytes = (size_t)value << 10;
    }
    if (moe_cache_env_i64("GGML_CUDA_MOE_CACHE_MAX_BATCH", 1, 8, value)) {
        config.max_batch = (int)value;
    }
    if (moe_cache_env_i64("GGML_CUDA_MOE_CACHE_MIN_SLOTS", 1, INT_MAX, value)) {
        config.min_slots = (int)value;
    }
    if (moe_cache_env_i64("GGML_CUDA_MOE_CACHE_INSERTS", 1, 1024, value)) {
        config.inserts_per_plan = (int)value;
    }
    if (moe_cache_env_i64("GGML_CUDA_MOE_CACHE_ADMIT_AFTER", 1, 255, value)) {
        config.admit_after = (int)value;
    }
    if (moe_cache_env_i64("GGML_CUDA_MOE_CACHE_THROTTLE", 1, 1024, value)) {
        config.readmit_after = (int)value;
    }
    if (moe_cache_env_i64("GGML_CUDA_MOE_CACHE_QUEUE", 1, 65536, value)) {
        config.queue_max = (int)value;
    }
    if (moe_cache_env_i64("GGML_CUDA_MOE_CACHE_QUEUE_MB", 1, 1024 * 1024, value)) {
        config.queue_mb = (size_t)value;
    }
    const bool has_stats_interval = moe_cache_env_i64(
            "GGML_CUDA_MOE_CACHE_STATS_INTERVAL_MS", 0, INT_MAX, value);
    if (has_stats_interval) {
        config.stats_interval_ms = (int)value;
    }
    if (!has_stats_interval &&
        moe_cache_env_i64("GGML_CUDA_MOE_CACHE_STATS", 0, INT_MAX, value)) {
        config.legacy_stats_every = (int)value;
    }
    if (moe_cache_env_i64("GGML_CUDA_MOE_CACHE_NDEV", 1, INT_MAX, value)) {
        config.max_devices = (int)value;
    }
    if (moe_cache_env_i64("GGML_CUDA_MOE_CACHE_SERIAL_FILL", 0, 1, value)) {
        config.serial_fill = value != 0;
    }
    if (moe_cache_env_i64("GGML_CUDA_MOE_CACHE_MIN_CC", 0, 999, value)) {
        config.min_compute_capability = (int)value;
    }
    if (const char * fail = getenv("GGML_CUDA_MOE_CACHE_FAIL")) {
        config.fail_stage = fail;
    }

    config.enabled = config.decode_enabled || config.prefetch_mb > 0;

    return config;
}

static bool moe_cache_fail(const moe_cache_session & session, const char * stage) {
    const std::string & value = session.config.fail_stage;
    if (value.empty()) {
        return false;
    }
    if (value == "all" || value == stage) {
        return true;
    }

    size_t begin = 0;
    while (begin < value.size()) {
        size_t end = value.find(',', begin);
        if (end == std::string::npos) {
            end = value.size();
        }
        if (value.compare(begin, end - begin, stage) == 0) {
            return true;
        }
        begin = end + 1;
    }
    return false;
}

static bool moe_cache_ranges_overlap(
        const void * lhs, size_t lhs_size, const void * rhs, size_t rhs_size) {
    if (!lhs || !rhs || lhs_size == 0 || rhs_size == 0) {
        return false;
    }
    const uintptr_t l = (uintptr_t)lhs;
    const uintptr_t r = (uintptr_t)rhs;
    return (l <= r ? r - l < lhs_size : l - r < rhs_size);
}

static uint64_t moe_cache_name_hash(const char * text) {
    uint64_t hash = 0xcbf29ce484222325ULL;
    while (*text) {
        hash ^= (unsigned char)*text++;
        hash *= 0x100000001b3ULL;
    }
    return hash;
}

static bool moe_cache_layer_number(const char * name, int & layer) {
    const char * marker = strstr(name, "blk.");
    if (!marker) {
        return false;
    }

    const char * first = marker + 4;
    char * end = nullptr;
    errno = 0;
    const long parsed = strtol(first, &end, 10);
    if (errno != 0 || end == first || parsed < 0 || parsed > INT_MAX) {
        return false;
    }
    if (*end != '.' && *end != '\0') {
        return false;
    }
    layer = (int)parsed;
    return true;
}

static moe_cache_partition_key moe_cache_partition_for(
        const char * name, const void * host_base, int & layer, bool & has_layer) {
    has_layer = moe_cache_layer_number(name, layer);
    if (has_layer) {
        // Even keys are stable layer partitions. A pool is already scoped to
        // one expert shape, so grouping gate/up/down tensors by layer prevents
        // the graph's sequential layer walk from becoming a cache-wide scan.
        return (moe_cache_partition_key)(uint32_t)layer << 1;
    }

    // Odd keys isolate tensors whose names do not expose a layer number. GGUF
    // tensor bases are address-stable for the scheduler session lifetime.
    uint64_t value = (uint64_t)(uintptr_t)host_base;
    value ^= value >> 33;
    value *= 0xff51afd7ed558ccdULL;
    value ^= moe_cache_name_hash(name);
    return (value << 1) | 1;
}

static bool moe_cache_type_supported(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q1_0:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_NVFP4:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ1_M:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_IQ4_XS:
            return true;
        default:
            return false;
    }
}

static void moe_cache_lru_remove(moe_cache_pool & pool, int index) {
    moe_cache_slot & slot = pool.slots[index];
    auto partition = pool.partition_residents.find(slot.partition);
    if (partition != pool.partition_residents.end()) {
        moe_cache_partition & state = partition->second;
        if (slot.partition_prev >= 0) {
            pool.slots[slot.partition_prev].partition_next = slot.partition_next;
        } else {
            state.lru_head = slot.partition_next;
        }
        if (slot.partition_next >= 0) {
            pool.slots[slot.partition_next].partition_prev = slot.partition_prev;
        } else {
            state.lru_tail = slot.partition_prev;
        }
    }
    slot.partition_prev = -1;
    slot.partition_next = -1;
}

static void moe_cache_lru_push_back(moe_cache_pool & pool, int index) {
    moe_cache_slot & slot = pool.slots[index];
    auto partition = pool.partition_residents.find(slot.partition);
    if (partition != pool.partition_residents.end()) {
        moe_cache_partition & state = partition->second;
        slot.partition_prev = state.lru_tail;
        slot.partition_next = -1;
        if (state.lru_tail >= 0) {
            pool.slots[state.lru_tail].partition_next = index;
        } else {
            state.lru_head = index;
        }
        state.lru_tail = index;
        slot.last_used = ++pool.lru_clock;
    }
}

static void moe_cache_map_erase(moe_cache_pool & pool, int index) {
    moe_cache_slot & slot = pool.slots[index];
    auto it = pool.map.find(slot.key);
    if (it != pool.map.end() && it->second == index) {
        pool.map.erase(it);
    }
}

static void moe_cache_rebalance_partitions(moe_cache_pool & pool) {
    const size_t n_partitions = pool.partition_residents.size();
    if (n_partitions == 0) {
        return;
    }
    const size_t base = (size_t)pool.n_slots / n_partitions;
    const size_t remainder = (size_t)pool.n_slots % n_partitions;
    for (auto & item : pool.partition_residents) {
        size_t rank = 0;
        for (const auto & other : pool.partition_residents) {
            if (other.first < item.first) {
                rank++;
            }
        }
        item.second.quota = base + (rank < remainder ? 1 : 0);
    }
}

static size_t moe_cache_partition_quota(
        const moe_cache_pool & pool, moe_cache_partition_key partition) {
    auto found = pool.partition_residents.find(partition);
    return found == pool.partition_residents.end()
        ? (size_t)pool.n_slots : found->second.quota;
}

static int moe_cache_partition_size(
        const moe_cache_pool & pool, moe_cache_partition_key partition) {
    auto found = pool.partition_residents.find(partition);
    return found == pool.partition_residents.end() ? 0 : found->second.residents;
}

static int moe_cache_lru_candidate(
        const moe_cache_pool & pool, moe_cache_partition_key requested,
        bool same_partition, bool donor_over_quota) {
    if (same_partition) {
        auto partition = pool.partition_residents.find(requested);
        int candidate = partition == pool.partition_residents.end()
            ? -1 : partition->second.lru_head;
        for (; candidate >= 0;
             candidate = pool.slots[candidate].partition_next) {
            const moe_cache_slot & slot = pool.slots[candidate];
            if (slot.state == moe_cache_slot_state::valid && slot.readers == 0) {
                return candidate;
            }
        }
        return -1;
    }

    int oldest = -1;
    uint64_t oldest_time = UINT64_MAX;
    for (const auto & partition : pool.partition_residents) {
        const moe_cache_partition & state = partition.second;
        if (donor_over_quota &&
            (state.residents <= 0 || (size_t)state.residents <= state.quota)) {
            continue;
        }
        for (int candidate = state.lru_head; candidate >= 0;
             candidate = pool.slots[candidate].partition_next) {
            const moe_cache_slot & slot = pool.slots[candidate];
            if (slot.state != moe_cache_slot_state::valid || slot.readers > 0) {
                continue;
            }
            if (slot.last_used < oldest_time) {
                oldest = candidate;
                oldest_time = slot.last_used;
            }
            break;
        }
    }
    return oldest;
}

static void moe_cache_slot_reset(moe_cache_pool & pool, int index, bool add_to_free) {
    moe_cache_slot & slot = pool.slots[index];
    if (slot.state == moe_cache_slot_state::valid) {
        moe_cache_lru_remove(pool, index);
    }
    moe_cache_map_erase(pool, index);
    if (slot.state != moe_cache_slot_state::free &&
        slot.partition != MOE_CACHE_NO_PARTITION) {
        auto partition = pool.partition_residents.find(slot.partition);
        if (partition != pool.partition_residents.end() && partition->second.residents > 0) {
            partition->second.residents--;
        }
    }
    slot.key = {};
    slot.partition = MOE_CACHE_NO_PARTITION;
    slot.generation++;
    slot.readers = 0;
    slot.state = moe_cache_slot_state::free;
    slot.last_used = 0;
    slot.partition_prev = -1;
    slot.partition_next = -1;
    if (add_to_free) {
        pool.free_slots.push_back(index);
    }
}

static bool moe_cache_cuda_ok(
        moe_cache_device & device, cudaError_t error, const char * operation, bool fatal) {
    if (error == cudaSuccess) {
        return true;
    }

    (void)cudaGetLastError();
    if (device.error_logs.fetch_add(1) < 8) {
        MOE_CACHE_LOG("[moe-cache] CUDA%d %s failed: %s\n",
                device.physical, operation, cudaGetErrorString(error));
    }
    if (fatal && error != cudaErrorMemoryAllocation) {
        device.dead.store(true);
    }
    return false;
}

static bool moe_cache_grow_device(
        moe_cache_device & device, void ** pointer, size_t & capacity,
        size_t required, const char * operation) {
    if (capacity >= required) {
        return true;
    }
    if (required > (std::numeric_limits<size_t>::max() - 256) / 2) {
        return false;
    }

    const size_t requested = required * 2 + 256;
    void * fresh = nullptr;
    if (!moe_cache_cuda_ok(device, cudaMalloc(&fresh, requested), operation, false)) {
        return false;
    }
    if (*pointer) {
        cudaFree(*pointer);
    }
    *pointer = fresh;
    capacity = requested;
    return true;
}

static size_t moe_cache_growth_capacity(size_t capacity, size_t required) {
    if (capacity >= required) {
        return capacity;
    }
    if (required > (std::numeric_limits<size_t>::max() - 256) / 2) {
        return 0;
    }
    return required * 2 + 256;
}

static bool moe_cache_scratch_requirements(
        int64_t n_in, int64_t n_out, moe_cache_scratch & result) {
    constexpr size_t max_rows = 64;
    if (n_in <= 0 || n_out <= 0 ||
        n_in > INT64_MAX - (MATRIX_ROW_PADDING - 1)) {
        return false;
    }
    const int64_t padded_n_in =
        ((n_in + MATRIX_ROW_PADDING - 1) / MATRIX_ROW_PADDING) * MATRIX_ROW_PADDING;
    if ((uint64_t)n_in > SIZE_MAX / (max_rows * sizeof(float)) ||
        (uint64_t)n_out > SIZE_MAX / (max_rows * sizeof(float)) ||
        (uint64_t)(padded_n_in / QK8_1) >
            SIZE_MAX / (max_rows * sizeof(block_q8_1))) {
        return false;
    }

    const size_t capacities[] = {
        moe_cache_growth_capacity(0, max_rows * sizeof(int32_t)),
        moe_cache_growth_capacity(0, max_rows * (size_t)n_in * sizeof(float)),
        moe_cache_growth_capacity(
                0, max_rows * (size_t)(padded_n_in / QK8_1) * sizeof(block_q8_1)),
        moe_cache_growth_capacity(0, max_rows * (size_t)n_out * sizeof(float)),
    };
    for (size_t capacity : capacities) {
        if (capacity == 0) {
            return false;
        }
    }
    result = {capacities[0], capacities[1], capacities[2], capacities[3]};
    return true;
}

static size_t moe_cache_scratch_total(
        const moe_cache_scratch & current,
        const moe_cache_scratch * additional = nullptr) {
    const size_t capacities[] = {
        additional ? std::max(current.ids, additional->ids) : current.ids,
        additional ? std::max(current.act, additional->act) : current.act,
        additional ? std::max(current.q8, additional->q8) : current.q8,
        additional ? std::max(current.out, additional->out) : current.out,
    };
    size_t total = 0;
    for (size_t capacity : capacities) {
        if (capacity > SIZE_MAX - total) {
            return SIZE_MAX;
        }
        total += capacity;
    }
    return total;
}

static bool moe_cache_grow_host(
        moe_cache_device & device, void ** pointer, size_t & capacity,
        size_t required, const char * operation) {
    if (capacity >= required) {
        return true;
    }
    if (required > (std::numeric_limits<size_t>::max() - 256) / 2) {
        return false;
    }

    const size_t requested = required * 2 + 256;
    void * fresh = nullptr;
    if (!moe_cache_cuda_ok(device, cudaMallocHost(&fresh, requested), operation, false)) {
        return false;
    }
    if (*pointer) {
        cudaFreeHost(*pointer);
    }
    *pointer = fresh;
    capacity = requested;
    return true;
}

static void moe_cache_worker(moe_cache_session * session, moe_cache_device * device) {
    char * stage = nullptr;
    size_t stage_capacity = 0;
    cudaStream_t stream = nullptr;

    for (;;) {
        moe_cache_job job;
        {
            std::unique_lock<std::mutex> lock(session->mu);
            session->cv.wait(lock, [&] {
                return session->stopping || device->dead.load() ||
                    !device->queue.empty();
            });
            if ((session->stopping || device->dead.load()) &&
                device->queue.empty()) {
                break;
            }

            job = device->queue.front();
            device->queue.pop_front();
            device->queued_bytes = job.bytes <= device->queued_bytes
                ? device->queued_bytes - job.bytes : 0;
            device->inflight = true;
            device->inflight_source = job.source;
            device->inflight_bytes = job.bytes;
        }

        cudaError_t error = cudaSuccess;
        ggml_cuda_set_device(device->physical);

        if (device->dead.load() || moe_cache_fail(*session, "insert")) {
            error = cudaErrorUnknown;
        }
        if (error == cudaSuccess && !stream) {
            int least_priority = 0;
            int greatest_priority = 0;
            error = cudaDeviceGetStreamPriorityRange(
                    &least_priority, &greatest_priority);
            if (error == cudaSuccess) {
                error = cudaStreamCreateWithPriority(
                        &stream, cudaStreamNonBlocking, least_priority);
            } else {
                (void)cudaGetLastError();
                error = cudaStreamCreateWithFlags(
                        &stream, cudaStreamNonBlocking);
            }
        }
        if (error == cudaSuccess && stage_capacity < job.bytes) {
            char * fresh = nullptr;
            cudaError_t alloc_error = cudaMallocHost((void **)&fresh, job.bytes);
            if (alloc_error == cudaSuccess) {
                if (stage) {
                    cudaFreeHost(stage);
                }
                stage = fresh;
                stage_capacity = job.bytes;
            } else {
                (void)cudaGetLastError();
            }
        }

        moe_cache_pool * pool = nullptr;
        char * destination = nullptr;
        {
            std::lock_guard<std::mutex> lock(session->mu);
            if (job.pool >= 0 && job.pool < (int)device->pools.size()) {
                pool = device->pools[job.pool].get();
                if (pool->slab && job.slot >= 0 && job.slot < pool->n_slots) {
                    destination = pool->slab + (size_t)job.slot * pool->expert_size;
                }
            }
        }

        if (error == cudaSuccess && !destination) {
            error = cudaErrorInvalidValue;
        }
        {
            std::unique_lock<std::mutex> fill_lock(
                    session->fill_mu, std::defer_lock);
            if (session->config.serial_fill) {
                fill_lock.lock();
            }
            if (error == cudaSuccess && stage && stage_capacity >= job.bytes) {
                memcpy(stage, job.source, job.bytes);
                error = cudaMemcpyAsync(
                        destination, stage, job.bytes, cudaMemcpyHostToDevice, stream);
                if (error == cudaSuccess) {
                    error = cudaStreamSynchronize(stream);
                }
            } else if (error == cudaSuccess) {
                error = cudaMemcpy(
                        destination, job.source, job.bytes, cudaMemcpyHostToDevice);
            }
        }

        {
            std::lock_guard<std::mutex> lock(session->mu);
            device->inflight = false;
            device->inflight_source = nullptr;
            device->inflight_bytes = 0;

            if (pool && job.slot >= 0 && job.slot < pool->n_slots) {
                moe_cache_slot & slot = pool->slots[job.slot];
                if (slot.state == moe_cache_slot_state::copying &&
                    slot.generation == job.generation && slot.key == job.key) {
                    if (error == cudaSuccess) {
                        slot.state = moe_cache_slot_state::valid;
                        moe_cache_lru_push_back(*pool, job.slot);
                        device->demand_count.erase(job.key);
                        device->fills++;
                    } else {
                        moe_cache_slot_reset(*pool, job.slot, true);
                        device->fill_failures++;
                    }
                }
            }
            session->idle_cv.notify_all();
        }

        if (error != cudaSuccess) {
            moe_cache_cuda_ok(*device, error, "expert fill", true);
            moe_cache_trim_session(*session, device->physical);
        }
    }

    if (stream) {
        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);
    }
    if (stage) {
        cudaFreeHost(stage);
    }
}

static bool moe_cache_start_worker(
        moe_cache_session & session, moe_cache_device & device) {
    if (device.worker_started) {
        return true;
    }
    try {
        device.worker = std::thread(moe_cache_worker, &session, &device);
        device.worker_started = true;
        return true;
    } catch (...) {
        device.dead.store(true);
        MOE_CACHE_LOG("[moe-cache] CUDA%d failed to start fill worker\n", device.physical);
        return false;
    }
}

static int moe_cache_find_pool(
        const moe_cache_device & device, size_t expert_size, int wtype) {
    for (int index = 0; index < (int)device.pools.size(); index++) {
        const moe_cache_pool & pool = *device.pools[index];
        if (pool.expert_size == expert_size && pool.wtype == wtype) {
            return index;
        }
    }
    return -1;
}

static bool moe_cache_prepare_budget(
        moe_cache_session & session, moe_cache_device & device) {
    if (device.budget_ready) {
        return device.budget_limit > 0;
    }
    device.budget_ready = true;

    ggml_cuda_set_device(device.physical);
    size_t free_memory = 0;
    size_t total_memory = 0;
    cudaError_t error = cudaMemGetInfo(&free_memory, &total_memory);
    if (!moe_cache_cuda_ok(device, error, "memory query", false)) {
        device.dead.store(true);
        return false;
    }

    const size_t reserve = session.config.reserve_mb << 20;
    size_t available = free_memory > reserve ? free_memory - reserve : 0;
    if (session.config.budget_mb > 0) {
        available = std::min(available, session.config.budget_mb << 20);
    }
    device.budget_limit = available;

    if (available == 0) {
        MOE_CACHE_LOG("[moe-cache] CUDA%d has no cache budget after %zu MiB reserve\n",
                device.physical, session.config.reserve_mb);
        return false;
    }
    return true;
}

static bool moe_cache_allocate_pool(
        moe_cache_session & session, moe_cache_device & device,
        moe_cache_shape & shape, size_t budget) {
    if (shape.expert_size == 0 || shape.n_tensors <= 0 || shape.n_expert <= 0) {
        return false;
    }
    shape.finished = true;

    int64_t max_entries = shape.n_tensors;
    if (max_entries > INT_MAX / shape.n_expert) {
        max_entries = INT_MAX;
    } else {
        max_entries *= shape.n_expert;
    }

    size_t slots_by_budget = budget / shape.expert_size;
    size_t slot_count = std::min<size_t>(slots_by_budget, (size_t)max_entries);
    const size_t type_size = ggml_type_size((ggml_type)shape.wtype);
    if (type_size == 0 || shape.expert_size % type_size != 0) {
        return false;
    }
    const size_t stride_blocks = shape.expert_size / type_size;
    if (stride_blocks == 0) {
        return false;
    }
    slot_count = std::min(slot_count, (size_t)INT_MAX / stride_blocks);
    if (slot_count > INT_MAX) {
        slot_count = INT_MAX;
    }
    if (slot_count < (size_t)session.config.min_slots) {
        return false;
    }

    ggml_cuda_set_device(device.physical);
    char * slab = nullptr;
    cudaError_t error = cudaSuccess;
    while (slot_count >= (size_t)session.config.min_slots) {
        if (moe_cache_fail(session, "slab")) {
            error = cudaErrorMemoryAllocation;
        } else {
            error = cudaMalloc((void **)&slab, slot_count * shape.expert_size);
        }
        if (error == cudaSuccess) {
            break;
        }
        (void)cudaGetLastError();
        slot_count /= 2;
    }
    if (error != cudaSuccess || !slab ||
        slot_count < (size_t)session.config.min_slots) {
        MOE_CACHE_LOG("[moe-cache] CUDA%d skipped %zu KiB expert pool: allocation failed\n",
                device.physical, shape.expert_size >> 10);
        return false;
    }

    std::unique_ptr<moe_cache_pool> pool(new (std::nothrow) moe_cache_pool());
    if (!pool) {
        cudaFree(slab);
        return false;
    }
    try {
        pool->expert_size = shape.expert_size;
        pool->wtype = shape.wtype;
        pool->slab = slab;
        pool->n_slots = (int)slot_count;
        pool->slots.resize(slot_count);
        pool->free_slots.reserve(slot_count);
        pool->map.reserve(slot_count);
        pool->partition_residents.reserve((size_t)shape.n_tensors);
        for (const auto & item : device.seen_tensors) {
            const moe_cache_seen_tensor & tensor = item.second;
            if (tensor.expert_size == shape.expert_size &&
                tensor.wtype == shape.wtype &&
                tensor.partition != MOE_CACHE_NO_PARTITION) {
                pool->partition_residents.emplace(
                        tensor.partition, moe_cache_partition{});
            }
        }
        moe_cache_rebalance_partitions(*pool);
        for (int index = (int)slot_count - 1; index >= 0; index--) {
            pool->free_slots.push_back(index);
        }
        device.pools.push_back(std::move(pool));
    } catch (...) {
        cudaFree(slab);
        return false;
    }

    shape.pool = (int)device.pools.size() - 1;
    const size_t allocated = slot_count * shape.expert_size;
    device.allocated_bytes += allocated;

    if (!moe_cache_start_worker(session, device)) {
        cudaFree(device.pools.back()->slab);
        device.pools.back()->slab = nullptr;
        device.pools.pop_back();
        device.allocated_bytes -= allocated;
        shape.pool = -1;
        return false;
    }

    if (!session.announced) {
        MOE_CACHE_LOG("[moe-cache] enabled: mode=%s budget=%s reserve=%zu MiB min-expert=%zu KiB admit=%d/%d\n",
                session.config.automatic ? "auto" : "on",
                session.config.budget_mb ? "fixed" : "free-minus-reserve",
                session.config.reserve_mb, session.config.min_expert_bytes >> 10,
                session.config.admit_after, session.config.readmit_after);
        session.announced = true;
    }
    MOE_CACHE_LOG("[moe-cache] CUDA%d pool[%d]: type=%s expert=%zu KiB slots=%zu partitions=%zu total=%zu MiB\n",
            device.physical, shape.pool, ggml_type_name((ggml_type)shape.wtype),
            shape.expert_size >> 10, slot_count,
            device.pools.back()->partition_residents.size(), allocated >> 20);

    return true;
}

static void moe_cache_build_pending(
        moe_cache_session & session, moe_cache_device & device) {
    if (!moe_cache_prepare_budget(session, device)) {
        for (moe_cache_shape & shape : device.shapes) {
            if (shape.n_tensors > 0) {
                shape.finished = true;
            }
        }
        return;
    }

    const size_t scratch_reserve =
        moe_cache_scratch_total(device.scratch_reserve);
    const size_t slab_limit =
        scratch_reserve < device.budget_limit
            ? device.budget_limit - scratch_reserve : 0;
    size_t remaining = slab_limit > device.allocated_bytes
        ? slab_limit - device.allocated_bytes : 0;
    if (remaining == 0) {
        for (moe_cache_shape & shape : device.shapes) {
            if (!shape.finished && shape.n_tensors > 0) {
                shape.finished = true;
            }
        }
        return;
    }

    std::vector<moe_cache_shape *> pending;
    double total_weight = 0.0;
    for (moe_cache_shape & shape : device.shapes) {
        if (!shape.finished && shape.n_tensors > 0) {
            pending.push_back(&shape);
            total_weight +=
                (double)shape.expert_size * (double)shape.n_tensors;
        }
    }
    std::sort(pending.begin(), pending.end(), [](const auto * lhs, const auto * rhs) {
        const double lhs_weight =
            (double)lhs->expert_size * (double)lhs->n_tensors;
        const double rhs_weight =
            (double)rhs->expert_size * (double)rhs->n_tensors;
        return lhs_weight > rhs_weight;
    });

    for (moe_cache_shape * shape : pending) {
        const double weight =
            (double)shape->expert_size * (double)shape->n_tensors;
        const size_t share = total_weight > 0.0
            ? (size_t)((double)remaining * weight / total_weight) : 0;
        const size_t before = device.allocated_bytes;
        moe_cache_allocate_pool(session, device, *shape, share);
        const size_t consumed = device.allocated_bytes - before;
        remaining = consumed <= remaining ? remaining - consumed : 0;
        total_weight -= weight;
    }
}

static int moe_cache_discover_pool(
        moe_cache_session & session, moe_cache_device & device,
        const void * host_base, size_t tensor_size, size_t expert_size,
        int wtype, int64_t n_expert, moe_cache_partition_key partition) {
    int pool = moe_cache_find_pool(device, expert_size, wtype);
    if (pool >= 0) {
        const bool first_visit = device.seen_tensors.emplace(
                host_base, moe_cache_seen_tensor{
                    tensor_size, expert_size, wtype, partition}).second;
        if (first_visit) {
            for (moe_cache_shape & shape : device.shapes) {
                if (shape.expert_size == expert_size && shape.wtype == wtype) {
                    shape.n_tensors++;
                    break;
                }
            }
        }
        const bool new_partition = device.pools[pool]->partition_residents.emplace(
                partition, moe_cache_partition{}).second;
        if (new_partition) {
            moe_cache_rebalance_partitions(*device.pools[pool]);
        }
        return pool;
    }

    moe_cache_shape * shape = nullptr;
    for (moe_cache_shape & candidate : device.shapes) {
        if (candidate.expert_size == expert_size && candidate.wtype == wtype) {
            shape = &candidate;
            break;
        }
    }
    if (!shape) {
        device.shapes.push_back({expert_size, wtype, n_expert, 0, -1, false});
        shape = &device.shapes.back();
    } else {
        shape->n_expert = std::max(shape->n_expert, n_expert);
    }

    const bool first_visit = device.seen_tensors.emplace(
            host_base, moe_cache_seen_tensor{
                tensor_size, expert_size, wtype, partition}).second;
    if (first_visit) {
        if (shape->n_tensors == 0 && shape->pool < 0) {
            shape->finished = false;
        }
        shape->n_tensors++;
        device.stable_visits = 0;
    } else {
        device.saw_repeat = true;
        device.stable_visits++;
    }

    if (!device.saw_repeat || device.stable_visits < 64) {
        return -1;
    }

    moe_cache_build_pending(session, device);
    return moe_cache_find_pool(device, expert_size, wtype);
}

static bool moe_cache_stats_requested(const moe_cache_config & config) {
    return config.stats_interval_ms >= 0 || config.legacy_stats_every > 0;
}

static void moe_cache_log_stats(moe_cache_device & device, bool requested) {
    size_t used = 0;
    size_t slots = 0;
    size_t partitions = 0;
    for (const auto & pool_ptr : device.pools) {
        const moe_cache_pool & pool = *pool_ptr;
        slots += pool.n_slots;
        used += pool.n_slots - pool.free_slots.size();
        partitions += pool.partition_residents.size();
    }
    const long long total = device.hits + device.misses;
    char text[1024];
    snprintf(text, sizeof(text),
            "[moe-cache] CUDA%d hits=%lld/%lld (%.1f%%) used=%zu/%zu partitions=%zu enqueued=%lld filled=%lld fill-fail=%lld evictions=%lld local=%lld reclaim=%lld skips=%lld admission=%lld queue=%zu jobs/%zu MiB dispatch-fail=%lld collect-fail=%lld prefill=%lld nodes/%lld rows prefill-fail=%lld hidden=%lld/%lld\n",
            device.physical, device.hits, total,
            total ? 100.0 * (double)device.hits / (double)total : 0.0,
            used, slots, partitions,
            device.inserts, device.fills, device.fill_failures,
            device.evictions, device.local_evictions,
            device.partition_reclaims, device.insert_skips,
            device.admission_skips, device.queue.size(), device.queued_bytes >> 20,
            device.dispatch_failures, device.collect_failures,
            device.prefill_nodes, device.prefill_rows, device.prefill_failures,
            device.prefill_fully_hidden, device.prefill_overlap_opportunities);
    if (requested) {
        MOE_CACHE_STATS_LOG("%s", text);
    } else {
        MOE_CACHE_LOG("%s", text);
    }
}

static void moe_cache_maybe_log_stats(
        const moe_cache_config & config, moe_cache_device & device,
        bool collected) {
    bool due = collected && config.legacy_stats_every > 0 &&
        device.collect_calls % config.legacy_stats_every == 0;
    if (config.stats_interval_ms > 0) {
        const int64_t now = ggml_time_us();
        if (device.last_stats_us == 0) {
            device.last_stats_us = now;
        } else if (now - device.last_stats_us >=
                   (int64_t)config.stats_interval_ms * 1000) {
            device.last_stats_us = now;
            due = true;
        }
    }
    if (due) {
        moe_cache_log_stats(device, true);
    }
}

static void * moe_cache_session_create(void * const * backends, int n_backends) {
    try {
        moe_cache_config config = moe_cache_read_config();

        std::unique_ptr<moe_cache_session> session(new (std::nothrow) moe_cache_session());
        if (!session) {
            return nullptr;
        }
        session->config = std::move(config);

        std::unordered_set<int> seen_devices;
        for (int index = 0; index < n_backends &&
                            (int)session->devices.size() < session->config.max_devices; index++) {
            ggml_backend_t backend = (ggml_backend_t)backends[index];
            if (!backend || !ggml_backend_is_cuda(backend)) {
                continue;
            }
            ggml_backend_cuda_context * context =
                (ggml_backend_cuda_context *)backend->context;
            const int physical = ggml_cuda_info().devices[context->device].physical_device;
            if (!seen_devices.insert(physical).second) {
                continue;
            }

            cudaDeviceProp properties;
            ggml_cuda_set_device(physical);
            cudaError_t error = cudaGetDeviceProperties(&properties, physical);
            if (error != cudaSuccess) {
                (void)cudaGetLastError();
                MOE_CACHE_LOG("[moe-cache] CUDA%d skipped: device query failed\n", physical);
                continue;
            }
            const int capability = properties.major * 100 + properties.minor * 10;
            if (capability < session->config.min_compute_capability) {
                MOE_CACHE_LOG("[moe-cache] CUDA%d skipped: compute capability %d.%d is below %d.%d\n",
                        physical, properties.major, properties.minor,
                        session->config.min_compute_capability / 100,
                        (session->config.min_compute_capability % 100) / 10);
                continue;
            }

            session->devices.emplace_back(new moe_cache_device(physical));
        }

        if (session->devices.empty()) {
            return nullptr;
        }

        // Keep an inert session even when the environment disables streaming.
        // The public scheduler API is applied after scheduler construction, so
        // it must be able to turn an explicitly disabled session back on.
        session->dormant.store(!session->config.enabled);

        moe_cache_session * result = session.get();
        {
            std::lock_guard<std::mutex> lock(g_registry_mu);
            g_sessions.insert(result);
            g_session_count.fetch_add(1, std::memory_order_release);
        }
        session.release();
        return result;
    } catch (...) {
        MOE_CACHE_LOG("[moe-cache] failed to create cache session\n");
        return nullptr;
    }
}

static void moe_cache_cancel_queue_locked(
        moe_cache_device & device, const void * base, size_t size, bool all) {
    for (auto it = device.queue.begin(); it != device.queue.end();) {
        if (all || moe_cache_ranges_overlap(it->source, it->bytes, base, size)) {
            device.queued_bytes = it->bytes <= device.queued_bytes
                ? device.queued_bytes - it->bytes : 0;
            if (it->pool >= 0 && it->pool < (int)device.pools.size()) {
                moe_cache_pool & pool = *device.pools[it->pool];
                if (it->slot >= 0 && it->slot < pool.n_slots) {
                    moe_cache_slot & slot = pool.slots[it->slot];
                    if (slot.state == moe_cache_slot_state::copying &&
                        slot.generation == it->generation && slot.key == it->key) {
                        moe_cache_slot_reset(pool, it->slot, true);
                    }
                }
            }
            it = device.queue.erase(it);
        } else {
            ++it;
        }
    }
}

static void moe_cache_free_device(moe_cache_device & device) {
    ggml_cuda_set_device(device.physical);
    if (device.compute_stream) {
        cudaStreamSynchronize(device.compute_stream);
    }
    if (device.prefetch_stream) {
        cudaStreamSynchronize(device.prefetch_stream);
    }
    for (auto & pool_ptr : device.pools) {
        if (pool_ptr->slab) {
            cudaFree(pool_ptr->slab);
            pool_ptr->slab = nullptr;
        }
    }
    if (device.d_ids) {
        cudaFree(device.d_ids);
        device.d_ids = nullptr;
    }
    if (device.d_act) {
        cudaFree(device.d_act);
        device.d_act = nullptr;
    }
    if (device.d_act_q8) {
        cudaFree(device.d_act_q8);
        device.d_act_q8 = nullptr;
    }
    if (device.d_out) {
        cudaFree(device.d_out);
        device.d_out = nullptr;
    }
    if (device.d_prefetch_weights) {
        cudaFree(device.d_prefetch_weights);
        device.d_prefetch_weights = nullptr;
    }
    if (device.h_ids) {
        cudaFreeHost(device.h_ids);
        device.h_ids = nullptr;
    }
    if (device.h_act) {
        cudaFreeHost(device.h_act);
        device.h_act = nullptr;
    }
    if (device.h_out) {
        cudaFreeHost(device.h_out);
        device.h_out = nullptr;
    }
    if (device.h_prefetch_weights) {
        cudaFreeHost(device.h_prefetch_weights);
        device.h_prefetch_weights = nullptr;
    }
    for (int slot = 0; slot < 2; slot++) {
        if (device.prefetch_ready[slot]) {
            cudaEventDestroy(device.prefetch_ready[slot]);
            device.prefetch_ready[slot] = nullptr;
        }
        if (device.prefetch_consumed[slot]) {
            cudaEventDestroy(device.prefetch_consumed[slot]);
            device.prefetch_consumed[slot] = nullptr;
        }
        device.prefetch_ready_recorded[slot] = false;
        device.prefetch_consumed_recorded[slot] = false;
    }
    if (device.prefetch_stream) {
        cudaStreamDestroy(device.prefetch_stream);
        device.prefetch_stream = nullptr;
    }
    if (device.compute_stream) {
        cudaStreamDestroy(device.compute_stream);
        device.compute_stream = nullptr;
    }
    device.pools.clear();
    device.allocated_bytes = 0;
    device.d_ids_cap = 0;
    device.d_act_cap = 0;
    device.act_q8_cap = 0;
    device.d_out_cap = 0;
    device.h_ids_cap = 0;
    device.h_act_cap = 0;
    device.h_out_cap = 0;
    device.prefetch_expert_size = 0;
}

static void moe_cache_session_destroy(void * opaque) {
    moe_cache_session * session = (moe_cache_session *)opaque;
    if (!session) {
        return;
    }

    {
        std::unique_lock<std::mutex> lock(session->mu);
        session->stopping = true;
        for (auto & device_ptr : session->devices) {
            moe_cache_cancel_queue_locked(*device_ptr, nullptr, 0, true);
        }
        session->cv.notify_all();
        session->idle_cv.wait(lock, [&] {
            return session->active_scopes == 0 && session->active_nodes == 0;
        });
    }

    for (auto & device_ptr : session->devices) {
        if (device_ptr->worker_started && device_ptr->worker.joinable()) {
            device_ptr->worker.join();
        }
    }
    {
        std::lock_guard<std::mutex> registry_lock(g_registry_mu);
        if (g_sessions.erase(session) > 0) {
            g_session_count.fetch_sub(1, std::memory_order_release);
        }
    }

    for (auto & device_ptr : session->devices) {
        if (device_ptr->nodes > 0 || device_ptr->dispatch_failures > 0 ||
            device_ptr->collect_failures > 0 || device_ptr->prefill_nodes > 0 ||
            device_ptr->prefill_failures > 0) {
            moe_cache_log_stats(
                    *device_ptr, moe_cache_stats_requested(session->config));
        }
    }

    for (auto & device_ptr : session->devices) {
        std::unique_lock<std::mutex> dispatch_lock(device_ptr->dispatch_mu);
        moe_cache_free_device(*device_ptr);
    }

    delete session;
}

static void moe_cache_session_configure(
        void * opaque, size_t cache_mib, size_t prefetch_mib,
        int stats_interval_ms) {
    moe_cache_session * session = (moe_cache_session *)opaque;
    if (!session) {
        return;
    }
    std::lock_guard<std::mutex> lock(session->mu);
    if (session->active_scopes != 0 || session->active_nodes != 0) {
        MOE_CACHE_LOG("[moe-cache] ignored configuration change during active compute\n");
        return;
    }
    session->config.decode_enabled = cache_mib > 0;
    session->config.automatic = false;
    session->config.budget_mb = cache_mib;
    session->config.prefetch_mb = prefetch_mib;
    session->config.stats_interval_ms = std::max(stats_interval_ms, -1);
    session->config.legacy_stats_every = 0;
    const int64_t now = ggml_time_us();
    for (auto & device_ptr : session->devices) {
        device_ptr->last_stats_us = now;
    }
    session->config.min_compute_capability = 700;
    session->config.enabled = cache_mib > 0 || prefetch_mib > 0;
    session->dormant.store(!session->config.enabled);
}

static void moe_cache_session_enter(void * opaque) {
    if (g_session_suppressed > 0) {
        g_session_suppressed++;
        return;
    }

    moe_cache_session * session = (moe_cache_session *)opaque;
    if (!session || session->dormant.load()) {
        if (g_session_stack.empty()) {
            return;
        }
        try {
            g_session_stack.push_back({session, nullptr});
        } catch (...) {
            g_session_suppressed++;
        }
        return;
    }
    std::lock_guard<std::mutex> lock(session->mu);
    if (session->dormant.load()) {
        if (g_session_stack.empty()) {
            return;
        }
        try {
            g_session_stack.push_back({session, nullptr});
        } catch (...) {
            g_session_suppressed++;
        }
        return;
    }
    if (session->stopping) {
        if (g_session_stack.empty()) {
            return;
        }
        try {
            g_session_stack.push_back({session, nullptr});
        } catch (...) {
            g_session_suppressed++;
        }
        return;
    }
    try {
        g_session_stack.push_back({session, session});
    } catch (...) {
        g_session_suppressed++;
        return;
    }
    session->active_scopes++;
}

static void moe_cache_session_leave(void * opaque) {
    if (g_session_suppressed > 0) {
        g_session_suppressed--;
        return;
    }
    moe_cache_session * expected = (moe_cache_session *)opaque;
    auto found = std::find_if(
            g_session_stack.rbegin(), g_session_stack.rend(),
            [expected](const moe_cache_scope_frame & frame) {
                return frame.requested == expected;
            });
    if (found == g_session_stack.rend()) {
        return;
    }
    moe_cache_session * active = found->active;
    g_session_stack.erase(std::next(found).base());
    if (active) {
        std::lock_guard<std::mutex> lock(active->mu);
        if (active->active_scopes > 0) {
            active->active_scopes--;
        }
        active->idle_cv.notify_all();
    }
}

static void * moe_cache_begin(
        const char * name, const void * host_base, size_t expert_size,
        int64_t n_in, int64_t n_out, int wtype, int64_t n_expert, int64_t n_tokens) {
    if (g_session_suppressed > 0 || g_session_stack.empty()) {
        return nullptr;
    }
    moe_cache_session * session = g_session_stack.back().active;
    if (!session || !session->config.decode_enabled ||
        session->stopping || session->dormant || !name || !host_base ||
        !strstr(name, "_exps") || n_tokens < 1 ||
        n_tokens > session->config.max_batch ||
        expert_size < session->config.min_expert_bytes ||
        n_in <= 0 || n_out <= 0 || n_expert <= 0 ||
        !moe_cache_type_supported((ggml_type)wtype)) {
        return nullptr;
    }

    const size_t row_size = ggml_row_size((ggml_type)wtype, n_in);
    if (row_size == 0 || (uint64_t)n_out > SIZE_MAX / row_size ||
        expert_size != (size_t)n_out * row_size ||
        (uint64_t)n_expert > SIZE_MAX / expert_size ||
        expert_size > SIZE_MAX / 64) {
        return nullptr;
    }
    const size_t tensor_size = (size_t)n_expert * expert_size;
    moe_cache_scratch scratch_requirements;
    if (!moe_cache_scratch_requirements(
            n_in, n_out, scratch_requirements)) {
        return nullptr;
    }
    int layer = -1;
    bool has_layer = false;
    const moe_cache_partition_key partition =
        moe_cache_partition_for(name, host_base, layer, has_layer);

    moe_cache_device * selected = nullptr;
    int pool_index = -1;
    moe_cache_pool * pool = nullptr;
    {
        std::unique_lock<std::mutex> lock(session->mu);
        if (session->stopping) {
            return nullptr;
        }

        int budget_devices = 0;
        int eligible_devices = 0;
        if ((size_t)session->config.min_slots > SIZE_MAX / expert_size) {
            return nullptr;
        }
        const size_t minimum_pool =
            expert_size * (size_t)session->config.min_slots;
        for (const auto & device_ptr : session->devices) {
            moe_cache_device & candidate = *device_ptr;
            if (candidate.dead.load() ||
                !moe_cache_prepare_budget(*session, candidate)) {
                continue;
            }
            budget_devices++;
            const size_t reserved = moe_cache_scratch_total(
                    candidate.scratch_reserve, &scratch_requirements);
            const size_t slab_limit =
                candidate.budget_limit > reserved
                    ? candidate.budget_limit - reserved : 0;
            if (slab_limit >= minimum_pool) {
                eligible_devices++;
            }
        }
        if (budget_devices == 0 ||
            (session->config.automatic && budget_devices < 2)) {
            if (session->config.prefetch_mb == 0) {
                session->dormant.store(true);
            }
            lock.unlock();
            if (session->config.prefetch_mb == 0) {
                for (const auto & device_ptr : session->devices) {
                    moe_cache_trim_session(*session, device_ptr->physical);
                }
            }
            return nullptr;
        }
        if (session->config.automatic && eligible_devices < 2) {
            return nullptr;
        }

        auto route_weight = [&](moe_cache_device & candidate) -> size_t {
            if (candidate.dead.load() || !candidate.budget_ready) {
                return 0;
            }
            const size_t reserved = moe_cache_scratch_total(
                    candidate.scratch_reserve, &scratch_requirements);
            const size_t slab_limit =
                candidate.budget_limit > reserved
                    ? candidate.budget_limit - reserved : 0;
            if (candidate.allocated_bytes > slab_limit) {
                return 0;
            }
            if (moe_cache_find_pool(candidate, expert_size, wtype) >= 0) {
                return slab_limit;
            }
            const size_t available = slab_limit - candidate.allocated_bytes;
            return available >= minimum_pool ? available : 0;
        };

        bool tensor_override = !has_layer;
        auto find_routed_device = [&](int physical) -> moe_cache_device * {
            for (const auto & device_ptr : session->devices) {
                if (device_ptr->physical == physical) {
                    return device_ptr.get();
                }
            }
            return nullptr;
        };

        auto tensor_route = session->tensor_devices.find(host_base);
        if (tensor_route != session->tensor_devices.end()) {
            selected = find_routed_device(tensor_route->second);
            if (!selected || selected->dead.load() ||
                route_weight(*selected) == 0) {
                session->tensor_devices.erase(tensor_route);
                selected = nullptr;
            } else {
                tensor_override = true;
            }
        }

        if (!selected && has_layer) {
            auto layer_route = session->layer_devices.find(layer);
            if (layer_route != session->layer_devices.end()) {
                selected = find_routed_device(layer_route->second);
                if (!selected || selected->dead.load()) {
                    session->layer_devices.erase(layer);
                    selected = nullptr;
                } else if (route_weight(*selected) == 0) {
                    selected = nullptr;
                    tensor_override = true;
                }
            }
        }

        uint64_t total_weight = 0;
        if (!selected) {
            for (const auto & device_ptr : session->devices) {
                const size_t weight = route_weight(*device_ptr);
                if (weight > UINT64_MAX - total_weight) {
                    total_weight = UINT64_MAX;
                } else {
                    total_weight += weight;
                }
            }
        }
        if (!selected && total_weight == 0) {
            return nullptr;
        }

        const uint64_t hash = has_layer
            ? ((uint64_t)(uint32_t)layer + 1) * 0x9e3779b97f4a7c15ULL
            : moe_cache_name_hash(name);
        if (!selected) {
            const uint64_t target = hash % total_weight;
            uint64_t cumulative = 0;
            for (const auto & device_ptr : session->devices) {
                moe_cache_device & candidate = *device_ptr;
                const size_t weight = route_weight(candidate);
                if (weight == 0) {
                    continue;
                }
                cumulative = weight > UINT64_MAX - cumulative
                    ? UINT64_MAX : cumulative + weight;
                if (target < cumulative) {
                    selected = &candidate;
                    break;
                }
            }
            if (!selected) {
                return nullptr;
            }
            try {
                if (tensor_override) {
                    session->tensor_devices.emplace(host_base, selected->physical);
                } else {
                    session->layer_devices.emplace(layer, selected->physical);
                }
            } catch (...) {
                return nullptr;
            }
        }

        const size_t selected_scratch = moe_cache_scratch_total(
                selected->scratch_reserve, &scratch_requirements);
        if (selected_scratch >= selected->budget_limit ||
            minimum_pool > selected->budget_limit - selected_scratch) {
            return nullptr;
        }
        selected->scratch_reserve.ids = std::max(
                selected->scratch_reserve.ids, scratch_requirements.ids);
        selected->scratch_reserve.act = std::max(
                selected->scratch_reserve.act, scratch_requirements.act);
        selected->scratch_reserve.q8 = std::max(
                selected->scratch_reserve.q8, scratch_requirements.q8);
        selected->scratch_reserve.out = std::max(
                selected->scratch_reserve.out, scratch_requirements.out);

        try {
            pool_index = moe_cache_discover_pool(
                    *session, *selected, host_base, tensor_size, expert_size,
                    wtype, n_expert, partition);
        } catch (...) {
            MOE_CACHE_LOG("[moe-cache] disabled one node after host allocation failure\n");
            return nullptr;
        }
        if (pool_index < 0 || pool_index >= (int)selected->pools.size() ||
            !selected->pools[pool_index]->slab) {
            return nullptr;
        }
        pool = selected->pools[pool_index].get();
        moe_cache_session::active_source * source = nullptr;
        try {
            source = &session->active_sources[host_base];
        } catch (...) {
            return nullptr;
        }
        source->bytes = std::max(source->bytes, tensor_size);
        source->references++;
        session->active_nodes++;
    }

    moe_cache_device & device = *selected;
    std::unique_lock<std::mutex> dispatch_lock;
    try {
        dispatch_lock = std::unique_lock<std::mutex>(
                device.dispatch_mu, std::try_to_lock);
    } catch (...) {
        std::lock_guard<std::mutex> lock(session->mu);
        auto source = session->active_sources.find(host_base);
        if (source != session->active_sources.end() && --source->second.references == 0) {
            session->active_sources.erase(source);
        }
        session->active_nodes--;
        session->idle_cv.notify_all();
        return nullptr;
    }
    if (!dispatch_lock.owns_lock() || device.dead.load()) {
        std::lock_guard<std::mutex> lock(session->mu);
        auto source = session->active_sources.find(host_base);
        if (source != session->active_sources.end() && --source->second.references == 0) {
            session->active_sources.erase(source);
        }
        session->active_nodes--;
        session->idle_cv.notify_all();
        return nullptr;
    }

    std::unique_ptr<moe_cache_node> node(new (std::nothrow) moe_cache_node());
    if (!node) {
        std::lock_guard<std::mutex> lock(session->mu);
        auto source = session->active_sources.find(host_base);
        if (source != session->active_sources.end() && --source->second.references == 0) {
            session->active_sources.erase(source);
        }
        session->active_nodes--;
        session->idle_cv.notify_all();
        return nullptr;
    }
    node->session = session;
    node->device = &device;
    node->pool = pool;
    node->pool_index = pool_index;
    node->host_base = host_base;
    node->expert_size = expert_size;
    node->n_in = n_in;
    node->n_out = n_out;
    node->n_expert = n_expert;
    node->wtype = wtype;
    node->partition = partition;
    node->dispatch_lock = std::move(dispatch_lock);
    return node.release();
}

static int moe_cache_plan(
        void * opaque, const int32_t * ids, int n_ids, int32_t * slot_indices) {
    moe_cache_node * node = (moe_cache_node *)opaque;
    if (!node || !ids || !slot_indices || n_ids < 0 || n_ids > 64 || node->planned) {
        return 0;
    }
    node->planned = true;
    for (int index = 0; index < n_ids; index++) {
        slot_indices[index] = -1;
    }

    moe_cache_session & session = *node->session;
    moe_cache_device & device = *node->device;
    moe_cache_pool & pool = *node->pool;
    int hits = 0;
    int inserts_left = session.config.inserts_per_plan;
    bool wake_worker = false;

    std::unique_lock<std::mutex> lock(session.mu);
    if (session.stopping) {
        return 0;
    }
    for (int index = 0; index < n_ids; index++) {
        const int32_t expert = ids[index];
        if (expert < 0 || expert >= node->n_expert || device.dead.load()) {
            continue;
        }

        const moe_cache_key key{node->host_base, expert};
        auto found = pool.map.find(key);
        if (found != pool.map.end()) {
            moe_cache_slot & slot = pool.slots[found->second];
            if (slot.state == moe_cache_slot_state::valid) {
                slot.readers++;
                moe_cache_lru_remove(pool, found->second);
                moe_cache_lru_push_back(pool, found->second);
                node->pins[node->n_pins++] = {found->second};
                slot_indices[index] = found->second;
                device.hits++;
                hits++;
            } else {
                device.misses++;
            }
            continue;
        }

        device.misses++;
        moe_cache_demand * demand = nullptr;
        try {
            demand = &device.demand_count[key];
        } catch (...) {
            device.insert_skips++;
            continue;
        }
        demand->expert_size = node->expert_size;
        if (demand->count < std::numeric_limits<uint16_t>::max()) {
            demand->count++;
        }
        const int partition_residents =
            moe_cache_partition_size(pool, node->partition);
        const size_t partition_quota =
            moe_cache_partition_quota(pool, node->partition);
        // Free capacity is shared. Once the pool is full, an under-quota layer
        // is admitted quickly so it can reclaim its fair share; a layer that
        // already owns its share must demonstrate repeated demand before it
        // displaces a local resident or a true surplus resident elsewhere.
        const int admit_after = !pool.free_slots.empty() ||
                (size_t)std::max(partition_residents, 0) < partition_quota
            ? session.config.admit_after
            : std::max(session.config.admit_after, session.config.readmit_after);
        if (demand->count < admit_after) {
            device.admission_skips++;
            continue;
        }
        const size_t queue_limit = session.config.queue_mb << 20;
        if (inserts_left <= 0 || (int)device.queue.size() >= session.config.queue_max ||
            node->expert_size > queue_limit - std::min(queue_limit, device.queued_bytes)) {
            device.insert_skips++;
            continue;
        }

        int slot_index = -1;
        if (!pool.free_slots.empty()) {
            slot_index = pool.free_slots.back();
            pool.free_slots.pop_back();
        } else {
            int candidate = -1;
            if ((size_t)std::max(partition_residents, 0) >= partition_quota) {
                const int local = moe_cache_lru_candidate(
                        pool, node->partition, true, false);
                const int surplus = moe_cache_lru_candidate(
                        pool, node->partition, false, true);
                if (local < 0) {
                    candidate = surplus;
                } else if (surplus < 0) {
                    candidate = local;
                } else {
                    candidate = pool.slots[surplus].last_used <
                            pool.slots[local].last_used
                        ? surplus : local;
                }
            } else {
                candidate = moe_cache_lru_candidate(
                        pool, node->partition, false, true);
            }
            if (candidate < 0 && partition_residents > 0) {
                candidate = moe_cache_lru_candidate(
                        pool, node->partition, true, false);
            }
            if (candidate < 0) {
                device.insert_skips++;
                continue;
            }
            const bool reclaimed =
                pool.slots[candidate].partition != node->partition;
            slot_index = candidate;
            moe_cache_slot_reset(pool, slot_index, false);
            device.evictions++;
            if (reclaimed) {
                device.partition_reclaims++;
            } else {
                device.local_evictions++;
            }
        }

        moe_cache_slot & slot = pool.slots[slot_index];
        slot.key = key;
        slot.partition = node->partition;
        slot.generation++;
        slot.readers = 0;
        slot.state = moe_cache_slot_state::copying;
        auto partition = pool.partition_residents.find(node->partition);
        if (partition == pool.partition_residents.end()) {
            moe_cache_slot_reset(pool, slot_index, true);
            device.insert_skips++;
            continue;
        }
        partition->second.residents++;
        try {
            const auto inserted = pool.map.emplace(key, slot_index);
            if (!inserted.second) {
                moe_cache_slot_reset(pool, slot_index, true);
                device.insert_skips++;
                continue;
            }

            const void * source =
                (const char *)node->host_base + (size_t)expert * node->expert_size;
            device.queue.push_back({
                    node->pool_index, slot_index, slot.generation,
                    key, source, node->expert_size});
            device.queued_bytes += node->expert_size;
        } catch (...) {
            moe_cache_slot_reset(pool, slot_index, true);
            device.insert_skips++;
            continue;
        }
        device.inserts++;
        inserts_left--;
        wake_worker = true;
    }
    device.nodes++;
    lock.unlock();
    if (wake_worker) {
        session.cv.notify_all();
    }
    return hits;
}

static int moe_cache_dispatch(
        void * opaque, int wtype, int64_t n_in, int64_t n_out, int n_hits,
        const int32_t * slot_indices, const float * const * act_rows) {
    moe_cache_node * node = (moe_cache_node *)opaque;
    if (!node || !node->planned || node->dispatched || n_hits <= 0 ||
        n_hits > 64 || n_hits != node->n_pins || !slot_indices || !act_rows ||
        wtype != node->wtype || n_in != node->n_in || n_out != node->n_out ||
        n_in > INT_MAX || n_out > INT_MAX ||
        n_in > INT64_MAX - (MATRIX_ROW_PADDING - 1)) {
        return 0;
    }

    moe_cache_session & session = *node->session;
    moe_cache_device & device = *node->device;
    moe_cache_pool & pool = *node->pool;
    if (device.dead.load() || moe_cache_fail(session, "dispatch")) {
        std::lock_guard<std::mutex> lock(session.mu);
        device.dispatch_failures++;
        return 0;
    }

    ggml_cuda_set_device(device.physical);
    if (!device.compute_stream) {
        if (!moe_cache_cuda_ok(device,
                    cudaStreamCreateWithFlags(&device.compute_stream, cudaStreamNonBlocking),
                    "compute stream creation", true)) {
            std::lock_guard<std::mutex> lock(session.mu);
            device.dispatch_failures++;
            return 0;
        }
    }

    bool shared_activation = true;
    for (int index = 1; index < n_hits; index++) {
        if (act_rows[index] != act_rows[0]) {
            shared_activation = false;
            break;
        }
    }
    const int activation_rows = shared_activation ? 1 : n_hits;
    for (int index = 0; index < n_hits; index++) {
        if (slot_indices[index] < 0 || slot_indices[index] >= pool.n_slots ||
            !act_rows[index]) {
            return 0;
        }
    }

    const int64_t padded_n_in =
        ((n_in + MATRIX_ROW_PADDING - 1) / MATRIX_ROW_PADDING) * MATRIX_ROW_PADDING;
    const size_t type_size = ggml_type_size((ggml_type)wtype);
    if (type_size == 0 || node->expert_size % type_size != 0 ||
        ggml_row_size((ggml_type)wtype, n_in) % type_size != 0 ||
        node->expert_size / type_size > INT_MAX ||
        ggml_row_size((ggml_type)wtype, n_in) / type_size > INT_MAX ||
        padded_n_in / QK8_1 > INT_MAX ||
        (uint64_t)activation_rows * (padded_n_in / QK8_1) > INT_MAX ||
        (uint64_t)n_out * n_hits > INT_MAX ||
        (uint64_t)(node->expert_size / type_size) * pool.n_slots > INT_MAX) {
        return 0;
    }

    if (n_in > INT64_MAX / activation_rows ||
        (uint64_t)n_in * activation_rows > SIZE_MAX / sizeof(float) ||
        n_out > INT64_MAX / n_hits ||
        (uint64_t)n_out * n_hits > SIZE_MAX / sizeof(float) ||
        (uint64_t)activation_rows * (padded_n_in / QK8_1) >
            SIZE_MAX / sizeof(block_q8_1)) {
        return 0;
    }
    const size_t ids_bytes = (size_t)n_hits * sizeof(int32_t);
    const size_t act_bytes = (size_t)activation_rows * n_in * sizeof(float);
    const size_t q8_bytes =
        (size_t)activation_rows * (padded_n_in / QK8_1) * sizeof(block_q8_1);
    const size_t out_bytes = (size_t)n_hits * n_out * sizeof(float);

    const size_t desired_caps[] = {
        moe_cache_growth_capacity(device.d_ids_cap, ids_bytes),
        moe_cache_growth_capacity(device.d_act_cap, act_bytes),
        moe_cache_growth_capacity(device.act_q8_cap, q8_bytes),
        moe_cache_growth_capacity(device.d_out_cap, out_bytes),
    };
    size_t scratch_bytes = 0;
    for (size_t capacity : desired_caps) {
        if (capacity == 0 || capacity > SIZE_MAX - scratch_bytes) {
            std::lock_guard<std::mutex> lock(session.mu);
            device.dispatch_failures++;
            return 0;
        }
        scratch_bytes += capacity;
    }
    {
        std::lock_guard<std::mutex> lock(session.mu);
        if (scratch_bytes > device.budget_limit ||
            device.allocated_bytes > device.budget_limit - scratch_bytes) {
            device.dispatch_failures++;
            return 0;
        }
    }

    if (!moe_cache_grow_host(device, (void **)&device.h_ids, device.h_ids_cap,
                             ids_bytes, "ids host allocation") ||
        !moe_cache_grow_device(device, (void **)&device.d_ids, device.d_ids_cap,
                               ids_bytes, "ids device allocation") ||
        !moe_cache_grow_host(device, (void **)&device.h_act, device.h_act_cap,
                             act_bytes, "activation host allocation") ||
        !moe_cache_grow_device(device, (void **)&device.d_act, device.d_act_cap,
                               act_bytes, "activation device allocation") ||
        !moe_cache_grow_device(device, &device.d_act_q8, device.act_q8_cap,
                               q8_bytes, "q8 activation allocation") ||
        !moe_cache_grow_device(device, (void **)&device.d_out, device.d_out_cap,
                               out_bytes, "output device allocation") ||
        !moe_cache_grow_host(device, (void **)&device.h_out, device.h_out_cap,
                             out_bytes, "output host allocation")) {
        std::lock_guard<std::mutex> lock(session.mu);
        device.dispatch_failures++;
        device.dead.store(true);
        return 0;
    }

    for (int index = 0; index < n_hits; index++) {
        device.h_ids[index] = slot_indices[index];
    }
    for (int index = 0; index < activation_rows; index++) {
        memcpy(device.h_act + (size_t)index * n_in,
               act_rows[shared_activation ? 0 : index], n_in * sizeof(float));
    }

    (void)cudaGetLastError();
    bool ok =
        moe_cache_cuda_ok(device, cudaMemcpyAsync(
                device.d_ids, device.h_ids, ids_bytes,
                cudaMemcpyHostToDevice, device.compute_stream), "ids upload", true) &&
        moe_cache_cuda_ok(device, cudaMemcpyAsync(
                device.d_act, device.h_act, act_bytes,
                cudaMemcpyHostToDevice, device.compute_stream), "activation upload", true);
    if (ok) {
        quantize_row_q8_1_cuda(
                device.d_act, nullptr, device.d_act_q8, (ggml_type)wtype,
                n_in, n_in, (int64_t)activation_rows * n_in,
                (int64_t)activation_rows * n_in, padded_n_in,
                activation_rows, 1, 1, device.compute_stream);
        ok = moe_cache_cuda_ok(
                device, cudaPeekAtLastError(), "activation quantization", true);
    }
    if (ok) {
        ggml_cuda_moe_cache_mmv(
                pool.slab, (ggml_type)wtype, (const char *)device.d_act_q8,
                device.d_ids, device.d_out, n_in, n_out, pool.n_slots,
                (int64_t)pool.expert_size, n_hits, activation_rows,
                device.compute_stream);
        ok = moe_cache_cuda_ok(
                device, cudaPeekAtLastError(), "expert matvec launch", true);
    }

    if (!ok) {
        cudaStreamSynchronize(device.compute_stream);
        std::lock_guard<std::mutex> lock(session.mu);
        device.dispatch_failures++;
        return 0;
    }

    node->dispatched = true;
    return 1;
}

static int moe_cache_collect(
        void * opaque, int n_hits, float * const * dst_rows, int64_t n_out) {
    moe_cache_node * node = (moe_cache_node *)opaque;
    if (!node || !node->dispatched || n_hits <= 0 || n_hits > 64 ||
        n_hits != node->n_pins || !dst_rows || n_out != node->n_out) {
        return 0;
    }
    for (int index = 0; index < n_hits; index++) {
        if (!dst_rows[index]) {
            return 0;
        }
    }

    moe_cache_session & session = *node->session;
    moe_cache_device & device = *node->device;
    ggml_cuda_set_device(device.physical);

    bool ok = !device.dead.load();
    if (moe_cache_fail(session, "collect")) {
        ok = false;
    }
    const size_t bytes = (size_t)n_hits * n_out * sizeof(float);
    if (ok) {
        ok = moe_cache_cuda_ok(device, cudaMemcpyAsync(
                device.h_out, device.d_out, bytes,
                cudaMemcpyDeviceToHost, device.compute_stream), "output download", true);
    }
    if (ok) {
        ok = moe_cache_cuda_ok(
                device, cudaStreamSynchronize(device.compute_stream),
                "output synchronization", true);
    } else {
        cudaStreamSynchronize(device.compute_stream);
    }
    node->dispatched = false;

    if (ok) {
        for (int index = 0; index < n_hits; index++) {
            memcpy(dst_rows[index], device.h_out + (size_t)index * n_out,
                   n_out * sizeof(float));
        }
    }

    {
        std::lock_guard<std::mutex> lock(session.mu);
        if (!ok) {
            device.collect_failures++;
        }
        device.collect_calls++;
        moe_cache_maybe_log_stats(session.config, device, true);
    }
    return ok ? 1 : 0;
}

static void moe_cache_end(void * opaque) {
    std::unique_ptr<moe_cache_node> node((moe_cache_node *)opaque);
    if (!node) {
        return;
    }

    if (node->dispatched) {
        ggml_cuda_set_device(node->device->physical);
        moe_cache_cuda_ok(
                *node->device, cudaStreamSynchronize(node->device->compute_stream),
                "end synchronization", true);
        node->dispatched = false;
    }

    moe_cache_session & session = *node->session;
    const bool trim = node->device->dead.load();
    {
        std::lock_guard<std::mutex> lock(session.mu);
        for (int index = 0; index < node->n_pins; index++) {
            const moe_cache_pin & pin = node->pins[index];
            if (pin.slot >= 0 && pin.slot < node->pool->n_slots) {
                moe_cache_slot & slot = node->pool->slots[pin.slot];
                if (slot.readers > 0) {
                    slot.readers--;
                }
            }
        }
        auto source = session.active_sources.find(node->host_base);
        if (source != session.active_sources.end()) {
            if (--source->second.references == 0) {
                session.active_sources.erase(source);
            }
        }
        session.active_nodes--;
        session.idle_cv.notify_all();
    }
    if (trim && node->dispatch_lock.owns_lock()) {
        node->dispatch_lock.unlock();
        moe_cache_trim_session(session, node->device->physical);
    }
}

struct moe_prefill_route {
    int32_t expert;
    int32_t token;
    int32_t route;
};

struct moe_cache_active_guard {
    moe_cache_session * session = nullptr;
    const void * source = nullptr;

    ~moe_cache_active_guard() {
        if (!session) {
            return;
        }
        std::lock_guard<std::mutex> lock(session->mu);
        auto found = session->active_sources.find(source);
        if (found != session->active_sources.end() &&
            --found->second.references == 0) {
            session->active_sources.erase(found);
        }
        session->active_nodes--;
        session->idle_cv.notify_all();
    }
};

static void moe_cache_release_prefetch_buffers(moe_cache_device & device) {
    if (device.prefetch_stream) {
        cudaStreamSynchronize(device.prefetch_stream);
    }
    if (device.compute_stream) {
        cudaStreamSynchronize(device.compute_stream);
    }
    if (device.d_prefetch_weights) {
        cudaFree(device.d_prefetch_weights);
        device.d_prefetch_weights = nullptr;
    }
    if (device.h_prefetch_weights) {
        cudaFreeHost(device.h_prefetch_weights);
        device.h_prefetch_weights = nullptr;
    }
    for (int slot = 0; slot < 2; slot++) {
        if (device.prefetch_ready[slot]) {
            cudaEventDestroy(device.prefetch_ready[slot]);
            device.prefetch_ready[slot] = nullptr;
        }
        if (device.prefetch_consumed[slot]) {
            cudaEventDestroy(device.prefetch_consumed[slot]);
            device.prefetch_consumed[slot] = nullptr;
        }
        device.prefetch_ready_recorded[slot] = false;
        device.prefetch_consumed_recorded[slot] = false;
    }
    device.prefetch_expert_size = 0;
}

static bool moe_cache_prepare_prefetch_buffers(
        moe_cache_session & session, moe_cache_device & device, size_t expert_size) {
    if (expert_size == 0 || expert_size > SIZE_MAX / 2 ||
        expert_size * 2 > (session.config.prefetch_mb << 20)) {
        return false;
    }
    if (device.prefetch_expert_size == expert_size &&
        device.h_prefetch_weights && device.d_prefetch_weights &&
        device.prefetch_stream && device.compute_stream) {
        return true;
    }

    ggml_cuda_set_device(device.physical);
    moe_cache_release_prefetch_buffers(device);

    cudaError_t error = cudaSuccess;
    if (!device.compute_stream) {
        error = cudaStreamCreateWithFlags(&device.compute_stream, cudaStreamNonBlocking);
    }
    if (error == cudaSuccess && !device.prefetch_stream) {
        error = cudaStreamCreateWithFlags(&device.prefetch_stream, cudaStreamNonBlocking);
    }
    if (error == cudaSuccess) {
        error = cudaMallocHost((void **)&device.h_prefetch_weights, expert_size * 2);
    }
    if (error == cudaSuccess) {
        error = cudaMalloc((void **)&device.d_prefetch_weights, expert_size * 2);
    }
    for (int slot = 0; slot < 2 && error == cudaSuccess; slot++) {
        error = cudaEventCreateWithFlags(&device.prefetch_ready[slot], cudaEventDisableTiming);
        if (error == cudaSuccess) {
            error = cudaEventCreateWithFlags(
                    &device.prefetch_consumed[slot], cudaEventDisableTiming);
        }
    }
    if (error != cudaSuccess) {
        (void)cudaGetLastError();
        MOE_CACHE_LOG("[moe-prefetch] CUDA%d cannot allocate two %zu KiB staging slots: %s\n",
                device.physical, expert_size >> 10, cudaGetErrorString(error));
        moe_cache_release_prefetch_buffers(device);
        return false;
    }

    device.prefetch_expert_size = expert_size;
    MOE_CACHE_LOG("[moe-prefetch] CUDA%d enabled: two %zu KiB pinned/GPU expert slots (%zu MiB host budget)\n",
            device.physical, expert_size >> 10, session.config.prefetch_mb);
    return true;
}

static int moe_cache_prefill(
        const char * name, const void * host_base, size_t expert_size,
        int64_t n_in, int64_t n_out, int wtype, int64_t n_expert,
        const void * ids_data, size_t ids_token_stride, size_t ids_route_stride,
        const void * act_data, size_t act_token_stride, size_t act_route_stride,
        void * dst_data, size_t dst_token_stride, size_t dst_route_stride,
        int64_t n_tokens, int64_t n_routes, int64_t n_act_routes) {
    if (g_session_suppressed > 0 || g_session_stack.empty()) {
        return 0;
    }
    moe_cache_session * session = g_session_stack.back().active;
    if (!session || session->config.prefetch_mb == 0 || session->stopping ||
        !name || !host_base || !ids_data || !act_data || !dst_data ||
        !strstr(name, "_exps") || n_tokens <= session->config.max_batch ||
        n_tokens <= 0 || n_routes <= 0 || n_act_routes <= 0 ||
        n_in <= 0 || n_out <= 0 ||
        n_expert <= 0 || n_expert > INT_MAX || n_tokens > INT_MAX ||
        n_routes > INT_MAX || (uint64_t)n_tokens * n_routes > INT_MAX ||
        !moe_cache_type_supported((ggml_type)wtype)) {
        return 0;
    }

    const size_t row_size = ggml_row_size((ggml_type)wtype, n_in);
    if (row_size == 0 || (uint64_t)n_out > SIZE_MAX / row_size ||
        expert_size != (size_t)n_out * row_size ||
        (uint64_t)n_expert > SIZE_MAX / expert_size ||
        expert_size > SIZE_MAX / 2 ||
        expert_size * 2 > (session->config.prefetch_mb << 20) ||
        n_in > INT64_MAX - (MATRIX_ROW_PADDING - 1)) {
        return 0;
    }

    moe_cache_device * device = nullptr;
    const uint64_t hash = moe_cache_name_hash(name);
    {
        std::lock_guard<std::mutex> lock(session->mu);
        for (size_t offset = 0; offset < session->devices.size(); offset++) {
            moe_cache_device * candidate =
                session->devices[(hash + offset) % session->devices.size()].get();
            if (!candidate->dead.load() && !candidate->prefetch_disabled.load()) {
                device = candidate;
                break;
            }
        }
        if (!device) {
            return 0;
        }
        try {
            moe_cache_session::active_source & active =
                session->active_sources[host_base];
            active.bytes = std::max(active.bytes, (size_t)n_expert * expert_size);
            active.references++;
            session->active_nodes++;
        } catch (...) {
            return 0;
        }
    }
    moe_cache_active_guard active_guard{session, host_base};

    std::unique_lock<std::mutex> dispatch_lock(
            device->dispatch_mu, std::try_to_lock);
    if (!dispatch_lock.owns_lock() || device->dead.load() ||
        device->prefetch_disabled.load()) {
        return 0;
    }

    try {
        std::vector<moe_prefill_route> routes;
        routes.reserve((size_t)n_tokens * n_routes);
        for (int64_t token = 0; token < n_tokens; token++) {
            for (int64_t route = 0; route < n_routes; route++) {
                const int32_t expert = *(const int32_t *)(
                        (const char *)ids_data + (size_t)token * ids_token_stride +
                        (size_t)route * ids_route_stride);
                if (expert < 0) {
                    memset((char *)dst_data + (size_t)token * dst_token_stride +
                           (size_t)route * dst_route_stride, 0,
                           (size_t)n_out * sizeof(float));
                    continue;
                }
                if (expert >= n_expert) {
                    return 0;
                }
                routes.push_back({expert, (int32_t)token, (int32_t)route});
            }
        }
        if (routes.empty()) {
            return 1;
        }
        std::sort(routes.begin(), routes.end(), [](const auto & lhs, const auto & rhs) {
            if (lhs.expert != rhs.expert) {
                return lhs.expert < rhs.expert;
            }
            if (lhs.token != rhs.token) {
                return lhs.token < rhs.token;
            }
            return lhs.route < rhs.route;
        });

        std::vector<size_t> groups;
        groups.push_back(0);
        for (size_t index = 1; index < routes.size(); index++) {
            if (routes[index].expert != routes[index - 1].expert) {
                groups.push_back(index);
            }
        }
        groups.push_back(routes.size());

        ggml_cuda_set_device(device->physical);
        if (!moe_cache_prepare_prefetch_buffers(*session, *device, expert_size)) {
            std::lock_guard<std::mutex> lock(session->mu);
            device->prefill_failures++;
            return 0;
        }

        constexpr int max_rows = 64;
        const int64_t padded_n_in =
            ((n_in + MATRIX_ROW_PADDING - 1) / MATRIX_ROW_PADDING) * MATRIX_ROW_PADDING;
        const size_t type_size = ggml_type_size((ggml_type)wtype);
        if (type_size == 0 || expert_size % type_size != 0 ||
            row_size % type_size != 0 || expert_size / type_size > INT_MAX ||
            row_size / type_size > INT_MAX || padded_n_in / QK8_1 > INT_MAX ||
            (uint64_t)max_rows * n_out > INT_MAX ||
            (uint64_t)(expert_size / type_size) * 2 > INT_MAX) {
            return 0;
        }

        const size_t ids_bytes = max_rows * sizeof(int32_t);
        const size_t act_bytes = max_rows * (size_t)n_in * sizeof(float);
        const size_t q8_bytes = max_rows * (size_t)(padded_n_in / QK8_1) *
                                sizeof(block_q8_1);
        const size_t out_bytes = max_rows * (size_t)n_out * sizeof(float);
        if (!moe_cache_grow_host(*device, (void **)&device->h_ids,
                                 device->h_ids_cap, ids_bytes, "prefill ids host allocation") ||
            !moe_cache_grow_device(*device, (void **)&device->d_ids,
                                   device->d_ids_cap, ids_bytes, "prefill ids device allocation") ||
            !moe_cache_grow_host(*device, (void **)&device->h_act,
                                 device->h_act_cap, act_bytes, "prefill activation host allocation") ||
            !moe_cache_grow_device(*device, (void **)&device->d_act,
                                   device->d_act_cap, act_bytes, "prefill activation device allocation") ||
            !moe_cache_grow_device(*device, &device->d_act_q8,
                                   device->act_q8_cap, q8_bytes, "prefill q8 allocation") ||
            !moe_cache_grow_device(*device, (void **)&device->d_out,
                                   device->d_out_cap, out_bytes, "prefill output device allocation") ||
            !moe_cache_grow_host(*device, (void **)&device->h_out,
                                 device->h_out_cap, out_bytes, "prefill output host allocation")) {
            std::lock_guard<std::mutex> lock(session->mu);
            device->prefill_failures++;
            return 0;
        }

        auto prefetch_ok = [&](cudaError_t error, const char * operation) {
            if (error == cudaSuccess) {
                return true;
            }
            (void)cudaGetLastError();
            if (device->error_logs.fetch_add(1) < 8) {
                MOE_CACHE_LOG("[moe-prefetch] CUDA%d %s failed: %s\n",
                        device->physical, operation, cudaGetErrorString(error));
            }
            return false;
        };

        auto load_group = [&](size_t group_index) {
            const int slot = (int)(group_index % 2);
            if (device->prefetch_ready_recorded[slot] &&
                !prefetch_ok(cudaEventSynchronize(device->prefetch_ready[slot]),
                             "staging slot wait")) {
                return false;
            }
            if (device->prefetch_consumed_recorded[slot] &&
                !prefetch_ok(cudaStreamWaitEvent(
                        device->prefetch_stream,
                        device->prefetch_consumed[slot], 0),
                        "weight slot reuse wait")) {
                return false;
            }
            const int32_t expert = routes[groups[group_index]].expert;
            char * stage = device->h_prefetch_weights + (size_t)slot * expert_size;
            char * target = device->d_prefetch_weights + (size_t)slot * expert_size;
            memcpy(stage, (const char *)host_base + (size_t)expert * expert_size,
                   expert_size);
            if (!prefetch_ok(cudaMemcpyAsync(
                    target, stage, expert_size, cudaMemcpyHostToDevice,
                    device->prefetch_stream), "expert upload") ||
                !prefetch_ok(cudaEventRecord(
                    device->prefetch_ready[slot], device->prefetch_stream),
                    "expert-ready record")) {
                return false;
            }
            device->prefetch_ready_recorded[slot] = true;
            return true;
        };

        if (moe_cache_fail(*session, "prefill") || !load_group(0)) {
            std::lock_guard<std::mutex> lock(session->mu);
            device->prefill_failures++;
            return 0;
        }

        bool next_loaded = false;
        long long hidden = 0;
        long long opportunities = 0;
        for (size_t group = 0; group + 1 < groups.size(); group++) {
            const int slot = (int)(group % 2);
            if (!prefetch_ok(cudaStreamWaitEvent(
                    device->compute_stream, device->prefetch_ready[slot], 0),
                    "compute weight wait")) {
                goto prefill_failure;
            }

            const size_t first = groups[group];
            const size_t end = groups[group + 1];
            next_loaded = false;
            bool next_hidden_counted = false;
            for (size_t chunk = first; chunk < end; chunk += max_rows) {
                const int rows = (int)std::min<size_t>(max_rows, end - chunk);
                for (int row = 0; row < rows; row++) {
                    const moe_prefill_route & item = routes[chunk + row];
                    device->h_ids[row] = slot;
                    memcpy(device->h_act + (size_t)row * n_in,
                           (const char *)act_data +
                               (size_t)item.token * act_token_stride +
                               (size_t)(item.route % n_act_routes) * act_route_stride,
                           (size_t)n_in * sizeof(float));
                }

                if (!prefetch_ok(cudaMemcpyAsync(
                        device->d_ids, device->h_ids,
                        (size_t)rows * sizeof(int32_t), cudaMemcpyHostToDevice,
                        device->compute_stream), "prefill ids upload") ||
                    !prefetch_ok(cudaMemcpyAsync(
                        device->d_act, device->h_act,
                        (size_t)rows * n_in * sizeof(float), cudaMemcpyHostToDevice,
                        device->compute_stream), "prefill activation upload")) {
                    goto prefill_failure;
                }
                quantize_row_q8_1_cuda(
                        device->d_act, nullptr, device->d_act_q8,
                        (ggml_type)wtype, n_in, n_in,
                        (int64_t)rows * n_in, (int64_t)rows * n_in,
                        padded_n_in, rows, 1, 1, device->compute_stream);
                if (!prefetch_ok(cudaPeekAtLastError(), "prefill activation quantization")) {
                    goto prefill_failure;
                }
                ggml_cuda_moe_cache_mmv(
                        device->d_prefetch_weights, (ggml_type)wtype,
                        (const char *)device->d_act_q8, device->d_ids,
                        device->d_out, n_in, n_out, 2, (int64_t)expert_size,
                        rows, rows, device->compute_stream);
                if (!prefetch_ok(cudaPeekAtLastError(), "prefill expert matvec")) {
                    goto prefill_failure;
                }

                if (!next_loaded && group + 2 < groups.size()) {
                    if (!load_group(group + 1)) {
                        goto prefill_failure;
                    }
                    next_loaded = true;
                    opportunities++;
                }

                if (!prefetch_ok(cudaMemcpyAsync(
                        device->h_out, device->d_out,
                        (size_t)rows * n_out * sizeof(float),
                        cudaMemcpyDeviceToHost, device->compute_stream),
                        "prefill output download") ||
                    !prefetch_ok(cudaStreamSynchronize(device->compute_stream),
                        "prefill compute synchronization")) {
                    goto prefill_failure;
                }
                if (next_loaded && !next_hidden_counted &&
                    cudaEventQuery(device->prefetch_ready[(group + 1) % 2]) == cudaSuccess) {
                    hidden++;
                    next_hidden_counted = true;
                }
                for (int row = 0; row < rows; row++) {
                    const moe_prefill_route & item = routes[chunk + row];
                    memcpy((char *)dst_data +
                               (size_t)item.token * dst_token_stride +
                               (size_t)item.route * dst_route_stride,
                           device->h_out + (size_t)row * n_out,
                           (size_t)n_out * sizeof(float));
                }
            }

            if (!prefetch_ok(cudaEventRecord(
                    device->prefetch_consumed[slot], device->compute_stream),
                    "weight-consumed record")) {
                goto prefill_failure;
            }
            device->prefetch_consumed_recorded[slot] = true;
            if (group + 2 < groups.size() && !next_loaded &&
                !load_group(group + 1)) {
                goto prefill_failure;
            }
        }

        {
            std::lock_guard<std::mutex> lock(session->mu);
            device->prefill_nodes++;
            device->prefill_rows += (long long)routes.size();
            device->prefill_overlap_opportunities += opportunities;
            device->prefill_fully_hidden += hidden;
            moe_cache_maybe_log_stats(session->config, *device, false);
            if (device->prefill_overlap_opportunities >= 32 &&
                device->prefill_fully_hidden == 0) {
                device->prefetch_disabled.store(true);
                MOE_CACHE_LOG("[moe-prefetch] CUDA%d disabled: no transfer was fully hidden in %lld measured opportunities\n",
                        device->physical, device->prefill_overlap_opportunities);
            }
        }
        return 1;

prefill_failure:
        cudaStreamSynchronize(device->compute_stream);
        cudaStreamSynchronize(device->prefetch_stream);
        {
            std::lock_guard<std::mutex> lock(session->mu);
            device->prefill_failures++;
        }
        return 0;
    } catch (...) {
        std::lock_guard<std::mutex> lock(session->mu);
        device->prefill_failures++;
        return 0;
    }
}

static void moe_cache_invalidate_session(
        moe_cache_session & session, const void * base, size_t size) {
    std::unique_lock<std::mutex> lock(session.mu);
    session.tensor_devices.erase(base);
    for (auto & device_ptr : session.devices) {
        moe_cache_cancel_queue_locked(*device_ptr, base, size, false);
    }

    session.idle_cv.wait(lock, [&] {
        for (const auto & active : session.active_sources) {
            if (active.second.references > 0 &&
                moe_cache_ranges_overlap(active.first, active.second.bytes, base, size)) {
                return false;
            }
        }
        for (const auto & device_ptr : session.devices) {
            if (device_ptr->inflight &&
                moe_cache_ranges_overlap(
                    device_ptr->inflight_source, device_ptr->inflight_bytes, base, size)) {
                return false;
            }
        }
        return true;
    });

    for (auto & device_ptr : session.devices) {
        moe_cache_cancel_queue_locked(*device_ptr, base, size, false);
        moe_cache_device & device = *device_ptr;
        for (auto & pool_ptr : device.pools) {
            moe_cache_pool & pool = *pool_ptr;
            for (int index = 0; index < pool.n_slots; index++) {
                moe_cache_slot & slot = pool.slots[index];
                const void * source = slot.key.expert >= 0
                    ? (const char *)slot.key.tensor +
                        (size_t)slot.key.expert * pool.expert_size
                    : nullptr;
                if (slot.state != moe_cache_slot_state::free &&
                    moe_cache_ranges_overlap(source, pool.expert_size, base, size)) {
                    moe_cache_slot_reset(pool, index, true);
                }
            }
        }
        for (auto it = device.seen_tensors.begin(); it != device.seen_tensors.end();) {
            if (moe_cache_ranges_overlap(it->first, it->second.bytes, base, size)) {
                session.tensor_devices.erase(it->first);
                for (moe_cache_shape & shape : device.shapes) {
                    if (shape.expert_size == it->second.expert_size &&
                        shape.wtype == it->second.wtype) {
                        shape.n_tensors = std::max<int64_t>(shape.n_tensors - 1, 0);
                        if (shape.n_tensors == 0 && shape.pool < 0) {
                            shape.finished = false;
                        }
                        break;
                    }
                }
                it = device.seen_tensors.erase(it);
                device.stable_visits = 0;
            } else {
                ++it;
            }
        }
        for (auto & pool_ptr : device.pools) {
            moe_cache_pool & pool = *pool_ptr;
            for (auto partition = pool.partition_residents.begin();
                 partition != pool.partition_residents.end();) {
                bool still_seen = false;
                if (partition->second.residents == 0) {
                    for (const auto & tensor : device.seen_tensors) {
                        if (tensor.second.expert_size == pool.expert_size &&
                            tensor.second.wtype == pool.wtype &&
                            tensor.second.partition == partition->first) {
                            still_seen = true;
                            break;
                        }
                    }
                }
                if (partition->second.residents == 0 && !still_seen) {
                    partition = pool.partition_residents.erase(partition);
                } else {
                    ++partition;
                }
            }
            moe_cache_rebalance_partitions(pool);
        }
        for (auto it = device.demand_count.begin(); it != device.demand_count.end();) {
            const void * source = it->first.expert >= 0
                ? (const char *)it->first.tensor +
                    (size_t)it->first.expert * it->second.expert_size
                : nullptr;
            if (moe_cache_ranges_overlap(
                    source, it->second.expert_size, base, size)) {
                it = device.demand_count.erase(it);
            } else {
                ++it;
            }
        }
    }
}

static void moe_cache_invalidate(const void * base, size_t size) {
    if (!base || size == 0 ||
        g_session_count.load(std::memory_order_acquire) == 0) {
        return;
    }
    std::lock_guard<std::mutex> registry_lock(g_registry_mu);
    for (moe_cache_session * session : g_sessions) {
        moe_cache_invalidate_session(*session, base, size);
    }
}

static size_t moe_cache_trim_session(
        moe_cache_session & session, int physical_device) {
    moe_cache_device * selected = nullptr;
    for (auto & device_ptr : session.devices) {
        if (device_ptr->physical == physical_device) {
            selected = device_ptr.get();
            break;
        }
    }
    if (!selected) {
        return 0;
    }

    std::unique_lock<std::mutex> dispatch_lock(selected->dispatch_mu);
    std::unique_lock<std::mutex> lock(session.mu);
    selected->dead.store(true);
    moe_cache_cancel_queue_locked(*selected, nullptr, 0, true);
    session.cv.notify_all();
    session.idle_cv.wait(lock, [&] {
        return !selected->inflight;
    });

    size_t freed = 0;
    for (const auto & pool_ptr : selected->pools) {
        if (pool_ptr->slab) {
            freed += (size_t)pool_ptr->n_slots * pool_ptr->expert_size;
        }
    }
    freed += selected->d_out_cap + selected->d_act_cap +
             selected->act_q8_cap + selected->d_ids_cap;
    if (selected->d_prefetch_weights) {
        freed += 2 * selected->prefetch_expert_size;
    }
    lock.unlock();
    moe_cache_free_device(*selected);

    if (freed > 0) {
        MOE_CACHE_LOG("[moe-cache] CUDA%d trimmed %zu MiB after a cache failure or allocator pressure\n",
                physical_device, freed >> 20);
    }
    return freed;
}

extern "C" size_t ggml_moe_cache_trim(int device) {
    if (g_session_count.load(std::memory_order_acquire) == 0) {
        return 0;
    }
    size_t freed = 0;
    std::lock_guard<std::mutex> registry_lock(g_registry_mu);
    for (moe_cache_session * session : g_sessions) {
        freed += moe_cache_trim_session(*session, device);
    }
    return freed;
}

void ggml_moe_cache_register(const void * owner) {
    if (ggml_moe_cache.owner && ggml_moe_cache.owner != owner) {
        return;
    }
    ggml_moe_cache.owner = owner;
    ggml_moe_cache.session_create = moe_cache_session_create;
    ggml_moe_cache.session_destroy = moe_cache_session_destroy;
    ggml_moe_cache.session_configure = moe_cache_session_configure;
    ggml_moe_cache.session_enter = moe_cache_session_enter;
    ggml_moe_cache.session_leave = moe_cache_session_leave;
    ggml_moe_cache.begin = moe_cache_begin;
    ggml_moe_cache.plan = moe_cache_plan;
    ggml_moe_cache.dispatch = moe_cache_dispatch;
    ggml_moe_cache.collect = moe_cache_collect;
    ggml_moe_cache.end = moe_cache_end;
    ggml_moe_cache.prefill = moe_cache_prefill;
    ggml_moe_cache.invalidate = moe_cache_invalidate;
}

#endif
