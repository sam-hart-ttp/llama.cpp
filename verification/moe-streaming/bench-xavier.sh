#!/usr/bin/env bash
#
# MoE-streaming benchmark harness for the Jetson AGX Xavier.
#
# Runs the same matched-comparison ladder used for the Thor validation
# (../../audit/thor-validation-2026-08-22.md) against a MoE GGUF, but sized
# for Xavier's much smaller unified-memory budget (~14 GiB usable RAM,
# SM 7.2, CUDA 11.4). Unlike Thor, a checkpoint here can exceed physical RAM
# and still load via mmap, backed by the NVMe drive; this harness treats
# that as a valid (slower) configuration rather than a failure, since
# demand-paged CPU experts are exactly what the streaming path is for.
#
# Usage:
#   bench-xavier.sh <model.gguf> [decode-cache-mib] [prefill-mib] [n-tokens]
#
# Environment:
#   BENCH_BUILD_DIR   build dir (default build-xavier)
#   BENCH_THREADS     CPU threads (default 6, leaves 2 cores for CUDA/dispatch)
#   BENCH_PROMPT      prompt text (default matches merge-gate.sh)
#   BENCH_TIMEOUT     per-run timeout, seconds (default 1800)
#   BENCH_NGL_FULL    -ngl value for the "full CUDA" arm (default 99)
#   BENCH_SKIP_FULL   set to 1 to skip the full-CUDA arm (checkpoint too big)
#   BENCH_CTX         context size (default 512; a short A/B run doesn't
#                     need the model's full trained context, and its KV
#                     cache can be several GiB on Xavier's tight budget)

set -u

model="${1:?usage: bench-xavier.sh <model.gguf> [decode-cache-mib] [prefill-mib] [n-tokens]}"
cache_mib="${2:-128}"
prefill_mib="${3:-16}"
n_tokens="${4:-128}"

repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$repo_root" || exit 1

build_dir="${BENCH_BUILD_DIR:-build-xavier}"
threads="${BENCH_THREADS:-6}"
prompt="${BENCH_PROMPT:-The purpose of a bounded expert cache is}"
timeout_s="${BENCH_TIMEOUT:-1800}"
ngl_full="${BENCH_NGL_FULL:-99}"
skip_full="${BENCH_SKIP_FULL:-0}"
ctx_size="${BENCH_CTX:-512}"

bin="$build_dir/bin/llama-completion"
if [ ! -x "$bin" ]; then
    echo "error: $bin not built (run cmake --build $build_dir)" >&2
    exit 1
fi
if [ ! -r "$model" ]; then
    echo "error: model not readable: $model" >&2
    exit 1
fi

model_bytes=$(stat -c%s "$model" 2>/dev/null || stat -f%z "$model")
mem_total_kib=$(awk '/MemTotal/ {print $2}' /proc/meminfo)

log_dir="$(mktemp -d)"
echo "logs kept in: $log_dir"

# poll_peak_rss <pid> <outfile>
# Samples VmHWM (kernel-maintained high-water RSS) until the pid exits.
# VmHWM only increases, so the last successful read is the peak.
poll_peak_rss() {
    local pid="$1" outfile="$2" hwm=0
    while [ -r "/proc/$pid/status" ]; do
        v=$(awk '/VmHWM/ {print $2}' "/proc/$pid/status" 2>/dev/null)
        [ -n "$v" ] && hwm="$v"
        sleep 0.5
    done
    echo "$hwm" > "$outfile"
}

# run_case <name> <extra llama-completion args...>
run_case() {
    local name="$1"; shift
    local log="$log_dir/$name.log"
    local hwm_file="$log_dir/$name.hwm"

    # Not wrapped in `timeout`: timeout forks the target as a child rather
    # than exec'ing it, so $! would be the wrapper's pid and VmHWM would
    # only ever reflect the tiny supervisor process, not the workload.
    "$bin" -m "$model" -p "$prompt" -n "$n_tokens" -c "$ctx_size" \
        -t "$threads" -s 1 -fa on -no-cnv "$@" > "$log" 2>&1 &
    local pid=$!
    ( sleep "$timeout_s"; kill -TERM "$pid" 2>/dev/null ) &
    local watchdog=$!
    poll_peak_rss "$pid" "$hwm_file" &
    local watcher=$!
    wait "$pid"
    local exit_code=$?
    kill "$watchdog" 2>/dev/null
    wait "$watcher" 2>/dev/null

    local peak_kib peak_mib prompt_tps decode_tps cache_line
    peak_kib=$(cat "$hwm_file" 2>/dev/null || echo 0)
    peak_mib=$(awk -v k="$peak_kib" 'BEGIN { printf "%.0f", k/1024 }')
    prompt_tps=$(grep 'prompt eval time' "$log" | tail -1 | sed -n 's/.*(\s*[0-9.]* ms per token,\s*\([0-9.]*\) tokens per second).*/\1/p')
    decode_tps=$(grep 'eval time' "$log" | grep -v 'prompt eval time' | tail -1 | sed -n 's/.*(\s*[0-9.]* ms per token,\s*\([0-9.]*\) tokens per second).*/\1/p')
    cache_line=$(grep '\[moe-cache\].*hits=[0-9]*/[0-9]*' "$log" | tail -1)

    printf '%-32s exit=%-4s prompt_tps=%-8s decode_tps=%-8s peak_rss_mib=%-8s %s\n' \
        "$name" "$exit_code" "${prompt_tps:-n/a}" "${decode_tps:-n/a}" "$peak_mib" "${cache_line:-}"

    if [ "$exit_code" -ne 0 ]; then
        echo "  -- last 15 log lines --"
        tail -15 "$log" | sed 's/^/  /'
    fi
}

echo "moe-streaming Xavier benchmark"
echo "repo:        $repo_root"
echo "build dir:   $build_dir"
echo "model:       $model ($((model_bytes / 1024 / 1024)) MiB)"
echo "MemTotal:    $((mem_total_kib / 1024)) MiB"
echo "decode cache: ${cache_mib} MiB   prefill: ${prefill_mib} MiB   tokens: $n_tokens   threads: $threads   ctx: $ctx_size"
echo

if [ "$skip_full" != "1" ]; then
    run_case "full-cuda"            -ngl "$ngl_full"
else
    echo "full-cuda                       skipped (BENCH_SKIP_FULL=1)"
fi

run_case "cpu-experts-baseline"     -cmoe --no-repack
run_case "cpu-experts-decode-cache" -cmoe --moe-cache-mib "$cache_mib" --moe-cache-stats
run_case "cpu-experts-prefill-only" -cmoe --moe-prefetch-mib "$prefill_mib" --moe-cache-stats
run_case "cpu-experts-cache+prefill" -cmoe --moe-cache-mib "$cache_mib" --moe-prefetch-mib "$prefill_mib" --moe-cache-stats

echo
echo "raw logs: $log_dir"
