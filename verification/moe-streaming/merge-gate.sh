#!/usr/bin/env bash
#
# Post-merge gate for the moe-streaming-research branch.
#
# Run this after every merge from upstream/master. It rebuilds the affected
# targets, runs the focused GPU regression, checks that the public controls are
# still advertised, checks the upstream invariant TAG sites for drift, and
# asserts a decode hit-rate floor.
#
# The hit-rate floor is the only stage that can detect scheduler placement
# drift. test-moe-cache drives the provider API directly, so it stays green
# even if expert MUL_MAT_ID nodes stop being routed to the CPU backend; in that
# case the cache is simply never called and streaming is silently lost.
#
# Usage:
#   verification/moe-streaming/merge-gate.sh [build-dir]
#
# Environment:
#   MOE_GATE_MODEL         path to a MoE GGUF. Without it the hit-rate floor
#                          stage reports SKIP and the gate exit code reflects
#                          the remaining stages only.
#   MOE_GATE_MIN_HITRATE   minimum decode hit rate, percent (default 20)
#   MOE_GATE_CACHE_MIB     decode cache budget, MiB (default 768)
#   MOE_GATE_TOKENS        decode tokens to time (default 256)
#   MOE_GATE_NGL           layers to offload to VRAM (default 0)
#   MOE_GATE_THREADS       CPU threads (default 6)
#   MOE_GATE_CUDA_DEVICE   CUDA_VISIBLE_DEVICES value (default 0)

set -u

build_dir="${1:-build-moe-cuda9}"
repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$repo_root" || exit 1

min_hitrate="${MOE_GATE_MIN_HITRATE:-20}"
cache_mib="${MOE_GATE_CACHE_MIB:-768}"
tokens="${MOE_GATE_TOKENS:-256}"
ngl="${MOE_GATE_NGL:-0}"
threads="${MOE_GATE_THREADS:-6}"
cuda_device="${MOE_GATE_CUDA_DEVICE:-0}"

# Number of [TAG_MUL_MAT_ID_CUDA_GRAPHS] sites upstream maintains. Bump this
# only after reading each new site and confirming it does not change where
# expert MUL_MAT_ID nodes execute.
expected_tag_sites=3

log_dir="$(mktemp -d)"
trap 'rm -rf "$log_dir"' EXIT

failures=0
skips=0

report() {
    # report <status> <stage> [detail]
    printf '%-5s %-22s %s\n' "$1" "$2" "${3:-}"
}

fail() {
    failures=$((failures + 1))
    report FAIL "$1" "${2:-}"
}

skip() {
    skips=$((skips + 1))
    report SKIP "$1" "${2:-}"
}

echo "moe-streaming merge gate"
echo "repo:      $repo_root"
echo "build dir: $build_dir"
echo "head:      $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo

# ---------------------------------------------------------------------------
# Stage 1: build
#
# llama-cli is deliberately not built. Since upstream PR #17824 the CLI is a
# client of the in-process server stack (llama-cli-impl links llama-server-impl)
# and PR #18670 gated it behind LLAMA_BUILD_SERVER, which also pulls in
# tools/ui. llama-completion is the ungated non-interactive generation tool and
# links only llama-common and llama, so it is what this gate uses.
# ---------------------------------------------------------------------------

targets="test-moe-cache test-arg-parser llama-completion"
if cmake --build "$build_dir" --target $targets -j "$threads" \
        > "$log_dir/build.log" 2>&1; then
    report OK build "$targets"
else
    fail build "see output below"
    tail -40 "$log_dir/build.log"
fi

# ---------------------------------------------------------------------------
# Stage 2: focused GPU regression
# ---------------------------------------------------------------------------

if [ -x "$build_dir/bin/test-moe-cache" ]; then
    CUDA_VISIBLE_DEVICES="$cuda_device" "$build_dir/bin/test-moe-cache" \
        > "$log_dir/gpu.log" 2>&1
    gpu_exit=$?
    n_ok=$(grep -c ': OK$' "$log_dir/gpu.log")
    n_skip=$(grep -c ': SKIP' "$log_dir/gpu.log")
    if [ "$gpu_exit" -eq 0 ]; then
        report OK test-moe-cache "$n_ok OK, $n_skip SKIP"
    else
        fail test-moe-cache "exit $gpu_exit, $n_ok OK, $n_skip SKIP"
        grep -vE ': (OK|SKIP)' "$log_dir/gpu.log" | tail -20
    fi
else
    fail test-moe-cache "binary not built"
fi

# ---------------------------------------------------------------------------
# Stage 3: argument parser
# ---------------------------------------------------------------------------

if [ -x "$build_dir/bin/test-arg-parser" ]; then
    if "$build_dir/bin/test-arg-parser" > "$log_dir/arg.log" 2>&1; then
        report OK test-arg-parser
    else
        # The upstream suite ends on a network download case. That failure is
        # pre-existing and unrelated, but the MoE assertions run before it.
        if grep -q 'cannot make GET request' "$log_dir/arg.log"; then
            report OK test-arg-parser "stopped at pre-existing network case"
        else
            fail test-arg-parser "see output below"
            tail -20 "$log_dir/arg.log"
        fi
    fi
else
    fail test-arg-parser "binary not built"
fi

# ---------------------------------------------------------------------------
# Stage 4: public controls still advertised
# ---------------------------------------------------------------------------

if [ -x "$build_dir/bin/llama-completion" ]; then
    "$build_dir/bin/llama-completion" --help > "$log_dir/help.log" 2>&1
    missing=""
    for flag in --moe-cache-mib --moe-prefetch-mib --moe-cache-stats \
                --moe-cache-stats-interval; do
        grep -q -- "$flag" "$log_dir/help.log" || missing="$missing $flag"
    done
    if [ -z "$missing" ]; then
        report OK moe-flags "4 controls advertised"
    else
        fail moe-flags "missing:$missing"
    fi
else
    fail moe-flags "llama-completion not built"
fi

# ---------------------------------------------------------------------------
# Stage 5: upstream invariant TAG sites
#
# Upstream marks cross-cutting CUDA invariants with greppable TAG_ comments.
# TAG_MUL_MAT_ID_CUDA_GRAPHS guards which MUL_MAT_ID dispatch paths permit CUDA
# graph capture. Our streaming path is not inside any captured graph, but that
# holds only while expert nodes execute on the CPU backend, so a change in
# these sites is worth reading by hand before trusting a merge.
# ---------------------------------------------------------------------------

tag_sites=$(grep -rn 'TAG_MUL_MAT_ID_CUDA_GRAPHS' ggml/src/ggml-cuda/ 2>/dev/null)
n_tag_sites=$(printf '%s\n' "$tag_sites" | grep -c 'TAG_MUL_MAT_ID_CUDA_GRAPHS')
if [ "$n_tag_sites" -eq "$expected_tag_sites" ]; then
    report OK tag-sites "$n_tag_sites sites, as recorded"
else
    fail tag-sites "$n_tag_sites sites, expected $expected_tag_sites"
    printf '%s\n' "$tag_sites"
    echo "  Read each site, confirm expert MUL_MAT_ID placement is unchanged,"
    echo "  then update expected_tag_sites in this script."
fi

# ---------------------------------------------------------------------------
# Stage 6: decode hit-rate floor
#
# This is the placement-drift assertion. hits=H/T in the teardown stats line
# counts routed expert rows the cache was asked about. T == 0 means the cache
# was never consulted, which is exactly the silent failure test-moe-cache
# cannot see.
# ---------------------------------------------------------------------------

model="${MOE_GATE_MODEL:-}"
if [ -z "$model" ]; then
    skip hit-rate-floor "set MOE_GATE_MODEL to a MoE GGUF"
elif [ ! -r "$model" ]; then
    skip hit-rate-floor "MOE_GATE_MODEL not readable: $model"
elif [ ! -x "$build_dir/bin/llama-completion" ]; then
    fail hit-rate-floor "llama-completion not built"
else
    CUDA_VISIBLE_DEVICES="$cuda_device" "$build_dir/bin/llama-completion" \
        -m "$model" \
        -p "The purpose of a bounded expert cache is" \
        -n "$tokens" -ngl "$ngl" -t "$threads" -s 1 -no-cnv \
        --moe-cache-mib "$cache_mib" --moe-cache-stats \
        > "$log_dir/decode.log" 2>&1
    decode_exit=$?

    stats_line=$(grep '\[moe-cache\].*hits=[0-9][0-9]*/[0-9][0-9]*' "$log_dir/decode.log" | tail -1)

    if [ "$decode_exit" -ne 0 ]; then
        fail hit-rate-floor "llama-completion exit $decode_exit"
        tail -20 "$log_dir/decode.log"
    elif [ -z "$stats_line" ]; then
        fail hit-rate-floor "no [moe-cache] stats line; provider never ran"
        tail -20 "$log_dir/decode.log"
    else
        hits=$(printf '%s\n' "$stats_line" | sed -n 's/.*hits=\([0-9]*\)\/\([0-9]*\).*/\1/p')
        total=$(printf '%s\n' "$stats_line" | sed -n 's/.*hits=\([0-9]*\)\/\([0-9]*\).*/\2/p')
        if [ -z "$total" ]; then
            fail hit-rate-floor "could not parse: $stats_line"
        elif [ "$total" -eq 0 ]; then
            fail hit-rate-floor \
                "0 routed rows reached the cache; expert MUL_MAT_ID placement drifted"
            printf '  %s\n' "$stats_line"
        else
            rate=$(awk -v h="$hits" -v t="$total" 'BEGIN { printf "%.1f", 100.0*h/t }')
            below=$(awk -v r="$rate" -v f="$min_hitrate" 'BEGIN { print (r < f) ? 1 : 0 }')
            if [ "$below" -eq 1 ]; then
                fail hit-rate-floor "$rate% hits, floor $min_hitrate%"
                printf '  %s\n' "$stats_line"
            else
                report OK hit-rate-floor "$rate% hits over $total rows, floor $min_hitrate%"
            fi
        fi
    fi
fi

# ---------------------------------------------------------------------------

echo
if [ "$failures" -eq 0 ]; then
    echo "gate: PASS ($skips skipped)"
    exit 0
fi
echo "gate: FAIL ($failures failed, $skips skipped)"
exit 1
