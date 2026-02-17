#!/bin/bash
# =============================================================================
# 08_benchmark_migraphx.sh - Benchmark ONNX model with MIGraphX EP
# =============================================================================
# Usage: ./08_benchmark_migraphx.sh <model_dir> [options]
#
# Benchmarks inference performance using ONNX Runtime with MIGraphX EP.
# Wraps benchmark_migraphx.py with shell-friendly interface.
#
# Options:
#   -n, --iterations <n>   Number of benchmark iterations (default: 100)
#   -w, --warmup <n>       Number of warmup iterations (default: 5)
#   -s, --seq-length <n>   Input sequence length (new tokens, default: 1)
#   -k, --kv-length <n>    KV cache length (context tokens, default: 0)
#   --exhaustive           Enable exhaustive tuning
#   --offload-copy         Use CPU memory during compilation
#   --no-cache             Disable model caching
#   -v, --verbose          Enable verbose logging
#   -q, --quiet            Minimal output, only show final results
#   --help                 Show this help
#
# Environment Variables:
#   ITERATIONS=<n>         Override default iterations
#   WARMUP=<n>             Override default warmup
#   SEQ_LENGTH=<n>         Override default sequence length
#   KV_LENGTH=<n>          Override default KV cache length
#
# Examples:
#   ./08_benchmark_migraphx.sh ./Llama3.1-8B-Instruct/onnx
#   ./08_benchmark_migraphx.sh ./onnx -n 500 -s 1 -k 512
#   ./08_benchmark_migraphx.sh ./onnx --seq-length 128 --quiet
# =============================================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Environment defaults
ITERATIONS="${ITERATIONS:-100}"
WARMUP="${WARMUP:-5}"
SEQ_LENGTH="${SEQ_LENGTH:-1}"
KV_LENGTH="${KV_LENGTH:-0}"

# Parse arguments
POSITIONAL=()
EXHAUSTIVE=false
OFFLOAD_COPY=false
NO_CACHE=false
VERBOSE=false
QUIET=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -n|--iterations)
            ITERATIONS="$2"
            shift 2
            ;;
        -w|--warmup)
            WARMUP="$2"
            shift 2
            ;;
        -s|--seq-length)
            SEQ_LENGTH="$2"
            shift 2
            ;;
        -k|--kv-length)
            KV_LENGTH="$2"
            shift 2
            ;;
        --exhaustive)
            EXHAUSTIVE=true
            shift
            ;;
        --offload-copy)
            OFFLOAD_COPY=true
            shift
            ;;
        --no-cache)
            NO_CACHE=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -q|--quiet)
            QUIET=true
            shift
            ;;
        --help|-h)
            head -35 "$0" | tail -32
            exit 0
            ;;
        -*)
            echo "Unknown option: $1"
            exit 1
            ;;
        *)
            POSITIONAL+=("$1")
            shift
            ;;
    esac
done

set -- "${POSITIONAL[@]}"
MODEL_DIR="${1:?Usage: $0 <model_dir> [options]}"

if [ ! -d "$MODEL_DIR" ]; then
    # Check if it's a direct path to model.onnx
    if [ -f "$MODEL_DIR" ]; then
        MODEL_DIR="$(dirname "$MODEL_DIR")"
    else
        echo "Error: Directory not found: $MODEL_DIR"
        exit 1
    fi
fi

# Verify benchmark script exists
BENCH_SCRIPT="$SCRIPT_DIR/benchmark_migraphx.py"
if [ ! -f "$BENCH_SCRIPT" ]; then
    echo "Error: benchmark_migraphx.py not found in $SCRIPT_DIR"
    exit 1
fi

# Build Python arguments
PYTHON_ARGS="$MODEL_DIR"
PYTHON_ARGS="$PYTHON_ARGS --iterations $ITERATIONS"
PYTHON_ARGS="$PYTHON_ARGS --warmup $WARMUP"
PYTHON_ARGS="$PYTHON_ARGS --seq-length $SEQ_LENGTH"
PYTHON_ARGS="$PYTHON_ARGS --kv-length $KV_LENGTH"

[ "$EXHAUSTIVE" = true ] && PYTHON_ARGS="$PYTHON_ARGS --exhaustive-tune"
[ "$OFFLOAD_COPY" = true ] && PYTHON_ARGS="$PYTHON_ARGS --offload-copy"
[ "$NO_CACHE" = true ] && PYTHON_ARGS="$PYTHON_ARGS --no-cache"
[ "$VERBOSE" = true ] && PYTHON_ARGS="$PYTHON_ARGS --verbose"
[ "$QUIET" = true ] && PYTHON_ARGS="$PYTHON_ARGS --quiet"

# Run benchmark
exec python3 "$BENCH_SCRIPT" $PYTHON_ARGS
