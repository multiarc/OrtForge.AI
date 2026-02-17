#!/bin/bash
# =============================================================================
# 09_run_inference_test.sh - Test inference with ONNX Runtime
# =============================================================================
# Usage: ./09_run_inference_test.sh <model_dir> [provider] [options]
# 
# Runs text generation to verify the model works correctly.
# Uses autoregressive generation with growing KV cache.
#
# Providers:
#   MIGraphXExecutionProvider  - AMD GPU with MIGraphX (default)
#   ROCMExecutionProvider      - AMD GPU with ROCm
#   CUDAExecutionProvider      - NVIDIA GPU
#   CPUExecutionProvider       - CPU fallback
#
# Options:
#   --prompt <text>      Custom prompt (default: "What is 2+2?")
#   --seq-length <n>     Static input sequence length (default: 256)
#                        Used for BOTH prefill and decode stages.
#                        Inputs are left-padded to this size.
#   --temperature <f>    Sampling temperature (default: 0.0 = greedy)
#   --verbose            Enable verbose ORT logging
#   --no-cache           Disable model caching
#   --exhaustive         Enable exhaustive tuning
#   --offload-copy       Use CPU memory during compilation
#   --help               Show this help
#
# KV Cache Strategy (FULLY STATIC shapes):
#   ALL shapes are FIXED to avoid MIGraphX recompilation and
#   hipHostRegister failures on small arrays.
#   
#   Fixed shapes:
#   BENCHMARK-COMPATIBLE SHAPES (the only shapes that work):
#   input=(1, 1), attn=(1, 257), kv=(1, h, 256, d)
#   Any other shape triggers hipHostRegister failures in MIGraphX.
#   Prefill is slow (1 token/step) but decode matches benchmark speed.
#   and copy it into the STATIC buffer at position filled_kv.
#
# Environment Variables:
#   VERBOSE=true           Enable verbose ORT + MIGraphX + HIP logging
#   MIGRAPHX_FP16=1        Enable FP16 mode (default: disabled for pre-FP16 models)
#   MIGRAPHX_SAVE_MODEL=1  Save compiled model
#
# Examples:
#   ./09_run_inference_test.sh ./Llama3.1-8B-Instruct/onnx
#   ./09_run_inference_test.sh ./onnx --prompt "Explain quantum computing"
#   ./09_run_inference_test.sh ./onnx --seq-length 256 --temperature 0.7
# =============================================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Parse arguments
POSITIONAL=()
PROMPT="What is 2+2?"
SEQ_LENGTH=256  # Default bucket size (max_output = seq_length)
TEMPERATURE=0.0
VERBOSE=false
NO_CACHE=false
EXHAUSTIVE=false
OFFLOAD_COPY=true  # Default to offload for large models
MIGRAPHX_FP16="${MIGRAPHX_FP16:-0}"
MIGRAPHX_SAVE="${MIGRAPHX_SAVE_MODEL:-1}"

while [[ $# -gt 0 ]]; do
    case $1 in
        --prompt)
            PROMPT="$2"
            shift 2
            ;;
        --seq-length)
            SEQ_LENGTH="$2"
            shift 2
            ;;
        --temperature)
            TEMPERATURE="$2"
            shift 2
            ;;
        --verbose|-v)
            VERBOSE=true
            shift
            ;;
        --no-cache)
            NO_CACHE=true
            shift
            ;;
        --exhaustive)
            EXHAUSTIVE=true
            shift
            ;;
        --offload-copy)
            OFFLOAD_COPY=true
            shift
            ;;
        --no-offload-copy)
            OFFLOAD_COPY=false
            shift
            ;;
        --help|-h)
            head -40 "$0" | tail -37
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
MODEL_DIR="${1:?Usage: $0 <model_dir> [provider] [options]}"
# MIGraphX provider - we use GPU OrtValues to avoid hipHostRegister issues
PROVIDER="${2:-MIGraphXExecutionProvider}"

if [ ! -d "$MODEL_DIR" ]; then
    echo "Error: Directory not found: $MODEL_DIR"
    exit 1
fi

echo "=============================================="
echo "ONNX Runtime Text Generation Test"
echo "=============================================="
echo "Model dir:   $MODEL_DIR"
echo "Provider:    $PROVIDER"
echo "Prompt:      \"$PROMPT\""
echo "Max context: $SEQ_LENGTH tokens"
echo "Max output:  $SEQ_LENGTH tokens"
echo "Temperature: $TEMPERATURE"
if [ "$PROVIDER" = "MIGraphXExecutionProvider" ]; then
    echo "FP16 convert: $MIGRAPHX_FP16"
    echo "Caching:     $([ "$NO_CACHE" = true ] && echo 'disabled' || echo 'enabled')"
    echo "Exhaustive:  $EXHAUSTIVE"
    echo "Offload:     $OFFLOAD_COPY"
fi
echo "=============================================="

# Auto-detect GPU target for ROCm
GPU_TARGET=$(rocminfo 2>/dev/null | grep -oP 'gfx\d+' | head -1 || echo "")
if [ -n "$GPU_TARGET" ]; then
    if [[ "$GPU_TARGET" == gfx11* ]]; then
        echo "Detected RDNA3 GPU: $GPU_TARGET"
    fi
fi

export MODEL_DIR PROVIDER PROMPT SEQ_LENGTH TEMPERATURE VERBOSE NO_CACHE EXHAUSTIVE OFFLOAD_COPY
export MIGRAPHX_FP16 MIGRAPHX_SAVE GPU_TARGET

# Run Python script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 "$SCRIPT_DIR/py/run_inference_test.py"
