#!/bin/bash
# =============================================================================
# 01_export_model.sh - Export HuggingFace model to ONNX for Inference
# =============================================================================
# Usage: ./01_export_model.sh <model_path> <output_dir> [options]
#
# Custom ONNX export with KV cache support using modern torch.export.
# Does NOT require optimum library.
#
# Options:
#   --opset <n>         ONNX opset version (default: 21)
#   --batch <n>         Batch size (default: 1)
#   --no-kv-cache       Disable KV cache (not recommended for inference)
#   --fp32              Export in FP32 instead of FP16
#   --help              Show this help
#
# Defaults optimized for LLM inference:
#   - KV cache: ENABLED (essential for efficient autoregressive generation)
#   - Precision: FP16 (faster, lower memory)
#   - Shapes: Dynamic (any batch/sequence length)
#
# Requirements:
#   pip install torch transformers onnx
#
# Examples:
#   ./01_export_model.sh ./Llama3.1-8B-Instruct/hf ./onnx
#   ./01_export_model.sh ./model/hf ./onnx --opset 21
# =============================================================================

set -e

# =============================================================================
# Parse arguments - DEFAULTS OPTIMIZED FOR INFERENCE
# =============================================================================
POSITIONAL=()
OPSET_VERSION="21"
BATCH_SIZE=1
WITH_KV_CACHE=true
USE_FP16=true

while [[ $# -gt 0 ]]; do
    case $1 in
        --opset)
            OPSET_VERSION="$2"
            shift 2
            ;;
        --batch)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --no-kv-cache)
            WITH_KV_CACHE=false
            shift
            ;;
        --fp32)
            USE_FP16=false
            shift
            ;;
        --help|-h)
            head -30 "$0" | tail -27
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

MODEL_PATH="${1:?Usage: $0 <model_path> <output_dir> [options]}"
OUTPUT_DIR="${2:?Usage: $0 <model_path> <output_dir> [options]}"

echo "=============================================="
echo "ONNX Model Export (Modern torch.export)"
echo "=============================================="
echo "Model path:    $MODEL_PATH"
echo "Output dir:    $OUTPUT_DIR"
echo "Opset version: $OPSET_VERSION"
echo "Precision:     $([ "$USE_FP16" = true ] && echo 'FP16' || echo 'FP32')"
echo "KV cache:      $([ "$WITH_KV_CACHE" = true ] && echo 'ENABLED ✓' || echo 'disabled')"
echo "=============================================="

mkdir -p "$OUTPUT_DIR"

# Export variables for Python
export MODEL_PATH OUTPUT_DIR OPSET_VERSION USE_FP16 WITH_KV_CACHE

# Run Python script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 "$SCRIPT_DIR/py/export_model.py"

echo ""
echo "Output files:"
ls -lh "$OUTPUT_DIR"
