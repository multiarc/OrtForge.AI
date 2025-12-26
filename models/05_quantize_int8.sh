#!/bin/bash
# =============================================================================
# 05_quantize_int8.sh - Quantize ONNX model to INT8 (dynamic quantization)
# =============================================================================
# Usage: ./05_quantize_int8.sh <input.onnx> <output.onnx>
# Example: ./05_quantize_int8.sh ./model.onnx ./model_int8.onnx
# =============================================================================

set -e

INPUT_FILE="${1:?Usage: $0 <input.onnx> <output.onnx>}"
OUTPUT_FILE="${2:?Usage: $0 <input.onnx> <output.onnx>}"

if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: File not found: $INPUT_FILE"
    exit 1
fi

echo "=============================================="
echo "Quantize to INT8 (Dynamic)"
echo "=============================================="
echo "Input:  $INPUT_FILE"
echo "Output: $OUTPUT_FILE"
echo "=============================================="

# Export variables for Python
export INPUT_FILE OUTPUT_FILE

# Run Python script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 "$SCRIPT_DIR/py/quantize_int8.py"

