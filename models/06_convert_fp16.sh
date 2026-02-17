#!/bin/bash
# =============================================================================
# 06_convert_fp16.sh - Convert ONNX model to FP16
# =============================================================================
# Usage: ./06_convert_fp16.sh <input.onnx> <output.onnx>
# Example: ./06_convert_fp16.sh ./model.onnx ./model_fp16.onnx
# =============================================================================

set -e

INPUT_FILE="${1:?Usage: $0 <input.onnx> <output.onnx>}"
OUTPUT_FILE="${2:?Usage: $0 <input.onnx> <output.onnx>}"

if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: File not found: $INPUT_FILE"
    exit 1
fi

echo "=============================================="
echo "Convert to FP16"
echo "=============================================="
echo "Input:  $INPUT_FILE"
echo "Output: $OUTPUT_FILE"
echo "=============================================="

# Export variables for Python
export INPUT_FILE OUTPUT_FILE

# Run Python script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 "$SCRIPT_DIR/py/convert_fp16.py"

