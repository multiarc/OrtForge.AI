#!/bin/bash
# =============================================================================
# 05_quantize_int4.sh - Quantize ONNX model to INT4 (4-bit weight quantization)
# =============================================================================
# Usage: ./05_quantize_int4.sh <input.onnx> <output.onnx> [block_size]
# Example: ./05_quantize_int4.sh ./model.onnx ./model_int4.onnx 128
#
# Requirements:
#   - ONNX Runtime 1.20+
#
# Block sizes: 32, 64, 128 (default), 256
#   - Smaller = better accuracy, larger model
#   - Larger = smaller model, may lose some accuracy
# =============================================================================

set -e

INPUT_FILE="${1:?Usage: $0 <input.onnx> <output.onnx> [block_size]}"
OUTPUT_FILE="${2:?Usage: $0 <input.onnx> <output.onnx> [block_size]}"
BLOCK_SIZE="${3:-128}"

if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: File not found: $INPUT_FILE"
    exit 1
fi

INPUT_DIR=$(dirname "$INPUT_FILE")
INPUT_BASE=$(basename "$INPUT_FILE" .onnx)

# Check for external data
EXTERNAL_DATA="$INPUT_DIR/${INPUT_BASE}.onnx.data"
EXTERNAL_DATA_ALT="$INPUT_DIR/${INPUT_BASE}.onnx_data"
HAS_EXTERNAL=false
if [ -f "$EXTERNAL_DATA" ] || [ -f "$EXTERNAL_DATA_ALT" ]; then
    HAS_EXTERNAL=true
fi

echo "=============================================="
echo "Quantize to INT4 (4-bit Weight Quantization)"
echo "=============================================="
echo "Input:      $INPUT_FILE"
echo "Output:     $OUTPUT_FILE"
echo "Block size: $BLOCK_SIZE"
echo "External:   $HAS_EXTERNAL"
echo "=============================================="

# Export variables for Python
export INPUT_FILE OUTPUT_FILE BLOCK_SIZE HAS_EXTERNAL

# Run Python script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 "$SCRIPT_DIR/py/quantize_int4.py"

echo ""
echo "Output files:"
ls -lh "$OUTPUT_FILE"* 2>/dev/null || echo "Check output directory for files"
