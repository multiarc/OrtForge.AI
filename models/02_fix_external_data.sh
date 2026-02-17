#!/bin/bash
# =============================================================================
# 02_fix_external_data.sh - Convert large ONNX model to use external data file
# =============================================================================
# Required for models > 2GB due to protobuf limits
# Usage: ./02_fix_external_data.sh <model.onnx>
# Example: ./02_fix_external_data.sh ./Llama3.1-8B-Instruct/onnx/model.onnx
# =============================================================================

set -e

MODEL_FILE="${1:?Usage: $0 <model.onnx>}"

if [ ! -f "$MODEL_FILE" ]; then
    echo "Error: File not found: $MODEL_FILE"
    exit 1
fi

OUTPUT_DIR=$(dirname "$MODEL_FILE")
BASENAME=$(basename "$MODEL_FILE" .onnx)
EXTERNAL_DATA_FILE="${BASENAME}.onnx.data"

echo "=============================================="
echo "Fix External Data"
echo "=============================================="
echo "Model file:    $MODEL_FILE"
echo "External data: $OUTPUT_DIR/$EXTERNAL_DATA_FILE"
echo "=============================================="

# Check file size
FILE_SIZE=$(stat -c%s "$MODEL_FILE")
FILE_SIZE_GB=$(echo "scale=2; $FILE_SIZE / 1024 / 1024 / 1024" | bc)
echo "Current file size: ${FILE_SIZE_GB} GB"

# Export variables for Python
export MODEL_FILE EXTERNAL_DATA_FILE FILE_SIZE

# Run Python script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 "$SCRIPT_DIR/py/fix_external_data.py"

echo ""
echo "Output files:"
ls -lh "$OUTPUT_DIR"/${BASENAME}*

