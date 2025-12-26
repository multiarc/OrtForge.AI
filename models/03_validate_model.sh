#!/bin/bash
# =============================================================================
# 03_validate_model.sh - Validate ONNX model
# =============================================================================
# Usage: ./03_validate_model.sh <model.onnx>
# Example: ./03_validate_model.sh ./Llama3.1-8B-Instruct/onnx/model.onnx
# =============================================================================

set -e

MODEL_FILE="${1:?Usage: $0 <model.onnx>}"

if [ ! -f "$MODEL_FILE" ]; then
    echo "Error: File not found: $MODEL_FILE"
    exit 1
fi

echo "=============================================="
echo "Validate ONNX Model"
echo "=============================================="
echo "Model: $MODEL_FILE"
echo "=============================================="

# Export variables for Python
export MODEL_FILE

# Run Python script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 "$SCRIPT_DIR/py/validate_model.py"

