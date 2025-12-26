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

python3 << EOF
import onnx
from onnxconverter_common import float16
from pathlib import Path

input_file = "$INPUT_FILE"
output_file = "$OUTPUT_FILE"

print("Loading model...")
model = onnx.load(input_file, load_external_data=True)

print("Converting to FP16...")
model_fp16 = float16.convert_float_to_float16(
    model,
    keep_io_types=True,  # Keep inputs/outputs as FP32 for compatibility
)

print("Saving model...")
onnx.save(model_fp16, output_file)

input_size = Path(input_file).stat().st_size / (1024**3)
output_size = Path(output_file).stat().st_size / (1024**3)
reduction = (1 - output_size / input_size) * 100

print(f"\n✅ Conversion complete!")
print(f"   Input size:  {input_size:.2f} GB")
print(f"   Output size: {output_size:.2f} GB")
print(f"   Reduction:   {reduction:.1f}%")
EOF

