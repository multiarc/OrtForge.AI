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

python3 << EOF
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
from onnxruntime.quantization.shape_inference import quant_pre_process
from pathlib import Path
import tempfile
import shutil
import os

input_file = "$INPUT_FILE"
output_file = "$OUTPUT_FILE"
input_path = Path(input_file)
output_path = Path(output_file)

print("Quantizing model to INT8...")
print("This may take a while for large models...")

# Check for external data
external_data_file = input_path.parent / (input_path.stem + ".onnx.data")
external_data_file_alt = input_path.parent / (input_path.stem + ".onnx_data")
has_external_data = external_data_file.exists() or external_data_file_alt.exists()

if has_external_data:
    print("Model has external data, using model path for quantization...")

# Try preprocessing first
try:
    print("Step 1: Preprocessing model...")
    preprocessed_file = str(input_path.parent / (input_path.stem + "_preprocessed.onnx"))
    
    quant_pre_process(
        input_model_path=input_file,
        output_model_path=preprocessed_file,
        skip_symbolic_shape=True,  # Skip if symbolic shape inference fails
    )
    quantize_input = preprocessed_file
    print("   Preprocessing complete")
except Exception as e:
    print(f"   Preprocessing skipped: {e}")
    quantize_input = input_file

# Perform quantization
try:
    print("Step 2: Quantizing to INT8...")
    quantize_dynamic(
        model_input=quantize_input,
        model_output=output_file,
        weight_type=QuantType.QInt8,
        extra_options={
            "MatMulConstBOnly": True,
        },
        use_external_data_format=has_external_data,
    )
except Exception as e:
    print(f"Dynamic quantization failed: {e}")
    print("Trying with per-channel quantization disabled...")
    try:
        quantize_dynamic(
            model_input=quantize_input,
            model_output=output_file,
            weight_type=QuantType.QInt8,
            per_channel=False,
            extra_options={
                "MatMulConstBOnly": True,
            },
            use_external_data_format=has_external_data,
        )
    except Exception as e2:
        print(f"Quantization failed: {e2}")
        print("\n❌ INT8 quantization is not supported for this model architecture.")
        print("   Consider using FP16 instead (06_convert_fp16.sh)")
        exit(1)

# Cleanup preprocessed file if it exists
preprocessed_path = input_path.parent / (input_path.stem + "_preprocessed.onnx")
if preprocessed_path.exists():
    os.remove(preprocessed_path)
    preprocessed_data = preprocessed_path.parent / (preprocessed_path.stem + ".onnx.data")
    if preprocessed_data.exists():
        os.remove(preprocessed_data)

# Calculate sizes
input_size = input_path.stat().st_size
if has_external_data:
    if external_data_file.exists():
        input_size += external_data_file.stat().st_size
    elif external_data_file_alt.exists():
        input_size += external_data_file_alt.stat().st_size

output_size = output_path.stat().st_size
output_data = output_path.parent / (output_path.stem + ".onnx.data")
if output_data.exists():
    output_size += output_data.stat().st_size

input_size_gb = input_size / (1024**3)
output_size_gb = output_size / (1024**3)
reduction = (1 - output_size / input_size) * 100 if input_size > 0 else 0

print(f"\n✅ Quantization complete!")
print(f"   Input size:  {input_size_gb:.2f} GB")
print(f"   Output size: {output_size_gb:.2f} GB")
print(f"   Reduction:   {reduction:.1f}%")
EOF

