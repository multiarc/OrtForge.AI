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

python3 << EOF
import onnx
from onnx.external_data_helper import convert_model_to_external_data
from pathlib import Path
import os
import sys

model_file = Path("$MODEL_FILE")
output_dir = model_file.parent
external_data_file = "$EXTERNAL_DATA_FILE"
file_size = $FILE_SIZE

# For very large files (>2GB), we need special handling
if file_size > 2 * 1024 * 1024 * 1024:
    print("Large model detected (>2GB). Using graph-only loading...")
    print("This preserves external data references without loading weights into memory.")
    
    try:
        # Load graph structure only (don't load external data into memory)
        model = onnx.load(str(model_file), load_external_data=False)
        
        # Check if model already references external data
        has_external_refs = False
        for tensor in model.graph.initializer:
            if tensor.HasField('data_location') and tensor.data_location == onnx.TensorProto.EXTERNAL:
                has_external_refs = True
                break
        
        if has_external_refs:
            print("✅ Model already uses external data references.")
            print("   External data file should contain the weights.")
            
            # Verify external data file exists
            ext_path = output_dir / external_data_file
            if ext_path.exists():
                ext_size = ext_path.stat().st_size
                print(f"   External data file: {ext_size / (1024**3):.2f} GB")
            else:
                print(f"⚠️  External data file not found: {ext_path}")
                print("   Model may be corrupted or missing weight data.")
                sys.exit(1)
        else:
            print("Model has embedded weights. Converting to external data format...")
            
            # Convert to external data
            convert_model_to_external_data(
                model,
                all_tensors_to_one_file=True,
                location=external_data_file,
                size_threshold=1024,
                convert_attribute=False
            )
            
            # Save the model with external data
            print(f"Saving model with external data: {external_data_file}")
            onnx.save_model(
                model,
                str(model_file),
                save_as_external_data=True,
                all_tensors_to_one_file=True,
                location=external_data_file,
                size_threshold=1024,
            )
            print("✅ Done!")
            
    except Exception as e:
        print(f"Error: {e}")
        print("")
        print("For models >2GB with embedded weights, try these alternatives:")
        print("1. Re-export the model with external data from the start")
        print("2. Use: python -m onnx.tools.update_inputs_outputs_dims")
        sys.exit(1)
else:
    print("Loading model (this may take a while for large models)...")
    model = onnx.load(str(model_file), load_external_data=True)

    print(f"Saving with external data: {external_data_file}")
    onnx.save_model(
        model,
        str(model_file),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=external_data_file,
        size_threshold=1024,
    )

    print("✅ Done!")
EOF

echo ""
echo "Output files:"
ls -lh "$OUTPUT_DIR"/${BASENAME}*

