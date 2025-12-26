#!/usr/bin/env python3
"""
quantize_int4.py - Quantize ONNX model to INT4 (4-bit weight quantization)
"""

import sys
import os
from pathlib import Path


def main():
    input_file = os.environ['INPUT_FILE']
    output_file = os.environ['OUTPUT_FILE']
    block_size = int(os.environ['BLOCK_SIZE'])
    has_external = os.environ['HAS_EXTERNAL'] == "true"

    input_path = Path(input_file)
    output_path = Path(output_file)

    # Check for INT4 support - use matmul_nbits_quantizer (correct module name)
    try:
        from onnxruntime.quantization import matmul_nbits_quantizer
        from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer, DefaultWeightOnlyQuantConfig
        print("✓ Found MatMulNBitsQuantizer")
    except ImportError as e:
        print(f"❌ INT4 quantization not available: {e}")
        print("")
        print("   Requires ONNX Runtime 1.20+")
        print("   pip install onnxruntime>=1.20")
        print("")
        print("   Or use INT8 quantization instead:")
        print("      ./05_quantize_int8.sh <input.onnx> <output.onnx>")
        print("")
        sys.exit(1)

    # Perform INT4 quantization
    print("")
    print("Performing INT4 quantization...")

    print("Step 1: Loading model...")
    import onnx
    try:
        model = onnx.load(str(input_path), load_external_data=True)
        print(f"   Loaded model with {len(model.graph.node)} nodes")
    except Exception as e:
        print(f"   Error loading model: {e}")
        sys.exit(1)

    print("Step 2: Checking model compatibility...")

    # Check if model has been optimized with FP16 Cast nodes inserted
    init_names = {init.name for init in model.graph.initializer}
    matmuls = [n for n in model.graph.node if n.op_type == 'MatMul']
    matmuls_with_const_weight = 0
    has_precision_cast = False

    for mm in matmuls:
        if len(mm.input) >= 2:
            weight_input = mm.input[1]
            if weight_input in init_names:
                matmuls_with_const_weight += 1
            if 'InsertedPrecisionFreeCast' in weight_input:
                has_precision_cast = True

    pct_quantizable = (matmuls_with_const_weight / len(matmuls) * 100) if matmuls else 0
    print(f"   MatMul nodes: {len(matmuls)}")
    print(f"   Quantizable:  {matmuls_with_const_weight} ({pct_quantizable:.0f}%)")

    if has_precision_cast or pct_quantizable < 50:
        print("")
        print("   ⚠ WARNING: This model appears to be FP16-optimized.")
        print("   The optimizer inserted Cast nodes that block weight quantization.")
        print("")
        print("   For INT4 quantization, use the base model BEFORE optimization:")
        print("      ./05_quantize_int4.sh ./path/to/model.onnx ./output_int4.onnx")
        print("")
        print("   Then optimize the INT4 model WITHOUT --float16:")
        print("      python3 -m onnxruntime.transformers.optimizer ...")
        print("")
        if pct_quantizable == 0:
            print("   ❌ No quantizable MatMul nodes found. Exiting.")
            sys.exit(1)
        print("   Continuing with partial quantization...")
        print("")

    print(f"Step 3: Creating INT4 quantizer (block_size={block_size})...")

    from onnxruntime.quantization import QuantFormat

    quantizer = MatMulNBitsQuantizer(
        model,
        block_size=block_size,
        is_symmetric=True,
        accuracy_level=4,
        op_types_to_quantize=("MatMul", "Gather"),  # Explicitly quantize MatMul and Gather ops
        quant_format=QuantFormat.QOperator,
    )

    print("Step 4: Running quantization...")
    print("   This may take several minutes for large models...")
    quantizer.process()

    print("Step 5: Saving quantized model...")
    use_external_out = has_external or (len(model.graph.initializer) > 100)
    quantizer.model.save_model_to_file(str(output_path), use_external_data_format=use_external_out)

    # Calculate and report sizes
    print("")
    print("Calculating size reduction...")

    def get_model_size(path):
        """Get total model size including external data."""
        p = Path(path)
        size = p.stat().st_size if p.exists() else 0
        for ext in ['.onnx.data', '.onnx_data', '_data']:
            ext_file = p.parent / (p.stem + ext)
            if ext_file.exists():
                size += ext_file.stat().st_size
                break
        return size

    input_size = get_model_size(input_path)
    output_size = get_model_size(output_path)

    input_gb = input_size / (1024**3)
    output_gb = output_size / (1024**3)
    reduction = (1 - output_size / input_size) * 100 if input_size > 0 else 0

    print(f"")
    print(f"✅ INT4 Quantization complete!")
    print(f"   Input size:  {input_gb:.2f} GB")
    print(f"   Output size: {output_gb:.2f} GB")
    print(f"   Reduction:   {reduction:.1f}%")
    print(f"   Expected:    ~75% reduction for INT4")


if __name__ == '__main__':
    main()
