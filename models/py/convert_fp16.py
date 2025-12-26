#!/usr/bin/env python3
"""
convert_fp16.py - Convert ONNX model to FP16
"""

import onnx
import os
from onnxconverter_common import float16
from pathlib import Path


def main():
    input_file = os.environ['INPUT_FILE']
    output_file = os.environ['OUTPUT_FILE']

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


if __name__ == '__main__':
    main()
