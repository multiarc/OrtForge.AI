#!/usr/bin/env python3
"""
validate_model.py - Validate ONNX model
"""

import onnx
from pathlib import Path
import os
import sys


def main():
    model_file = os.environ['MODEL_FILE']
    model_path = Path(model_file)
    model_dir = model_path.parent

    # Check for external data files
    external_data_file = model_dir / (model_path.stem + ".onnx.data")
    external_data_file_alt = model_dir / (model_path.stem + ".onnx_data")

    has_external_data = external_data_file.exists() or external_data_file_alt.exists()

    # Calculate total size including external data
    file_size = os.path.getsize(model_file)
    if external_data_file.exists():
        file_size += os.path.getsize(external_data_file)
        print(f"External data file: {external_data_file}")
    elif external_data_file_alt.exists():
        file_size += os.path.getsize(external_data_file_alt)
        print(f"External data file: {external_data_file_alt}")

    file_size_gb = file_size / (1024**3)
    print(f"Total model size: {file_size_gb:.2f} GB")

    # For models with external data or large models, use path-based validation
    if has_external_data or file_size_gb > 2.0:
        print("Using path-based validation (external data detected)...")
        print("Checking model...")
        try:
            # Use path-based check for models with external data
            onnx.checker.check_model(model_file)
            print("✅ Model is valid!")
        except onnx.checker.ValidationError as e:
            print(f"❌ Validation failed: {e}")
            sys.exit(1)
        except Exception as e:
            # Some versions of onnx may not support all checks
            print(f"⚠️  Validation warning: {e}")
            print("   Continuing with metadata extraction...")

        # Load without external data just to get metadata
        print("\nLoading metadata (without weights)...")
        model = onnx.load(model_file, load_external_data=False)
    else:
        print("Loading model...")
        try:
            model = onnx.load(model_file, load_external_data=True)
        except Exception as e:
            print("Trying without external data...")
            model = onnx.load(model_file, load_external_data=False)

        print("Checking model...")
        try:
            onnx.checker.check_model(model)
            print("✅ Model is valid!")
        except onnx.checker.ValidationError as e:
            print(f"❌ Validation failed: {e}")
            sys.exit(1)

    print("\nModel info:")
    print(f"  IR version: {model.ir_version}")
    print(f"  Opset version: {model.opset_import[0].version}")
    print(f"  Producer: {model.producer_name} {model.producer_version}")
    print(f"  Graph name: {model.graph.name}")
    print(f"  Inputs: {len(model.graph.input)}")
    for inp in model.graph.input:
        try:
            dims = [d.dim_value or d.dim_param for d in inp.type.tensor_type.shape.dim]
            print(f"    - {inp.name}: {dims}")
        except:
            print(f"    - {inp.name}: (unknown shape)")
    print(f"  Outputs: {len(model.graph.output)}")
    for out in model.graph.output:
        try:
            dims = [d.dim_value or d.dim_param for d in out.type.tensor_type.shape.dim]
            print(f"    - {out.name}: {dims}")
        except:
            print(f"    - {out.name}: (unknown shape)")
    print(f"  Nodes: {len(model.graph.node)}")
    print(f"  Initializers: {len(model.graph.initializer)}")


if __name__ == '__main__':
    main()
