#!/usr/bin/env python3
"""
optimize_model.py - Optimize ONNX model for ONNX Runtime inference

This script optimizes ONNX models for ONNX Runtime execution (CPU or GPU EP).
It fuses attention patterns into efficient operators (MultiHeadAttention/GQA)
which MIGraphX can then accelerate with Flash Attention kernels.
"""

import os
import sys
from pathlib import Path


def main():
    # Input parameters
    input_file = os.environ['INPUT_FILE']
    output_file = os.environ['OUTPUT_FILE']
    model_type = os.environ['MODEL_TYPE']
    num_heads = int(os.environ['NUM_HEADS'])
    hidden_size = int(os.environ['HIDDEN_SIZE'])
    num_kv_heads = int(os.environ['NUM_KV_HEADS'])
    opt_level = int(os.environ['OPT_LEVEL'])
    skip_fp16 = os.environ['SKIP_FP16'] == "true"
    use_gpu = os.environ['USE_GPU'] == "true"
    attention_type = os.environ['ATTENTION_TYPE']

    input_path = Path(input_file)
    output_path = Path(output_file)
    input_dir = input_path.parent

    # Check for external data files
    external_data_files = list(input_dir.glob(f"{input_path.stem}*.data")) + \
                          list(input_dir.glob(f"{input_path.stem}*_data"))
    has_external_data = len(external_data_files) > 0

    # Calculate total model size
    total_size = input_path.stat().st_size
    for ext_file in external_data_files:
        total_size += ext_file.stat().st_size
    total_size_gb = total_size / (1024**3)

    # Force external data for large models
    use_external = has_external_data or total_size_gb > 1.5

    print(f"Configuration:")
    print(f"  Model type:      {model_type}")
    print(f"  Num heads:       {num_heads}")
    print(f"  Num KV heads:    {num_kv_heads}")
    print(f"  Hidden size:     {hidden_size}")
    print(f"  Model size:      {total_size_gb:.2f} GB")
    print(f"  External data:   {use_external}")
    print(f"  Use GPU:         {use_gpu}")
    print(f"  FP16:            {not skip_fp16}")
    print(f"  Opt level:       {opt_level}")
    print(f"  Attention type:  {attention_type}")
    print()

    try:
        from onnxruntime.transformers import optimizer
        from onnxruntime.transformers.fusion_options import FusionOptions, AttentionOpType

        # Create FusionOptions with attention fusion enabled
        fusion_options = FusionOptions(model_type)

        # Enable attention fusion for MIGraphX Flash Attention
        fusion_options.enable_attention = True
        fusion_options.use_multi_head_attention = True
        fusion_options.enable_rotary_embeddings = True  # Important for LLaMA RoPE
        fusion_options.enable_shape_inference = True

        # Set attention operator type based on model architecture
        if attention_type == "auto":
            # Auto-detect: Use GQA if num_kv_heads < num_heads (LLaMA 3.x uses GQA)
            if num_kv_heads < num_heads:
                print(f"  Detected GQA (KV heads {num_kv_heads} < Q heads {num_heads})")
                fusion_options.attention_op_type = AttentionOpType.GroupQueryAttention
            else:
                print(f"  Using MultiHeadAttention (standard MHA)")
                fusion_options.attention_op_type = AttentionOpType.MultiHeadAttention
        elif attention_type == "GroupQueryAttention":
            fusion_options.attention_op_type = AttentionOpType.GroupQueryAttention
        elif attention_type == "MultiHeadAttention":
            fusion_options.attention_op_type = AttentionOpType.MultiHeadAttention
        elif attention_type == "PagedAttention":
            fusion_options.attention_op_type = AttentionOpType.PagedAttention
        else:
            fusion_options.attention_op_type = AttentionOpType.Attention

        print(f"  Attention op:    {fusion_options.attention_op_type}")
        print()

        # Run optimizer
        print("Optimizing model...")
        print("   (This may take several minutes for large models)")
        optimized_model = optimizer.optimize_model(
            input=input_file,
            model_type=model_type,
            num_heads=num_heads,
            hidden_size=hidden_size,
            optimization_options=fusion_options,
            opt_level=opt_level,
            use_gpu=use_gpu,
            only_onnxruntime=True,  # Use only ONNX Runtime optimizations
        )

        # Convert to FP16 if enabled (skip symbolic inference for large models)
        if not skip_fp16:
            print("Converting to FP16...")
            try:
                optimized_model.convert_float_to_float16(
                    keep_io_types=True,  # Keep input/output as FP32 for compatibility
                    use_symbolic_shape_infer=(total_size_gb < 2.0),  # Skip for large models
                )
            except Exception as e:
                print(f"   Warning: FP16 conversion had issues: {e}")
                print("   Continuing with partial FP16 conversion...")

        # Save model with external data for large models
        print(f"Saving to {output_file}...")
        if use_external:
            print("   Using external data format (model > 2GB)")
            # Create external data filename
            external_data_name = output_path.stem + ".onnx.data"
            optimized_model.save_model_to_file(
                str(output_file),
                use_external_data_format=True,
                all_tensors_to_one_file=True,
                location=external_data_name,
                size_threshold=1024,  # Externalize tensors > 1KB
                convert_attribute=False,
            )
        else:
            optimized_model.save_model_to_file(str(output_file))

        # Report fusion results
        print()
        print("=" * 50)
        print("Optimization Results")
        print("=" * 50)

        # Count fused operators
        import onnx
        model = onnx.load(output_file, load_external_data=False)
        op_counts = {}
        for node in model.graph.node:
            op_counts[node.op_type] = op_counts.get(node.op_type, 0) + 1

        # Report attention-related ops
        attention_ops = ['Attention', 'MultiHeadAttention', 'GroupQueryAttention', 'PagedAttention']
        found_attention = False
        for op in attention_ops:
            if op in op_counts:
                print(f"  ✅ {op}: {op_counts[op]} (FUSED - Flash Attention compatible)")
                found_attention = True

        if not found_attention:
            # Check for unfused attention pattern
            unfused_ops = ['MatMul', 'Softmax']
            if all(op in op_counts for op in unfused_ops):
                print(f"  ⚠️  No fused attention operators found")
                print(f"     MatMul: {op_counts.get('MatMul', 0)}, Softmax: {op_counts.get('Softmax', 0)}")
                print(f"     Attention patterns may not have been fused")

        # Report total ops
        total_ops = sum(op_counts.values())
        print(f"\n  Total operators: {total_ops}")

        # Top operators
        sorted_ops = sorted(op_counts.items(), key=lambda x: -x[1])[:10]
        print(f"  Top operators:")
        for op, count in sorted_ops:
            print(f"    {op}: {count}")

        # Calculate output size
        print()
        out_path = Path(output_file)
        out_size = out_path.stat().st_size
        ext_data_path = out_path.parent / (out_path.stem + ".onnx.data")
        if ext_data_path.exists():
            ext_size = ext_data_path.stat().st_size
            print(f"  Output model:    {out_size / (1024**2):.1f} MB")
            print(f"  External data:   {ext_size / (1024**3):.2f} GB")
            print(f"  Total size:      {(out_size + ext_size) / (1024**3):.2f} GB")
        else:
            print(f"  Output size:     {out_size / (1024**3):.2f} GB")

        print()
        print("✅ Optimization complete!")

    except Exception as e:
        print(f"❌ Optimization failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
