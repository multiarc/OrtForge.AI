#!/usr/bin/env python3
"""
export_model.py - Export HuggingFace model to ONNX for Inference

Custom ONNX export with KV cache support using modern torch.export.
Does NOT require optimum library.

IMPORTANT NOTES:
- Exports on CPU by default (set FORCE_CPU_EXPORT=false to use GPU)
- CPU export is RECOMMENDED for stability, especially with PyTorch nightly builds
- ONNX models are device-agnostic: CPU-exported models run fine on GPU/MIGraphX
- For PyTorch stable (non-nightly), either CPU or GPU export works
- For ROCm nightly builds, CPU export avoids potential issues
"""

import sys
import os
import json
import gc
import torch
import onnx
from pathlib import Path
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from transformers.cache_utils import DynamicCache, DynamicLayer


# ============================================================================
# Export-friendly wrapper that takes flat tensor inputs
# Based on Optimum's approach: flatten KV cache to individual tensors
# ============================================================================
class OnnxExportWrapper(torch.nn.Module):
    """
    Wrapper for ONNX export that converts flat KV cache tensors to DynamicCache.

    Input signature:
        - input_ids: (batch, seq_len) - token IDs
        - attention_mask: (batch, seq_len) - attention mask
        - past_seq_len: (256,) - padded tensor with past sequence length in first element (used to compute position_ids)
        - past_kv_flat: tuple of 2*num_layers tensors, each (batch, num_kv_heads, past_seq, head_dim)

    Output signature:
        - logits: (batch, seq_len, vocab_size)
        - present_kv_flat: tuple of 2*num_layers tensors

    Note: position_ids is computed internally from past_seq_len[0] to avoid MIGraphX
    hipHostRegister failures. The past_seq_len input is padded to 256 elements (2048 bytes)
    to meet MIGraphX minimum buffer size requirements for hipHostRegister.
    """

    def __init__(self, model, num_layers, num_kv_heads, head_dim, dtype):
        super().__init__()
        self.model = model
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype

    def forward(self, input_ids, attention_mask, past_seq_len_tensor, past_kv_flat):
        """
        Forward pass with flat KV cache tensors as a tuple.
        Computes position_ids internally to avoid hipHostRegister issues with small buffers.

        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)
            past_seq_len_tensor: (256,) padded tensor with past sequence length in first element
            past_kv_flat: tuple of KV cache tensors
        """
        # Reconstruct DynamicCache from flat tensors
        past_key_values = DynamicCache()

        if past_kv_flat is not None and len(past_kv_flat) > 0:
            for i in range(self.num_layers):
                key = past_kv_flat[2 * i]      # (batch, num_kv_heads, past_seq, head_dim)
                value = past_kv_flat[2 * i + 1]
                past_key_values.update(key, value, i)

        # Compute position_ids internally from past_seq_len
        # past_seq_len_tensor is padded to 256 elements to avoid hipHostRegister failures
        # Extract the first element using pure tensor operations (no .item() to avoid CPU copy)
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]

        # Extract scalar using tensor indexing (stays on device, no CPU transfer)
        past_seq_len_scalar = past_seq_len_tensor[0:1]  # (1,) tensor

        # Create position_ids: [past_seq_len, past_seq_len+1, ..., past_seq_len+seq_len-1]
        # Use broadcasting to add past_seq_len to arange
        position_ids = torch.arange(0, seq_len, dtype=torch.long, device=input_ids.device).unsqueeze(0)
        position_ids = position_ids + past_seq_len_scalar  # Broadcasting addition
        position_ids = position_ids.expand(batch_size, -1)

        # Call model with computed position_ids
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )

        logits = outputs.logits
        present_kv = outputs.past_key_values

        # Flatten present_key_values for output
        flat_outputs = [logits]
        for i in range(len(present_kv.layers)):
            layer = present_kv.layers[i]
            flat_outputs.append(layer.keys)   # (batch, num_kv_heads, total_seq, head_dim)
            flat_outputs.append(layer.values)

        return tuple(flat_outputs)


def main():
    # Read from environment variables
    model_path = os.environ['MODEL_PATH']
    output_dir = Path(os.environ['OUTPUT_DIR'])
    opset_version = int(os.environ['OPSET_VERSION'])
    use_fp16 = os.environ['USE_FP16'] == "true"
    with_kv_cache = os.environ['WITH_KV_CACHE'] == "true"

    print(f"[1/6] Loading model configuration...")
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    # Extract model info
    model_type = getattr(config, 'model_type', 'unknown')
    hidden_size = getattr(config, 'hidden_size', 0)
    num_heads = getattr(config, 'num_attention_heads', 0)
    num_kv_heads = getattr(config, 'num_key_value_heads', num_heads)
    num_layers = getattr(config, 'num_hidden_layers', 0)
    vocab_size = getattr(config, 'vocab_size', 0)
    max_position = getattr(config, 'max_position_embeddings', 4096)
    head_dim = hidden_size // num_heads

    variants = {
        2048: "Llama 3.2 1B",
        3072: "Llama 3.2 3B",
        4096: "Llama 3.1 8B / Mistral 7B",
        8192: "Llama 3.1 70B",
        16384: "Llama 3.1 405B",
    }
    model_variant = variants.get(hidden_size, f"Unknown ({model_type})")

    print(f"    Model:          {model_variant}")
    print(f"    Type:           {model_type}")
    print(f"    Hidden size:    {hidden_size}")
    print(f"    Attention:      {num_heads} heads, {num_kv_heads} KV heads")
    print(f"    Head dim:       {head_dim}")
    print(f"    Layers:         {num_layers}")
    print(f"    Vocab:          {vocab_size}")

    print(f"\n[2/6] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        fix_mistral_regex=True,  # Fix incorrect regex pattern in Llama/Mistral tokenizers
    )
    tokenizer.save_pretrained(output_dir)

    print(f"\n[3/6] Loading model ({'FP16' if use_fp16 else 'FP32'})...")
    dtype = torch.float16 if use_fp16 else torch.float32

    # Device selection for export
    # Note: ONNX export only traces the graph - optimization happens at inference time
    # GPU export is faster for large models but may have stability issues with nightly builds
    force_cpu_export = os.environ.get('FORCE_CPU_EXPORT', 'false') == 'true'

    if force_cpu_export:
        device = "cpu"
        print(f"    Using CPU for export (stable)")
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"    Using GPU for export (faster, uses ROCm)")
        if device == "cuda":
            print(f"    Note: If export fails, try FORCE_CPU_EXPORT=true")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
        use_cache=with_kv_cache,
        attn_implementation="eager",  # Required for ONNX export
    )
    model.eval()
    model.to(device)

    print(f"    Device: {device}")
    print(f"    Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")

    print(f"\n[4/6] Creating export wrapper...")

    wrapper = OnnxExportWrapper(model, num_layers, num_kv_heads, head_dim, dtype)
    wrapper.eval()

    print(f"    ✓ Export wrapper created")
    print(f"    KV cache: {num_layers} layers × 2 (key + value) = {2 * num_layers} tensors")

    print(f"\n[5/6] Preparing ONNX export...")

    # Create dummy inputs
    batch_size = 1
    seq_len = 256  # Must be >= MIN_SEQ_LEN to satisfy Dim constraints
    past_seq_len = 512 if with_kv_cache else 0

    dummy_input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    # Use seq_len for attention_mask to match dynamic_shapes (batch handling requires consistent dims)
    dummy_attention_mask = torch.ones((batch_size, seq_len), dtype=torch.int64, device=device)
    # past_seq_len as padded tensor (256 elements = 2048 bytes to avoid hipHostRegister failures)
    # Only first element is used; rest is padding
    dummy_past_seq_len = torch.zeros(256, dtype=torch.int64, device=device)
    dummy_past_seq_len[0] = past_seq_len

    # Create KV cache inputs as a tuple
    past_kv_list = []

    # Use past_seq_len as scalar input instead of position_ids array
    # This avoids hipHostRegister failures on small buffers
    input_names = ["input_ids", "attention_mask", "past_seq_len"]
    output_names = ["logits"]

    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
        # past_seq_len is a fixed-size padded tensor (256 elements) - no dynamic axes
        "logits": {0: "batch_size", 1: "sequence_length"},
    }

    if with_kv_cache and past_seq_len > 0:
        kv_shape = (batch_size, num_kv_heads, past_seq_len, head_dim)
        print(f"    KV cache input shape: {kv_shape}")

        for i in range(num_layers):
            # Input past KV
            key_name = f"past_key_values.{i}.key"
            value_name = f"past_key_values.{i}.value"
            input_names.extend([key_name, value_name])

            past_kv_list.append(torch.randn(kv_shape, dtype=dtype, device=device))
            past_kv_list.append(torch.randn(kv_shape, dtype=dtype, device=device))

            dynamic_axes[key_name] = {0: "batch_size", 2: "past_sequence_length"}
            dynamic_axes[value_name] = {0: "batch_size", 2: "past_sequence_length"}

            # Output present KV
            present_key_name = f"present.{i}.key"
            present_value_name = f"present.{i}.value"
            output_names.extend([present_key_name, present_value_name])

            dynamic_axes[present_key_name] = {0: "batch_size", 2: "total_sequence_length"}
            dynamic_axes[present_value_name] = {0: "batch_size", 2: "total_sequence_length"}

    past_kv_tuple = tuple(past_kv_list) if past_kv_list else ()
    dummy_inputs = (dummy_input_ids, dummy_attention_mask, dummy_past_seq_len, past_kv_tuple)

    print(f"    Input tensors: {len(input_names)}")
    print(f"    Output tensors: {len(output_names)}")
    print(f"    past_seq_len (scalar): {past_seq_len} (position_ids computed internally)")

    # Verify wrapper works
    print(f"\n    Verifying wrapper forward pass...")
    with torch.no_grad():
        test_output = wrapper(dummy_input_ids, dummy_attention_mask, dummy_past_seq_len, past_kv_tuple)
        print(f"    ✓ Forward pass successful")
        print(f"    Logits shape: {test_output[0].shape}")
        if with_kv_cache:
            print(f"    Present KV[0].key shape: {test_output[1].shape}")
            expected_kv_len = past_seq_len + seq_len
            actual_kv_len = test_output[1].shape[2]
            if actual_kv_len == expected_kv_len:
                print(f"    ✓ KV cache outputs ALL positions: {actual_kv_len} = {past_seq_len} + {seq_len}")
            else:
                print(f"    ⚠ KV cache length mismatch: {actual_kv_len} (expected {expected_kv_len})")

    print(f"\n[6/6] Exporting to ONNX (opset {opset_version})...")
    print(f"    This may take several minutes for large models...")

    output_file = output_dir / "model.onnx"

    # Use dynamo=True for opset 21 with dynamic_shapes
    from torch.export import Dim

    # CRITICAL: MIGraphX hipHostRegister bug - even 1024 bytes may fail
    # HIP memory pool seems to have 2KB minimum allocation
    # Testing with 256 elements = 2048 bytes
    MIN_SEQ_LEN = 256  # Minimum sequence length to avoid hipHostRegister failure

    batch_dim = Dim("batch_size", min=1, max=64)
    seq_dim = Dim("sequence_length", min=MIN_SEQ_LEN, max=4096)
    past_seq_dim = Dim("past_sequence_length", min=0, max=131072)

    # Build dynamic_shapes matching input structure: (input_ids, attention_mask, position_ids, past_kv_tuple)
    kv_dynamic_shapes = []
    if with_kv_cache and past_seq_len > 0:
        for i in range(num_layers):
            kv_dynamic_shapes.append({0: batch_dim, 2: past_seq_dim})  # key
            kv_dynamic_shapes.append({0: batch_dim, 2: past_seq_dim})  # value

    # CRITICAL: All current sequence dimensions must use the same seq_dim
    # past_seq_len is a scalar (no dynamic shape)
    # position_ids is computed internally from past_seq_len to avoid hipHostRegister bug
    dynamic_shapes_tuple = (
        {0: batch_dim, 1: seq_dim},           # input_ids
        {0: batch_dim, 1: seq_dim},           # attention_mask (must match input_ids dim)
        None,                                  # past_seq_len (scalar, no dynamic shape)
        tuple(kv_dynamic_shapes),             # past_kv_flat tuple
    )

    # Export with dynamo=True (modern torch.export path)
    # If this fails with nightly builds, try: dynamo=False with old export path
    use_dynamo = os.environ.get('USE_DYNAMO', 'true') == 'true'

    if use_dynamo:
        print(f"    Using dynamo export (torch.export path, recommended)")
        torch.onnx.export(
            wrapper,
            dummy_inputs,
            str(output_file),
            input_names=input_names,
            output_names=output_names,
            opset_version=opset_version,
            dynamo=True,
            dynamic_shapes=dynamic_shapes_tuple,
            external_data=True,
            report=True,
        )
        print(f"    ✓ ONNX export complete (dynamo, opset {opset_version})")
    else:
        print(f"    Using legacy export (fallback for nightly issues)")
        torch.onnx.export(
            wrapper,
            dummy_inputs,
            str(output_file),
            input_names=input_names,
            output_names=output_names,
            opset_version=opset_version,
            dynamic_axes=dynamic_axes,
            do_constant_folding=False,
            external_data=True,
        )
        print(f"    ✓ ONNX export complete (legacy, opset {opset_version})")

    # Verify ONNX model
    print(f"\n    Verifying ONNX model...")
    try:
        onnx_model = onnx.load(str(output_file), load_external_data=False)
        onnx.checker.check_model(onnx_model)
        print(f"    ✓ ONNX model structure is valid")

        print(f"\n    ONNX Model Inputs ({len(onnx_model.graph.input)}):")
        for inp in onnx_model.graph.input[:5]:
            print(f"      - {inp.name}")
        if len(onnx_model.graph.input) > 5:
            print(f"      ... and {len(onnx_model.graph.input) - 5} more")

        print(f"\n    ONNX Model Outputs ({len(onnx_model.graph.output)}):")
        for out in onnx_model.graph.output[:5]:
            print(f"      - {out.name}")
        if len(onnx_model.graph.output) > 5:
            print(f"      ... and {len(onnx_model.graph.output) - 5} more")

    except Exception as e:
        print(f"    ⚠ Could not verify: {e}")

    # Calculate sizes
    data_files = list(output_dir.glob("model*.onnx*"))
    total_size = sum(f.stat().st_size for f in data_files if f.exists())

    # Save export info
    export_info = {
        "export_method": "torch.onnx.export with OnnxExportWrapper",
        "shape_mode": "dynamic",
        "precision": "fp16" if use_fp16 else "fp32",
        "opset_version": opset_version,
        "with_kv_cache": with_kv_cache,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "hidden_size": hidden_size,
        "vocab_size": vocab_size,
        "max_position_embeddings": max_position,
        "model_variant": model_variant,
        "model_type": model_type,
        "input_names": input_names,
        "output_names": output_names,
        "dynamic_dims": {
            "batch_size": "Variable batch size (1-64)",
            "sequence_length": f"Current input sequence length ({MIN_SEQ_LEN}-4096, min=64 avoids MIGraphX hipHostRegister bug)",
            "past_sequence_length": "Previous tokens in KV cache (0-131072)",
            "total_sequence_length": "past_sequence_length + sequence_length",
        },
        "kv_cache_info": {
            "shape": f"(batch_size, {num_kv_heads}, sequence_length, {head_dim})",
            "num_layers": num_layers,
            "inputs_per_layer": 2,
            "total_kv_inputs": 2 * num_layers,
        } if with_kv_cache else None,
    }

    with open(output_dir / "export_info.json", "w") as f:
        json.dump(export_info, f, indent=2)

    # Clean up
    del model, wrapper
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print("✅ Export complete!")
    print(f"{'='*60}")
    print(f"   Output directory: {output_dir}")
    print(f"   Total size: {total_size / (1024**3):.2f} GB")
    print(f"   position_ids: INCLUDED (enables full KV cache output)")
    if with_kv_cache:
        print(f"   KV cache: {num_layers} layers × 2 (key+value)")
        print(f"   KV shape: (batch, {num_kv_heads}, seq_len, {head_dim})")
    print(f"\n   Dynamic dimensions:")
    print(f"   - batch_size: 1-64")
    print(f"   - sequence_length: {MIN_SEQ_LEN}-4096 (min={MIN_SEQ_LEN} avoids MIGraphX hipHostRegister bug)")
    print(f"   - past_sequence_length: 0-131072 (KV cache)")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
