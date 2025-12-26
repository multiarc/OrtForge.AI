#!/bin/bash
# =============================================================================
# 01_export_model.sh - Export HuggingFace model to ONNX for Inference
# =============================================================================
# Usage: ./01_export_model.sh <model_path> <output_dir> [options]
#
# Custom ONNX export with KV cache support using modern torch.export.
# Does NOT require optimum library.
#
# Options:
#   --opset <n>         ONNX opset version (default: 21)
#   --batch <n>         Batch size (default: 1)
#   --no-kv-cache       Disable KV cache (not recommended for inference)
#   --fp32              Export in FP32 instead of FP16
#   --help              Show this help
#
# Defaults optimized for LLM inference:
#   - KV cache: ENABLED (essential for efficient autoregressive generation)
#   - Precision: FP16 (faster, lower memory)
#   - Shapes: Dynamic (any batch/sequence length)
#
# Requirements:
#   pip install torch transformers onnx
#
# Examples:
#   ./01_export_model.sh ./Llama3.1-8B-Instruct/hf ./onnx
#   ./01_export_model.sh ./model/hf ./onnx --opset 21
# =============================================================================

set -e

# =============================================================================
# Parse arguments - DEFAULTS OPTIMIZED FOR INFERENCE
# =============================================================================
POSITIONAL=()
OPSET_VERSION="21"
BATCH_SIZE=1
WITH_KV_CACHE=true
USE_FP16=true

while [[ $# -gt 0 ]]; do
    case $1 in
        --opset)
            OPSET_VERSION="$2"
            shift 2
            ;;
        --batch)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --no-kv-cache)
            WITH_KV_CACHE=false
            shift
            ;;
        --fp32)
            USE_FP16=false
            shift
            ;;
        --help|-h)
            head -30 "$0" | tail -27
            exit 0
            ;;
        -*)
            echo "Unknown option: $1"
            exit 1
            ;;
        *)
            POSITIONAL+=("$1")
            shift
            ;;
    esac
done

set -- "${POSITIONAL[@]}"

MODEL_PATH="${1:?Usage: $0 <model_path> <output_dir> [options]}"
OUTPUT_DIR="${2:?Usage: $0 <model_path> <output_dir> [options]}"

echo "=============================================="
echo "ONNX Model Export (Modern torch.export)"
echo "=============================================="
echo "Model path:    $MODEL_PATH"
echo "Output dir:    $OUTPUT_DIR"
echo "Opset version: $OPSET_VERSION"
echo "Precision:     $([ "$USE_FP16" = true ] && echo 'FP16' || echo 'FP32')"
echo "KV cache:      $([ "$WITH_KV_CACHE" = true ] && echo 'ENABLED ✓' || echo 'disabled')"
echo "=============================================="

mkdir -p "$OUTPUT_DIR"

# Export variables for Python
export MODEL_PATH OUTPUT_DIR OPSET_VERSION USE_FP16 WITH_KV_CACHE

python3 << 'PYEOF'
import sys
import os
import json
import gc
import torch
import onnx
from pathlib import Path
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from transformers.cache_utils import DynamicCache, DynamicLayer

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
device = "cuda" if torch.cuda.is_available() else "cpu"

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


# ============================================================================
# Export-friendly wrapper that takes flat tensor inputs
# Based on Optimum's approach: flatten KV cache to individual tensors
# ============================================================================
class OnnxExportWrapper(torch.nn.Module):
    """
    Wrapper for ONNX export that converts flat KV cache tensors to DynamicCache.
    
    Input signature (all tensors - export friendly):
        - input_ids: (batch, seq_len)
        - attention_mask: (batch, total_seq_len)
        - position_ids: (batch, seq_len) - REQUIRED for proper KV cache output
        - past_kv_flat: tuple of 2*num_layers tensors, each (batch, num_kv_heads, past_seq, head_dim)
    
    Output signature:
        - logits: (batch, seq_len, vocab_size)
        - present_kv_flat: tuple of 2*num_layers tensors
    
    NOTE: position_ids is essential - without it, model may only output KV for last position!
    """
    
    def __init__(self, model, num_layers, num_kv_heads, head_dim, dtype):
        super().__init__()
        self.model = model
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
    
    def forward(self, input_ids, attention_mask, position_ids, past_kv_flat):
        """
        Forward pass with flat KV cache tensors as a tuple.
        position_ids ensures model computes KV for ALL input positions.
        """
        # Reconstruct DynamicCache from flat tensors
        past_key_values = DynamicCache()
        
        if past_kv_flat is not None and len(past_kv_flat) > 0:
            for i in range(self.num_layers):
                key = past_kv_flat[2 * i]      # (batch, num_kv_heads, past_seq, head_dim)
                value = past_kv_flat[2 * i + 1]
                past_key_values.update(key, value, i)
        
        # Call model with position_ids to ensure KV is computed for all positions
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


print(f"\n[4/6] Creating export wrapper...")

wrapper = OnnxExportWrapper(model, num_layers, num_kv_heads, head_dim, dtype)
wrapper.eval()

print(f"    ✓ Export wrapper created")
print(f"    KV cache: {num_layers} layers × 2 (key + value) = {2 * num_layers} tensors")

print(f"\n[5/6] Preparing ONNX export...")

# Create dummy inputs
batch_size = 1
seq_len = 4  # Current input sequence length
past_seq_len = 8 if with_kv_cache else 0
total_seq_len = seq_len + past_seq_len

dummy_input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
dummy_attention_mask = torch.ones((batch_size, total_seq_len), dtype=torch.int64, device=device)
# position_ids: tells model which positions we're computing (essential for KV cache!)
dummy_position_ids = torch.arange(past_seq_len, past_seq_len + seq_len, device=device).unsqueeze(0)

# Create KV cache inputs as a tuple
past_kv_list = []

input_names = ["input_ids", "attention_mask", "position_ids"]
output_names = ["logits"]

dynamic_axes = {
    "input_ids": {0: "batch_size", 1: "sequence_length"},
    "attention_mask": {0: "batch_size", 1: "total_sequence_length"},
    "position_ids": {0: "batch_size", 1: "sequence_length"},
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
dummy_inputs = (dummy_input_ids, dummy_attention_mask, dummy_position_ids, past_kv_tuple)

print(f"    Input tensors: {len(input_names)}")
print(f"    Output tensors: {len(output_names)}")
print(f"    Position IDs: {dummy_position_ids.tolist()} (ensures KV for all positions)")

# Verify wrapper works
print(f"\n    Verifying wrapper forward pass...")
with torch.no_grad():
    test_output = wrapper(dummy_input_ids, dummy_attention_mask, dummy_position_ids, past_kv_tuple)
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

batch_dim = Dim("batch_size", min=1, max=64)
seq_dim = Dim("sequence_length", min=1, max=4096)
past_seq_dim = Dim("past_sequence_length", min=1, max=131072)
total_seq_dim = Dim("total_sequence_length", min=1, max=135168)

# Build dynamic_shapes matching input structure: (input_ids, attention_mask, position_ids, past_kv_tuple)
kv_dynamic_shapes = []
if with_kv_cache and past_seq_len > 0:
    for i in range(num_layers):
        kv_dynamic_shapes.append({0: batch_dim, 2: past_seq_dim})  # key
        kv_dynamic_shapes.append({0: batch_dim, 2: past_seq_dim})  # value

dynamic_shapes_tuple = (
    {0: batch_dim, 1: seq_dim},           # input_ids
    {0: batch_dim, 1: total_seq_dim},     # attention_mask
    {0: batch_dim, 1: seq_dim},           # position_ids (same dims as input_ids)
    tuple(kv_dynamic_shapes),             # past_kv_flat tuple
)

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
        "sequence_length": "Current input sequence length (1-4096)",
        "past_sequence_length": "Previous tokens in KV cache (1-131072)",
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
print(f"   - sequence_length: 1-4096 (current input)")
print(f"   - past_sequence_length: 1-131072 (KV cache)")
print(f"{'='*60}")
PYEOF

echo ""
echo "Output files:"
ls -lh "$OUTPUT_DIR"
