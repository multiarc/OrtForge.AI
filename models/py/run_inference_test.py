#!/usr/bin/env python3
"""
run_inference_test.py - Test inference with ONNX Runtime

Runs text generation to verify the model works correctly.
Uses autoregressive generation with growing KV cache.
"""

import os
import sys
import onnxruntime as ort
import numpy as np
from pathlib import Path
import time
import json
import subprocess
from transformers import AutoTokenizer

# Get environment variables
model_dir = Path(os.environ['MODEL_DIR'])
provider = os.environ['PROVIDER']
prompt = os.environ.get('PROMPT', 'What is 2+2?')
seq_length = int(os.environ.get('SEQ_LENGTH', '256'))  # Bucket size
# Max output = bucket size (KV cache = 2*bucket covers input + output)
max_tokens = seq_length
max_kv_len = seq_length  # Maximum KV cache length
temperature = float(os.environ.get('TEMPERATURE', '0.0'))
verbose = os.environ.get('VERBOSE', 'false') == 'true'
no_cache = os.environ.get('NO_CACHE', 'false') == 'true'
exhaustive = os.environ.get('EXHAUSTIVE', 'false') == 'true'
offload_copy = os.environ.get('OFFLOAD_COPY', 'true') == 'true'
migraphx_fp16 = os.environ.get('MIGRAPHX_FP16', '0') == '1'
migraphx_save = os.environ.get('MIGRAPHX_SAVE', '1') == '1'
gpu_target = os.environ.get('GPU_TARGET', '')

# Configure logging
log_level = 0 if verbose else 2
ort.set_default_logger_severity(log_level)

if gpu_target:
    print(f"GPU target: {gpu_target}")

# Load export info if available
export_info = {}
export_info_path = model_dir / "export_info.json"
if export_info_path.exists():
    with open(export_info_path) as f:
        export_info = json.load(f)
    print(f"Export info: {export_info.get('shape_mode', 'unknown')} shapes")
    if export_info.get('model_variant'):
        print(f"Model: {export_info['model_variant']}")

# Find model file
model_file = None
for candidate in ["model.onnx", "model_optimized.onnx"]:
    if (model_dir / candidate).exists():
        model_file = model_dir / candidate
        break

if model_file is None:
    onnx_files = list(model_dir.glob("*.onnx"))
    if onnx_files:
        model_file = onnx_files[0]

if model_file is None:
    print(f"Error: No .onnx file found in {model_dir}")
    sys.exit(1)

print(f"\nModel file: {model_file}")
print(f"Available providers: {ort.get_available_providers()}")

# Check GPU memory before loading
try:
    result = subprocess.run(['rocm-smi', '--showmeminfo', 'vram'],
                          capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        print("\nGPU Memory before model load:")
        for line in result.stdout.strip().split('\n'):
            if 'Used' in line or 'GPU' in line:
                print(f"  {line.strip()}")
except:
    pass

# Enable verbose logging for debugging
if verbose:
    # ORT verbose logging
    os.environ['ORT_LOG_LEVEL'] = 'VERBOSE'
    # MIGraphX verbose logging
    os.environ['MIGRAPHX_TRACE_COMPILE'] = '1'
    os.environ['MIGRAPHX_TRACE_EVAL'] = '1'
    os.environ['MIGRAPHX_TRACE_GPU_ALLOC'] = '1'
    # HIP verbose
    os.environ['AMD_LOG_LEVEL'] = '4'
    os.environ['HIP_TRACE_API'] = '1'

# Configure session options
sess_options = ort.SessionOptions()
sess_options.log_severity_level = 0 if verbose else log_level  # 0=VERBOSE
sess_options.log_verbosity_level = 10 if verbose else 0

# CRITICAL: Disable graph optimizations to avoid hipHostRegister issues
# MIGraphX's optimization may be inserting problematic copy operations
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
print("Graph optimizations: DISABLED (workaround for MIGraphX hipHostRegister bug)")

# Enable profiling for detailed timing
if verbose:
    sess_options.enable_profiling = True
    print("Verbose logging enabled (ORT + MIGraphX + HIP)")

# Configure provider options
if provider == "MIGraphXExecutionProvider":
    cache_path = str(model_dir / "migraphx_cache")

    # Define minimum sequence length to avoid hipHostRegister bug
    MIN_SEQ_FOR_MIGRAPHX = 256  # 2048 bytes minimum

    # MIGraphX options MUST be strings, not booleans/integers
    # ALWAYS enable offload_copy to fix hipHostRegister failures on small buffers
    # (attention_mask at 4KB fails GPU registration without this)
    # MIGraphX provider options - trying to work around hipHostRegister bug
    provider_options = {
        'device_id': '0',
        'migraphx_fp16_enable': '1' if migraphx_fp16 else '0',
        'migraphx_exhaustive_tune': '0',  # Disable exhaustive tuning
        'migraphx_offload_copy': '1',  # Should handle small buffers
        # Note: migraphx_enable_gpu is not a valid option, removed
    }

    print(f"\nAttempting workaround for MIGraphX hipHostRegister bug...")
    print(f"  - Graph optimizations disabled")
    print(f"  - offload_copy enabled")
    print(f"  - Testing with min buffer size: {MIN_SEQ_FOR_MIGRAPHX * 8} bytes")

    if not no_cache:
        os.makedirs(cache_path, exist_ok=True)
        provider_options['migraphx_model_cache_dir'] = cache_path
        print(f"MIGraphX cache: {cache_path}")

    print(f"\nMIGraphX options:")
    for k, v in provider_options.items():
        print(f"  {k}: {v}")

    providers = [provider]
    provider_options_list = [provider_options]

elif provider == "ROCMExecutionProvider":
    providers = [provider]
    provider_options_list = [{
        'device_id': 0,
        'tunable_op_enable': True,
        'tunable_op_tuning_enable': False,
    }]
elif provider == "CUDAExecutionProvider":
    providers = [provider]
    provider_options_list = [{'device_id': 0}]
else:
    providers = [provider]
    provider_options_list = [{}]

# Check if we should use IOBinding to avoid hipHostRegister
use_io_binding = provider == "MIGraphXExecutionProvider"
if use_io_binding:
    print("\nUsing IOBinding to pre-allocate inputs on GPU (avoids hipHostRegister)")

# Create session
print(f"\nCreating session with {provider}...")
print("  (First run may take time for MIGraphX compilation)")

start_load = time.time()

try:
    print(f"\nAttempting to create session with providers: {providers}")
    print(f"Provider options: {provider_options_list}")

    session = ort.InferenceSession(
        str(model_file),
        sess_options,
        providers=providers,
        provider_options=provider_options_list
    )
    load_time = time.time() - start_load
    print(f"Session created in {load_time:.2f}s")

except Exception as e:
    print(f"❌ {provider} failed: {e}")
    print(f"\n   For MIGraphX issues, try:")
    print(f"   1. Check GPU target matches: rocminfo | grep gfx")
    print(f"   2. Try CPU provider: ./09_run_inference_test.sh {model_dir} CPUExecutionProvider")
    print(f"\n   Full error:")
    import traceback
    traceback.print_exc()
    raise

# Verify which provider is actually being used
actual_providers = session.get_providers()
print(f"Session providers: {actual_providers}")

if provider != "CPUExecutionProvider" and actual_providers == ['CPUExecutionProvider']:
    print(f"⚠️  WARNING: Requested {provider} but fell back to CPU!")
    print("   This may indicate the model has unsupported operators.")
else:
    print(f"✅ Running on: {actual_providers[0]}")

# Check GPU memory after loading
if provider != "CPUExecutionProvider":
    try:
        result = subprocess.run(['rocm-smi', '--showmeminfo', 'vram'],
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("\nGPU Memory after model load:")
            for line in result.stdout.strip().split('\n'):
                if 'Used' in line or 'GPU' in line:
                    print(f"  {line.strip()}")
    except:
        pass

# Get model input/output info
model_inputs = session.get_inputs()
model_outputs = session.get_outputs()

print(f"\nModel inputs ({len(model_inputs)}):")
has_kv_cache = False
num_layers = export_info.get('num_layers', 32)
num_kv_heads = export_info.get('num_kv_heads', 8)
head_dim = export_info.get('head_dim', 128)

# Check for expected inputs
expected_inputs = ['input_ids', 'attention_mask', 'past_seq_len']
actual_input_names = [inp.name for inp in model_inputs]
has_past_seq_len = 'past_seq_len' in actual_input_names
has_position_ids = 'position_ids' in actual_input_names

print(f"  Expected new signature (past_seq_len): {has_past_seq_len}")
print(f"  Old signature (position_ids): {has_position_ids}")

if not has_past_seq_len and has_position_ids:
    print(f"\n⚠️  WARNING: Model still has old signature!")
    print(f"  You need to RE-EXPORT the model with the updated export_model.py")
    print(f"  Current model was exported before the past_seq_len changes.")

for inp in model_inputs[:5]:
    shape_str = str(inp.shape)
    is_dynamic = any(isinstance(d, str) or d is None or d == -1 for d in inp.shape)
    print(f"  {inp.name}: {shape_str} {'[dynamic]' if is_dynamic else '[fixed]'}")
    if 'past_key' in inp.name or 'cache' in inp.name:
        has_kv_cache = True

if len(model_inputs) > 5:
    print(f"  ... and {len(model_inputs) - 5} more")

print(f"\nModel outputs ({len(model_outputs)}):")
for out in model_outputs[:3]:
    print(f"  {out.name}: {out.shape}")
if len(model_outputs) > 3:
    print(f"  ... and {len(model_outputs) - 3} more")

# Load tokenizer
print("\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Detect model type from tokenizer/config
model_type = "unknown"
try:
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    model_type = getattr(config, 'model_type', 'unknown')
except:
    pass

# Fallback detection from tokenizer
if model_type == "unknown":
    if hasattr(tokenizer, 'name_or_path'):
        name_lower = tokenizer.name_or_path.lower()
        if 'llama' in name_lower:
            model_type = 'llama'
        elif 'mistral' in name_lower:
            model_type = 'mistral'
        elif 'qwen' in name_lower:
            model_type = 'qwen2'
        elif 'phi' in name_lower:
            model_type = 'phi3'

print(f"Detected model type: {model_type}")

# Detect model dtype
model_dtype = np.float16  # Default for modern models
for inp in model_inputs:
    if "float16" in str(inp.type).lower():
        model_dtype = np.float16
        break
    elif "float32" in str(inp.type).lower():
        model_dtype = np.float32
print(f"Model dtype: {model_dtype}")

# Format prompt using chat template
print(f"\n{'='*60}")
print("USER PROMPT:")
print(f"{'='*60}")
print(prompt)
print(f"{'='*60}")

# Apply chat template if available
messages = [{"role": "user", "content": prompt}]
formatted_prompt = None

if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template is not None:
    try:
        # Use tokenizer's built-in chat template
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        print(f"\nUsing tokenizer chat template")
    except Exception as e:
        print(f"Chat template failed: {e}, using raw prompt")

# Fallback: manual templates for common models
if formatted_prompt is None:
    if model_type in ['llama', 'llama3']:
        # Llama 3.x format
        formatted_prompt = (
            f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            f"{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        print(f"\nUsing Llama 3 chat format")
    elif model_type == 'mistral':
        # Mistral format
        formatted_prompt = f"[INST] {prompt} [/INST]"
        print(f"\nUsing Mistral chat format")
    elif model_type == 'qwen2':
        # Qwen2 format
        formatted_prompt = (
            f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        )
        print(f"\nUsing Qwen2 chat format")
    elif model_type == 'phi3':
        # Phi-3 format
        formatted_prompt = f"<|user|>\n{prompt}<|end|>\n<|assistant|>\n"
        print(f"\nUsing Phi-3 chat format")
    else:
        # Generic fallback
        formatted_prompt = prompt
        print(f"\nUsing raw prompt (no chat template)")

print(f"\nFORMATTED PROMPT:")
print("-" * 60)
print(formatted_prompt[:500] + "..." if len(formatted_prompt) > 500 else formatted_prompt)
print("-" * 60)

# Tokenize formatted prompt
inputs = tokenizer(formatted_prompt, return_tensors="np", add_special_tokens=False)
input_ids = inputs["input_ids"].astype(np.int64)
raw_prompt_len = input_ids.shape[1]
print(f"Formatted prompt tokens: {raw_prompt_len}")

# Truncate if prompt exceeds max context
if seq_length > 0 and raw_prompt_len > seq_length:
    print(f"WARNING: Prompt ({raw_prompt_len}) exceeds max context ({seq_length}), truncating")
    input_ids = input_ids[:, -seq_length:]  # Keep last seq_length tokens
    raw_prompt_len = input_ids.shape[1]

prompt_len = raw_prompt_len
print(f"Prompt length: {prompt_len}")

# Sampling function
def sample_token(logits, temperature=0.0):
    """Sample next token from logits."""
    if temperature <= 0:
        # Greedy
        return np.argmax(logits)
    else:
        # Temperature sampling
        logits = logits / temperature
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        return np.random.choice(len(probs), p=probs)

# ============================================================
# AUTOREGRESSIVE GENERATION
# ============================================================
# Use larger batch sizes for prefill to avoid reshape errors with MIGraphX.
# After export fix: attention_mask must have same sequence dimension as input_ids.
# We use a static shape for all operations to avoid recompilation.
#
# filled_kv tracks how many positions contain valid data (0 to KV_LEN).

print(f"\nGenerating up to {max_tokens} tokens...")
print("-" * 60)

generated_ids = input_ids[0].tolist()
eos_token_id = tokenizer.eos_token_id

# Use a larger sequence length for both prefill and decode to avoid reshape errors
# This allows batched prefill processing while maintaining consistent shapes
# CRITICAL: MIGraphX hipHostRegister bug - HIP memory pool minimum is ~2KB
# Using 256 elements (2048 bytes) to avoid the bug
# Note: MIN_SEQ_FOR_MIGRAPHX is defined earlier if using MIGraphX
if 'MIN_SEQ_FOR_MIGRAPHX' not in locals():
    MIN_SEQ_FOR_MIGRAPHX = 256  # Default if not using MIGraphX
PREFILL_SEQ_LEN = max(MIN_SEQ_FOR_MIGRAPHX, min(256, seq_length))
DECODE_SEQ_LEN = PREFILL_SEQ_LEN       # Use same shape for decode (not optimal but avoids recompile)
KV_LEN = seq_length                     # e.g., 256 - KV cache size

print(f"Static shapes: prefill_seq={PREFILL_SEQ_LEN}, decode_seq={DECODE_SEQ_LEN}, kv={KV_LEN}")
print(f"Note: decode uses same seq length as prefill to avoid MIGraphX recompilation")

# Pre-allocate buffers with static shapes
# For prefill: process multiple tokens at once
# For decode: use same shape but only fill first position (inefficient but avoids recompile)
# Note: position_ids is computed internally in ONNX graph from past_seq_len scalar
prefill_input_ids = np.zeros((1, PREFILL_SEQ_LEN), dtype=np.int64)
prefill_attention_mask = np.zeros((1, PREFILL_SEQ_LEN), dtype=np.int64)

decode_input_ids = np.zeros((1, DECODE_SEQ_LEN), dtype=np.int64)
decode_attention_mask = np.zeros((1, DECODE_SEQ_LEN), dtype=np.int64)

print(f"Buffers: prefill={prefill_input_ids.shape}, decode={decode_input_ids.shape}")

# Fixed-size KV cache buffer
kv_cache = {}
for layer_idx in range(num_layers):
    kv_cache[layer_idx] = {
        'key': np.zeros((1, num_kv_heads, KV_LEN, head_dim), dtype=model_dtype),
        'value': np.zeros((1, num_kv_heads, KV_LEN, head_dim), dtype=model_dtype),
    }

print(f"KV cache allocated: {num_layers} layers, shape per layer: {kv_cache[0]['key'].shape}")

# Track how many positions are filled (valid data in the static buffer)
filled_kv = 0  # 0 to KV_LEN

# Timing
total_start = time.time()
decode_times = []
new_token_ids = []
prompt_tokens = generated_ids.copy()

def run_batch_prefill(tokens, start_position, kv_cache, filled_kv):
    """
    Run inference for a batch of tokens during prefill.

    Args:
        tokens: List of token IDs (up to PREFILL_SEQ_LEN)
        start_position: Starting position index
        kv_cache: KV cache dict (will be updated)
        filled_kv: Current filled positions in KV cache

    Returns:
        logits (for last token), kv_cache, new_filled_kv
    """
    batch_size = len(tokens)
    assert batch_size <= PREFILL_SEQ_LEN, f"Batch {batch_size} exceeds {PREFILL_SEQ_LEN}"

    # Fill buffers
    prefill_input_ids.fill(0)
    prefill_attention_mask.fill(0)

    for i, token_id in enumerate(tokens):
        prefill_input_ids[0, i] = token_id
        prefill_attention_mask[0, i] = 1  # Mark valid positions

    # Create past_seq_len padded tensor (256 elements = 2048 bytes to avoid hipHostRegister failures)
    # Only first element is used; rest is padding
    past_seq_len_padded = np.zeros(256, dtype=np.int64)
    past_seq_len_padded[0] = start_position

    # Build feed dict
    feed_dict = {
        "input_ids": prefill_input_ids,
        "attention_mask": prefill_attention_mask,
        "past_seq_len": past_seq_len_padded,
    }

    for inp in model_inputs:
        if "past_key_values" in inp.name:
            layer_idx = int(inp.name.split('.')[1])
            if ".key" in inp.name:
                feed_dict[inp.name] = kv_cache[layer_idx]['key']
            elif ".value" in inp.name:
                feed_dict[inp.name] = kv_cache[layer_idx]['value']

    # Run inference
    outputs = session.run(None, feed_dict)

    # Extract and store KV cache updates
    output_idx = 1
    for layer_idx in range(num_layers):
        out_key = outputs[output_idx]
        out_value = outputs[output_idx + 1]

        # Copy new KV entries (only valid positions)
        for i in range(batch_size):
            if filled_kv + i < KV_LEN:
                kv_cache[layer_idx]['key'][:, :, filled_kv + i, :] = out_key[:, :, i, :]
                kv_cache[layer_idx]['value'][:, :, filled_kv + i, :] = out_value[:, :, i, :]

        output_idx += 2

    new_filled = min(filled_kv + batch_size, KV_LEN)

    # Return logits for last token
    logits = outputs[0]
    return logits[0, batch_size - 1, :], kv_cache, new_filled


def run_single_decode(token_id, position, kv_cache, filled_kv):
    """
    Run inference for single token during decode phase.
    Uses same shape as prefill (DECODE_SEQ_LEN) but only fills first position.

    Args:
        token_id: Token ID to process
        position: Position index
        kv_cache: KV cache dict (will be updated)
        filled_kv: Current filled positions in KV cache

    Returns:
        logits, kv_cache, new_filled_kv
    """
    # Fill buffers (only first position used)
    decode_input_ids.fill(0)
    decode_attention_mask.fill(0)

    decode_input_ids[0, 0] = token_id
    decode_attention_mask[0, 0] = 1  # Only current token is valid

    # Create past_seq_len padded tensor (256 elements = 2048 bytes to avoid hipHostRegister failures)
    # Only first element is used; rest is padding
    past_seq_len_padded = np.zeros(256, dtype=np.int64)
    past_seq_len_padded[0] = position

    # Build feed dict
    feed_dict = {
        "input_ids": decode_input_ids,
        "attention_mask": decode_attention_mask,
        "past_seq_len": past_seq_len_padded,
    }

    for inp in model_inputs:
        if "past_key_values" in inp.name:
            layer_idx = int(inp.name.split('.')[1])
            if ".key" in inp.name:
                feed_dict[inp.name] = kv_cache[layer_idx]['key']
            elif ".value" in inp.name:
                feed_dict[inp.name] = kv_cache[layer_idx]['value']

    # Run inference
    outputs = session.run(None, feed_dict)

    # Extract and store KV cache update
    output_idx = 1
    for layer_idx in range(num_layers):
        out_key = outputs[output_idx]
        out_value = outputs[output_idx + 1]

        # Copy new KV entry (only first position is valid)
        if filled_kv < KV_LEN:
            kv_cache[layer_idx]['key'][:, :, filled_kv, :] = out_key[:, :, 0, :]
            kv_cache[layer_idx]['value'][:, :, filled_kv, :] = out_value[:, :, 0, :]

        output_idx += 2

    new_filled = min(filled_kv + 1, KV_LEN)

    # Return logits for the token
    logits = outputs[0]
    return logits[0, 0, :], kv_cache, new_filled


# ========== PREFILL (BATCHED) ==========
# Process prompt in batches for faster prefill
prefill_start = time.time()

n_prompt = len(prompt_tokens)
print(f"[Prefill: {n_prompt} tokens in batches of {PREFILL_SEQ_LEN}]")

position = 0
for i in range(0, n_prompt, PREFILL_SEQ_LEN):
    batch_tokens = prompt_tokens[i:i + PREFILL_SEQ_LEN]
    logits, kv_cache, filled_kv = run_batch_prefill(batch_tokens, position, kv_cache, filled_kv)
    position += len(batch_tokens)

    if (i + len(batch_tokens)) % (PREFILL_SEQ_LEN * 2) == 0 or i + len(batch_tokens) >= n_prompt:
        print(f"  [Prefill: {i + len(batch_tokens)}/{n_prompt}, KV: {filled_kv}/{KV_LEN}]", end='\r')

print()  # Newline
prefill_time = time.time() - prefill_start
print(f"[Prefill complete: {len(prompt_tokens)} tokens in {prefill_time*1000:.0f}ms")
print(f" Throughput: {len(prompt_tokens)/prefill_time:.1f} tok/s]")
print(f"[KV filled: {filled_kv}/{KV_LEN}]")
print("\nASSISTANT:")
print("-" * 60)

# Sample first token from prefill logits
next_token_id = sample_token(logits, temperature)
generated_ids.append(int(next_token_id))
new_token_ids.append(int(next_token_id))

# Print first token
token_str = tokenizer.decode([next_token_id], skip_special_tokens=True)
sys.stdout.write(token_str)
sys.stdout.flush()

# Track position for decode
current_position = len(prompt_tokens)

# ========== DECODE ==========
# Each decode step processes one token (uses same shape as prefill for consistency)
for step in range(max_tokens - 1):  # -1 because we already generated 1
    # Check stopping conditions
    if next_token_id == eos_token_id:
        break
    if tokenizer.decode([next_token_id]) in ['<|eot_id|>', '<|end|>', '<|im_end|>', '</s>']:
        break

    # Check if KV buffer is full
    if filled_kv >= KV_LEN:
        print(f"\n[KV buffer full at {KV_LEN}, stopping]")
        break

    step_start = time.time()

    # Process single token (uses DECODE_SEQ_LEN shape)
    logits, kv_cache, filled_kv = run_single_decode(
        next_token_id, current_position, kv_cache, filled_kv
    )

    decode_times.append(time.time() - step_start)
    current_position += 1

    # Sample next token
    next_token_id = sample_token(logits, temperature)
    generated_ids.append(int(next_token_id))
    new_token_ids.append(int(next_token_id))

    # Print token
    token_str = tokenizer.decode([next_token_id], skip_special_tokens=True)
    sys.stdout.write(token_str)
    sys.stdout.flush()

print()  # New line

total_time = time.time() - total_start
print()
print("-" * 60)

# ============================================================
# RESULTS
# ============================================================
# Generated tokens count excludes padding
generated_tokens = len(new_token_ids)

# Decode only the assistant's response (new tokens)
assistant_response = tokenizer.decode(new_token_ids, skip_special_tokens=True).strip()

print(f"\n{'='*60}")
print("ASSISTANT RESPONSE (clean):")
print(f"{'='*60}")
print(assistant_response)
print(f"{'='*60}")

# Performance stats
print(f"\n{'='*60}")
print("PERFORMANCE SUMMARY")
print(f"{'='*60}")
print(f"Provider:           {actual_providers[0]}")
print(f"Model type:         {model_type}")
print(f"Shapes:             prefill_seq={PREFILL_SEQ_LEN}, decode_seq={DECODE_SEQ_LEN}, kv={KV_LEN}")
print(f"KV filled:          {filled_kv}/{KV_LEN}")
print(f"Prompt tokens:      {raw_prompt_len}")
print(f"Generated tokens:   {generated_tokens}")
print(f"Total context:      {raw_prompt_len + generated_tokens}")
print(f"Temperature:        {temperature}")
print(f"-" * 60)
print(f"Model load time:    {load_time*1000:.0f} ms")
if prefill_time > 0:
    print(f"Prefill time:       {prefill_time*1000:.0f} ms ({raw_prompt_len/prefill_time:.1f} tok/s)")
if decode_times:
    avg_decode = np.mean(decode_times) * 1000
    print(f"Avg decode time:    {avg_decode:.2f} ms/token")
    print(f"Decode throughput:  {1000/avg_decode:.1f} tokens/sec")
if total_time > 0 and generated_tokens > 0:
    print(f"Total gen time:     {total_time*1000:.0f} ms")
    print(f"Overall tok/sec:    {generated_tokens/total_time:.1f}")
print(f"{'='*60}")

# Check stopping reason
if new_token_ids and new_token_ids[-1] == eos_token_id:
    print("\n✅ Generation stopped at EOS token")
elif generated_tokens >= max_tokens:
    print(f"\n✅ Generation stopped at max output ({max_tokens} tokens)")
else:
    print("\n✅ Generation stopped at model stop token")

print("\n✅ Text generation complete!")
