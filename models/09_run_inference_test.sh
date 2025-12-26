#!/bin/bash
# =============================================================================
# 09_run_inference_test.sh - Test inference with ONNX Runtime
# =============================================================================
# Usage: ./09_run_inference_test.sh <model_dir> [provider] [options]
# 
# Runs text generation to verify the model works correctly.
# Uses autoregressive generation with growing KV cache.
#
# Providers:
#   MIGraphXExecutionProvider  - AMD GPU with MIGraphX (default)
#   ROCMExecutionProvider      - AMD GPU with ROCm
#   CUDAExecutionProvider      - NVIDIA GPU
#   CPUExecutionProvider       - CPU fallback
#
# Options:
#   --prompt <text>      Custom prompt (default: "What is 2+2?")
#   --seq-length <n>     Static input sequence length (default: 256)
#                        Used for BOTH prefill and decode stages.
#                        Inputs are left-padded to this size.
#   --temperature <f>    Sampling temperature (default: 0.0 = greedy)
#   --verbose            Enable verbose ORT logging
#   --no-cache           Disable model caching
#   --exhaustive         Enable exhaustive tuning
#   --offload-copy       Use CPU memory during compilation
#   --help               Show this help
#
# KV Cache Strategy (FULLY STATIC shapes):
#   ALL shapes are FIXED to avoid MIGraphX recompilation and
#   hipHostRegister failures on small arrays.
#   
#   Fixed shapes:
#   - input_ids: (1, SEQ_LEN) - always 1 (matches benchmark)
#   - position_ids: (1, SEQ_LEN) - always 1
#   - attention_mask: (1, ATTN_LEN) - always 257 (KV_LEN + SEQ_LEN)
#   - past_key_values: (1, h, KV_LEN, d) - always 256
#
#   Model outputs KV of shape (KV_LEN + SEQ_LEN), we extract new KV
#   and copy it into the STATIC buffer at position filled_kv.
#
# Environment Variables:
#   VERBOSE=true           Enable verbose ORT + MIGraphX + HIP logging
#   MIGRAPHX_FP16=1        Enable FP16 mode (default: disabled for pre-FP16 models)
#   MIGRAPHX_SAVE_MODEL=1  Save compiled model
#
# Examples:
#   ./09_run_inference_test.sh ./Llama3.1-8B-Instruct/onnx
#   ./09_run_inference_test.sh ./onnx --prompt "Explain quantum computing"
#   ./09_run_inference_test.sh ./onnx --seq-length 256 --temperature 0.7
# =============================================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Parse arguments
POSITIONAL=()
PROMPT="What is 2+2?"
SEQ_LENGTH=256  # Default bucket size (max_output = seq_length)
TEMPERATURE=0.0
VERBOSE=false
NO_CACHE=false
EXHAUSTIVE=false
OFFLOAD_COPY=true  # Default to offload for large models
MIGRAPHX_FP16="${MIGRAPHX_FP16:-0}"
MIGRAPHX_SAVE="${MIGRAPHX_SAVE_MODEL:-1}"

while [[ $# -gt 0 ]]; do
    case $1 in
        --prompt)
            PROMPT="$2"
            shift 2
            ;;
        --seq-length)
            SEQ_LENGTH="$2"
            shift 2
            ;;
        --temperature)
            TEMPERATURE="$2"
            shift 2
            ;;
        --verbose|-v)
            VERBOSE=true
            shift
            ;;
        --no-cache)
            NO_CACHE=true
            shift
            ;;
        --exhaustive)
            EXHAUSTIVE=true
            shift
            ;;
        --offload-copy)
            OFFLOAD_COPY=true
            shift
            ;;
        --no-offload-copy)
            OFFLOAD_COPY=false
            shift
            ;;
        --help|-h)
            head -40 "$0" | tail -37
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
MODEL_DIR="${1:?Usage: $0 <model_dir> [provider] [options]}"
# MIGraphX provider - we use GPU OrtValues to avoid hipHostRegister issues
PROVIDER="${2:-MIGraphXExecutionProvider}"

if [ ! -d "$MODEL_DIR" ]; then
    echo "Error: Directory not found: $MODEL_DIR"
    exit 1
fi

echo "=============================================="
echo "ONNX Runtime Text Generation Test"
echo "=============================================="
echo "Model dir:   $MODEL_DIR"
echo "Provider:    $PROVIDER"
echo "Prompt:      \"$PROMPT\""
echo "Max context: $SEQ_LENGTH tokens"
echo "Max output:  $SEQ_LENGTH tokens"
echo "Temperature: $TEMPERATURE"
if [ "$PROVIDER" = "MIGraphXExecutionProvider" ]; then
    echo "FP16 convert: $MIGRAPHX_FP16"
    echo "Caching:     $([ "$NO_CACHE" = true ] && echo 'disabled' || echo 'enabled')"
    echo "Exhaustive:  $EXHAUSTIVE"
    echo "Offload:     $OFFLOAD_COPY"
fi
echo "=============================================="

# Auto-detect GPU target for ROCm
GPU_TARGET=$(rocminfo 2>/dev/null | grep -oP 'gfx\d+' | head -1 || echo "")
if [ -n "$GPU_TARGET" ]; then
    if [[ "$GPU_TARGET" == gfx11* ]]; then
        echo "Detected RDNA3 GPU: $GPU_TARGET"
    fi
fi

export MODEL_DIR PROVIDER PROMPT SEQ_LENGTH TEMPERATURE VERBOSE NO_CACHE EXHAUSTIVE OFFLOAD_COPY
export MIGRAPHX_FP16 MIGRAPHX_SAVE GPU_TARGET

python3 << 'PYTHON_SCRIPT'
import os
import sys
import onnxruntime as ort
import numpy as np
from pathlib import Path
import time
import json
import subprocess
from transformers import AutoTokenizer

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
    exit(1)

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

# Enable profiling for detailed timing
if verbose:
    sess_options.enable_profiling = True
    print("Verbose logging enabled (ORT + MIGraphX + HIP)")

# Configure provider options
if provider == "MIGraphXExecutionProvider":
    cache_path = str(model_dir / "migraphx_cache")
    
    # MIGraphX options MUST be strings, not booleans/integers
    # ALWAYS enable offload_copy to fix hipHostRegister failures on small buffers
    # (attention_mask at 4KB fails GPU registration without this)
    provider_options = {
        'device_id': '0',
        'migraphx_fp16_enable': '1' if migraphx_fp16 else '0',
        'migraphx_exhaustive_tune': '1' if exhaustive else '0',
        'migraphx_offload_copy': '1',  # Required for reliable inference
    }
    
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

# Create session
print(f"\nCreating session with {provider}...")
print("  (First run may take time for MIGraphX compilation)")

start_load = time.time()

try:
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
# FULLY STATIC shapes to avoid MIGraphX recompilation:
# - input_ids: (1, SEQ_LEN) - always 1 (matches benchmark)
# - position_ids: (1, SEQ_LEN) - always 1
# - attention_mask: (1, ATTN_LEN) - always 257 (KV_LEN + SEQ_LEN)
# - past_key_values: (1, h, KV_LEN, d) - always 256
#
# filled_kv tracks how many positions contain valid data (0 to KV_LEN).
# attention_mask marks filled_kv positions + valid input tokens as 1.

print(f"\nGenerating up to {max_tokens} tokens...")
print("-" * 60)

generated_ids = input_ids[0].tolist()
eos_token_id = tokenizer.eos_token_id

# MATCH BENCHMARK SHAPES EXACTLY to use the same compiled MIGraphX program
# Benchmark uses: seq_len=1, kv_len=256, attn_len=257
# This avoids hipHostRegister failures that occur with different shapes
SEQ_LEN = 1             # Always process 1 token at a time (like benchmark)
KV_LEN = seq_length     # e.g., 256 - KV cache size  
ATTN_LEN = KV_LEN + SEQ_LEN  # e.g., 257 - attention covers past + current

print(f"Using benchmark-compatible shapes: seq_len={SEQ_LEN}, kv_len={KV_LEN}, attn_len={ATTN_LEN}")

# Pre-allocate buffers with EXACT benchmark shapes
input_ids_buffer = np.zeros((1, SEQ_LEN), dtype=np.int64)
position_ids_buffer = np.zeros((1, SEQ_LEN), dtype=np.int64)
attention_mask_buffer = np.zeros((1, ATTN_LEN), dtype=np.int64)

print(f"Pre-allocated buffers: input_ids={input_ids_buffer.shape}, position_ids={position_ids_buffer.shape}, attention_mask={attention_mask_buffer.shape}")

# Fixed-size KV cache buffer (matches benchmark: kv_len=256)
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

def run_single_token(token_id, position, kv_cache, filled_kv):
    """
    Run inference for a SINGLE token - matches benchmark_migraphx.py exactly.
    
    Uses fixed shapes: seq_len=1, kv_len=256, attn_len=257
    This ensures we use the same compiled MIGraphX program as the benchmark.
    
    Args:
        token_id: Single token ID to process
        position: Position index for this token
        kv_cache: KV cache dict with shape (1, h, KV_LEN, d)
        filled_kv: Number of valid positions in KV cache (0 to KV_LEN)
    
    Returns:
        logits, updated_kv_cache, new_filled_kv
    """
    # Set input_ids: single token
    input_ids_buffer[0, 0] = token_id
    
    # Set position_ids: position for this token
    position_ids_buffer[0, 0] = position
    
    # Attention mask: (1, ATTN_LEN=257) = (1, KV_LEN + SEQ_LEN)
    # First KV_LEN positions are for past KV, last SEQ_LEN positions are for current input
    # Mark filled_kv past positions + 1 current token as attended
    attention_mask_buffer.fill(0)
    attention_mask_buffer[0, :filled_kv] = 1  # Past KV positions
    attention_mask_buffer[0, KV_LEN:KV_LEN + SEQ_LEN] = 1  # Current token position
    
    # Build feed dict
    feed_dict = {}
    for inp in model_inputs:
        if inp.name == "input_ids":
            feed_dict[inp.name] = input_ids_buffer
        elif inp.name == "attention_mask":
            feed_dict[inp.name] = attention_mask_buffer
        elif inp.name == "position_ids":
            feed_dict[inp.name] = position_ids_buffer
        elif "past_key_values" in inp.name:
            layer_idx = int(inp.name.split('.')[1])
            if ".key" in inp.name:
                feed_dict[inp.name] = kv_cache[layer_idx]['key']
            elif ".value" in inp.name:
                feed_dict[inp.name] = kv_cache[layer_idx]['value']
    
    # Debug first few calls
    if filled_kv < 3:
        print(f"\n  [DEBUG] filled_kv={filled_kv}, token_id={token_id}, position={position}")
        print(f"  [DEBUG] input_ids: {input_ids_buffer.shape}, value={input_ids_buffer[0,0]}")
        print(f"  [DEBUG] position_ids: {position_ids_buffer.shape}, value={position_ids_buffer[0,0]}")
        print(f"  [DEBUG] attention_mask: {attention_mask_buffer.shape}, sum={attention_mask_buffer.sum()}")
        print(f"  [DEBUG] kv_cache[0].key: {kv_cache[0]['key'].shape}")
    
    # Run inference
    outputs = session.run(None, feed_dict)
    
    # Model outputs KV with shape (1, h, KV_LEN + SEQ_LEN, d) = (1, h, 257, d)
    # The new KV for this token is at position KV_LEN (index 256)
    output_idx = 1
    out_kv_len = outputs[1].shape[2]
    
    if filled_kv < 3:
        print(f"  [DEBUG] Output KV shape: {outputs[1].shape}")
    
    # Update KV cache: copy new token's KV from output position KV_LEN to filled_kv position
    for layer_idx in range(num_layers):
        out_key = outputs[output_idx]
        out_value = outputs[output_idx + 1]
        
        if filled_kv < KV_LEN:
            # Copy new KV (at output position KV_LEN) to buffer position filled_kv
            kv_cache[layer_idx]['key'][:, :, filled_kv, :] = out_key[:, :, KV_LEN, :]
            kv_cache[layer_idx]['value'][:, :, filled_kv, :] = out_value[:, :, KV_LEN, :]
        else:
            # KV cache full - would need sliding window (stop for now)
            pass
        
        output_idx += 2
    
    # Update filled count
    new_filled_kv = min(filled_kv + 1, KV_LEN)
    
    # Logits - single token output
    logits = outputs[0]
    token_logits = logits[0, -1, :]
    
    return token_logits, kv_cache, new_filled_kv


# ========== PREFILL ==========
# Process tokens ONE AT A TIME to match benchmark shapes exactly
# This uses the same compiled MIGraphX program as the benchmark
prefill_start = time.time()

print(f"[Prefill: {len(prompt_tokens)} tokens (one-by-one, matching benchmark shapes)]")

for i, token_id in enumerate(prompt_tokens):
    logits, kv_cache, filled_kv = run_single_token(
        token_id, i, kv_cache, filled_kv
    )
    if (i + 1) % 10 == 0 or i == len(prompt_tokens) - 1:
        print(f"  [Prefill: {i+1}/{len(prompt_tokens)} tokens, KV: {filled_kv}/{KV_LEN}]", end='\r')

print()  # Newline after progress
prefill_time = time.time() - prefill_start
print(f"[Prefill complete: {len(prompt_tokens)} tokens in {prefill_time*1000:.0f}ms]")
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
# Each decode step adds one token - uses same shapes as benchmark
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
    
    # Process single token (same shapes as benchmark)
    logits, kv_cache, filled_kv = run_single_token(
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
print(f"Static shapes:      seq={SEQ_LEN}, kv={KV_LEN}, attn={ATTN_LEN} (matches benchmark)")
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
PYTHON_SCRIPT
