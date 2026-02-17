# ONNX Model Export and Optimization Scripts

Scripts for exporting HuggingFace models to ONNX and running inference with MIGraphX/ROCm.

## Requirements

```bash
pip install torch transformers onnx onnxruntime onnxconverter-common
```

For MIGraphX support, ensure ROCm and MIGraphX are installed.

## Quick Start

### Full Pipeline (Recommended)

```bash
# Make scripts executable
chmod +x *.sh

# Export and test with MIGraphX (default GPU workflow)
./export_pipeline.sh /path/to/Llama3.1-8B-Instruct/hf ./Llama3.1-8B-Instruct/onnx

# Pre-compile for common KV cache lengths (recommended for production)
./export_pipeline.sh ./model/hf ./model/onnx --precompile

# Benchmark with specific context length
./export_pipeline.sh ./model/hf ./model/onnx --benchmark-only --kv-length 512 -n 500

# CPU target with optimization
./export_pipeline.sh ./model/hf ./model/onnx --cpu
```

## Default Settings (Optimized for Inference)

All exports use these **inference-optimized defaults**:

| Setting | Default | Description |
|---------|---------|-------------|
| **KV Cache** | ✅ ENABLED | Essential for efficient autoregressive generation |
| **Precision** | FP16 | Faster inference, lower memory |
| **Shapes** | Dynamic | Any batch/sequence length at runtime |
| **Caching** | ✅ ENABLED | MIGraphX compiled models cached in `migraphx_cache/` |

```python
import onnxruntime as ort

session = ort.InferenceSession(
    'model.onnx',
    providers=['MIGraphXExecutionProvider'],
    provider_options=[{
        'device_id': 0,
        'migraphx_model_cache_dir': './migraphx_cache',
    }]
)

# Works with any sequence length
outputs = session.run(None, {
    'input_ids': input_ids,        # shape: (batch, any_seq_len)
    'attention_mask': attention_mask,
    # ... KV cache tensors ...
})
```

## MIGraphX Shape Compilation

**Important:** MIGraphX requires fixed shapes at compile time. Each unique `(seq_length, kv_length)` combination requires a separate compiled model (~3 min each for 8B models).

### Automatic Caching

The MIGraphX EP automatically caches compiled models. First inference with new shapes triggers compilation; subsequent runs use the cache.

### Pre-compilation (Recommended for Production)

Pre-compile common shapes to avoid runtime compilation delays:

```bash
# Pre-compile with defaults (buckets 0-64K, seq-lengths 1,4,16,64)
python precompile_shapes.py ./Llama3.1-8B-Instruct/onnx

# Custom buckets (smaller set for faster compilation)
python precompile_shapes.py ./onnx --buckets "0,512,1024,2048,4096,8192"

# Custom sequence lengths
python precompile_shapes.py ./onnx --seq-lengths "1,4" --buckets "0,1024,4096,16384"
```

### Shape Bucketing Strategy

For efficient production use, implement shape bucketing:

```python
BUCKETS = [0, 128, 256, 512, 1024, 2048, 4096]

def get_bucket(actual_kv_length):
    """Find smallest bucket >= actual_length"""
    for b in BUCKETS:
        if b >= actual_kv_length:
            return b
    return BUCKETS[-1]

# Pad KV cache to bucket size for cache hits
kv_length = get_bucket(actual_context_length)
```

## Workflows

### GPU Target (Default)

```
Export (dynamic) → Validate → Test (MIGraphX EP) → Benchmark
```

```bash
./export_pipeline.sh ./model/hf ./model/onnx

# With pre-compilation:
./export_pipeline.sh ./model/hf ./model/onnx --precompile

# With custom benchmark settings:
./export_pipeline.sh ./model/hf ./model/onnx --seq-length 1 --kv-length 512 -n 500
```

### CPU Target

```
Export (dynamic) → Validate → Optimize (FP16) → Test
```

```bash
./export_pipeline.sh ./model/hf ./model/onnx --cpu
```

### INT4/INT8 Quantization (CPU Only)

```bash
# INT4 (~75% size reduction)
./export_pipeline.sh ./model/hf ./model/onnx --int4

# INT8 (~50% size reduction)
./export_pipeline.sh ./model/hf ./model/onnx --int8
```

**Note**: Quantized models use operators MIGraphX doesn't support. Use CPU for quantized inference.

## Benchmark Script

The Python benchmark script provides detailed performance metrics:

```bash
# Basic benchmark (100 iterations)
python benchmark_migraphx.py ./Llama3.1-8B-Instruct/onnx

# With context (simulates decoding with 512-token history)
python benchmark_migraphx.py ./onnx --seq-length 1 --kv-length 512

# Extended benchmark with verbose logging
python benchmark_migraphx.py ./onnx -n 500 --verbose

# Quick test with minimal output
python benchmark_migraphx.py ./onnx -n 50 --quiet
```

### Benchmark Options

| Option | Default | Description |
|--------|---------|-------------|
| `-n, --iterations` | 100 | Number of benchmark iterations |
| `-w, --warmup` | 5 | Warmup iterations |
| `--seq-length` | 1 | Input sequence length (new tokens) |
| `--kv-length` | 0 | KV cache length (context tokens) |
| `--exhaustive-tune` | off | Exhaustive MIGraphX tuning |
| `--offload-copy` | off | Use CPU memory during compilation |
| `-v, --verbose` | off | Verbose ORT logging |
| `-q, --quiet` | off | Minimal output |
| `--no-cache` | off | Disable model caching |

### Benchmark Output

```
============================================================
Results
============================================================
Iterations:      100
Input tokens:    1
Context tokens:  512

Average latency: 25.43ms
Std deviation:   1.23ms
Min latency:     23.12ms
Max latency:     31.45ms

P50 latency:     25.21ms
P90 latency:     26.89ms
P99 latency:     29.12ms

Throughput:      39.3 inferences/sec
Tokens/sec:      39.3 (output tokens)
```

## Scripts Reference

| Script | Description |
|--------|-------------|
| `export_pipeline.sh` | **Main orchestration script** - runs full workflow |
| `01_export_model.sh` | Export HuggingFace model to ONNX (dynamic shapes) |
| `02_fix_external_data.sh` | Convert large models (>2GB) to external data format |
| `03_validate_model.sh` | Validate ONNX model structure |
| `04_optimize_model.sh` | Optimize for ONNX Runtime (attention fusion + FP16) |
| `05_quantize_int4.sh` | INT4 weight quantization |
| `05_quantize_int8.sh` | INT8 dynamic quantization |
| `06_convert_fp16.sh` | Convert weights to FP16 (standalone) |
| `precompile_shapes.py` | **Pre-compile MIGraphX for multiple shapes** |
| `08_benchmark_migraphx.sh` | Benchmark wrapper script |
| `09_run_inference_test.sh` | Quick inference test |
| `benchmark_migraphx.py` | **Python benchmark script** with detailed metrics |

## Manual Step-by-Step

```bash
chmod +x *.sh

# 1. Export model to ONNX (FP16 + KV cache by default)
./01_export_model.sh /path/to/model/hf ./output

# 2. Fix external data (if model > 2GB)
./02_fix_external_data.sh ./output/model.onnx

# 3. Validate
./03_validate_model.sh ./output/model.onnx

# 4. Test inference with MIGraphX
./09_run_inference_test.sh ./output MIGraphXExecutionProvider

# 5. Pre-compile common shapes (uses defaults: buckets 0-64K, seq 1,4,16,64)
python precompile_shapes.py ./output

# 6. Benchmark with context
python benchmark_migraphx.py ./output --seq-length 1 --kv-length 512 -n 100
```

## Pipeline Options

### Target Selection

| Option | Description |
|--------|-------------|
| `--gpu` | Target GPU with MIGraphX (default) |
| `--cpu` | Target CPU |
| `--int4` | INT4 quantization (CPU only) |
| `--int8` | INT8 quantization (CPU only) |

### Export Options

| Option | Description |
|--------|-------------|
| `--opset <n>` | ONNX opset version (default: auto-detect, max 21) |
| `--no-kv-cache` | Disable KV cache (not recommended for inference) |
| `--fp32` | Export in FP32 instead of FP16 |

### MIGraphX Options

| Option | Description |
|--------|-------------|
| `--precompile` | Pre-compile for common KV cache lengths |
| `--exhaustive` | Enable exhaustive tuning (slower compile, faster inference) |
| `--offload-copy` | Use CPU memory during compilation |

### Benchmarking Options

| Option | Description |
|--------|-------------|
| `--seq-length <n>` | Input sequence length (default: 1) |
| `--kv-length <n>` | KV cache length / context (default: 0) |
| `--iterations <n>` | Benchmark iterations (default: 100) |
| `--skip-benchmark` | Skip benchmarking step |
| `--benchmark-only` | Only run benchmark (model must exist) |
| `--verbose` | Enable verbose logging |

### Other Options

| Option | Description |
|--------|-------------|
| `--dry-run` | Show what would be executed |
| `-h, --help` | Show help |

## Environment Variables

### MIGraphX Options

| Variable | Default | Description |
|----------|---------|-------------|
| `MIGRAPHX_FP16` | `0` | Enable FP16 conversion (not needed for FP16 models) |

### Benchmark Options

| Variable | Default | Description |
|----------|---------|-------------|
| `SEQ_LENGTH` | `1` | Input sequence length |
| `KV_LENGTH` | `0` | KV cache length |
| `ITERATIONS` | `100` | Number of iterations |
| `WARMUP` | `5` | Warmup iterations |

## Examples

```bash
# Basic export and test (FP16 + KV cache enabled by default)
./export_pipeline.sh ./model/hf ./model/onnx

# Export with pre-compilation for production
./export_pipeline.sh ./model/hf ./model/onnx --precompile

# Benchmark with 512-token context (simulates decoding)
python benchmark_migraphx.py ./model/onnx --seq-length 1 --kv-length 512 -n 500

# Pre-compile with defaults (9 buckets × 4 seq-lengths = 36 shapes)
python precompile_shapes.py ./model/onnx

# Quick inference test with verbose logging
./09_run_inference_test.sh ./model/onnx MIGraphXExecutionProvider --verbose

# Export without KV cache (not recommended)
./01_export_model.sh ./model/hf ./output --no-kv-cache

# Export in FP32 precision
./01_export_model.sh ./model/hf ./output --fp32
```

## Supported Models (Auto-Detected)

| Model | hidden_size | num_heads | num_kv_heads | num_layers |
|-------|-------------|-----------|--------------|------------|
| **Llama 3.2 1B** | 2048 | 32 | 8 | 16 |
| **Llama 3.2 3B** | 3072 | 24 | 8 | 28 |
| **Llama 3.1 8B** | 4096 | 32 | 8 | 32 |
| **Llama 3.1 70B** | 8192 | 64 | 8 | 80 |
| **Llama 3.1 405B** | 16384 | 128 | 8 | 126 |
| **Mistral 7B** | 4096 | 32 | 8 | 32 |

## Execution Providers

| Provider | Use Case |
|----------|----------|
| `MIGraphXExecutionProvider` | AMD GPUs with MIGraphX (recommended) |
| `ROCMExecutionProvider` | AMD GPUs with ROCm (deprecated in ORT 1.23+) |
| `CUDAExecutionProvider` | NVIDIA GPUs |
| `CPUExecutionProvider` | CPU fallback |

## Troubleshooting

### Model > 2GB protobuf error
```bash
./02_fix_external_data.sh ./output/model.onnx
```

### MIGraphX falls back to CPU
Check if all operators are supported:
```bash
python benchmark_migraphx.py ./model/onnx --verbose 2>&1 | grep -i "fallback\|cpu"
```

### Slow first inference
MIGraphX JIT-compiles on first run. Pre-compile to avoid:
```bash
python precompile_shapes.py ./model/onnx
```

### INT4 not working with MIGraphX
INT4 uses `GatherBlockQuantized` which MIGraphX doesn't support. Use CPU:
```bash
./09_run_inference_test.sh ./model/onnx CPUExecutionProvider
```

### Different KV lengths cause recompilation
MIGraphX requires fixed shapes. Use shape bucketing:
```bash
# Pre-compile all default shapes
python precompile_shapes.py ./model/onnx

# Then pad actual KV cache to nearest bucket at runtime
```

### Out of memory during compilation
Use offload copy to use CPU memory during compilation:
```bash
python benchmark_migraphx.py ./model/onnx --offload-copy
# Or
./export_pipeline.sh ./model/hf ./model/onnx --offload-copy
```

### Verbose logging for debugging
```bash
python benchmark_migraphx.py ./model/onnx --verbose
# Or
./09_run_inference_test.sh ./model/onnx MIGraphXExecutionProvider --verbose
```
