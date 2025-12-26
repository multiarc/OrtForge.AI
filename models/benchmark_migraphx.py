#!/usr/bin/env python3
"""MIGraphX benchmark script for ONNX models with KV cache."""

import argparse
import json
import os
import time
import numpy as np
import onnxruntime as ort

# Log severity levels:
# 0 = VERBOSE (all messages)
# 1 = INFO
# 2 = WARNING (default, shows WARNING and above)
# 3 = ERROR
# 4 = FATAL


def detect_model_dtype(model_path):
    """Detect if model uses FP16 or FP32 by checking input types."""
    import onnx
    model = onnx.load(model_path, load_external_data=False)
    
    for inp in model.graph.input:
        elem_type = inp.type.tensor_type.elem_type
        # Check tensor inputs (skip int64 inputs like input_ids)
        if elem_type == onnx.TensorProto.FLOAT16:
            return np.float16
        elif elem_type == onnx.TensorProto.FLOAT:
            return np.float32
    
    # Default to float16 for modern models
    return np.float16


def main():
    parser = argparse.ArgumentParser(description="Benchmark MIGraphX inference")
    parser.add_argument("model_dir", help="Directory containing model.onnx and export_info.json")
    parser.add_argument("--iterations", "-n", type=int, default=100, help="Number of benchmark iterations (default: 100)")
    parser.add_argument("--warmup", "-w", type=int, default=5, help="Number of warmup iterations (default: 5)")
    parser.add_argument("--seq-length", type=int, default=256, 
                       help="Bucket size: prompt padded to this, KV cache = 2×this (default: 256)")
    parser.add_argument("--no-cache", action="store_true", help="Disable model caching")
    parser.add_argument("--convert-fp16", action="store_true", help="Force FP32->FP16 conversion (not needed if model is already FP16)")
    parser.add_argument("--exhaustive-tune", action="store_true", help="Enable exhaustive tuning")
    parser.add_argument("--offload-copy", action="store_true", help="Use CPU memory during compilation (reduces GPU memory usage)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging (shows all ORT messages)")
    parser.add_argument("--log-level", type=int, default=2, choices=[0, 1, 2, 3, 4],
                       help="Log severity level: 0=VERBOSE, 1=INFO, 2=WARNING (default), 3=ERROR, 4=FATAL")
    parser.add_argument("--quiet", "-q", action="store_true", help="Only show final results (no per-iteration output)")
    args = parser.parse_args()
    
    # Configure logging - must be done before creating any session
    log_level = 0 if args.verbose else args.log_level
    ort.set_default_logger_severity(log_level)
    log_level_names = {0: "VERBOSE", 1: "INFO", 2: "WARNING", 3: "ERROR", 4: "FATAL"}
    if not args.quiet:
        print(f"ORT Log Level: {log_level_names.get(log_level, log_level)}")

    model_path = os.path.join(args.model_dir, "model.onnx")
    info_path = os.path.join(args.model_dir, "export_info.json")
    cache_path = os.path.join(args.model_dir, "migraphx_cache")

    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return 1

    if not os.path.exists(info_path):
        print(f"Error: Export info not found at {info_path}")
        return 1

    with open(info_path) as f:
        info = json.load(f)

    num_layers = info["num_layers"]
    num_kv_heads = info["num_kv_heads"]
    head_dim = info["head_dim"]
    
    # Detect model dtype
    model_dtype = detect_model_dtype(model_path)
    dtype_name = "FP16" if model_dtype == np.float16 else "FP32"
    
    # Benchmark simulates decode step (seq_len=1, kv_len=bucket)
    # This represents generating tokens after a prompt of `bucket` tokens
    seq_len = 1  # Decode: one token at a time
    kv_len = args.seq_length  # KV cache = past context (the prompt)
    
    print("=" * 60)
    print("MIGraphX Benchmark (Decode Phase)")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Model dtype: {dtype_name}")
    print(f"Layers: {num_layers}, KV Heads: {num_kv_heads}, Head Dim: {head_dim}")
    print(f"Decode: seq_len=1, kv_len={kv_len}")
    print(f"  (simulates generating after {kv_len}-token prompt)")
    print(f"Iterations: {args.iterations} (warmup: {args.warmup})")
    print(f"Force FP16 conversion: {args.convert_fp16}")
    print(f"Caching: {not args.no_cache}")
    print(f"Exhaustive Tune: {args.exhaustive_tune}")
    print(f"Offload Copy (CPU compile): {args.offload_copy}")
    print()

    # Configure provider - only enable fp16 conversion if explicitly requested
    # Models already in FP16 don't need conversion (saves memory)
    provider_options = {
        "device_id": "0",
        "migraphx_fp16_enable": "1" if args.convert_fp16 else "0",
        "migraphx_exhaustive_tune": "1" if args.exhaustive_tune else "0",
        "migraphx_offload_copy": "1" if args.offload_copy else "0",
    }

    if not args.no_cache:
        os.makedirs(cache_path, exist_ok=True)
        provider_options["migraphx_model_cache_dir"] = cache_path
        print(f"Cache path: {cache_path}")

    # Create session - MIGraphX only, no CPU fallback
    print("\nCreating session (MIGraphX only, no fallback)...")
    t0 = time.time()
    sess_options = ort.SessionOptions()
    sess_options.log_severity_level = log_level
    sess_options.log_verbosity_level = 10 if args.verbose else 0  # Higher = more verbose
    
    try:
        session = ort.InferenceSession(
            model_path,
            sess_options,
            providers=["MIGraphXExecutionProvider"],
            provider_options=[provider_options],
        )
    except Exception as e:
        print(f"\nERROR: MIGraphX session creation failed!")
        print(f"Exception: {e}")
        print("\nThis means MIGraphX is not working properly.")
        return 1
    
    session_time = time.time() - t0
    print(f"Session created in {session_time:.2f}s")
    
    active_providers = session.get_providers()
    print(f"Active providers: {active_providers}")
    
    if "MIGraphXExecutionProvider" not in active_providers:
        print("\nERROR: MIGraphX is not active!")
        return 1

    # Build inputs for decode benchmark
    # Only include inputs that the model actually expects
    model_inputs = session.get_inputs()
    input_names = [inp.name for inp in model_inputs]
    
    dtype = model_dtype
    attn_len = seq_len + kv_len  # attention covers current + past
    
    feed = {}
    
    if "input_ids" in input_names:
        feed["input_ids"] = np.ones((1, seq_len), dtype=np.int64)
    
    if "attention_mask" in input_names:
        feed["attention_mask"] = np.ones((1, attn_len), dtype=np.int64)
    
    if "position_ids" in input_names:
        # Position for decode = kv_len (next position after past context)
        feed["position_ids"] = np.array([[kv_len]], dtype=np.int64)

    # KV cache tensors (filled with random data to simulate real cache)
    for i in range(num_layers):
        key_name = f"past_key_values.{i}.key"
        value_name = f"past_key_values.{i}.value"
        if key_name in input_names:
            feed[key_name] = np.random.randn(1, num_kv_heads, kv_len, head_dim).astype(dtype)
        if value_name in input_names:
            feed[value_name] = np.random.randn(1, num_kv_heads, kv_len, head_dim).astype(dtype)

    # Calculate memory footprint
    total_bytes = sum(v.nbytes for v in feed.values())
    print(f"\nInputs: {len(feed)} tensors, {total_bytes / 1024 / 1024:.2f} MB")

    # Warmup
    print(f"Running {args.warmup} warmup iterations...")
    warmup_times = []
    for i in range(args.warmup):
        t0 = time.time()
        outputs = session.run(None, feed)
        warmup_times.append(time.time() - t0)
        if not args.quiet:
            print(f"  Warmup {i+1}: {warmup_times[-1]*1000:.2f}ms")
    
    print(f"Warmup avg: {np.mean(warmup_times)*1000:.2f}ms")
    print(f"Output shape: {outputs[0].shape}")

    # Benchmark
    print(f"\nBenchmarking ({args.iterations} iterations)...")
    times = []
    
    # Progress reporting
    report_interval = max(1, args.iterations // 10)  # Report ~10 times
    
    for i in range(args.iterations):
        t0 = time.time()
        outputs = session.run(None, feed)
        elapsed = time.time() - t0
        times.append(elapsed)
        
        if not args.quiet and ((i + 1) % report_interval == 0 or i == 0):
            avg_so_far = np.mean(times) * 1000
            print(f"  [{i+1}/{args.iterations}] Current: {elapsed*1000:.2f}ms, Avg: {avg_so_far:.2f}ms")

    # Results
    times_ms = np.array(times) * 1000
    avg_ms = np.mean(times_ms)
    min_ms = np.min(times_ms)
    max_ms = np.max(times_ms)
    std_ms = np.std(times_ms)
    p50_ms = np.percentile(times_ms, 50)
    p90_ms = np.percentile(times_ms, 90)
    p99_ms = np.percentile(times_ms, 99)

    print()
    print("=" * 60)
    print("Results (Decode Phase)")
    print("=" * 60)
    print(f"Iterations:      {args.iterations}")
    print(f"Decode shape:    seq={seq_len}, kv={kv_len}")
    print(f"Context length:  {kv_len} tokens")
    print()
    print(f"Average latency: {avg_ms:.2f}ms")
    print(f"Std deviation:   {std_ms:.2f}ms")
    print(f"Min latency:     {min_ms:.2f}ms")
    print(f"Max latency:     {max_ms:.2f}ms")
    print()
    print(f"P50 latency:     {p50_ms:.2f}ms")
    print(f"P90 latency:     {p90_ms:.2f}ms")
    print(f"P99 latency:     {p99_ms:.2f}ms")
    print()
    print(f"Throughput:      {1000/avg_ms:.1f} inferences/sec")
    print(f"Tokens/sec:      {args.seq_length * 1000/avg_ms:.1f} (output tokens)")
    print()

    return 0


if __name__ == "__main__":
    exit(main())
