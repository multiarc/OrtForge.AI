# RDNA3 GPU Compatibility Guide

## Problem Overview

The `GroupQueryAttention` operator in ONNX Runtime ROCm is optimized specifically for AMD's CDNA2 and CDNA3 data center architectures (MI250X, MI300 series). Consumer RDNA3 GPUs like the **RX 7900 XTX** are not supported by this operator, resulting in the following errors:

```
GroupQueryAttention currently only supports ck_tile fmha backend which only supports CDNA2 and CDNA3 archs.
GroupQueryAttention running on an unsuppoted GPU may result in hipErrorNoBinaryForGpu or hipErrorSharedObjectInitFailedshared error.
```

## Architecture Differences

- **RDNA3**: Consumer gaming GPU architecture (RX 7900 XTX, RX 7800 XT, etc.)
- **CDNA2/CDNA3**: Data center compute architectures (MI250X, MI300 series, etc.)

## Solutions

### Option 1: Environment Variable Override (Recommended)

Try this first - it tricks ROCm into thinking your RDNA3 GPU is a CDNA3 GPU:

```bash
export HSA_OVERRIDE_GFX_VERSION=10.3.0
# Then run your application
```

### Option 2: Use RDNA3 Compatible Mode (Built-in)

The OrtRuntimeFactory now includes RDNA3 compatibility mode by default. You can also explicitly choose:

```csharp
// Explicit RDNA3 compatibility mode
var session = OrtRuntimeFactory.CreateSession(modelPath, GpuCompatibilityMode.RDNA3Compatible);

// Or CPU-only for maximum compatibility
var session = OrtRuntimeFactory.CreateSession(modelPath, GpuCompatibilityMode.CpuOnly);

// Or standard mode for CDNA2/CDNA3 GPUs
var session = OrtRuntimeFactory.CreateSession(modelPath, GpuCompatibilityMode.Standard);
```

### Option 3: Limit ROCm Visibility

If you have multiple GPUs and some are unsupported:

```bash
export HIP_VISIBLE_DEVICES=0  # Only use first GPU
export ROCR_VISIBLE_DEVICES="0,GPU-your-gpu-uuid"
```

## Performance Expectations

| Mode | GPU Usage | CPU Usage | Performance | Compatibility |
|------|-----------|-----------|-------------|---------------|
| Standard | Full | Fallback | Best | CDNA2/3 only |
| RDNA3Compatible | Partial | Fallback | Good | RDNA3 + CDNA |
| CpuOnly | None | Full | Slower | Universal |

## Compatibility Settings Explained

### RDNA3Compatible Mode
- Uses `GraphOptimizationLevel.ORT_ENABLE_BASIC` to avoid problematic operator fusions
- Maintains ROCm + CPU execution provider setup for automatic fallback
- Allows unsupported operators (like GroupQueryAttention) to fall back to CPU
- Maintains GPU acceleration for supported operations

### What Runs Where
- **GPU (ROCm)**: Matrix operations, embeddings, most computations
- **CPU (Fallback)**: GroupQueryAttention operators, unsupported ops
- **Hybrid**: Tensors automatically transferred between devices

## Troubleshooting

### If you still get errors:
1. Verify ROCm installation: `rocminfo`
2. Check GPU visibility: `echo $HIP_VISIBLE_DEVICES`
3. Try CPU-only mode for testing
4. Enable ONNX Runtime logging for detailed operator placement

### Performance Optimization
- Use Float16 models when possible (faster on GPU)
- Monitor GPU utilization: `rocm-smi`
- Consider batch size adjustments for RDNA3

## Model Compatibility

| Model Type | RDNA3 Compatibility | Notes |
|------------|-------------------|-------|
| Llama 3.2 | ✅ Good | Uses GQA, benefits from hybrid execution |
| Llama 3.1 | ✅ Good | Uses GQA, benefits from hybrid execution |
| BGE-M3 | ✅ Excellent | No GQA operators |
| Reranker | ✅ Excellent | No GQA operators |

## Future Improvements

AMD is working on broader RDNA support in ROCm. Monitor these repositories:
- [ROCm ONNX Runtime](https://github.com/ROCm/onnxruntime)
- [Composable Kernels](https://github.com/ROCm/composable_kernel)

## Getting Help

If you continue experiencing issues:
1. Check ROCm version compatibility
2. Verify your ONNX model doesn't require CDNA-specific features
3. Consider using models exported specifically for RDNA3
