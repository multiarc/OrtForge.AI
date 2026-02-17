#!/bin/bash
# =============================================================================
# check_migraphx_support.sh - Check MIGraphX compatibility and operator support
# =============================================================================
# Usage: ./check_migraphx_support.sh [model.onnx]
#
# Without arguments: runs GPU and MIGraphX diagnostics only
# With model path: also checks operator support for the model
# =============================================================================

set -e

MODEL_FILE="${1:-}"

echo "=============================================="
echo "MIGraphX Compatibility Check"
echo "=============================================="

# GPU Information
echo ""
echo "[1] GPU Information"
echo "----------------------------------------------"
GPU_TARGET=$(rocminfo 2>/dev/null | grep -oP 'gfx\d+' | head -1 || echo "unknown")
GPU_NAME=$(rocminfo 2>/dev/null | grep "Marketing Name:" | head -1 | cut -d: -f2 | xargs || echo "unknown")
echo "GPU Target: $GPU_TARGET"
echo "GPU Name:   $GPU_NAME"

# ROCm Version
echo ""
echo "[2] ROCm / MIGraphX Version"
echo "----------------------------------------------"
ROCM_VERSION=$(cat /opt/rocm/.info/version 2>/dev/null || echo "not found")
echo "ROCm: $ROCM_VERSION"

if command -v migraphx-driver &> /dev/null; then
    MIGRAPHX_VERSION=$(migraphx-driver --version 2>/dev/null | head -1 || echo "error")
    echo "MIGraphX: $MIGRAPHX_VERSION"
else
    echo "MIGraphX: migraphx-driver not found"
fi

# ONNX Runtime
echo ""
echo "[3] ONNX Runtime"
echo "----------------------------------------------"
python3 -c "
import onnxruntime as ort
print(f'Version: {ort.__version__}')
print(f'Providers: {ort.get_available_providers()}')
has_migraphx = 'MIGraphXExecutionProvider' in ort.get_available_providers()
print(f'MIGraphX EP: {\"✓ Available\" if has_migraphx else \"✗ Not available\"}')" 2>/dev/null || echo "ONNX Runtime not installed"

# Simple MIGraphX test
echo ""
echo "[4] MIGraphX Compilation Test"
echo "----------------------------------------------"
python3 << 'PYTEST'
import os
import sys

try:
    import onnxruntime as ort
    import tempfile
    import numpy as np
    
    # Create minimal ONNX model for testing (use opset 17 for max compatibility)
    import onnx
    from onnx import helper, TensorProto
    
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 4])
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 4])
    relu_node = helper.make_node('Relu', ['X'], ['Y'])
    graph = helper.make_graph([relu_node], 'test', [X], [Y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)])
    model.ir_version = 8  # Compatible with older ONNX Runtime builds
    
    with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
        onnx.save(model, f.name)
        temp_path = f.name
    
    # Test MIGraphX
    sess_options = ort.SessionOptions()
    provider_options = {'device_id': 0, 'migraphx_fp16_enable': False}
    
    session = ort.InferenceSession(
        temp_path,
        sess_options,
        providers=['MIGraphXExecutionProvider'],
        provider_options=[provider_options]
    )
    
    # Run inference
    x = np.random.randn(1, 4).astype(np.float32)
    result = session.run(None, {'X': x})
    
    os.unlink(temp_path)
    
    actual = session.get_providers()
    if 'MIGraphXExecutionProvider' in actual:
        print("✓ MIGraphX compilation: SUCCESS")
        print("✓ MIGraphX inference: SUCCESS")
    else:
        print(f"⚠ Fell back to: {actual}")
        
except Exception as e:
    print(f"✗ MIGraphX test failed: {e}")
    import traceback
    traceback.print_exc()
PYTEST

# Check for model file
if [ -z "$MODEL_FILE" ]; then
    echo ""
    echo "=============================================="
    echo "Done (no model specified)"
    echo "=============================================="
    echo ""
    echo "To check operator support for a model:"
    echo "  $0 <model.onnx>"
    exit 0
fi

if [ ! -f "$MODEL_FILE" ]; then
    echo ""
    echo "Error: File not found: $MODEL_FILE"
    exit 1
fi

echo ""
echo "=============================================="
echo "Model Operator Support Check"
echo "=============================================="
echo "Model: $MODEL_FILE"

# Method 1: Try to parse with migraphx-driver
echo ""
echo "Method 1: migraphx-driver parse test"
echo "----------------------------------------------"
if command -v migraphx-driver &> /dev/null; then
    echo "Running: migraphx-driver read --onnx $MODEL_FILE"
    migraphx-driver read --onnx "$MODEL_FILE" 2>&1 | head -100 || true
else
    echo "migraphx-driver not found"
fi

# Method 2: Check operators against known MIGraphX support list
echo ""
echo "Method 2: Operator analysis"
echo "----------------------------------------------"

python3 << EOF
import onnx
import os

model_path = "$MODEL_FILE"

print(f"Loading model: {model_path}")
model = onnx.load(model_path, load_external_data=False)

# Count operators
op_counts = {}
for node in model.graph.node:
    op_counts[node.op_type] = op_counts.get(node.op_type, 0) + 1

print(f"\nModel has {len(model.graph.node)} nodes, {len(op_counts)} unique operators")

# Known MIGraphX supported operators (as of MIGraphX 2.x)
# This list is approximate - check MIGraphX docs for exact support
MIGRAPHX_SUPPORTED = {
    # Basic
    'Add', 'Sub', 'Mul', 'Div', 'Pow', 'Sqrt', 'Exp', 'Log',
    'Abs', 'Neg', 'Ceil', 'Floor', 'Round',
    'Relu', 'LeakyRelu', 'Elu', 'Selu', 'Sigmoid', 'Tanh', 'Softmax', 'LogSoftmax',
    'Clip', 'Min', 'Max', 'Sum', 'Mean',
    # Reduction
    'ReduceSum', 'ReduceMean', 'ReduceMax', 'ReduceMin', 'ReduceProd',
    'ReduceL1', 'ReduceL2', 'ReduceLogSum', 'ReduceLogSumExp',
    # Matrix
    'MatMul', 'Gemm', 'MatMulInteger',
    # Convolution
    'Conv', 'ConvTranspose', 'AveragePool', 'MaxPool', 'GlobalAveragePool', 'GlobalMaxPool',
    # Normalization
    'BatchNormalization', 'InstanceNormalization', 'LRN',
    'LayerNormalization',  # Limited support
    # Shape
    'Reshape', 'Flatten', 'Squeeze', 'Unsqueeze', 'Transpose',
    'Concat', 'Split', 'Slice', 'Gather', 'GatherElements',
    'Shape', 'Size', 'Tile', 'Expand', 'Pad',
    # Cast/Convert
    'Cast', 'CastLike',
    # Logic
    'Equal', 'Less', 'Greater', 'LessOrEqual', 'GreaterOrEqual',
    'And', 'Or', 'Not', 'Xor', 'Where',
    # Other common
    'Identity', 'Dropout', 'Constant', 'ConstantOfShape',
    'Range', 'Einsum',
    # Attention (limited)
    'Attention', 'MultiHeadAttention',
}

# Operators with known issues in MIGraphX
MIGRAPHX_PROBLEMATIC = {
    'SimplifiedLayerNormalization',  # May not be supported
    'RotaryEmbedding',               # Custom op
    'GatherND',                      # Limited support
    'ScatterND',                     # Limited support
    'NonZero',                       # Dynamic output shape
    'Loop', 'If', 'Scan',            # Control flow
    'LSTM', 'GRU', 'RNN',            # Recurrent (limited)
    'Resize',                        # Some modes not supported
    'GridSample',                    # Limited
}

print("\n" + "=" * 60)
print("OPERATOR SUPPORT ANALYSIS")
print("=" * 60)

supported = {}
unsupported = {}
problematic = {}
unknown = {}

for op, count in sorted(op_counts.items(), key=lambda x: -x[1]):
    if op in MIGRAPHX_SUPPORTED:
        supported[op] = count
    elif op in MIGRAPHX_PROBLEMATIC:
        problematic[op] = count
    elif op.startswith('com.') or op.startswith('ai.') or 'Custom' in op:
        unsupported[op] = count
    else:
        unknown[op] = count

print(f"\n✅ SUPPORTED ({len(supported)} types, {sum(supported.values())} nodes):")
for op, count in sorted(supported.items(), key=lambda x: -x[1])[:15]:
    print(f"   {op}: {count}")
if len(supported) > 15:
    print(f"   ... and {len(supported) - 15} more")

if problematic:
    print(f"\n⚠️  PROBLEMATIC ({len(problematic)} types, {sum(problematic.values())} nodes):")
    for op, count in sorted(problematic.items(), key=lambda x: -x[1]):
        print(f"   {op}: {count}")

if unsupported:
    print(f"\n❌ UNSUPPORTED ({len(unsupported)} types, {sum(unsupported.values())} nodes):")
    for op, count in sorted(unsupported.items(), key=lambda x: -x[1]):
        print(f"   {op}: {count}")

if unknown:
    print(f"\n❓ UNKNOWN STATUS ({len(unknown)} types, {sum(unknown.values())} nodes):")
    for op, count in sorted(unknown.items(), key=lambda x: -x[1]):
        print(f"   {op}: {count}")

# Check for dynamic shapes (problematic for MIGraphX)
print("\n" + "=" * 60)
print("DYNAMIC SHAPE ANALYSIS")
print("=" * 60)

dynamic_inputs = []
for inp in model.graph.input:
    shape = []
    if inp.type.tensor_type.shape.dim:
        for dim in inp.type.tensor_type.shape.dim:
            if dim.dim_param:
                shape.append(dim.dim_param)
            elif dim.dim_value:
                shape.append(dim.dim_value)
            else:
                shape.append('?')
        if any(isinstance(s, str) for s in shape):
            dynamic_inputs.append((inp.name, shape))

if dynamic_inputs:
    print("⚠️  Model has dynamic input shapes:")
    for name, shape in dynamic_inputs:
        print(f"   {name}: {shape}")
    print("\n   MIGraphX requires fixed shapes. Dynamic shapes may cause issues.")
else:
    print("✅ All inputs have fixed shapes")

# Check data types
print("\n" + "=" * 60)
print("DATA TYPE ANALYSIS")
print("=" * 60)

dtype_map = {
    1: 'float32', 2: 'uint8', 3: 'int8', 4: 'uint16', 5: 'int16',
    6: 'int32', 7: 'int64', 9: 'bool', 10: 'float16', 11: 'double',
    12: 'uint32', 13: 'uint64', 14: 'complex64', 15: 'complex128',
    16: 'bfloat16'
}

initializer_dtypes = {}
for init in model.graph.initializer:
    dtype = dtype_map.get(init.data_type, f'unknown({init.data_type})')
    initializer_dtypes[dtype] = initializer_dtypes.get(dtype, 0) + 1

print("Initializer (weight) data types:")
for dtype, count in sorted(initializer_dtypes.items(), key=lambda x: -x[1]):
    print(f"   {dtype}: {count}")

if 'float16' in initializer_dtypes:
    print("\n⚠️  Model has FP16 weights - ensure MIGraphX FP16 mode is enabled")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

total_nodes = len(model.graph.node)
supported_nodes = sum(supported.values())
problematic_nodes = sum(problematic.values())
unsupported_nodes = sum(unsupported.values())
unknown_nodes = sum(unknown.values())

print(f"Total nodes: {total_nodes}")
print(f"Likely supported: {supported_nodes} ({100*supported_nodes/total_nodes:.1f}%)")
print(f"Potentially problematic: {problematic_nodes} ({100*problematic_nodes/total_nodes:.1f}%)")
print(f"Likely unsupported: {unsupported_nodes} ({100*unsupported_nodes/total_nodes:.1f}%)")
print(f"Unknown: {unknown_nodes} ({100*unknown_nodes/total_nodes:.1f}%)")

if problematic_nodes > 0 or unsupported_nodes > 0 or unknown_nodes > total_nodes * 0.1:
    print("\n⚠️  This model may have compatibility issues with MIGraphX")
    print("   Try:")
    print("   1. Check if operators are supported in your MIGraphX version")
    print("   2. Use CPU provider for testing: CPUExecutionProvider")
    print("   3. File an issue with MIGraphX for unsupported operators")
EOF

echo ""
echo "=============================================="
echo "Done"
echo "=============================================="

