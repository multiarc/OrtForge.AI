#!/bin/bash
# =============================================================================
# 04_optimize_model.sh - Optimize ONNX model for ONNX Runtime inference
# =============================================================================
# Usage: ./04_optimize_model.sh <input.onnx> <output.onnx> [model_type]
#
# This script optimizes ONNX models for ONNX Runtime execution (CPU or GPU EP).
# It fuses attention patterns into efficient operators (MultiHeadAttention/GQA)
# which MIGraphX can then accelerate with Flash Attention kernels.
#
# Environment Variables:
#   SKIP_FP16=true       - Skip FP16 conversion (for quantized models)
#   OPT_LEVEL=<0-2>      - Optimization level (default: 1)
#   USE_GPU=true         - Use GPU for optimization (enables more fusions)
#   ATTENTION_TYPE=<type> - Force attention type: MultiHeadAttention, GroupQueryAttention
#
# Model parameters are auto-detected from config.json in the model directory.
# =============================================================================

set -e

INPUT_FILE="${1:?Usage: $0 <input.onnx> <output.onnx> [model_type]}"
OUTPUT_FILE="${2:?Usage: $0 <input.onnx> <output.onnx> [model_type]}"
MODEL_TYPE="${3:-gpt_neox}"  # gpt_neox is compatible with LLaMA

if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: File not found: $INPUT_FILE"
    exit 1
fi

INPUT_DIR=$(dirname "$INPUT_FILE")
INPUT_BASE=$(basename "$INPUT_FILE" .onnx)

# Settings from environment
SKIP_FP16="${SKIP_FP16:-false}"
OPT_LEVEL="${OPT_LEVEL:-1}"

# =============================================================================
# Auto-detect model configuration
# =============================================================================
CONFIG_FILE="$INPUT_DIR/config.json"
if [ -f "$CONFIG_FILE" ]; then
    echo "Auto-detecting model parameters from config.json..."
    
    DETECTED_PARAMS=$(python3 << EOF
import json
with open("$CONFIG_FILE", "r") as f:
    config = json.load(f)

hidden_size = config.get("hidden_size", 4096)
num_heads = config.get("num_attention_heads", 32)
num_kv_heads = config.get("num_key_value_heads", num_heads)
num_layers = config.get("num_hidden_layers", 32)

# Model variant
variants = {2048: "Llama_3.2_1B", 3072: "Llama_3.2_3B", 4096: "Llama_3.1_8B", 
            8192: "Llama_3.1_70B", 16384: "Llama_3.1_405B"}
variant = variants.get(hidden_size, f"Unknown_{hidden_size}")

print(f'MODEL_VARIANT="{variant}"')
print(f'NUM_HEADS="{num_heads}"')
print(f'HIDDEN_SIZE="{hidden_size}"')
print(f'NUM_KV_HEADS="{num_kv_heads}"')
print(f'NUM_LAYERS="{num_layers}"')
EOF
)
    eval "$DETECTED_PARAMS"
else
    echo "No config.json found, using defaults..."
    NUM_HEADS="32"
    HIDDEN_SIZE="4096"
    MODEL_VARIANT="Unknown"
fi

# =============================================================================
# Check for quantized models (skip FP16)
# =============================================================================
IS_QUANTIZED=false
if [[ "$INPUT_BASE" == *"int4"* ]] || [[ "$INPUT_BASE" == *"int8"* ]]; then
    IS_QUANTIZED=true
    SKIP_FP16=true
fi

# Check for quantization ops in model
if [ "$IS_QUANTIZED" = false ]; then
    QUANT_CHECK=$(python3 -c "
import onnx
model = onnx.load('$INPUT_FILE', load_external_data=False)
quant_ops = {'MatMulNBits', 'QLinearMatMul', 'MatMulInteger', 'DequantizeLinear'}
print('QUANTIZED' if set(n.op_type for n in model.graph.node) & quant_ops else '')
" 2>/dev/null || echo "")
    [ "$QUANT_CHECK" = "QUANTIZED" ] && IS_QUANTIZED=true && SKIP_FP16=true
fi

# =============================================================================
# Print configuration
# =============================================================================
echo ""
echo "=============================================="
echo "Optimize ONNX Model for ONNX Runtime"
echo "=============================================="
echo "Input:        $INPUT_FILE"
echo "Output:       $OUTPUT_FILE"
echo "Model:        $MODEL_VARIANT"
echo "Heads:        $NUM_HEADS (KV: ${NUM_KV_HEADS:-$NUM_HEADS})"
echo "Hidden size:  $HIDDEN_SIZE"
echo "----------------------------------------------"
echo "FP16:         $([ "$SKIP_FP16" = true ] && echo 'disabled' || echo 'enabled')"
echo "Quantized:    $([ "$IS_QUANTIZED" = true ] && echo 'yes' || echo 'no')"
echo "Opt level:    $OPT_LEVEL"
echo "=============================================="
echo ""

# =============================================================================
# Check external data
# =============================================================================
USE_EXTERNAL=""
if [ -f "$INPUT_DIR/${INPUT_BASE}.onnx.data" ] || [ -f "$INPUT_DIR/${INPUT_BASE}.onnx_data" ]; then
    echo "External data detected, will preserve in output..."
    USE_EXTERNAL="--use_external_data_format"
fi

# Check for oversized model
ONNX_SIZE=$(stat -c%s "$INPUT_FILE" 2>/dev/null || stat -f%z "$INPUT_FILE" 2>/dev/null || echo "0")
if [ "$ONNX_SIZE" -gt 2147483648 ]; then
    echo "⚠️  ONNX file exceeds 2GB protobuf limit!"
    echo "   Run: ./02_fix_external_data.sh $INPUT_FILE"
    exit 1
fi

# =============================================================================
# GPU/Provider settings
# =============================================================================
USE_GPU="${USE_GPU:-true}"
ATTENTION_TYPE="${ATTENTION_TYPE:-auto}"

# Check for MIGraphX provider
if [ "$USE_GPU" = true ]; then
    HAS_MIGRAPHX=$(python3 -c "import onnxruntime as ort; print('yes' if 'MIGraphXExecutionProvider' in ort.get_available_providers() else 'no')" 2>/dev/null || echo "no")
    if [ "$HAS_MIGRAPHX" = "yes" ]; then
        echo "MIGraphX EP detected - will optimize for Flash Attention"
        PROVIDER="MIGraphXExecutionProvider"
    else
        echo "MIGraphX not available, using CPU optimization"
        USE_GPU=false
        PROVIDER="CPUExecutionProvider"
    fi
else
    PROVIDER="CPUExecutionProvider"
fi

# =============================================================================
# Run optimizer with FusionOptions for efficient attention
# =============================================================================
echo ""
echo "Running ONNX Runtime transformer optimizer..."
echo "   Enabling attention fusion for MIGraphX Flash Attention support"
echo ""

# Export variables for Python
export INPUT_FILE OUTPUT_FILE MODEL_TYPE NUM_HEADS HIDDEN_SIZE NUM_KV_HEADS
export OPT_LEVEL SKIP_FP16 USE_GPU ATTENTION_TYPE

# Run Python script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 "$SCRIPT_DIR/py/optimize_model.py"

if [ $? -eq 0 ]; then
    echo ""
    ls -lh "$OUTPUT_FILE"
else
    echo "❌ Optimization failed"
    exit 1
fi
