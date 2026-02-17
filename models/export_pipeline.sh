#!/bin/bash
# =============================================================================
# export_pipeline.sh - ONNX Export and Inference Pipeline
# =============================================================================
# Usage: ./export_pipeline.sh <model_path> <output_dir> [options]
#
# Workflows:
#   GPU (default):  Export → Validate → Test (MIGraphX EP) → Benchmark
#   CPU (--cpu):    Export → Validate → Optimize (FP16) → Test
#   INT4 (--int4):  Export → Validate → INT4 Quantize → Optimize → Test
#   INT8 (--int8):  Export → Validate → INT8 Quantize → Optimize → Test
#
# Defaults (optimized for inference):
#   - KV cache: ENABLED (essential for autoregressive generation)
#   - Precision: FP16 (faster, lower memory)
#   - Shapes: Dynamic (any batch/sequence length)
#
# Options:
#   --gpu              Target MIGraphX (default)
#   --cpu              Target ONNX Runtime CPU
#   --int4             INT4 quantization (CPU only)
#   --int8             INT8 quantization (CPU only)
#   --opset <n>        ONNX opset version (default: auto-detect)
#   --no-kv-cache      Disable KV cache (not recommended)
#   --fp32             Export in FP32 instead of FP16
#   --skip-benchmark   Skip benchmark step
#   --benchmark-only   Only run benchmark (model must exist)
#   --precompile       Pre-compile MIGraphX for common shapes
#   --buckets <list>   Bucket sizes for precompile (default: 256)
#   --seq-length <n>   Bucket size for testing (default: 256)
#                      KV cache = 2 × seq-length, max output = seq-length
#   --iterations <n>   Benchmark iterations (default: 100)
#   --exhaustive       Enable exhaustive MIGraphX tuning
#   --offload-copy     Use CPU memory for MIGraphX compilation
#   --verbose          Enable verbose logging
#   --dry-run          Show commands without executing
#   -h, --help         Show this help
#
# Examples:
#   ./export_pipeline.sh ./Llama3.1-8B/hf ./Llama3.1-8B/onnx
#   ./export_pipeline.sh ./model/hf ./model/onnx --precompile
#   ./export_pipeline.sh ./model/hf ./model/onnx --benchmark-only -n 500
# =============================================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Colors
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; NC='\033[0m'

print_header() { echo -e "\n${BLUE}══════════════════════════════════════════════════════════════════${NC}\n${BLUE}  $1${NC}\n${BLUE}══════════════════════════════════════════════════════════════════${NC}"; }
print_step() { echo -e "${CYAN}▶ $1${NC}"; }
print_ok() { echo -e "${GREEN}✅ $1${NC}"; }
print_warn() { echo -e "${YELLOW}⚠️  $1${NC}"; }
print_err() { echo -e "${RED}❌ $1${NC}"; }

show_help() { head -45 "$0" | tail -43; exit 0; }

# =============================================================================
# Defaults - OPTIMIZED FOR INFERENCE
# =============================================================================
TARGET="gpu"
OPSET=""
NO_KV_CACHE=false
USE_FP32=false
SKIP_BENCHMARK=false
BENCHMARK_ONLY=false
PRECOMPILE=false
DRY_RUN=false
SEQ_LENGTH=256      # Bucket size (KV cache = 2 × this, max output = this)
BUCKETS="256"       # Bucket sizes for precompile
ITERATIONS=100
EXHAUSTIVE=false
OFFLOAD_COPY=true   # Default: offload to CPU during compile
VERBOSE=false

# =============================================================================
# Parse Arguments
# =============================================================================
POSITIONAL=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu)          TARGET="gpu"; shift ;;
        --cpu)          TARGET="cpu"; shift ;;
        --int4)         TARGET="int4"; shift ;;
        --int8)         TARGET="int8"; shift ;;
        --opset)        OPSET="$2"; shift 2 ;;
        --no-kv-cache)  NO_KV_CACHE=true; shift ;;
        --fp32)         USE_FP32=true; shift ;;
        --skip-benchmark) SKIP_BENCHMARK=true; shift ;;
        --benchmark-only) BENCHMARK_ONLY=true; shift ;;
        --precompile)   PRECOMPILE=true; shift ;;
        --buckets)      BUCKETS="$2"; shift 2 ;;
        --seq-length|-s) SEQ_LENGTH="$2"; shift 2 ;;
        --iterations|-n) ITERATIONS="$2"; shift 2 ;;
        --exhaustive)   EXHAUSTIVE=true; shift ;;
        --offload-copy) OFFLOAD_COPY=true; shift ;;
        --no-offload-copy) OFFLOAD_COPY=false; shift ;;
        --verbose|-v)   VERBOSE=true; shift ;;
        --dry-run)      DRY_RUN=true; shift ;;
        -h|--help)      show_help ;;
        -*)             print_err "Unknown option: $1"; exit 1 ;;
        *)              POSITIONAL+=("$1"); shift ;;
    esac
done
set -- "${POSITIONAL[@]}"

if [ ${#POSITIONAL[@]} -lt 2 ]; then
    print_err "Usage: $0 <model_path> <output_dir> [options]"
    exit 1
fi

MODEL_PATH="$1"
OUTPUT_DIR="$2"

# =============================================================================
# Validate
# =============================================================================
if [ "$BENCHMARK_ONLY" = false ]; then
    [ ! -d "$MODEL_PATH" ] && print_err "Model path not found: $MODEL_PATH" && exit 1
fi

for script in 01_export_model.sh 03_validate_model.sh 08_benchmark_migraphx.sh 09_run_inference_test.sh; do
    [ ! -x "$SCRIPT_DIR/$script" ] && chmod +x "$SCRIPT_DIR/$script"
done

mkdir -p "$OUTPUT_DIR"

# =============================================================================
# Auto-detect ONNX opset version if not specified
# =============================================================================
if [ -z "$OPSET" ]; then
    OPSET=$(python3 -c "
import onnx
latest = onnx.defs.onnx_opset_version()
print(min(latest, 21))
" 2>/dev/null || echo "21")
    OPSET_SOURCE="auto-detected"
else
    OPSET_SOURCE="specified"
fi

# =============================================================================
# Configuration Summary
# =============================================================================
print_header "Pipeline Configuration"
echo ""
echo "   Model:      $MODEL_PATH"
echo "   Output:     $OUTPUT_DIR"
echo "   Target:     $TARGET"
echo "   Opset:      $OPSET ($OPSET_SOURCE)"
echo "   Precision:  $([ "$USE_FP32" = true ] && echo 'FP32' || echo 'FP16 ✓')"
echo "   KV cache:   $([ "$NO_KV_CACHE" = true ] && echo 'disabled' || echo 'ENABLED ✓')"
echo "   Shapes:     dynamic"
echo ""
echo "   Inference settings:"
echo "   - Bucket size:  $SEQ_LENGTH (prompt length / context length)"
echo "   - Iterations:   $ITERATIONS"
[ "$EXHAUSTIVE" = true ] && echo "   - Exhaustive tuning: enabled"
[ "$OFFLOAD_COPY" = true ] && echo "   - Offload copy: enabled (CPU memory during compile)"
[ "$PRECOMPILE" = true ] && echo "   - Pre-compile buckets: $BUCKETS"
echo ""

case $TARGET in
    gpu)
        if [ "$PRECOMPILE" = true ]; then
            echo "   Workflow: Export → Validate → Pre-compile → Test → Benchmark"
        else
            echo "   Workflow: Export → Validate → Test (MIGraphX EP) → Benchmark"
        fi
        echo ""
        echo "   Optimized for inference:"
        echo "   - KV cache enabled for efficient autoregressive generation"
        echo "   - FP16 precision for speed and lower memory"
        echo "   - Pre-allocated KV cache (2 × bucket size)"
        ;;
    cpu)  echo "   Workflow: Export → Validate → Optimize → Test" ;;
    int4) echo "   Workflow: Export → Validate → INT4 Quantize → Optimize → Test" ;;
    int8) echo "   Workflow: Export → Validate → INT8 Quantize → Optimize → Test" ;;
esac
echo ""

# =============================================================================
# Helper
# =============================================================================
run_cmd() {
    local desc="$1"; shift
    print_step "$desc"
    if [ "$DRY_RUN" = true ]; then
        echo "   [DRY RUN] $*"
    else
        "$@" || { print_err "$desc failed"; exit 1; }
    fi
    print_ok "$desc"
}

# =============================================================================
# Build common benchmark arguments
# =============================================================================
build_bench_args() {
    local args=""
    args="$args --seq-length $SEQ_LENGTH"
    args="$args --iterations $ITERATIONS"
    [ "$EXHAUSTIVE" = true ] && args="$args --exhaustive-tune"
    [ "$OFFLOAD_COPY" = true ] && args="$args --offload-copy"
    [ "$VERBOSE" = true ] && args="$args --verbose"
    echo "$args"
}

# =============================================================================
# Skip to benchmark if requested
# =============================================================================
if [ "$BENCHMARK_ONLY" = true ]; then
    MODEL_ONNX="$OUTPUT_DIR/model.onnx"
    [ ! -f "$MODEL_ONNX" ] && print_err "Model not found: $MODEL_ONNX" && exit 1
    
    print_header "Benchmark Only Mode"
    
    BENCH_ARGS=$(build_bench_args)
    run_cmd "Benchmark" python3 "$SCRIPT_DIR/benchmark_migraphx.py" "$OUTPUT_DIR" $BENCH_ARGS
    
    print_ok "Benchmark complete!"
    exit 0
fi

# =============================================================================
# Step 1: Export (Optimized for Inference)
# =============================================================================
print_header "Step 1: Export Model"

MODEL_ONNX="$OUTPUT_DIR/model.onnx"

build_export_args() {
    local args=""
    [ -n "$OPSET" ] && args="$args --opset $OPSET"
    [ "$NO_KV_CACHE" = true ] && args="$args --no-kv-cache"
    [ "$USE_FP32" = true ] && args="$args --fp32"
    echo "$args"
}

if [ -f "$MODEL_ONNX" ]; then
    print_warn "Model exists: $MODEL_ONNX"
    read -p "   Re-export? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -f "$MODEL_ONNX" "$MODEL_ONNX.data" "${MODEL_ONNX}_data"
        EXPORT_ARGS=$(build_export_args)
        run_cmd "Export to ONNX (FP16 + KV cache)" "$SCRIPT_DIR/01_export_model.sh" "$MODEL_PATH" "$OUTPUT_DIR" $EXPORT_ARGS
    else
        print_ok "Using existing model"
    fi
else
    EXPORT_ARGS=$(build_export_args)
    run_cmd "Export to ONNX (FP16 + KV cache)" "$SCRIPT_DIR/01_export_model.sh" "$MODEL_PATH" "$OUTPUT_DIR" $EXPORT_ARGS
fi

# =============================================================================
# Step 2: Validate
# =============================================================================
print_header "Step 2: Validate Model"
run_cmd "Validate ONNX" "$SCRIPT_DIR/03_validate_model.sh" "$MODEL_ONNX"

# =============================================================================
# Step 3+: Target-specific workflow
# =============================================================================
case $TARGET in
    # =========================================================================
    # GPU: ONNX Runtime with MIGraphXExecutionProvider
    # =========================================================================
    gpu)
        STEP=3
        
        # Pre-compile FIRST if requested (so test uses cached shapes)
        if [ "$PRECOMPILE" = true ]; then
            print_header "Step $STEP: Pre-compile MIGraphX (Cache Shapes)"
            echo "   Pre-compiling shapes for bucket: $BUCKETS"
            echo "   KV cache sizes: $(echo $BUCKETS | tr ',' '\n' | while read b; do echo -n "$((b*2)) "; done)"
            echo ""
            
            if [ -f "$SCRIPT_DIR/precompile_shapes.py" ]; then
                PRECOMPILE_ARGS="$OUTPUT_DIR --buckets $BUCKETS"
                [ "$EXHAUSTIVE" = true ] && PRECOMPILE_ARGS="$PRECOMPILE_ARGS --exhaustive-tune"
                [ "$OFFLOAD_COPY" = false ] && PRECOMPILE_ARGS="$PRECOMPILE_ARGS --no-offload-copy"
                [ "$VERBOSE" = true ] && PRECOMPILE_ARGS="$PRECOMPILE_ARGS --verbose"
                
                run_cmd "Pre-compile shapes" python3 "$SCRIPT_DIR/precompile_shapes.py" $PRECOMPILE_ARGS
            else
                print_warn "precompile_shapes.py not found, skipping pre-compilation"
            fi
            STEP=$((STEP + 1))
        fi
        
        print_header "Step $STEP: Test Inference (MIGraphX EP)"
        echo "   Bucket size: $SEQ_LENGTH (prompt padded to this)"
        echo "   KV cache:    $((SEQ_LENGTH * 2)) (pre-allocated)"
        echo "   Max output:  $SEQ_LENGTH tokens"
        echo ""
        
        # Build test args
        TEST_ARGS="--seq-length $SEQ_LENGTH"
        [ "$EXHAUSTIVE" = true ] && TEST_ARGS="$TEST_ARGS --exhaustive"
        [ "$OFFLOAD_COPY" = true ] && TEST_ARGS="$TEST_ARGS --offload-copy"
        [ "$VERBOSE" = true ] && TEST_ARGS="$TEST_ARGS --verbose"
        
        run_cmd "Test inference" "$SCRIPT_DIR/09_run_inference_test.sh" "$OUTPUT_DIR" "MIGraphXExecutionProvider" $TEST_ARGS
        STEP=$((STEP + 1))
        
        if [ "$SKIP_BENCHMARK" = false ]; then
            print_header "Step $STEP: Benchmark"
            
            BENCH_ARGS=$(build_bench_args)
            run_cmd "Benchmark" python3 "$SCRIPT_DIR/benchmark_migraphx.py" "$OUTPUT_DIR" $BENCH_ARGS
        fi
        
        BEST_MODEL="$MODEL_ONNX"
        ;;
    
    # =========================================================================
    # CPU: ONNX Runtime with FP16 optimization
    # =========================================================================
    cpu)
        print_header "Step 3: Optimize for ONNX Runtime CPU"
        echo "   Fusing attention patterns and converting to FP16..."
        
        MODEL_OPT="$OUTPUT_DIR/model_optimized.onnx"
        
        if [ -f "$MODEL_OPT" ]; then
            print_warn "Optimized model exists: $MODEL_OPT"
            read -p "   Re-run optimization? [y/N] " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                rm -f "$MODEL_OPT" "$MODEL_OPT.data" "${MODEL_OPT}_data"
                USE_GPU=false run_cmd "Optimize (attention fusion + FP16)" "$SCRIPT_DIR/04_optimize_model.sh" "$MODEL_ONNX" "$MODEL_OPT" "gpt_neox"
            else
                print_ok "Using existing optimized model"
            fi
        else
            USE_GPU=false run_cmd "Optimize (attention fusion + FP16)" "$SCRIPT_DIR/04_optimize_model.sh" "$MODEL_ONNX" "$MODEL_OPT" "gpt_neox"
        fi
        
        if [ ! -f "$MODEL_OPT" ]; then
            print_err "Optimized model not found: $MODEL_OPT"
            exit 1
        fi
        
        print_header "Step 4: Inference Test"
        run_cmd "Test inference" "$SCRIPT_DIR/09_run_inference_test.sh" "$OUTPUT_DIR" "CPUExecutionProvider" --seq-length $SEQ_LENGTH
        
        BEST_MODEL="$MODEL_OPT"
        ;;
    
    # =========================================================================
    # INT4: Quantize then optimize
    # =========================================================================
    int4)
        print_header "Step 3: INT4 Quantization"
        
        MODEL_INT4="$OUTPUT_DIR/model_int4.onnx"
        run_cmd "Quantize to INT4" "$SCRIPT_DIR/05_quantize_int4.sh" "$MODEL_ONNX" "$MODEL_INT4" 128
        
        print_header "Step 4: Optimize INT4 Model"
        MODEL_OPT="$OUTPUT_DIR/model_int4_optimized.onnx"
        SKIP_FP16=true run_cmd "Optimize (no FP16)" "$SCRIPT_DIR/04_optimize_model.sh" "$MODEL_INT4" "$MODEL_OPT"
        
        print_header "Step 5: Inference Test"
        run_cmd "Test inference" "$SCRIPT_DIR/09_run_inference_test.sh" "$OUTPUT_DIR" "CPUExecutionProvider" --seq-length $SEQ_LENGTH
        
        BEST_MODEL="$MODEL_OPT"
        ;;
    
    # =========================================================================
    # INT8: Quantize then optimize
    # =========================================================================
    int8)
        print_header "Step 3: INT8 Quantization"
        
        MODEL_INT8="$OUTPUT_DIR/model_int8.onnx"
        run_cmd "Quantize to INT8" "$SCRIPT_DIR/05_quantize_int8.sh" "$MODEL_ONNX" "$MODEL_INT8"
        
        print_header "Step 4: Optimize INT8 Model"
        MODEL_OPT="$OUTPUT_DIR/model_int8_optimized.onnx"
        SKIP_FP16=true run_cmd "Optimize (no FP16)" "$SCRIPT_DIR/04_optimize_model.sh" "$MODEL_INT8" "$MODEL_OPT"
        
        print_header "Step 5: Inference Test"
        run_cmd "Test inference" "$SCRIPT_DIR/09_run_inference_test.sh" "$OUTPUT_DIR" "CPUExecutionProvider" --seq-length $SEQ_LENGTH
        
        BEST_MODEL="$MODEL_OPT"
        ;;
esac

# =============================================================================
# Summary
# =============================================================================
print_header "Pipeline Complete"
echo ""
echo "   Best model: $BEST_MODEL"
echo ""
echo "   Output files:"
ls -lh "$OUTPUT_DIR"/*.onnx "$OUTPUT_DIR"/*.data 2>/dev/null | sed 's/^/     /' || true
echo ""

# Show cache directory if present
if [ -d "$OUTPUT_DIR/migraphx_cache" ]; then
    echo "   MIGraphX cache:"
    ls -lh "$OUTPUT_DIR/migraphx_cache"/*.mxr 2>/dev/null | head -5 | sed 's/^/     /' || echo "     (empty)"
    CACHE_COUNT=$(ls "$OUTPUT_DIR/migraphx_cache"/*.mxr 2>/dev/null | wc -l || echo "0")
    [ "$CACHE_COUNT" -gt 5 ] && echo "     ... and $((CACHE_COUNT - 5)) more"
    echo ""
fi

case $TARGET in
    gpu)
        echo "   Usage with ONNX Runtime (Python):"
        echo "   ────────────────────────────────────────────────────────────"
        echo "   import onnxruntime as ort"
        echo "   "
        echo "   session = ort.InferenceSession("
        echo "       '$BEST_MODEL',"
        echo "       providers=['MIGraphXExecutionProvider'],"
        echo "       provider_options=[{"
        echo "           'device_id': 0,"
        echo "           'migraphx_model_cache_dir': '$OUTPUT_DIR/migraphx_cache',"
        echo "       }]"
        echo "   )"
        echo "   "
        echo "   # Use pre-compiled bucket size for KV cache"
        echo "   # KV cache = 2 × bucket, max output = bucket"
        echo "   outputs = session.run(None, {"
        echo "       'input_ids': input_ids,        # (1, bucket_size)"
        echo "       'attention_mask': attn_mask,   # (1, bucket_size + kv_cache_size)"
        echo "       # ... KV cache tensors (1, heads, kv_cache_size, head_dim) ..."
        echo "   })"
        echo "   ────────────────────────────────────────────────────────────"
        echo ""
        echo "   Quick test:"
        echo "   ./09_run_inference_test.sh $OUTPUT_DIR --seq-length 256"
        echo ""
        if [ "$PRECOMPILE" != true ]; then
            echo "   Pre-compile for production (recommended):"
            echo "   python precompile_shapes.py $OUTPUT_DIR --buckets '256,512,1024'"
        fi
        ;;
    cpu|int4|int8)
        echo "   Usage: Load $BEST_MODEL with ONNX Runtime CPUExecutionProvider"
        ;;
esac
echo ""
