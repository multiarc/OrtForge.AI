#!/bin/bash

# RDNA3 GPU Compatibility Test Script
# Tests different execution modes to find the best configuration for your system

echo "🔧 RDNA3 GPU Compatibility Test"
echo "================================"

# Check ROCm installation
echo "📋 Checking ROCm installation..."
if command -v rocminfo &> /dev/null; then
    echo "✅ ROCm found"
    rocminfo | grep "Name:" | head -5
else
    echo "❌ ROCm not found or not in PATH"
    exit 1
fi

# Check GPU visibility
echo ""
echo "📋 Checking GPU visibility..."
if [ -n "$HIP_VISIBLE_DEVICES" ]; then
    echo "HIP_VISIBLE_DEVICES: $HIP_VISIBLE_DEVICES"
else
    echo "HIP_VISIBLE_DEVICES: not set (all GPUs visible)"
fi

if [ -n "$ROCR_VISIBLE_DEVICES" ]; then
    echo "ROCR_VISIBLE_DEVICES: $ROCR_VISIBLE_DEVICES"
else
    echo "ROCR_VISIBLE_DEVICES: not set (all devices visible)"
fi

# Test with different configurations
echo ""
echo "🧪 Testing execution modes..."

# Test 1: Environment variable override
echo ""
echo "Test 1: HSA_OVERRIDE_GFX_VERSION=10.3.0"
export HSA_OVERRIDE_GFX_VERSION=10.3.0
echo "Environment variable set. Try running your application now."

# Test 2: Check if integrated GPU is enabled
echo ""
echo "Test 2: Checking for integrated GPU interference..."
rocminfo | grep -i "integrated" && echo "⚠️ Warning: Integrated GPU detected. Consider disabling in BIOS or using ROCR_VISIBLE_DEVICES to exclude it."

# Test 3: Build and run a simple test
echo ""
echo "Test 3: Building and testing with different compatibility modes..."
echo "Building test project..."

cd "$(dirname "$0")"
if dotnet build OrtForge.AI.Agent.Console/OrtForge.AI.Agent.Console.csproj -c Release -v q; then
    echo "✅ Build successful"
    echo ""
    echo "🚀 Ready to test! Try running with:"
    echo "   1. Standard mode (will likely fail on RDNA3)"
    echo "   2. RDNA3 compatible mode (recommended)"
    echo "   3. CPU-only mode (fallback)"
    echo ""
    echo "The runtime factory now defaults to RDNA3 compatible mode."
else
    echo "❌ Build failed. Check your .NET installation."
fi

echo ""
echo "✨ Test complete. If you still have issues:"
echo "   1. Try HSA_OVERRIDE_GFX_VERSION=10.3.0"
echo "   2. Use CPU-only mode for testing"
echo "   3. Check the RDNA3_GPU_COMPATIBILITY.md guide"
