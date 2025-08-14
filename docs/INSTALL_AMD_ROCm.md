# Install AMD ROCm accelerator on Linux/WSL environment.
Beware of if you have integrated AMD graphics (most likely you do with AMD CPUs), you must turn it off in order for ROCm accelerators to function with ONNX Runtime.

Here is the instruction on how to install version 6.4.2 of ROCm, and it works with an open source AMD driver on Ubuntu 24.04.
```bash
wget https://repo.radeon.com/amdgpu-install/6.4.2/ubuntu/noble/amdgpu-install_6.4.60402-1_all.deb
sudo apt update
sudo apt install ./amdgpu-install_6.4.60402-1_all.deb
sudo amdgpu-install --usecase=rocm,hiplibsdk,graphics,opencl -y --vulkan=amdvlk --no-dkms
```

Sample for version 6.4.3 
```bash
wget https://repo.radeon.com/amdgpu-install/6.4.3/ubuntu/noble/amdgpu-install_6.4.60403-1_all.deb
sudo apt update
sudo apt install ./amdgpu-install_6.4.60403-1_all.deb
sudo amdgpu-install --usecase=rocm,hiplibsdk,graphics,opencl -y --vulkan=amdvlk --no-dkms
```

And to check if the installation succeeded.
```bash
rocminfo #make note of your GPU uuid, to whitelist only CPU and discreet GPU on the next step
```

`rocminfo` DOESN'T fail if integrated GPU is enabled, but a lot of features may not be supported to a point when it will crash a driver at runtime.
Your options are: disable iGPU in UEFI/BIOS or export environment variable to whitelist CPU and discreet GPU only.
```bash
export ROCR_VISIBLE_DEVICES="0,GPU-deadbeefdeadbeef" #0 - CPU, GPU-deadbeefdeadbeef - GPU.
```

The source for instruction was taken from version 6.4.1 — it does not exist for higher versions. But it works with pretty much all versions.

## Instructions source
https://rocm.docs.amd.com/projects/install-on-linux/en/docs-6.4.1/install/install-methods/amdgpu-installer/amdgpu-installer-ubuntu.html

# Building ONNX Runtime for ROCm

The build process for ROCm target accelerator is extremely heavy and may take 3+ hours on Ryzen 9 9950X and peaks at ~50 Gb memory usage (with 96 Gb total RAM).
Considering the above, choose your targets from the beginning. I recommend building all targets in one go (Python and .NET) — this will save a lot of time.

Clone repo
```bash
git clone --recursive https://github.com/ROCm/onnxruntime.git
git checkout tags/v1.22.1
cd onnxruntime
```

Build for .NET only to run models
```bash
./build.sh --update --build --config Release --build_nuget --parallel --use_rocm --rocm_home /opt/rocm --skip_tests
```

Build for .NET and for Python stack with PyTorch and any other toolset that may utilize GPU accelerators on AMD 

```bash
python3 -m venv .
source ./bin/activate
pip install 'cmake>=3.28,<4'
pip install -r requirements.txt
pip install setuptools
./build.sh --update --build --config Release --build_wheel --build_nuget --parallel --use_rocm --rocm_home /opt/rocm --skip_tests
```

Install wheel for python to use in the venv
```bash
pip install ./build/Linux/Release/dist/*.whl
```
Instructions primary source
https://onnxruntime.ai/docs/build/eps.html#amd-rocm

### Pre-built .NET packages are linked to the repo

### Optimum[onnx] CLI can use ROCm but would actually call accelerator/target as CUDA and work for parts of workloads, please hold on tight and brace yourself, this may get fixed at some point in the future.
Also, AMD has a CUDA translation layer for non-precompiled code, so it may simply work sometimes.
```text
  .-'---`-.
,'          `.
|             \
|              \
\           _  \
,\  _    ,'-,/-)\
( * \ \,' ,' ,'-)
 `._,)     -',-')
   \/         ''/
    )        / /
   /       ,'-'
```