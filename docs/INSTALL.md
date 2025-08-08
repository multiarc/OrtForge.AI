# Install AMD ROCm accelerator on Linux/WSL environment.
Beware than if you have integrated AMD graphics (most likely you do) you must turn it off in order for ROCm accelerators to function with ONNX Runtime.

Here is the instruction on how to install version 6.4.2 of ROCm and it works with open source AMD driver on Ubuntu 24.04.
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

And to check if installation succeded.
```bash
rocminfo
```

This command NOT fail if integrated GPU is enabled and may display it as legitimate accelerator but in reality it will rail at runtime, and will fail entire process.

The source for instruction was taken from version 6.4.1 (it does not exist for higher versions) but works with further versions.

## Instructions source
https://rocm.docs.amd.com/projects/install-on-linux/en/docs-6.4.1/install/install-methods/amdgpu-installer/amdgpu-installer-ubuntu.html

# Building ONNX Runtime for ROCm

The build process for ROCm target accelerator is extreemly heavy and may tooke 3+ hours on Ryzen 9 9950X and peaks at ~50 Gb memory usage (wih 96 Gb total RAM).
Considering above, choose your targets from the beginning, I recommend to build all targets in one go (Python and .NET) - this will save a lot of time.

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

### Pre-built .NET packages should be linked to the repo


### Optimum[onnx] CLI will utilize ROCm but would actually call accelerator/target as CUDA and not work for all workloads, please hold on tight and brace yourself, this may get fixed at some point in the future
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