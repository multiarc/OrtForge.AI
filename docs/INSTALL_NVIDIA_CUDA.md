# Install Nvidia CUDA accelerator on Linux WSL environment.

1. Update drivers to the latest on Windows.
2. Install CUDA Toolkit 13.0.
3. Install ONNX Runtime for CUDA.

```bash
sudo apt-key del 7fa2af80
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-13-0
```

## Instructions source
https://docs.nvidia.com/cuda/wsl-user-guide/index.html#getting-started-with-cuda-on-wsl