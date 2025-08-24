# Install Optimum CLI for model conversion and optimization

```bash
sudo apt update
sudo apt install build-essential flex bison libssl-dev libelf-dev bc python3 pahole cpio python3.12-venv python3-pip
mkdir optimum
cd optimum
python3 -m venv .
source ./bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.4
pip install onnxruntime_genai onnx-ir
#ROCm
python3 -m onnxruntime_genai.models.builder -i . -o ./onnx_opt_i4 -p int4 -e rocm
#CUDA
python3 -m onnxruntime_genai.models.builder -i . -o ./onnx_opt_i4 -p int4 -e cuda 
```

To install AMD GPU support for onnx runtime to run and optimize models, please follow the instructions in [AMD GPU Support](INSTALL_AMD_ROCm.md)

Optimize a model for inference on GPU using FP16 precision
```bash
optimum-cli export onnx --model . --dtype fp16 --task default --device cuda --optimize O4 ./onnx_fp16
```