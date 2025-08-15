# Install Optimum CLI for model conversion and optimization

```bash
sudo apt update
sudo apt install build-essential flex bison libssl-dev libelf-dev bc python3 pahole cpio python3.12-venv python3-pip
mkdir optimum
cd optimum
python3 -m venv .
source ./bin/activate
pip install optimum
pip install optimum[exporters,onnxruntime,sentence_transformers,amd]
pip install accelerate
```

To install AMD GPU support to run models, please follow the instructions in [AMD GPU Support](INSTALL_AMD_ROCm.md) 