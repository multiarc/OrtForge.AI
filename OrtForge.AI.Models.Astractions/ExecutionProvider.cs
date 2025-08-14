namespace OrtForge.AI.Models.Astractions;

[Flags]
public enum ExecutionProvider
{
    CPU =       0b1,
    CUDA =      0b10,
    ROCm =      0b100,
    MIGraphX =  0b1000,
    TensorRT =  0b10000,
    OpenVINO =  0b100000,
    oneDNN =    0b1000000,
    DirectML =  0b10000000,
    NNAPI =     0b100000000,
    CoreML =    0b1000000000
}