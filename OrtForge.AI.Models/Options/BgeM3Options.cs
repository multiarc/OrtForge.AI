using Microsoft.ML.OnnxRuntime.Tensors;
using OrtForge.AI.Models.Abstractions;

namespace OrtForge.AI.Models.Options;

public class BgeM3Options : BaseModelOptions
{
    public required TensorElementType TensorElementType { get; init; }
}