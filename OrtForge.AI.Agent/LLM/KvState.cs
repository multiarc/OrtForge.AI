using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace OrtForge.AI.Agent.LLM;

public sealed class KvArena : IDisposable
{
    private readonly Dictionary<string, OrtValue> _kvTensorPool = new();

    public OrtValue GetOrCreateKvTensor(string name, long[] shape, TensorElementType elementType)
    {
        if (_kvTensorPool.TryGetValue(name, out var existingTensor))
        {
            // Verify element type and shape; reallocate if mismatched
            var existingInfo = existingTensor.GetTensorTypeAndShape();
            var existingShape = existingInfo.Shape;
            var existingType = existingInfo.ElementDataType;

            bool typeMismatch = existingType != elementType;
            bool shapeMismatch = existingShape.Length != shape.Length
                                 || !existingShape.SequenceEqual(shape);

            if (!typeMismatch && !shapeMismatch)
            {
                return existingTensor;
            }

            // Dispose and replace with new allocation
            existingTensor.Dispose();
            _kvTensorPool.Remove(name);
        }

        var tensor = OrtValue.CreateAllocatedTensorValue(OrtAllocator.DefaultInstance, elementType, shape);
        _kvTensorPool[name] = tensor;
        return tensor;
    }

    public void Dispose()
    {
        foreach (var value in _kvTensorPool.Values)
        {
            value.Dispose();
        }

        _kvTensorPool.Clear();
    }
}

public sealed class KvState
{
    public readonly Dictionary<string, OrtValue> Tensors = new();
    public KvArena KvArena { get; }

    public KvState(KvArena kvArena)
    {
        KvArena = kvArena;
    }

    public void AddTensor(string name, OrtValue tensor)
    {
        Tensors[name] = tensor;
    }
        
    public OrtValue? GetTensor(string name)
    {
        return Tensors.GetValueOrDefault(name);
    }
}