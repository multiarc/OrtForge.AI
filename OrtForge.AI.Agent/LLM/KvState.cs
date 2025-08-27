using Microsoft.ML.OnnxRuntime;

namespace OrtForge.AI.Agent.LLM;

/// <summary>
/// Simplified KV cache state that holds tensor references.
/// ONNX Runtime's allocator handles memory pooling and reuse efficiently.
/// </summary>
public sealed class KvState : IDisposable
{
    public readonly Dictionary<string, OrtValue> Tensors = new();
    
    /// <summary>
    /// Tracks the accumulated sequence length for proper KV cache sizing.
    /// This is the total length of all tokens processed so far.
    /// </summary>
    public int AccumulatedSequenceLength { get; private set; }

    public KvState(int initialSequenceLength = 0)
    {
        AccumulatedSequenceLength = initialSequenceLength;
    }

    public void AddTensor(string name, OrtValue tensor)
    {
        Tensors[name] = tensor;
    }
        
    public OrtValue? GetTensor(string name)
    {
        return Tensors.GetValueOrDefault(name);
    }
    
    /// <summary>
    /// Updates the accumulated sequence length after processing tokens.
    /// </summary>
    /// <param name="additionalTokens">Number of tokens processed in this step</param>
    public void UpdateSequenceLength(int additionalTokens)
    {
        AccumulatedSequenceLength += additionalTokens;
    }
    
    public void Dispose()
    {
        foreach (var tensor in Tensors.Values)
        {
            tensor?.Dispose();
        }
        Tensors.Clear();
    }
}