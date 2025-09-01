namespace OrtForge.AI.Agent.LLM;

/// <summary>
/// Centralized KV cache state with authoritative sequence length management.
/// This is the single source of truth for sequence length tracking.
/// </summary>
public sealed class KvState : IDisposable
{
    public List<LlamaSession.OutputKvTensor> Tensors { get; }
    private int _accumulatedSequenceLength;
    
    /// <summary>
    /// The authoritative sequence length - total tokens processed so far.
    /// This is the single source of truth for all sequence length calculations.
    /// </summary>
    public int AccumulatedSequenceLength 
    { 
        get => _accumulatedSequenceLength;
        private set 
        { 
            if (value < 0)
                throw new ArgumentException("Sequence length cannot be negative", nameof(value));
            _accumulatedSequenceLength = value;
        }
    }

    public KvState(List<LlamaSession.OutputKvTensor> mappedOutputs, int initialSequenceLength = 0)
    {
        AccumulatedSequenceLength = initialSequenceLength;
        Tensors = mappedOutputs;
    }
    
    /// <summary>
    /// Calculate the total sequence length after adding new tokens
    /// </summary>
    /// <param name="newTokenCount">Number of new tokens to add</param>
    /// <returns>The total sequence length after adding new tokens</returns>
    public int CalculateTotalLengthAfterTokens(int newTokenCount)
    {
        if (newTokenCount < 0)
            throw new ArgumentException("New token count cannot be negative", nameof(newTokenCount));
        return AccumulatedSequenceLength + newTokenCount;
    }

    public void Dispose()
    {
        foreach (var tensor in Tensors)
        {
            tensor.Tensor.Dispose();
        }

        Tensors.Clear();
    }
}