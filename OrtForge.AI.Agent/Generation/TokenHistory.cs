namespace OrtForge.AI.Agent.Generation;

/// <summary>
/// Maintains a sliding window of recent tokens for repetition penalty purposes.
/// This allows repetition penalties to be applied across conversation turns.
/// </summary>
public sealed class TokenHistory
{
    private readonly Queue<int> _tokens = new();
    
    public TokenHistory(int maxSize = 128)
    {
        if (maxSize <= 0)
            throw new ArgumentException("Max size must be positive", nameof(maxSize));
        MaxSize = maxSize;
    }
    
    public int MaxSize { get; }
    public int Count => _tokens.Count;
    
    public void AddToken(int token)
    {
        _tokens.Enqueue(token);
        while (_tokens.Count > MaxSize)
        {
            _tokens.Dequeue();
        }
    }
    
    public void AddTokens(IEnumerable<int> tokens)
    {
        foreach (var token in tokens)
        {
            AddToken(token);
        }
    }
    
    public List<int> GetTokens()
    {
        return _tokens.ToList();
    }
    
    public void Clear()
    {
        _tokens.Clear();
    }
}

