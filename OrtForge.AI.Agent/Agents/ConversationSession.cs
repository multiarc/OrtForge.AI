using System.Text;
using OrtForge.AI.Agent.Generation;
using OrtForge.AI.Agent.LLM;
using OrtForge.AI.Agent.Tokenization;

namespace OrtForge.AI.Agent.Agents;

public sealed class ConversationSession : IDisposable
{
    private readonly TokenizerService _tokenizer;
    private readonly List<(string role, string content)> _history = new();
    private KvState? _kvState;

    private bool _isSystemPromptProcessed = false;
    
    public string SessionId { get; } = Guid.NewGuid().ToString("N")[..8];
    public IReadOnlyList<(string role, string content)> History => _history;
    public int TotalTokensProcessed => _kvState?.AccumulatedSequenceLength ?? 0;
    public bool IsInitialized => _isSystemPromptProcessed;
    

    public int MaxHistoryLength { get; set; } = 20; // Keep last N messages
    public int MaxTokensBeforeTruncation { get; set; } = 4096; // Truncate when approaching context limit
    public bool EnableSummarization { get; set; } = true;

    public ConversationSession(TokenizerService tokenizer)
    {
        _tokenizer = tokenizer;
    }


    public async Task<KvState> InitializeSystemPromptAsync(
        LlamaSession llmSession, 
        IReadOnlyList<string> retrievedContext, 
        bool enableTools = false)
    {
        if (_isSystemPromptProcessed)
        {
            return _kvState ?? throw new InvalidOperationException("System prompt processed but KV state is null");
        }


        var systemPrompt = AgentOrchestrator.BuildSystemPrompt(retrievedContext, enableTools);
        var systemTokens = _tokenizer.EncodeToIds(systemPrompt);
        
        _kvState = new KvState();
        

        var inputIds = systemTokens.Select(id => (long)id).ToArray();
        
        var outputs = await llmSession.RunOptimizedStep(inputIds, _kvState, 0, inputIds.Length);
        

        _kvState = outputs.KvCache;
        _isSystemPromptProcessed = true;
        
        outputs.Dispose();
        return _kvState;
    }


    public async Task<(int[] newTokens, KvState kvState)> AddMessageAsync(
        string role, 
        string content,
        LlamaSession? llmSession = null)
    {

        await TruncateIfNeededAsync(llmSession);
        

        _history.Add((role, content));
        

        var messagePrompt = FormatMessage(role, content);
        var messageTokens = _tokenizer.EncodeToIds(messagePrompt);
        
        if (_kvState == null)
        {
            throw new InvalidOperationException("Session not initialized. Call InitializeSystemPromptAsync first.");
        }
        

        if (llmSession != null)
        {
            var inputIds = messageTokens.Select(id => (long)id).ToArray();
            var currentSeqLength = _kvState.AccumulatedSequenceLength;
            var totalSeqLength = currentSeqLength + inputIds.Length;
            
            Console.WriteLine($"🔍 AddMessageAsync: role={role}, currentSeqLength={currentSeqLength}, inputLength={inputIds.Length}, totalSeqLength={totalSeqLength}");
            
            // Debug: Check actual tensor shapes before processing
            var firstTensor = _kvState.Tensors.FirstOrDefault();
            if (firstTensor.Value != null)
            {
                var shape = firstTensor.Value.GetTensorTypeAndShape().Shape;
                Console.WriteLine($"🔍   Before: Sample tensor {firstTensor.Key}: shape=[{string.Join(",", shape)}]");
            }
            
            var outputs = await llmSession.RunOptimizedStep(inputIds, _kvState, 0, totalSeqLength);
            _kvState = outputs.KvCache;
            
            Console.WriteLine($"🔍 After processing: newSeqLength={_kvState.AccumulatedSequenceLength}");
            outputs.Dispose();
        }
        
        return (messageTokens, _kvState);
    }


    public KvState GetCurrentKvState()
    {
        if (_kvState == null)
            throw new InvalidOperationException("Session not initialized");
            
        // Debug: Check for tensor/metadata mismatch
        Console.WriteLine($"🔍 GetCurrentKvState: AccumulatedSequenceLength={_kvState.AccumulatedSequenceLength}");
        var firstTensor = _kvState.Tensors.FirstOrDefault();
        if (firstTensor.Value != null)
        {
            var shape = firstTensor.Value.GetTensorTypeAndShape().Shape;
            Console.WriteLine($"🔍   Sample tensor {firstTensor.Key}: shape=[{string.Join(",", shape)}]");
        }
        
        return _kvState;
    }

    public void UpdateKvState(KvState newKvState)
    {
        Console.WriteLine($"🔍 UpdateKvState: oldSeqLength={_kvState?.AccumulatedSequenceLength ?? 0}, newSeqLength={newKvState.AccumulatedSequenceLength}");
        
        // Debug: Check actual tensor shapes
        var firstTensor = newKvState.Tensors.FirstOrDefault();
        if (firstTensor.Value != null)
        {
            var shape = firstTensor.Value.GetTensorTypeAndShape().Shape;
            Console.WriteLine($"🔍   Incoming tensor {firstTensor.Key}: shape=[{string.Join(",", shape)}]");
        }
        
        _kvState = newKvState;
    }

    public void AddToHistory(string role, string content)
    {
        _history.Add((role, content));
    }




    private async Task TruncateIfNeededAsync(LlamaSession? llmSession)
    {
        if (_history.Count <= MaxHistoryLength && 
            TotalTokensProcessed <= MaxTokensBeforeTruncation)
        {
            return;
        }

        if (EnableSummarization && llmSession != null)
        {
            await SummarizeAndTruncateAsync(llmSession);
        }
        else
        {
            // Simple truncation - keep only recent messages
            SimpleTruncate();
        }
    }


    private void SimpleTruncate()
    {
        var messagesToKeep = MaxHistoryLength / 2;
        if (_history.Count > messagesToKeep)
        {
            _history.RemoveRange(0, _history.Count - messagesToKeep);
            

            _kvState?.Dispose();
            _kvState = null;
            _isSystemPromptProcessed = false;
        }
    }


    private Task SummarizeAndTruncateAsync(LlamaSession llmSession)
    {
        SimpleTruncate();
        return Task.CompletedTask;
    }


    private static string FormatMessage(string role, string content)
    {
        return $"<|start_header_id|>{role}<|end_header_id|>\n\n{content}\n<|eot_id|>";
    }




    public void Dispose()
    {
        _kvState?.Dispose();
    }
}
