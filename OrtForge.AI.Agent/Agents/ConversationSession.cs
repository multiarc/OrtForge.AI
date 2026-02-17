using System.Runtime.CompilerServices;
using System.Text;
using Microsoft.ML.Tokenizers;
using OrtForge.AI.Agent.Generation;
using OrtForge.AI.Agent.LLM;
using OrtForge.AI.Agent.Tokenization;

namespace OrtForge.AI.Agent.Agents;

public sealed class ConversationSession : IDisposable
{
    private readonly TokenizerService _tokenizer;
    private readonly LlamaSession _llm;
    private readonly InferenceConfig _inferenceConfig;
    private KvState _kvState;
    private bool _isSystemPromptProcessed;
    private readonly TokenHistory _tokenHistory;
    public StringBuilder EntireConversation { get; } = new();

    public ConversationSession(LlamaSession llm, TokenizerService tokenizer, InferenceConfig inferenceConfig, int repetitionPenaltyWindowSize = 128)
    {
        _llm = llm;
        _inferenceConfig = inferenceConfig;
        _tokenizer = tokenizer;
        _kvState = new KvState([]);
        _tokenHistory = new TokenHistory(repetitionPenaltyWindowSize);
    }

    public string SessionId { get; } = Guid.NewGuid().ToString("N")[..8];
    public bool IsInitialized => _isSystemPromptProcessed;

    public void Dispose()
    {
        _kvState.Dispose();
    }

    public async IAsyncEnumerable<string> GenerateNextResponseAsync(string prompt,
        Func<string, string>? toolExecutor = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        EntireConversation.Append(prompt);
        var toolState = new ToolCallState();
        var inputIds = _tokenizer.EncodeToIds(prompt).Select(x => (long)x).ToArray();
        var isFirstToken = true;

        for (int token = 0; token < _inferenceConfig.MaxTokens; token++)
        {
            using var outputs =
                await _llm.RunOptimizedStepAsync(inputIds, _kvState, _kvState.AccumulatedSequenceLength + inputIds.Length,
                    cancellationToken);
            
            // Dispose previous KV state to prevent memory leak
            var oldKvState = _kvState;
            _kvState = outputs.KvCache;
            oldKvState.Dispose();
            
            // Use sliding window token history for repetition penalty
            var nextToken = GetNextTokenSample(outputs, _tokenHistory.GetTokens());
            var tokenText = _tokenizer.DecodeFromIds([nextToken]);
            EntireConversation.Append(tokenText);
            
            if (IsStopToken(nextToken))
            {
                _isSystemPromptProcessed = true;
                // Append the stop token text to conversation for proper multi-turn format
                EntireConversation.Append("<|eot_id|>");
                yield break;
            }
            
            // Add to sliding window for cross-turn repetition penalty
            _tokenHistory.AddToken(nextToken);
            
            //inject current token into next inference step
            inputIds = [nextToken];

            if (toolExecutor != null)
            {
                toolState.AppendToken(tokenText);
                var pendingCall = toolState.GetNextPendingCall();
                if (pendingCall != null)
                {
                    //TODO
                }
            }

            // Mark session as initialized after first token generated
            if (isFirstToken)
            {
                _isSystemPromptProcessed = true;
                isFirstToken = false;
            }

            yield return tokenText;
        }
    }
    
    private bool IsStopToken(int tokenId) => _inferenceConfig.StopTokenIds.Contains(tokenId);
    private int GetNextTokenSample(LlamaSession.StepOutputs outputs, List<int> previousTurnTokens)
    {
        var span = outputs.GetLogitsSpan();
        var logitsShape = outputs.Logits.GetTensorTypeAndShape().Shape;
        Span<float> logitsForSampling;
        if (logitsShape.Length == 3) // [batch, seq_len, vocab]
        {
            var seqLen = (int)logitsShape[1];
            var vocabSize = (int)logitsShape[2];
                
            var lastTokenStart = (seqLen - 1) * vocabSize;
            logitsForSampling = span.Slice(lastTokenStart, vocabSize);
        }
        else if (logitsShape.Length == 2) // [batch, vocab] - generation step
        {
            var vocabSize = (int)logitsShape[1];
                
            logitsForSampling = span.Slice(0, vocabSize);
        }
        else
        {
            throw new InvalidOperationException("Unexpected logits shape.");
        }
            
        return Sampling.Sample(logitsForSampling, _inferenceConfig, previousTurnTokens);
    }
}
