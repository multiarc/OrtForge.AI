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
    public StringBuilder EntireConversation { get; } = new();

    public ConversationSession(LlamaSession llm, TokenizerService tokenizer, InferenceConfig inferenceConfig)
    {
        _llm = llm;
        _inferenceConfig = inferenceConfig;
        _tokenizer = tokenizer;
        _kvState = new KvState([]);
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
        var generatedTokens = new List<int>();
        var toolState = new ToolCallState();
        var inputIds = _tokenizer.EncodeToIds(prompt).Select(x => (long)x).ToArray();

        for (int token = 0; token < _inferenceConfig.MaxTokens; token++)
        {
            using var outputs =
                await _llm.RunOptimizedStepAsync(inputIds, _kvState, _kvState.AccumulatedSequenceLength + inputIds.Length,
                    cancellationToken);
            _kvState = outputs.KvCache;
            var nextToken = GetNextTokenSample(outputs, generatedTokens);
            var tokenText = _tokenizer.DecodeFromIds([nextToken]);
            EntireConversation.Append(tokenText);
            
            if (IsStopToken(nextToken))
            {
                yield break;
            }
            
            generatedTokens.Add(nextToken);
            
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
