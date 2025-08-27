using System.Text;
using Microsoft.ML.OnnxRuntime.Tensors;
using OrtForge.AI.Agent.Generation;
using OrtForge.AI.Agent.LLM;
using OrtForge.AI.Agent.Rag;
using OrtForge.AI.Agent.Tokenization;
using OrtForge.AI.Models.Models;

namespace OrtForge.AI.Agent.Agents;

public sealed class AgentOrchestrator
{
    private readonly LlamaSession _llm;
    private readonly TokenizerService _tokenizer;
    private readonly BgeM3Model _embeddings;
    private readonly BgeRerankerM3? _reranker;
    private readonly InMemoryVectorStore _vec;

    public AgentOrchestrator(LlamaSession llm, TokenizerService tokenizer, BgeM3Model embeddings, InMemoryVectorStore vec, BgeRerankerM3? reranker = null)
    {
        _llm = llm;
        _tokenizer = tokenizer;
        _embeddings = embeddings;
        _reranker = reranker;
        _vec = vec;
    }

    public async Task<string> ChatTurnAsync(string user, IReadOnlyList<(string role, string content)> history, InferenceConfig? config = null, Func<string, string>? toolExecutor = null)
    {
        config = LlamaOptimizations.GetOptimalConfigForModel(_llm.ModelName, config);
        
        var queryVec = await _embeddings.CreateEmbeddingAsync(user);
        var candidateResults = _vec.TopK(queryVec, 10).ToList(); // Get more candidates for reranking
        
        var retrieved = candidateResults.Select(x => x.Text).ToList();
        
        // Apply reranking if available
        if (_reranker != null && candidateResults.Count > 1)
        {
            var rerankedResults = new List<(float score, string text)>();
            foreach (var candidate in candidateResults)
            {
                var score = await _reranker.GetRerankingScoreAsync(user, candidate.Text);
                rerankedResults.Add((score: score, text: candidate.Text));
            }
            
            // Sort by reranking score and take top 5
            retrieved = rerankedResults
                .OrderByDescending(x => x.score)
                .Take(5)
                .Select(x => x.text)
                .ToList();
        }
        else
        {
            // Fall back to similarity-based ranking, take top 5
            retrieved = retrieved.Take(5).ToList();
        }

        var prompt = BuildPrompt(history, user, retrieved, toolExecutor != null);
        Console.WriteLine($"DEBUG: Built prompt ({prompt.Length} chars):\n{prompt}\n--- END PROMPT ---");
        
        var inputIds = _tokenizer.EncodeToIds(prompt);
        Console.WriteLine($"DEBUG: Tokenized to {inputIds.Length} tokens: [{string.Join(", ", inputIds.Take(10))}...]");

        var idsArray = inputIds.Select(id => (long)id).ToArray();

        var kv = new KvState(); // Simplified - no KvArena needed
        var response = new StringBuilder();
        var generatedTokens = new List<int>();
        var sequenceLength = inputIds.Length;
        var toolState = new ToolCallState();

        int GetNextSample(LlamaSession.StepOutputs outputs, int vocab)
        {
            var span = outputs.GetLogitsSpan();
            var logitsShape = outputs.Logits.GetTensorTypeAndShape().Shape;
            
            // For logits shape [batch, seq_len, vocab], we need the last token's logits
            Span<float> logitsForSampling;
            if (logitsShape.Length == 3) // [batch, seq_len, vocab]
            {
                var batchSize = (int)logitsShape[0];
                var seqLen = (int)logitsShape[1];
                var vocabSize = (int)logitsShape[2];
                
                // FIXED: Use vocabSize consistently for calculations
                // Take logits for the last token position: span[(seqLen-1) * vocabSize : seqLen * vocabSize]
                var lastTokenStart = (seqLen - 1) * vocabSize;
                logitsForSampling = span.Slice(lastTokenStart, vocabSize);
                
                Console.WriteLine($"DEBUG: Sampling from logits shape [{batchSize}, {seqLen}, {vocabSize}], using slice [{lastTokenStart}:{lastTokenStart + vocabSize}]");
            }
            else if (logitsShape.Length == 2) // [batch, vocab] - generation step
            {
                // For single token generation, logits are already [batch, vocab]
                var batchSize = (int)logitsShape[0];
                var vocabSize = (int)logitsShape[1];
                
                // Take logits for batch 0
                logitsForSampling = span.Slice(0, vocabSize);
                
                Console.WriteLine($"DEBUG: Sampling from logits shape [{batchSize}, {vocabSize}], using full vocab span");
            }
            else
            {
                // Fallback: assume span is already the right size [vocab]
                logitsForSampling = span;
                Console.WriteLine($"DEBUG: Using fallback logits sampling, span length: {span.Length}");
            }
            
            // Check for NaN/Inf values in logits that would cause bad sampling
            var hasNan = false;
            var hasInf = false;
            for (int i = 0; i < logitsForSampling.Length; i++)
            {
                if (float.IsNaN(logitsForSampling[i])) hasNan = true;
                if (float.IsInfinity(logitsForSampling[i])) hasInf = true;
            }
            
            if (hasNan || hasInf)
            {
                Console.WriteLine($"WARNING: Logits contain NaN: {hasNan}, Inf: {hasInf} - this will cause bad sampling!");
            }
            
            // Debug: Check logits values for anomalies
            var maxLogit = float.NegativeInfinity;
            var minLogit = float.PositiveInfinity;
            var sumLogits = 0.0f;
            for (int i = 0; i < logitsForSampling.Length; i++)
            {
                var logit = logitsForSampling[i];
                if (logit > maxLogit) maxLogit = logit;
                if (logit < minLogit) minLogit = logit;
                sumLogits += logit;
            }
            
            Console.WriteLine($"DEBUG: Logits range [{minLogit:F3}, {maxLogit:F3}], avg: {sumLogits / logitsForSampling.Length:F3}");
            
            var previousTokensSpan = generatedTokens.Count > 0 ? generatedTokens.ToArray().AsSpan() : ReadOnlySpan<int>.Empty;
            var sampledToken = Sampling.Sample(logitsForSampling, config, previousTokensSpan);
            
            Console.WriteLine($"DEBUG: Sampled token {sampledToken} from {logitsForSampling.Length} vocab options");
            return sampledToken;
        }
        
        for (int step = 0; step < config.MaxTokens; step++)
        {
            // First step: use full prompt, subsequent steps: use only the last generated token
            var currentInput = step == 0 ? idsArray : new long[] { generatedTokens[^1] };
            
            var outputs = await _llm.RunOptimizedStep(currentInput, kv, step, sequenceLength + generatedTokens.Count);
            
            var newKv = outputs.KvCache;

            var logitsShape = outputs.Logits.GetTensorTypeAndShape().Shape;
            var vocab = (int)logitsShape[^1];
            
            var nextId = GetNextSample(outputs, vocab);
            
            generatedTokens.Add(nextId);

            var tokenText = _tokenizer.DecodeFromIds(new[] { nextId });
            Console.WriteLine($"DEBUG: Generated token ID {nextId} -> '{tokenText}' (step {step})");
            
            // Check for immediate repetition (same token repeated)
            if (generatedTokens.Count >= 3)
            {
                var recent = generatedTokens.TakeLast(3).ToArray();
                if (recent[0] == recent[1] && recent[1] == recent[2])
                {
                    Console.WriteLine($"WARNING: Token {recent[0]} repeated 3 times in a row! Breaking to prevent infinite loop.");
                    break;
                }
            }
            
            response.Append(tokenText);
            
            if (toolExecutor != null)
            {
                toolState.AppendToken(tokenText);
                
                var pendingCall = toolState.GetNextPendingCall();
                if (pendingCall != null)
                {
                    var (injectedText, injectedTokens) = ExecuteToolCall(pendingCall, toolExecutor, toolState);
                    if (!string.IsNullOrEmpty(injectedText))
                    {
                        response.Append(injectedText);
                        generatedTokens.AddRange(injectedTokens);
                        
                        var injectArray = injectedTokens.Select(token => (long)token).ToArray();
                        
                        var injectOutputs = await _llm.RunOptimizedStep(injectArray, newKv, step, sequenceLength + generatedTokens.Count);
                        outputs.Dispose();
                        outputs = injectOutputs;
                        newKv = injectOutputs.KvCache;
                    }
                }
            }

            if (IsStopToken(nextId, config) || IsStopSequence(response.ToString(), config)) break;

            kv = newKv;
            outputs.Dispose();
        }
        
        kv.Dispose(); // Clean up KV tensors

        return response.ToString();
    }

    internal static bool IsStopToken(int tokenId, InferenceConfig config) => config.StopTokenIds.Contains(tokenId);

    internal static bool IsStopSequence(string text, InferenceConfig config)
    {
        return config.StopSequences.Any(seq => text.Contains(seq));
    }

    private (string injectedText, int[] injectedTokens) ExecuteToolCall(ToolCall toolCall, Func<string, string> toolExecutor, ToolCallState toolState)
    {
        try
        {
            toolState.UpdateCallStatus(toolCall, ToolCallStatus.Executing);
            
            var result = toolExecutor.Invoke(toolCall.Arguments);
            
            toolState.UpdateCallStatus(toolCall, ToolCallStatus.Completed, result);
            
            var injectedText = $"\n<|tool_result|>\n{result}\n<|/tool_result|>\n";
            var injectedTokens = _tokenizer.EncodeToIds(injectedText);
            
            return (injectedText, injectedTokens);
        }
        catch (Exception ex)
        {
            var errorMessage = $"Tool execution failed: {ex.Message}";
            toolState.UpdateCallStatus(toolCall, ToolCallStatus.Failed, error: errorMessage);
            
            var injectedText = $"\n<|tool_result|>\nError: {errorMessage}\n<|/tool_result|>\n";
            var injectedTokens = _tokenizer.EncodeToIds(injectedText);
            
            return (injectedText, injectedTokens);
        }
    }

    internal static string BuildPrompt(IReadOnlyList<(string role, string content)> history, string user, IReadOnlyList<string> retrieved, bool enableTools = false)
    {
        var sb = new StringBuilder();
        
        // Use a simpler, more compatible prompt format
        sb.AppendLine("You are a helpful assistant. Use context when relevant and cite sources.");
        
        if (enableTools)
        {
            sb.AppendLine();
            sb.AppendLine("When you need to use a tool, format it as:");
            sb.AppendLine("TOOL_CALL");
            sb.AppendLine("name: tool_name");
            sb.AppendLine("args: tool_arguments");
            sb.AppendLine("END_TOOL_CALL");
            sb.AppendLine();
            sb.AppendLine("The tool result will be provided in TOOL_RESULT...END_TOOL_RESULT tags.");
        }
        
        if (retrieved.Count > 0)
        {
            sb.AppendLine();
            sb.AppendLine("Context:");
            foreach (var ctx in retrieved) 
            {
                sb.AppendLine($"- {ctx}");
            }
            sb.AppendLine();
        }
        
        // Add conversation history in a simple format
        foreach (var (role, content) in history)
        {
            sb.AppendLine($"{role.ToUpperInvariant()}: {content}");
        }
        
        sb.AppendLine($"USER: {user}");
        sb.Append("ASSISTANT:");
        return sb.ToString();
    }
}


