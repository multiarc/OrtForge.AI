using System.Text;
using Microsoft.ML.OnnxRuntime.Tensors;
using OrtForge.AI.Agent.Generation;
using OrtForge.AI.Agent.LLM;
using OrtForge.AI.Agent.Rag;
using OrtForge.AI.Agent.Tokenization;
using OrtForge.AI.Agent.Tools;
using OrtForge.AI.Models.Models;

namespace OrtForge.AI.Agent.Agents;

public sealed class AgentOrchestrator
{
    private readonly LlamaSession _llm;
    private readonly TokenizerService _tokenizer;
    private readonly BgeM3Model? _embeddings;
    private readonly BgeRerankerM3? _reranker;
    private readonly InMemoryVectorStore? _vec;
    private readonly ToolInjectionManager _toolInjectionManager;

    public AgentOrchestrator(LlamaSession llm, TokenizerService tokenizer, BgeM3Model? embeddings = null, InMemoryVectorStore? vec = null, BgeRerankerM3? reranker = null)
    {
        _llm = llm;
        _tokenizer = tokenizer;
        _embeddings = embeddings;
        _reranker = reranker;
        _vec = vec;
        _toolInjectionManager = new ToolInjectionManager(tokenizer);
    }

    public async Task<string> ChatTurnAsync(string user, IReadOnlyList<(string role, string content)> history, InferenceConfig? config = null, Func<string, string>? toolExecutor = null, ConversationSession? session = null)
    {
        config = config != null ? MergeConfigs(_llm.OptimalConfig, config) : _llm.OptimalConfig;

        List<string> retrieved;
        
        if (_embeddings == null || _vec == null)
        {
            retrieved = [];
        }
        else
        {

            var queryVec = await _embeddings.CreateEmbeddingAsync(user);
            var candidateResults = _vec.TopK(queryVec, 10).ToList();

            retrieved = candidateResults.Select(x => x.Text).ToList();

            if (_reranker != null && candidateResults.Count > 1)
            {
                var rerankedResults = new List<(float score, string text)>();
                foreach (var candidate in candidateResults)
                {
                    var score = await _reranker.GetRerankingScoreAsync(user, candidate.Text);
                    rerankedResults.Add((score: score, text: candidate.Text));
                }

                retrieved = rerankedResults
                    .OrderByDescending(x => x.score)
                    .Take(5)
                    .Select(x => x.text)
                    .ToList();
            }
            else
            {
                retrieved = retrieved.Take(5).ToList();
            }
        }

        KvState kv;
        long[] idsArray;
        
        if (session != null)
        {
            if (!session.IsInitialized)
            {
                await session.InitializeSystemPromptAsync(_llm, retrieved, toolExecutor != null);
            }
            
            await session.AddMessageAsync("user", user, _llm);
            
            var assistantStartTokens = _tokenizer.EncodeToIds("<|start_header_id|>assistant<|end_header_id|>\n\n");
            idsArray = assistantStartTokens.Select(id => (long)id).ToArray();
            kv = session.GetCurrentKvState();
        }
        else
        {
            var prompt = BuildPrompt(history, user, retrieved, toolExecutor != null);
            var inputIds = _tokenizer.EncodeToIds(prompt);
            idsArray = inputIds.Select(id => (long)id).ToArray();
            kv = new KvState();
        }
        var response = new StringBuilder();
        var generatedTokens = new List<int>();
        var toolState = new ToolCallState();
        var recentTokensForStopCheck = new StringBuilder();
        


        int GetNextSample(LlamaSession.StepOutputs outputs, int vocab)
        {
            var span = outputs.GetLogitsSpan();
            var logitsShape = outputs.Logits.GetTensorTypeAndShape().Shape;
            

            
            Span<float> logitsForSampling;
            if (logitsShape.Length == 3) // [batch, seq_len, vocab]
            {
                var batchSize = (int)logitsShape[0];
                var seqLen = (int)logitsShape[1];
                var vocabSize = (int)logitsShape[2];
                
                var lastTokenStart = (seqLen - 1) * vocabSize;
                logitsForSampling = span.Slice(lastTokenStart, vocabSize);
            }
            else if (logitsShape.Length == 2) // [batch, vocab] - generation step
            {
                var batchSize = (int)logitsShape[0];
                var vocabSize = (int)logitsShape[1];
                
                logitsForSampling = span.Slice(0, vocabSize);
            }
            else
            {
                if (span.Length >= vocab)
                {
                    logitsForSampling = span.Slice(span.Length - vocab, vocab);
                }
                else
                {
                    logitsForSampling = span;
                }
            }
            
            var previousTokensSpan = generatedTokens.Count > 0 ? generatedTokens.ToArray().AsSpan() : ReadOnlySpan<int>.Empty;
            return Sampling.Sample(logitsForSampling, config, previousTokensSpan);
        }
        
        for (int step = 0; step < config.MaxTokens; step++)
        {
            var currentInput = step == 0 ? idsArray : [generatedTokens[^1]];
            
            var tokensToProcess = currentInput.Length;
            var totalSeqLen = kv.CalculateTotalLengthAfterTokens(tokensToProcess);
            
            var outputs = await _llm.RunOptimizedStep(currentInput, kv, step, totalSeqLen);
            var newKv = outputs.KvCache;

            var logitsShape = outputs.Logits.GetTensorTypeAndShape().Shape;
            var vocab = (int)logitsShape[^1];
            
            var nextId = GetNextSample(outputs, vocab);
            
            generatedTokens.Add(nextId);

            var tokenText = _tokenizer.DecodeFromIds([nextId]);
            
            Console.Write(tokenText);
            
            response.Append(tokenText);
            recentTokensForStopCheck.Append(tokenText);
            
            if (recentTokensForStopCheck.Length > 100)
            {
                recentTokensForStopCheck.Remove(0, recentTokensForStopCheck.Length - 100);
            }
            
            if (toolExecutor != null && _toolInjectionManager.IsInjectionPointSafe(step, step > 0))
            {
                toolState.AppendToken(tokenText);
                
                var pendingCall = toolState.GetNextPendingCall();
                if (pendingCall != null)
                {
                    var injectionResult = await _toolInjectionManager.ExecuteAndInjectAsync(
                        pendingCall, toolExecutor, toolState, _llm, 
                        newKv, step, totalSeqLen);
                    
                    if (injectionResult.Success)
                    {
                        Console.Write(injectionResult.InjectedText);
                        
                        response.Append(injectionResult.InjectedText);
                        generatedTokens.AddRange(injectionResult.InjectedTokens);
                        
                        outputs.Dispose();
                        newKv = injectionResult.UpdatedKvState;
                        
                        if (!newKv.ValidateSequenceLength(injectionResult.NewSequenceLength))
                        {
                            Console.WriteLine("⚠️  Sequence length inconsistency detected after tool injection");
                        }
                    }
                    else
                    {
                        Console.WriteLine($"⚠️  Tool injection failed: {injectionResult.ErrorMessage}");
                        var errorText = $"\n[Tool execution failed: {injectionResult.ErrorMessage}]\n";
                        Console.Write(errorText);
                        response.Append(errorText);
                    }
                }
            }

            kv = newKv;
            
            if (IsStopToken(nextId, config) || IsStopSequence(recentTokensForStopCheck.ToString(), config))
            {
                outputs.Dispose();
                break;
            }
            
            outputs.Dispose();
        }
        
        if (session != null)
        {
            session.UpdateKvState(kv);
            session.AddToHistory("assistant", response.ToString());
        }
        else
        {
            kv.Dispose();
        }

        if (!response.ToString().EndsWith('\n'))
        {
            Console.WriteLine();
        }

        return response.ToString();
    }
    
    /// <summary>
    /// Efficiently merge user config with pre-computed optimal config
    /// </summary>
    private static InferenceConfig MergeConfigs(InferenceConfig optimalConfig, InferenceConfig userConfig)
    {
        return optimalConfig with
        {
            Temperature = userConfig.Temperature,
            TopK = userConfig.TopK,
            TopP = userConfig.TopP,
            RepetitionPenalty = userConfig.RepetitionPenalty,
            FrequencyPenalty = userConfig.FrequencyPenalty,
            PresencePenalty = userConfig.PresencePenalty,
            MaxTokens = userConfig.MaxTokens,
            Seed = userConfig.Seed,
            UseGreedy = userConfig.UseGreedy,
            MinP = userConfig.MinP,
            TfsZ = userConfig.TfsZ,
            TypicalP = userConfig.TypicalP,
            StopTokenIds = optimalConfig.StopTokenIds.Concat(userConfig.StopTokenIds).ToHashSet(),
            StopSequences = optimalConfig.StopSequences.Concat(userConfig.StopSequences).ToArray()
        };
    }

    internal static bool IsStopToken(int tokenId, InferenceConfig config) => config.StopTokenIds.Contains(tokenId);

    internal static bool IsStopSequence(string text, InferenceConfig config)
    {
        return config.StopSequences.Any(seq => text.Contains(seq));
    }

    internal static string BuildSystemPrompt(IReadOnlyList<string> retrieved, bool enableTools = false)
    {
        var sb = new StringBuilder();
        

        sb.AppendLine("<|begin_of_text|>");
        sb.AppendLine("<|start_header_id|>system<|end_header_id|>");
        sb.AppendLine();
        

        sb.AppendLine("You are an AI assistant specialized in answering questions based on provided context information.");
        sb.AppendLine();
        
        if (enableTools)
        {
            sb.AppendLine();
            sb.AppendLine("## Tool Usage:");
            sb.AppendLine("When you need to use a tool, format it as:");
            sb.AppendLine("```");
            sb.AppendLine("TOOL_CALL");
            sb.AppendLine("name: tool_name");
            sb.AppendLine("args: tool_arguments");
            sb.AppendLine("END_TOOL_CALL");
            sb.AppendLine("```");
            sb.AppendLine("The tool result will be provided in TOOL_RESULT...END_TOOL_RESULT tags.");
        }
        
        if (retrieved.Count > 0)
        {
            sb.AppendLine();
            sb.AppendLine("## Available Context:");
            for (int i = 0; i < retrieved.Count; i++)
            {
                sb.AppendLine($"**Source {i + 1}:**");
                sb.AppendLine($"> {retrieved[i]}");
                sb.AppendLine();
            }
        }
        

        sb.AppendLine("<|eot_id|>");
        
        return sb.ToString();
    }

    internal static string BuildPrompt(IReadOnlyList<(string role, string content)> history, string user, IReadOnlyList<string> retrieved, bool enableTools = false)
    {
        var sb = new StringBuilder();
        

        sb.AppendLine("<|begin_of_text|>");
        sb.AppendLine("<|start_header_id|>system<|end_header_id|>");
        sb.AppendLine();
        

        sb.AppendLine("You are an AI assistant specialized in answering questions based on provided context information.");
        sb.AppendLine();
        
        if (enableTools)
        {
            sb.AppendLine();
            sb.AppendLine("## Tool Usage:");
            sb.AppendLine("When you need to use a tool, format it as:");
            sb.AppendLine("```");
            sb.AppendLine("TOOL_CALL");
            sb.AppendLine("name: tool_name");
            sb.AppendLine("args: tool_arguments");
            sb.AppendLine("END_TOOL_CALL");
            sb.AppendLine("```");
            sb.AppendLine("The tool result will be provided in TOOL_RESULT...END_TOOL_RESULT tags.");
        }
        
        if (retrieved.Count > 0)
        {
            sb.AppendLine();
            sb.AppendLine("## Available Context:");
            for (int i = 0; i < retrieved.Count; i++)
            {
                sb.AppendLine($"**Source {i + 1}:**");
                sb.AppendLine($"> {retrieved[i]}");
                sb.AppendLine();
            }
        }
        

        sb.AppendLine("<|eot_id|>");
        

        foreach (var (role, content) in history)
        {
            if (role.Equals("user", StringComparison.OrdinalIgnoreCase))
            {
                sb.AppendLine("<|start_header_id|>user<|end_header_id|>");
                sb.AppendLine();
                sb.AppendLine(content);
                sb.AppendLine("<|eot_id|>");
            }
            else if (role.Equals("assistant", StringComparison.OrdinalIgnoreCase))
            {
                sb.AppendLine("<|start_header_id|>assistant<|end_header_id|>");
                sb.AppendLine();
                sb.AppendLine(content);
                sb.AppendLine("<|eot_id|>");
            }
        }
        

        sb.AppendLine("<|start_header_id|>user<|end_header_id|>");
        sb.AppendLine();
        sb.AppendLine(user);
        sb.AppendLine("<|eot_id|>");
        

        sb.AppendLine("<|start_header_id|>assistant<|end_header_id|>");
        sb.AppendLine();
        
        return sb.ToString();
    }
}


