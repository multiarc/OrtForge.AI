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

    public async Task<string> ChatTurnAsync(string user, IReadOnlyList<(string role, string content)> history, InferenceConfig? config = null, Func<string, string>? toolExecutor = null, ConversationSession? session = null)
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
        var currentSeqLength = session != null ? kv.AccumulatedSequenceLength : idsArray.Length;
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
            }
            else if (logitsShape.Length == 2) // [batch, vocab] - generation step
            {
                // For single token generation, logits are already [batch, vocab]
                var batchSize = (int)logitsShape[0];
                var vocabSize = (int)logitsShape[1];
                
                // Take logits for batch 0
                logitsForSampling = span.Slice(0, vocabSize);
            }
            else
            {
                // Fallback: assume span is already the right size [vocab]
                logitsForSampling = span;
            }
            
            var previousTokensSpan = generatedTokens.Count > 0 ? generatedTokens.ToArray().AsSpan() : ReadOnlySpan<int>.Empty;
            return Sampling.Sample(logitsForSampling, config, previousTokensSpan);
        }
        
        for (int step = 0; step < config.MaxTokens; step++)
        {
            // First step: use full prompt, subsequent steps: use only the last generated token
            var currentInput = step == 0 ? idsArray : new long[] { generatedTokens[^1] };
            
            // Update sequence length for the tokens we're about to process
            var tokensToProcess = currentInput.Length;
            var totalSeqLen = currentSeqLength + tokensToProcess;
            
            var outputs = await _llm.RunOptimizedStep(currentInput, kv, step, totalSeqLen);
            var newKv = outputs.KvCache;

            var logitsShape = outputs.Logits.GetTensorTypeAndShape().Shape;
            var vocab = (int)logitsShape[^1];
            
            var nextId = GetNextSample(outputs, vocab);
            
            generatedTokens.Add(nextId);

            var tokenText = _tokenizer.DecodeFromIds(new[] { nextId });
            
            // Stream token output to console immediately
            Console.Write(tokenText);
            
            // Check for immediate repetition (same token repeated)
            if (generatedTokens.Count >= 3)
            {
                var recent = generatedTokens.TakeLast(3).ToArray();
                if (recent[0] == recent[1] && recent[1] == recent[2])
                {
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
                        // Stream injected text immediately as well
                        Console.Write(injectedText);
                        
                        response.Append(injectedText);
                        generatedTokens.AddRange(injectedTokens);
                        
                        var injectArray = injectedTokens.Select(token => (long)token).ToArray();
                        
                        var injectSeqLen = totalSeqLen + injectArray.Length;
                        var injectOutputs = await _llm.RunOptimizedStep(injectArray, newKv, step, injectSeqLen);
                        currentSeqLength = injectSeqLen;
                        outputs.Dispose();
                        outputs = injectOutputs;
                        newKv = injectOutputs.KvCache;
                    }
                }
            }

            if (IsStopToken(nextId, config) || IsStopSequence(response.ToString(), config)) break;

            kv = newKv;
            currentSeqLength = totalSeqLen; // Update our sequence length tracker
            outputs.Dispose();
        }
        
        if (session != null)
        {
            session.UpdateKvState(kv);
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

    internal static string BuildSystemPrompt(IReadOnlyList<string> retrieved, bool enableTools = false)
    {
        var sb = new StringBuilder();
        

        sb.AppendLine("<|begin_of_text|>");
        sb.AppendLine("<|start_header_id|>system<|end_header_id|>");
        sb.AppendLine();
        

        sb.AppendLine("You are an AI assistant specialized in answering questions based on provided context information. Follow these instructions strictly:");
        sb.AppendLine();
        sb.AppendLine("## Core Instructions:");
        sb.AppendLine("- **ONLY respond as the assistant** - never generate or fill in user messages, questions, or responses");
        sb.AppendLine("- **Always format your response in markdown** with proper headings, lists, code blocks, and emphasis");
        sb.AppendLine("- **Base your answers primarily on the provided context** - if context doesn't contain the answer, clearly state this");
        sb.AppendLine("- **Cite sources explicitly** when referencing context information");
        sb.AppendLine("- **Accept and process markdown-formatted input** from users");
        sb.AppendLine();
        sb.AppendLine("## Response Format Requirements:");
        sb.AppendLine("- Use **bold** for emphasis and key points");
        sb.AppendLine("- Use `code formatting` for technical terms, file names, and code snippets");
        sb.AppendLine("- Use proper markdown headers (##, ###) to structure your response");
        sb.AppendLine("- Use bullet points or numbered lists when presenting multiple items");
        sb.AppendLine("- Include relevant code blocks with proper language specification when applicable");
        sb.AppendLine();
        sb.AppendLine("## Context Usage:");
        sb.AppendLine("- Analyze the provided context thoroughly before responding");
        sb.AppendLine("- Quote relevant portions using markdown blockquotes (>) when appropriate");
        sb.AppendLine("- If multiple context sources conflict, acknowledge and explain the differences");
        sb.AppendLine("- If context is insufficient, explicitly state what additional information would be needed");
        
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
        

        sb.AppendLine("You are an AI assistant specialized in answering questions based on provided context information. Follow these instructions strictly:");
        sb.AppendLine();
        sb.AppendLine("## Core Instructions:");
        sb.AppendLine("- **ONLY respond as the assistant** - never generate or fill in user messages, questions, or responses");
        sb.AppendLine("- **Always format your response in markdown** with proper headings, lists, code blocks, and emphasis");
        sb.AppendLine("- **Base your answers primarily on the provided context** - if context doesn't contain the answer, clearly state this");
        sb.AppendLine("- **Cite sources explicitly** when referencing context information");
        sb.AppendLine("- **Accept and process markdown-formatted input** from users");
        sb.AppendLine();
        sb.AppendLine("## Response Format Requirements:");
        sb.AppendLine("- Use **bold** for emphasis and key points");
        sb.AppendLine("- Use `code formatting` for technical terms, file names, and code snippets");
        sb.AppendLine("- Use proper markdown headers (##, ###) to structure your response");
        sb.AppendLine("- Use bullet points or numbered lists when presenting multiple items");
        sb.AppendLine("- Include relevant code blocks with proper language specification when applicable");
        sb.AppendLine();
        sb.AppendLine("## Context Usage:");
        sb.AppendLine("- Analyze the provided context thoroughly before responding");
        sb.AppendLine("- Quote relevant portions using markdown blockquotes (>) when appropriate");
        sb.AppendLine("- If multiple context sources conflict, acknowledge and explain the differences");
        sb.AppendLine("- If context is insufficient, explicitly state what additional information would be needed");
        
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


