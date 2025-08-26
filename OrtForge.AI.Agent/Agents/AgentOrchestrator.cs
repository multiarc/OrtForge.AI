using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Microsoft.ML.OnnxRuntime.Tensors;
using OrtAgent.Core.Generation;
using OrtAgent.Core.LLM;
using OrtAgent.Core.Rag;
using OrtAgent.Core.Tokenization;

namespace OrtAgent.Core.Agents;

public sealed class AgentOrchestrator
{
    private readonly LlamaSession _llm;
    private readonly TokenizerService _tokenizer;
    private readonly EmbeddingService _embeddings;
    private readonly InMemoryVectorStore _vec;

    public AgentOrchestrator(LlamaSession llm, TokenizerService tokenizer, EmbeddingService embeddings, InMemoryVectorStore vec)
    {
        _llm = llm;
        _tokenizer = tokenizer;
        _embeddings = embeddings;
        _vec = vec;
    }

    public string ChatTurn(string user, IReadOnlyList<(string role, string content)> history, InferenceConfig? config = null, Func<string, string>? toolExecutor = null)
    {
        config = LlamaOptimizations.GetOptimalConfigForModel(_llm.ModelName, config);
        
        var queryVec = _embeddings.EmbedTokenIds(_tokenizer.EncodeToIds(user));
        var retrieved = _vec.TopK(queryVec, 5).Select(x => x.Text).ToList();

        var prompt = BuildPrompt(history, user, retrieved, toolExecutor != null);
        var inputIds = _tokenizer.EncodeToIds(prompt);

        var idsTensor = new DenseTensor<int>(new[] { 1, inputIds.Length });
        for (int i = 0; i < inputIds.Length; i++) idsTensor[0, i] = inputIds[i];

        var kv = LlamaSession.KvState.Empty;
        var response = new StringBuilder();
        var generatedTokens = new List<int>();
        var sequenceLength = inputIds.Length;
        var toolState = new ToolCallState();
        
        for (int step = 0; step < config.MaxTokens; step++)
        {
            var outputs = _llm.RunOptimizedStep(idsTensor, kv, step, sequenceLength + generatedTokens.Count);
            
            var newKv = outputs.KvCache;

            var logitsShape = outputs.Logits.GetTensorTypeAndShape().Shape;
            var vocab = (int)logitsShape[^1];
            var span = outputs.GetLogitsSpan();
            var logitsLast = span.Slice(span.Length - vocab, vocab);
            
            var previousTokensSpan = generatedTokens.Count > 0 ? generatedTokens.ToArray().AsSpan() : ReadOnlySpan<int>.Empty;
            var nextId = Sampling.Sample(logitsLast, config, previousTokensSpan);
            
            generatedTokens.Add(nextId);

            var tokenText = _tokenizer.DecodeFromIds(new[] { nextId });
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
                        
                        var injectTensor = new DenseTensor<int>(new[] { 1, injectedTokens.Length });
                        for (int i = 0; i < injectedTokens.Length; i++) injectTensor[0, i] = injectedTokens[i];
                        
                        var injectOutputs = _llm.RunOptimizedStep(injectTensor, newKv, step, sequenceLength + generatedTokens.Count);
                        outputs.Dispose();
                        outputs = injectOutputs;
                        newKv = injectOutputs.KvCache;
                    }
                }
            }

            if (IsStopToken(nextId, config) || IsStopSequence(response.ToString(), config)) break;

            idsTensor = new DenseTensor<int>(new[] { 1, 1 });
            idsTensor[0, 0] = nextId;
            
            kv?.Dispose();
            kv = newKv;
            outputs.Dispose();
        }

        kv?.Dispose();
        return response.ToString();
    }

    public IEnumerable<string> ChatTurnStream(string user, IReadOnlyList<(string role, string content)> history, InferenceConfig? config = null, Func<string, string>? toolExecutor = null)
    {
        config = LlamaOptimizations.GetOptimalConfigForModel(_llm.ModelName, config);
        
        var queryVec = _embeddings.EmbedTokenIds(_tokenizer.EncodeToIds(user));
        var retrieved = _vec.TopK(queryVec, 5).Select(x => x.Text).ToList();

        var prompt = BuildPrompt(history, user, retrieved, toolExecutor != null);
        var inputIds = _tokenizer.EncodeToIds(prompt);

        var idsTensor = new DenseTensor<int>(new[] { 1, inputIds.Length });
        for (int i = 0; i < inputIds.Length; i++) idsTensor[0, i] = inputIds[i];

        var kv = LlamaSession.KvState.Empty;
        var response = new StringBuilder();
        var generatedTokens = new List<int>();
        var sequenceLength = inputIds.Length;
        var toolState = new ToolCallState();
        
        for (int step = 0; step < config.MaxTokens; step++)
        {
            var outputs = _llm.RunOptimizedStep(idsTensor, kv, step, sequenceLength + generatedTokens.Count);
            
            var newKv = outputs.KvCache;

            var logitsShape = outputs.Logits.GetTensorTypeAndShape().Shape;
            var vocab = (int)logitsShape[^1];
            var span = outputs.GetLogitsSpan();
            var logitsLast = span.Slice(span.Length - vocab, vocab);
            
            var previousTokensSpan = generatedTokens.Count > 0 ? generatedTokens.ToArray().AsSpan() : ReadOnlySpan<int>.Empty;
            var nextId = Sampling.Sample(logitsLast, config, previousTokensSpan);
            
            generatedTokens.Add(nextId);

            var tokenText = _tokenizer.DecodeFromIds(new[] { nextId });
            response.Append(tokenText);
            yield return tokenText;
            
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
                        
                        var injectTensor = new DenseTensor<int>(new[] { 1, injectedTokens.Length });
                        for (int i = 0; i < injectedTokens.Length; i++) injectTensor[0, i] = injectedTokens[i];
                        
                        var injectOutputs = _llm.RunOptimizedStep(injectTensor, newKv, step, sequenceLength + generatedTokens.Count);
                        outputs.Dispose();
                        outputs = injectOutputs;
                        newKv = injectOutputs.KvCache;
                        
                        yield return injectedText;
                    }
                }
            }

            if (IsStopToken(nextId, config) || IsStopSequence(response.ToString(), config)) break;

            idsTensor = new DenseTensor<int>(new[] { 1, 1 });
            idsTensor[0, 0] = nextId;
            
            kv?.Dispose();
            kv = newKv;
            outputs.Dispose();
        }

        kv?.Dispose();
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
        sb.AppendLine("<|system|>You are a helpful assistant. Use context when relevant and cite sources.");
        
        if (enableTools)
        {
            sb.AppendLine();
            sb.AppendLine("When you need to use a tool, format it as:");
            sb.AppendLine("<|tool_call|>");
            sb.AppendLine("name: tool_name");
            sb.AppendLine("args: tool_arguments");
            sb.AppendLine("<|/tool_call|>");
            sb.AppendLine();
            sb.AppendLine("The tool result will be provided in <|tool_result|>...<|/tool_result|> tags.");
        }
        
        sb.AppendLine("</s>");
        
        if (retrieved.Count > 0)
        {
            sb.AppendLine("<|context|>");
            foreach (var ctx in retrieved) sb.AppendLine(ctx);
            sb.AppendLine("</context>");
        }
        
        foreach (var (role, content) in history)
        {
            sb.Append("<|").Append(role).Append("|>").Append(content).AppendLine("</s>");
        }
        
        sb.Append("<|user|>").Append(user).AppendLine("</s>");
        sb.Append("<|assistant|>");
        return sb.ToString();
    }
}


