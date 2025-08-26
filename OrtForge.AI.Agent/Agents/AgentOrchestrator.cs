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

    public string ChatTurn(string user, IReadOnlyList<(string role, string content)> history, Func<string, string>? toolExecutor = null)
    {
        var queryVec = _embeddings.EmbedTokenIds(_tokenizer.EncodeToIds(user));
        var retrieved = _vec.TopK(queryVec, 5).Select(x => x.Text).ToList();

        var prompt = BuildPrompt(history, user, retrieved);
        var inputIds = _tokenizer.EncodeToIds(prompt);

        var idsTensor = new DenseTensor<int>(new[] { 1, inputIds.Length });
        for (int i = 0; i < inputIds.Length; i++) idsTensor[0, i] = inputIds[i];

        var kv = LlamaSession.KvState.Empty;
        var response = new StringBuilder();

        for (int step = 0; step < 2048; step++)
        {
            using var inputs = LlamaSession.StepInputs.Create(idsTensor, kv);
            var outputs = _llm.RunStep(inputs);
            
            var newKv = outputs.KvCache;

            var logitsShape = outputs.Logits.GetTensorTypeAndShape().Shape;
            var vocab = (int)logitsShape[^1];
            var span = outputs.GetLogitsSpan();
            var logitsLast = span.Slice(span.Length - vocab, vocab);
            var nextId = Sampling.TopK(logitsLast, k: 40, temperature: 0.7);

            var tokenText = _tokenizer.DecodeFromIds(new[] { nextId });
            response.Append(tokenText);

            if (IsStopToken(nextId)) break;

            idsTensor = new DenseTensor<int>(new[] { 1, 1 });
            idsTensor[0, 0] = nextId;

            if (toolExecutor != null && IsToolCallStart(tokenText))
            {
                var (toolName, toolArgs) = ParseToolCall(response.ToString());
                var toolResult = toolExecutor.Invoke(toolArgs);
                var toolInject = $"\n[T-RESULT]\n{toolResult}\n[/T-RESULT]\n";
                var injectIds = _tokenizer.EncodeToIds(toolInject);
                var injectTensor = new DenseTensor<int>(new[] { 1, injectIds.Length });
                for (int i = 0; i < injectIds.Length; i++) injectTensor[0, i] = injectIds[i];
                using var injectInputs = LlamaSession.StepInputs.Create(injectTensor, newKv);
                var injectOutputs = _llm.RunStep(injectInputs);
                outputs.Dispose();
                outputs = injectOutputs;
                newKv = injectOutputs.KvCache;
            }
            
            kv?.Dispose();
            kv = newKv;
            outputs.Dispose();
        }

        kv?.Dispose();
        return response.ToString();
    }

    internal static bool IsStopToken(int tokenId) => tokenId == 2 || tokenId == 0;

    internal static bool IsToolCallStart(string decoded) => decoded.Contains("[T-CALL]");

    internal static (string name, string args) ParseToolCall(string text)
    {
        var start = text.LastIndexOf("[T-CALL]");
        if (start < 0) return ("", "");
        var end = text.IndexOf("[/T-CALL]", start, StringComparison.Ordinal);
        var body = end > start ? text.Substring(start + 8, end - (start + 8)) : string.Empty;
        return ("tool", body);
    }

    internal static string BuildPrompt(IReadOnlyList<(string role, string content)> history, string user, IReadOnlyList<string> retrieved)
    {
        var sb = new StringBuilder();
        sb.AppendLine("<|system|>You are a helpful assistant. Use context when relevant and cite sources.</s>");
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


