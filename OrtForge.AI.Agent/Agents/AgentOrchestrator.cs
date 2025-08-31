using System.Runtime.CompilerServices;
using System.Text;
using OrtForge.AI.Agent.Generation;
using OrtForge.AI.Agent.Rag;
using OrtForge.AI.Models.Models;

namespace OrtForge.AI.Agent.Agents;

public sealed class AgentOrchestrator
{
    private readonly BgeM3Model? _embeddings;
    private readonly BgeRerankerM3? _reranker;
    private readonly InMemoryVectorStore? _vec;

    public AgentOrchestrator(BgeM3Model? embeddings = null, InMemoryVectorStore? vec = null, BgeRerankerM3? reranker = null)
    {
        _embeddings = embeddings;
        _reranker = reranker;
        _vec = vec;
    }

    public async IAsyncEnumerable<string> ChatTurnAsync(ConversationSession session, string user,
        Func<string, string>? toolExecutor = null, [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        List<string> retrieved;

        if (_embeddings == null || _vec == null)
        {
            retrieved = [];
        }
        else
        {
            var queryVec = await _embeddings.CreateEmbeddingAsync(user, cancellationToken: cancellationToken);
            var candidateResults = _vec.TopK(queryVec, 10).ToList();

            retrieved = candidateResults.Select(x => x.Text).ToList();

            if (_reranker != null && candidateResults.Count > 1)
            {
                var rerankedResults = new List<(float score, string text)>();
                foreach (var candidate in candidateResults)
                {
                    var score = await _reranker.GetRerankingScoreAsync(user, candidate.Text, cancellationToken: cancellationToken);
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

        var prompt = !session.IsInitialized
            ? BuildSystemPrompt(retrieved, user, toolExecutor != null)
            : BuildChatTurnPrompt(retrieved, user, toolExecutor != null);

        await foreach (var token in session.GenerateNextResponseAsync(prompt, toolExecutor, cancellationToken))
        {
            yield return token;
        }
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

    internal static string BuildSystemPrompt(IReadOnlyList<string> retrieved, string firstUserMessage, bool enableTools = false)
    {
        var sb = new StringBuilder();
        sb.AppendLine("<|begin_of_text|><|start_header_id|>system<|end_header_id|>");
        sb.AppendLine("Answer questions best to your knowledge.");
        sb.AppendLine("<|eot_id|>");
        sb.AppendLine("<|start_header_id|>user<|end_header_id|>");
        sb.AppendLine(firstUserMessage);
        if (retrieved.Count > 0)
        {
            sb.AppendLine("## Available Context:");
            for (int i = 0; i < retrieved.Count; i++)
            {
                sb.AppendLine($"**Source {i + 1}:**");
                sb.AppendLine($"> {retrieved[i]}");
            }
        }
        sb.AppendLine("<|eot_id|>");
        
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
        
        sb.AppendLine("<|start_header_id|>assistant<|end_header_id|>");
        
        return sb.ToString();
    }

    internal static string BuildChatTurnPrompt(IReadOnlyList<string> retrieved, string user, bool enableTools = false)
    {
        var sb = new StringBuilder();
        sb.AppendLine("<|start_header_id|>user<|end_header_id|>");
        sb.AppendLine(user);
        if (retrieved.Count > 0)
        {
            sb.AppendLine("## Available Context:");
            for (int i = 0; i < retrieved.Count; i++)
            {
                sb.AppendLine($"**Source {i + 1}:**");
                sb.AppendLine($"> {retrieved[i]}");
            }
        }
        sb.AppendLine("<|eot_id|>");
        
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
        
        sb.AppendLine("<|start_header_id|>assistant<|end_header_id|>");
        return sb.ToString();
    }
}


