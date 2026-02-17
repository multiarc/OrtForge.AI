using System.Diagnostics;
using System.Text.Json;
using OrtForge.AI.Agent.Agents;
using OrtForge.AI.Agent.Generation;
using OrtForge.AI.Agent.LLM;
using OrtForge.AI.Agent.Tokenization;

namespace OrtForge.AI.Agent.TestApp;

public sealed class PerformanceTestRunner
{
    public record TestResult(
        string Prompt,
        string Category,
        double TimeToFirstTokenMs,
        double TokensPerSecond,
        int TotalTokens,
        double TotalTimeMs,
        string Response,
        bool HitMaxTokens,
        bool StoppedNaturally);

    public record BenchmarkSummary(
        string ModelPath,
        string ConfigName,
        double AverageTimeToFirstTokenMs,
        double AverageTokensPerSecond,
        int TotalPrompts,
        double TotalDurationMs,
        List<TestResult> Results);

    private static readonly Dictionary<string, string[]> TestPrompts = new()
    {
        ["Factual"] =
        [
            "What is the capital of France?",
            "What is 2 + 2?",
            "How many days are in a week?",
            "What color is the sky?",
            "Who wrote Romeo and Juliet?"
        ],
        ["Math"] =
        [
            "What is 15 multiplied by 7?",
            "If I have 3 apples and buy 5 more, how many do I have?",
            "What is the next number: 2, 4, 6, 8, ?",
            "Is 17 a prime number? Answer yes or no."
        ],
        ["Coding"] =
        [
            "Write a Python function that adds two numbers.",
            "Write hello world in JavaScript.",
            "What does 'print' do in Python?"
        ],
        ["Creative"] =
        [
            "Write one sentence about the ocean.",
            "Name three colors.",
            "Complete: The quick brown fox..."
        ]
    };

    private static readonly string[][] MultiTurnConversation =
    [
        ["My name is Alice.", "What is my name?", "Tell me a joke."]
    ];

    private readonly LlamaSession _llm;
    private readonly TokenizerService _tokenizer;
    private readonly string _modelPath;

    public PerformanceTestRunner(LlamaSession llm, TokenizerService tokenizer, string modelPath)
    {
        _llm = llm;
        _tokenizer = tokenizer;
        _modelPath = modelPath;
    }

    public async Task<BenchmarkSummary> RunBenchmarksAsync(
        InferenceConfig config,
        string configName = "Default",
        CancellationToken cancellationToken = default)
    {
        var results = new List<TestResult>();
        var overallStopwatch = Stopwatch.StartNew();

        Console.WriteLine();
        Console.WriteLine("╔══════════════════════════════════════════════════════════════╗");
        Console.WriteLine("║           OrtForge.AI Inference Benchmark                    ║");
        Console.WriteLine("╚══════════════════════════════════════════════════════════════╝");
        Console.WriteLine();
        Console.WriteLine($"  Model: {Path.GetFileName(_modelPath)}");
        Console.WriteLine($"  Config: {configName} (Temp={config.Temperature}, TopK={config.TopK}, TopP={config.TopP})");
        Console.WriteLine($"  Model Type: {_llm.ModelType}");
        Console.WriteLine($"  Max Tokens: {config.MaxTokens}");
        Console.WriteLine($"  Stop Token IDs: [{string.Join(", ", config.StopTokenIds)}]");
        Console.WriteLine();

        // Run single-turn tests
        foreach (var (category, prompts) in TestPrompts)
        {
            Console.WriteLine($"┌─ Category: {category} ─────────────────────────────────────────┐");
            
            foreach (var prompt in prompts)
            {
                if (cancellationToken.IsCancellationRequested)
                    break;

                var result = await RunSinglePromptAsync(prompt, category, config, cancellationToken);
                results.Add(result);
                PrintResult(result);
            }
            
            Console.WriteLine("└──────────────────────────────────────────────────────────────┘");
            Console.WriteLine();
        }

        // Run multi-turn conversation test
        Console.WriteLine("┌─ Category: Multi-turn ────────────────────────────────────────┐");
        var multiTurnResults = await RunMultiTurnTestAsync(MultiTurnConversation[0], config, cancellationToken);
        foreach (var result in multiTurnResults)
        {
            results.Add(result);
            PrintResult(result);
        }
        Console.WriteLine("└──────────────────────────────────────────────────────────────┘");
        Console.WriteLine();

        overallStopwatch.Stop();

        var summary = new BenchmarkSummary(
            ModelPath: _modelPath,
            ConfigName: configName,
            AverageTimeToFirstTokenMs: results.Count > 0 ? results.Average(r => r.TimeToFirstTokenMs) : 0,
            AverageTokensPerSecond: results.Count > 0 ? results.Average(r => r.TokensPerSecond) : 0,
            TotalPrompts: results.Count,
            TotalDurationMs: overallStopwatch.Elapsed.TotalMilliseconds,
            Results: results);

        PrintSummary(summary);

        return summary;
    }

    private async Task<TestResult> RunSinglePromptAsync(
        string prompt,
        string category,
        InferenceConfig config,
        CancellationToken cancellationToken)
    {
        using var session = new ConversationSession(_llm, _tokenizer, config);
        var agent = new AgentOrchestrator();

        var stopwatch = Stopwatch.StartNew();
        var firstTokenTime = TimeSpan.Zero;
        var tokenCount = 0;
        var response = new System.Text.StringBuilder();
        var isFirstToken = true;

        await foreach (var token in agent.ChatTurnAsync(session, prompt, cancellationToken: cancellationToken))
        {
            if (isFirstToken)
            {
                firstTokenTime = stopwatch.Elapsed;
                isFirstToken = false;
            }
            tokenCount++;
            response.Append(token);
        }

        stopwatch.Stop();

        var totalTimeMs = stopwatch.Elapsed.TotalMilliseconds;
        var generationTimeMs = totalTimeMs - firstTokenTime.TotalMilliseconds;
        var tokensPerSecond = generationTimeMs > 0 && tokenCount > 1 
            ? (tokenCount - 1) / (generationTimeMs / 1000.0) 
            : 0;

        var hitMaxTokens = tokenCount >= config.MaxTokens;
        var stoppedNaturally = !hitMaxTokens && tokenCount > 0;

        return new TestResult(
            Prompt: prompt,
            Category: category,
            TimeToFirstTokenMs: firstTokenTime.TotalMilliseconds,
            TokensPerSecond: tokensPerSecond,
            TotalTokens: tokenCount,
            TotalTimeMs: totalTimeMs,
            Response: response.ToString().Trim(),
            HitMaxTokens: hitMaxTokens,
            StoppedNaturally: stoppedNaturally);
    }

    private async Task<List<TestResult>> RunMultiTurnTestAsync(
        string[] turns,
        InferenceConfig config,
        CancellationToken cancellationToken)
    {
        var results = new List<TestResult>();
        using var session = new ConversationSession(_llm, _tokenizer, config);
        var agent = new AgentOrchestrator();

        for (int i = 0; i < turns.Length; i++)
        {
            var prompt = turns[i];
            var stopwatch = Stopwatch.StartNew();
            var firstTokenTime = TimeSpan.Zero;
            var tokenCount = 0;
            var response = new System.Text.StringBuilder();
            var isFirstToken = true;

            await foreach (var token in agent.ChatTurnAsync(session, prompt, cancellationToken: cancellationToken))
            {
                if (isFirstToken)
                {
                    firstTokenTime = stopwatch.Elapsed;
                    isFirstToken = false;
                }
                tokenCount++;
                response.Append(token);
            }

            stopwatch.Stop();

            var totalTimeMs = stopwatch.Elapsed.TotalMilliseconds;
            var generationTimeMs = totalTimeMs - firstTokenTime.TotalMilliseconds;
            var tokensPerSecond = generationTimeMs > 0 && tokenCount > 1 
                ? (tokenCount - 1) / (generationTimeMs / 1000.0) 
                : 0;

            var hitMaxTokens = tokenCount >= config.MaxTokens;
            var stoppedNaturally = !hitMaxTokens && tokenCount > 0;

            results.Add(new TestResult(
                Prompt: $"[Turn {i + 1}] {prompt}",
                Category: "Multi-turn",
                TimeToFirstTokenMs: firstTokenTime.TotalMilliseconds,
                TokensPerSecond: tokensPerSecond,
                TotalTokens: tokenCount,
                TotalTimeMs: totalTimeMs,
                Response: response.ToString().Trim(),
                HitMaxTokens: hitMaxTokens,
                StoppedNaturally: stoppedNaturally));
        }

        return results;
    }

    private static void PrintResult(TestResult result)
    {
        var promptDisplay = result.Prompt.Length > 45 
            ? result.Prompt[..42] + "..." 
            : result.Prompt;
        
        var stopStatus = result.StoppedNaturally ? "EOS" : (result.HitMaxTokens ? "MAX" : "???");
        
        Console.WriteLine($"│  \"{promptDisplay}\"");
        Console.WriteLine($"│    TTFT: {result.TimeToFirstTokenMs,7:F1}ms | TPS: {result.TokensPerSecond,6:F1} | Tokens: {result.TotalTokens,4} | Stop: {stopStatus} | Total: {result.TotalTimeMs,7:F0}ms");
        
        // Show full response (sanitized)
        var fullResponse = SanitizeForDisplay(result.Response);
        
        // Word wrap at ~70 chars for readability
        var lines = WordWrap(fullResponse, 68);
        Console.WriteLine($"│    Response:");
        foreach (var line in lines)
        {
            Console.WriteLine($"│      {line}");
        }
        Console.WriteLine("│");
    }

    private static List<string> WordWrap(string text, int maxWidth)
    {
        var lines = new List<string>();
        if (string.IsNullOrEmpty(text))
        {
            lines.Add("(empty)");
            return lines;
        }

        var words = text.Split(' ', StringSplitOptions.RemoveEmptyEntries);
        var currentLine = new System.Text.StringBuilder();

        foreach (var word in words)
        {
            if (currentLine.Length + word.Length + 1 > maxWidth)
            {
                if (currentLine.Length > 0)
                {
                    lines.Add(currentLine.ToString());
                    currentLine.Clear();
                }
            }
            
            if (currentLine.Length > 0)
                currentLine.Append(' ');
            currentLine.Append(word);
        }

        if (currentLine.Length > 0)
            lines.Add(currentLine.ToString());

        return lines;
    }

    private static string SanitizeForDisplay(string text)
    {
        if (string.IsNullOrEmpty(text))
            return "(empty)";
        
        // Remove common special tokens
        var sanitized = text
            .Replace("<|begin_of_text|>", "")
            .Replace("<|end_of_text|>", "")
            .Replace("<|start_header_id|>", "")
            .Replace("<|end_header_id|>", "")
            .Replace("<|eot_id|>", "")
            .Replace("<|im_start|>", "")
            .Replace("<|im_end|>", "");
        
        // Remove control characters and normalize whitespace
        var chars = new System.Text.StringBuilder(sanitized.Length);
        foreach (var c in sanitized)
        {
            if (char.IsControl(c) || c == '\r' || c == '\n' || c == '\t')
            {
                chars.Append(' ');
            }
            else if (char.IsHighSurrogate(c) || char.IsLowSurrogate(c))
            {
                // Skip unpaired surrogates that might render as garbage
                continue;
            }
            else if (c >= 0x4E00 && c <= 0x9FFF)
            {
                // Skip CJK characters that are likely tokenizer artifacts (like 醴)
                continue;
            }
            else
            {
                chars.Append(c);
            }
        }
        
        // Collapse multiple spaces
        var result = System.Text.RegularExpressions.Regex.Replace(chars.ToString().Trim(), @"\s+", " ");
        return string.IsNullOrWhiteSpace(result) ? "(special tokens only)" : result;
    }

    private static void PrintSummary(BenchmarkSummary summary)
    {
        var stoppedNaturally = summary.Results.Count(r => r.StoppedNaturally);
        var hitMaxTokens = summary.Results.Count(r => r.HitMaxTokens);
        
        Console.WriteLine("╔══════════════════════════════════════════════════════════════╗");
        Console.WriteLine("║                         SUMMARY                              ║");
        Console.WriteLine("╠══════════════════════════════════════════════════════════════╣");
        Console.WriteLine($"║  Average TTFT:     {summary.AverageTimeToFirstTokenMs,8:F1} ms                           ║");
        Console.WriteLine($"║  Average TPS:      {summary.AverageTokensPerSecond,8:F1} tokens/sec                     ║");
        Console.WriteLine($"║  Total Prompts:    {summary.TotalPrompts,8}                                 ║");
        Console.WriteLine($"║  Stopped (EOS):    {stoppedNaturally,8}                                 ║");
        Console.WriteLine($"║  Hit Max Tokens:   {hitMaxTokens,8}                                 ║");
        Console.WriteLine($"║  Total Duration:   {summary.TotalDurationMs / 1000.0,8:F2} sec                            ║");
        Console.WriteLine("╚══════════════════════════════════════════════════════════════╝");
        
        if (hitMaxTokens > stoppedNaturally)
        {
            Console.WriteLine();
            Console.WriteLine("⚠️  WARNING: Most responses hit max token limit without natural stop.");
            Console.WriteLine("   This may indicate:");
            Console.WriteLine("   1. Model is a BASE model (not instruction-tuned)");
            Console.WriteLine("   2. Stop token IDs are incorrect for this model");
            Console.WriteLine("   3. Chat template doesn't match model's training format");
            Console.WriteLine();
            Console.WriteLine("   Verify your model is 'Meta-Llama-3.1-8B-Instruct' (not base)");
        }
    }

    public static void ExportToJson(BenchmarkSummary summary, string filePath)
    {
        var options = new JsonSerializerOptions 
        { 
            WriteIndented = true,
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase
        };
        var json = JsonSerializer.Serialize(summary, options);
        File.WriteAllText(filePath, json);
        Console.WriteLine($"\nResults exported to: {filePath}");
    }

    public static async Task RunComparisonBenchmarkAsync(
        LlamaSession llm,
        TokenizerService tokenizer,
        string modelPath,
        string? exportPath = null,
        int? maxTokens = null,
        CancellationToken cancellationToken = default)
    {
        var runner = new PerformanceTestRunner(llm, tokenizer, modelPath);
        
        var configs = new Dictionary<string, InferenceConfig>
        {
            ["Greedy"] = InferenceConfig.Greedy,
            ["Default"] = InferenceConfig.Default,
            ["Precise"] = InferenceConfig.Precise
        };

        var allSummaries = new List<BenchmarkSummary>();

        foreach (var (name, config) in configs)
        {
            if (cancellationToken.IsCancellationRequested)
                break;

            // Merge with model-specific optimal config
            var mergedConfig = LlamaOptimizations.GetOptimalConfigForModel(llm.ModelType, config);
            if (maxTokens.HasValue)
            {
                mergedConfig = mergedConfig with { MaxTokens = maxTokens.Value };
            }
            var summary = await runner.RunBenchmarksAsync(mergedConfig, name, cancellationToken);
            allSummaries.Add(summary);

            Console.WriteLine("\nPress any key to continue to next config (or Ctrl+C to stop)...\n");
            if (Console.KeyAvailable)
                Console.ReadKey(true);
        }

        if (!string.IsNullOrEmpty(exportPath))
        {
            var combinedPath = Path.Combine(
                Path.GetDirectoryName(exportPath) ?? ".", 
                $"benchmark_comparison_{DateTime.Now:yyyyMMdd_HHmmss}.json");
            
            var options = new JsonSerializerOptions 
            { 
                WriteIndented = true,
                PropertyNamingPolicy = JsonNamingPolicy.CamelCase
            };
            var json = JsonSerializer.Serialize(allSummaries, options);
            File.WriteAllText(combinedPath, json);
            Console.WriteLine($"\nComparison results exported to: {combinedPath}");
        }
    }
}

