using OrtForge.AI.Agent.Agents;
using OrtForge.AI.Agent.Generation;
using OrtForge.AI.Agent.LLM;
using OrtForge.AI.Agent.Runtime;
using OrtForge.AI.Agent.Tokenization;

namespace OrtForge.AI.Agent.TestApp;

internal static class Program
{
    private static async Task Main(string[] args)
    {
        // Check for --help flag
        if (args.Length == 0 || args.Contains("--help") || args.Contains("-h"))
        {
            PrintUsage();
            return;
        }

        // Parse arguments
        var benchmarkMode = args.Contains("--benchmark");
        var compareMode = args.Contains("--compare");
        var jsonExport = args.FirstOrDefault(a => a.StartsWith("--export="))?.Replace("--export=", "");
        var configArg = args.FirstOrDefault(a => a.StartsWith("--config="))?.Replace("--config=", "");
        var maxTokensArg = args.FirstOrDefault(a => a.StartsWith("--max-tokens="))?.Replace("--max-tokens=", "");
        int? maxTokens = int.TryParse(maxTokensArg, out var mt) ? mt : null; // null = use config default (no override)
        var debugPrompts = args.Contains("--debug-prompts");
        
        // Filter out flags to get positional arguments
        var positionalArgs = args.Where(a => !a.StartsWith("--") && !a.StartsWith("-")).ToArray();
        
        if (positionalArgs.Length < 2)
        {
            Console.WriteLine("Error: Missing required arguments.");
            PrintUsage();
            return;
        }

        var llmPath = positionalArgs[0].Trim();
        var tokenizerPath = positionalArgs[1].Trim();
        
        Console.WriteLine($"LLM: {llmPath}");
        Console.WriteLine($"Tokenizer: {tokenizerPath}");

        using var llmSession = OrtRuntimeFactory.CreateSession(llmPath);
        var modelType = ModelTypeExtensions.ParseFromString(llmPath);
        Console.WriteLine($"Detected model type: {modelType}");
        
        // Debug: Print model inputs/outputs
        if (args.Contains("--debug-model"))
        {
            Console.WriteLine("\n=== MODEL METADATA ===");
            Console.WriteLine("Inputs:");
            foreach (var input in llmSession.InputMetadata)
            {
                var dims = string.Join(", ", input.Value.Dimensions);
                Console.WriteLine($"  {input.Key}: [{dims}] ({input.Value.ElementDataType})");
            }
            Console.WriteLine("\nOutputs:");
            foreach (var output in llmSession.OutputMetadata)
            {
                var dims = string.Join(", ", output.Value.Dimensions);
                Console.WriteLine($"  {output.Key}: [{dims}] ({output.Value.ElementDataType})");
            }
            Console.WriteLine("=== END MODEL METADATA ===\n");
        }
        
        using var llama = new LlamaSession(llmSession, modelType);
        var tok = TokenizerService.FromHuggingFace(tokenizerPath);

        // Run benchmark mode
        if (benchmarkMode || compareMode)
        {
            await RunBenchmarkMode(llama, tok, llmPath, compareMode, jsonExport, configArg, maxTokens, debugPrompts);
            return;
        }

        // Interactive chat mode
        await RunInteractiveMode(llama, tok);
    }

    private static void PrintUsage()
    {
        Console.WriteLine(@"
OrtForge.AI TestApp - LLM Inference Testing

Usage:
  OrtForge.AI.Agent.TestApp <llm.onnx> <tokenizer.json> [options]

Arguments:
  llm.onnx              Path to the ONNX model file
  tokenizer.json        Path to the tokenizer file (HuggingFace JSON or SentencePiece BPE)

Options:
  --benchmark           Run performance benchmark with predefined prompts
  --compare             Run benchmarks with multiple inference configs (Greedy, Default, Precise)
  --config=<name>       Specify config for benchmark: Greedy, Default, Precise, Creative
  --max-tokens=<n>      Maximum tokens to generate per response (default: 128 for benchmarks)
  --debug-prompts       Show the chat template format being used
  --export=<path>       Export benchmark results to JSON file
  --help, -h            Show this help message

Examples:
  # Interactive chat mode
  OrtForge.AI.Agent.TestApp model.onnx tokenizer.json

  # Run benchmark with default config
  OrtForge.AI.Agent.TestApp model.onnx tokenizer.json --benchmark

  # Run benchmark with limited tokens for quick testing
  OrtForge.AI.Agent.TestApp model.onnx tokenizer.json --benchmark --max-tokens=64

  # Run benchmark with Greedy config and export results
  OrtForge.AI.Agent.TestApp model.onnx tokenizer.json --benchmark --config=Greedy --export=results.json

  # Compare all configs
  OrtForge.AI.Agent.TestApp model.onnx tokenizer.json --compare
");
    }

    private static async Task RunBenchmarkMode(
        LlamaSession llama,
        TokenizerService tok,
        string llmPath,
        bool compareMode,
        string? jsonExport,
        string? configArg,
        int? maxTokens,
        bool debugPrompts)
    {
        if (compareMode)
        {
            await PerformanceTestRunner.RunComparisonBenchmarkAsync(llama, tok, llmPath, jsonExport, maxTokens);
            return;
        }

        // Single config benchmark
        var config = GetConfigByName(configArg ?? "Default");
        var configName = configArg ?? "Default";
        
        // Merge with model-specific optimal config
        var mergedConfig = LlamaOptimizations.GetOptimalConfigForModel(llama.ModelType, config);
        if (maxTokens.HasValue)
        {
            mergedConfig = mergedConfig with { MaxTokens = maxTokens.Value };
        }
        
        if (debugPrompts)
        {
            // Show a sample prompt for debugging
            var samplePrompt = AgentOrchestrator.BuildSystemPrompt([], "What is 2+2?");
            Console.WriteLine();
            Console.WriteLine("=== DEBUG: Sample Prompt Format ===");
            Console.WriteLine(samplePrompt.Replace("\n", "\\n\n"));
            Console.WriteLine("=== END DEBUG ===");
            Console.WriteLine();
            
            // Show actual token IDs
            var tokenIds = tok.EncodeToIds(samplePrompt);
            Console.WriteLine("=== DEBUG: Token IDs (first 50) ===");
            Console.WriteLine($"Total tokens: {tokenIds.Length}");
            var first50 = tokenIds.Take(50).ToArray();
            Console.WriteLine($"IDs: [{string.Join(", ", first50)}]");
            
            // Check if special tokens are recognized
            var specialTokenTest = tok.EncodeToIds("<|begin_of_text|>");
            Console.WriteLine($"\n<|begin_of_text|> encodes to: [{string.Join(", ", specialTokenTest)}]");
            
            var eotTest = tok.EncodeToIds("<|eot_id|>");
            Console.WriteLine($"<|eot_id|> encodes to: [{string.Join(", ", eotTest)}]");
            
            var headerTest = tok.EncodeToIds("<|start_header_id|>system<|end_header_id|>");
            Console.WriteLine($"<|start_header_id|>system<|end_header_id|> encodes to: [{string.Join(", ", headerTest)}]");
            Console.WriteLine("=== END TOKEN DEBUG ===");
            Console.WriteLine();
        }
        
        var runner = new PerformanceTestRunner(llama, tok, llmPath);
        var summary = await runner.RunBenchmarksAsync(mergedConfig, configName);
        
        if (!string.IsNullOrEmpty(jsonExport))
        {
            PerformanceTestRunner.ExportToJson(summary, jsonExport);
        }
    }

    private static InferenceConfig GetConfigByName(string name)
    {
        return name.ToLowerInvariant() switch
        {
            "greedy" => InferenceConfig.Greedy,
            "precise" => InferenceConfig.Precise,
            "creative" => InferenceConfig.Creative,
            _ => InferenceConfig.Default
        };
    }

    private static async Task RunInteractiveMode(LlamaSession llama, TokenizerService tok)
    {
        var agent = new AgentOrchestrator();
        using var session = new ConversationSession(llama, tok, llama.OptimalConfig);
        
        Console.WriteLine("🤖 OrtForge.AI Chat");
        Console.WriteLine("💬 Enter your message (empty line to quit):");
        Console.WriteLine();
        
        while (true)
        {
            Console.Write("🧑 > ");
            var user = Console.ReadLine();
            if (string.IsNullOrWhiteSpace(user)) 
            {
                Console.WriteLine("👋 Goodbye!");
                break;
            }
            
            Console.WriteLine();
            Console.Write("🤖 Assistant: ");
            
            try
            {
                await foreach (var token in agent.ChatTurnAsync(session, user!))
                {
                    Console.Write(token);
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine();
                Console.WriteLine($"❌ Error: {ex.Message}");
                Console.WriteLine($"❌ Stack trace: {ex.StackTrace}");
            }
            
            Console.WriteLine();
        }
        
        Console.WriteLine("===============CHAT HISTORY================");
        Console.WriteLine(session.EntireConversation.ToString());
        Console.WriteLine("===========================================");
        Console.WriteLine("Press any key to exit...");
        Console.ReadKey();
    }
}


