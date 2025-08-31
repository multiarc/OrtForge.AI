using OrtForge.AI.Agent.Agents;
using OrtForge.AI.Agent.LLM;
using OrtForge.AI.Agent.Runtime;
using OrtForge.AI.Agent.Tokenization;

namespace OrtForge.AI.Agent.TestApp;

internal static class Program
{
    private static async Task Main(string[] args)
    {
        if (args.Length < 4)
        {
            Console.WriteLine("Usage: OrtAgent.Console <llm.onnx> <tokenizer.json|sentencepiece.bpe.model> <embedding.onnx> <embedding_tokenizer.model> [reranker.onnx] [reranker_tokenizer.model]");
            return;
        }

        var llmPath = args[0].Trim();
        var tokenizerPath = args[1].Trim();
        // var embPath = args[2].Trim();
        // var embTokenizerPath = args[3].Trim();
        // var rerankerPath = args.Length > 4 ? args[4].Trim() : null;
        // var rerankerTokenizerPath = args.Length > 5 ? args[5].Trim() : null;
        
        Console.WriteLine($"LLM: {llmPath}");
        Console.WriteLine($"Tokenizer: {tokenizerPath}");
        // System.Console.WriteLine($"Embedding: {embPath}");
        // System.Console.WriteLine($"Embedding Tokenizer: {embTokenizerPath}");
        // System.Console.WriteLine($"Reranker: {rerankerPath}");
        // System.Console.WriteLine($"Reranker Tokenizer: {rerankerTokenizerPath}");

        using var llmSession = OrtRuntimeFactory.CreateSession(llmPath);
        // Auto-detect model type from path, or specify explicitly
        var modelType = ModelTypeExtensions.ParseFromString(llmPath);
        Console.WriteLine($"Detected model type: {modelType}");
        using var llama = new LlamaSession(llmSession, modelType);
        
        // // Initialize embedding model with BgeM3Model
        // var embeddingOptions = new BgeM3Options
        // {
        //     ModelPath = embPath,
        //     TokenizerModelPath = embTokenizerPath,
        //     TensorElementType = TensorElementType.Float16
        // };
        // using var embeddingModel = new BgeM3Model(embeddingOptions);
        // embeddingModel.Initialize(providers: ExecutionProvider.CPU | ExecutionProvider.ROCm);
        //
        // // Initialize reranker if provided
        // BgeRerankerM3? rerankerModel = null;
        // if (!string.IsNullOrEmpty(rerankerPath) && !string.IsNullOrEmpty(rerankerTokenizerPath))
        // {
        //     var rerankerOptions = new BgeM3Options
        //     {
        //         ModelPath = rerankerPath,
        //         TokenizerModelPath = rerankerTokenizerPath,
        //         TensorElementType = TensorElementType.Float16
        //     };
        //     rerankerModel = new BgeRerankerM3(rerankerOptions);
        //     rerankerModel.Initialize(providers: ExecutionProvider.CPU | ExecutionProvider.ROCm);
        // }

        var tok = TokenizerService.FromHuggingFace(tokenizerPath);
        //var vec = new InMemoryVectorStore();
        var agent = new AgentOrchestrator(/*, embeddingModel, vec, rerankerModel*/);
        
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

        // Dispose models
        //embeddingModel.Dispose();
        //rerankerModel?.Dispose();
    }
}


