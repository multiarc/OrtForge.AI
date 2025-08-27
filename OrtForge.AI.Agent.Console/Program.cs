using Microsoft.ML.OnnxRuntime.Tensors;
using OrtForge.AI.Agent.Agents;
using OrtForge.AI.Agent.LLM;
using OrtForge.AI.Agent.Rag;
using OrtForge.AI.Agent.Runtime;
using OrtForge.AI.Agent.Tokenization;
using OrtForge.AI.Models.Astractions;
using OrtForge.AI.Models.Models;
using OrtForge.AI.Models.Options;

namespace OrtForge.AI.Agent.Console;

internal static class Program
{
    private static async Task Main(string[] args)
    {
        if (args.Length < 4)
        {
            System.Console.WriteLine("Usage: OrtAgent.Console <llm.onnx> <tokenizer.json|sentencepiece.bpe.model> <embedding.onnx> <embedding_tokenizer.model> [reranker.onnx] [reranker_tokenizer.model]");
            return;
        }

        var llmPath = args[0].Trim();
        var tokenizerPath = args[1].Trim();
        var embPath = args[2].Trim();
        var embTokenizerPath = args[3].Trim();
        var rerankerPath = args.Length > 4 ? args[4].Trim() : null;
        var rerankerTokenizerPath = args.Length > 5 ? args[5].Trim() : null;
        
        System.Console.WriteLine($"LLM: {llmPath}");
        System.Console.WriteLine($"Tokenizer: {tokenizerPath}");
        System.Console.WriteLine($"Embedding: {embPath}");
        System.Console.WriteLine($"Embedding Tokenizer: {embTokenizerPath}");
        System.Console.WriteLine($"Reranker: {rerankerPath}");
        System.Console.WriteLine($"Reranker Tokenizer: {rerankerTokenizerPath}");

        using var llmSession = OrtRuntimeFactory.CreateSession(llmPath);
        using var llama = new LlamaSession(llmSession);
        
        // Initialize embedding model with BgeM3Model
        var embeddingOptions = new BgeM3Options
        {
            ModelPath = embPath,
            TokenizerModelPath = embTokenizerPath,
            TensorElementType = TensorElementType.Float16
        };
        using var embeddingModel = new BgeM3Model(embeddingOptions);
        embeddingModel.Initialize(providers: ExecutionProvider.CPU | ExecutionProvider.ROCm);
        
        // Initialize reranker if provided
        BgeRerankerM3? rerankerModel = null;
        if (!string.IsNullOrEmpty(rerankerPath) && !string.IsNullOrEmpty(rerankerTokenizerPath))
        {
            var rerankerOptions = new BgeM3Options
            {
                ModelPath = rerankerPath,
                TokenizerModelPath = rerankerTokenizerPath,
                TensorElementType = TensorElementType.Float16
            };
            rerankerModel = new BgeRerankerM3(rerankerOptions);
            rerankerModel.Initialize(providers: ExecutionProvider.CPU | ExecutionProvider.ROCm);
        }

        var tok = TokenizerService.FromHuggingFace(tokenizerPath);
        var vec = new InMemoryVectorStore();
        var agent = new AgentOrchestrator(llama, tok, embeddingModel, vec, rerankerModel);

        System.Console.WriteLine("Enter your message (empty line to quit):");
        while (true)
        {
            System.Console.Write("> ");
            var user = System.Console.ReadLine();
            if (string.IsNullOrWhiteSpace(user)) break;
            var answer = await agent.ChatTurnAsync(user!, Array.Empty<(string role, string content)>());
            System.Console.WriteLine();
            System.Console.WriteLine($"Assistant: {answer}");
        }
        
        // Dispose models
        embeddingModel.Dispose();
        rerankerModel?.Dispose();
    }
}


