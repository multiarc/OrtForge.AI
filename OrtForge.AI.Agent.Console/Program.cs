using OrtAgent.Core.Agents;
using OrtAgent.Core.LLM;
using OrtAgent.Core.Rag;
using OrtAgent.Core.Runtime;
using OrtAgent.Core.Tokenization;

namespace OrtAgent.ConsoleApp;

internal static class Program
{
    private static void Main(string[] args)
    {
        if (args.Length < 2)
        {
            System.Console.WriteLine("Usage: OrtAgent.Console <llm.onnx> <tokenizer.json|sentencepiece.bpe.model> [embedding.onnx]");
            return;
        }

        var llmPath = args[0];
        var tokenizerPath = args[1];
        var embPath = args.Length > 2 ? args[2] : args[0]; // allow same model for quick test

        using var llmSession = OrtRuntimeFactory.CreateSession(llmPath);
        using var embSession = OrtRuntimeFactory.CreateSession(embPath);
        using var llama = new LlamaSession(llmSession);
        using var embed = new EmbeddingService(embSession);
        var tok = tokenizerPath.EndsWith(".json", StringComparison.OrdinalIgnoreCase)
            ? TokenizerService.FromJson(tokenizerPath)
            : TokenizerService.FromPretrained(tokenizerPath);
        var vec = new InMemoryVectorStore();
        var agent = new AgentOrchestrator(llama, tok, embed, vec);

        System.Console.WriteLine("Enter your message (empty line to quit):");
        while (true)
        {
            System.Console.Write("> ");
            var user = System.Console.ReadLine();
            if (string.IsNullOrWhiteSpace(user)) break;
            var answer = agent.ChatTurn(user!, Array.Empty<(string role, string content)>());
            System.Console.WriteLine();
            System.Console.WriteLine($"Assistant: {answer}");
        }
    }
}


