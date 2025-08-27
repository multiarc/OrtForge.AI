using System.Text;
using Microsoft.ML.OnnxRuntime.Tensors;
using OrtForge.AI.Agent.Agents;
using OrtForge.AI.Agent.Generation;
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
        
        using var session = new ConversationSession(tok);
        
        System.Console.WriteLine("🤖 OrtForge.AI Chat - Llama 3.2 Agent with Session Management");
        System.Console.WriteLine("💬 Enter your message (empty line to quit):");
        System.Console.WriteLine();
        
        bool isFirstMessage = true;
        
        while (true)
        {
            System.Console.Write("🧑 > ");
            var user = System.Console.ReadLine();
            if (string.IsNullOrWhiteSpace(user)) 
            {
                System.Console.WriteLine("👋 Goodbye!");
                break;
            }
            
            System.Console.WriteLine();
            System.Console.Write("🤖 Assistant: ");
            
            try
            {
                if (isFirstMessage)
                {
                    var retrieved = new List<string>();
                    
                    await session.InitializeSystemPromptAsync(llama, retrieved, enableTools: false);
                    isFirstMessage = false;
                }
                
                await session.AddMessageAsync("user", user!, llama);
                
                var assistantStartTokens = tok.EncodeToIds("<|start_header_id|>assistant<|end_header_id|>\n\n");
                
                if (assistantStartTokens?.Length > 0)
                {
                    var answer = await GenerateResponseAsync(llama, tok, assistantStartTokens, session.GetCurrentKvState());
                    
                    if (!string.IsNullOrEmpty(answer))
                    {
                        await session.AddMessageAsync("assistant", answer, null);
                    }
                }
            }
            catch (Exception ex)
            {
                System.Console.WriteLine();
                System.Console.WriteLine($"❌ Error: {ex.Message}");
                System.Console.WriteLine($"❌ Stack trace: {ex.StackTrace}");
            }
            
            System.Console.WriteLine();
        }
        
        // Dispose models
        embeddingModel.Dispose();
        rerankerModel?.Dispose();
    }

    private static async Task<string> GenerateResponseAsync(LlamaSession llama, TokenizerService tokenizer, int[] startTokens, KvState kvState)
    {
        if (startTokens == null || startTokens.Length == 0)
            return string.Empty;
            
        if (kvState == null)
            throw new ArgumentNullException(nameof(kvState));
            
        var config = LlamaOptimizations.GetOptimalConfigForModel(llama.ModelName);
        var response = new StringBuilder();
        var generatedTokens = new List<int>();

        var idsArray = startTokens.Select(id => (long)id).ToArray();
        
        for (int step = 0; step < config.MaxTokens; step++)
        {
            var currentInput = step == 0 ? idsArray : new long[] { generatedTokens[^1] };
            
            using var currentInputIds = Microsoft.ML.OnnxRuntime.OrtValue.CreateTensorValueFromMemory<long>(currentInput, new long[] { 1, currentInput.Length });
            var stepInputs = new LlamaSession.StepInputs(currentInputIds, kvState, null, null);
            
            var outputs = await llama.RunStepAsync(stepInputs);
            
            var logitsShape = outputs.Logits.GetTensorTypeAndShape().Shape;
            var vocab = (int)logitsShape[^1];
            
            var nextId = GetNextToken(outputs, vocab, config, generatedTokens);
            generatedTokens.Add(nextId);

            var tokenText = tokenizer.DecodeFromIds(new[] { nextId });
            System.Console.Write(tokenText);
            response.Append(tokenText);

            bool shouldStop = config.StopTokenIds.Contains(nextId) || 
                             config.StopSequences.Any(seq => response.ToString().Contains(seq));

            kvState = outputs.KvCache;
            outputs.Dispose();
            
            if (shouldStop) break;
        }

        System.Console.WriteLine();
        return response.ToString();
    }

    private static int GetNextToken(LlamaSession.StepOutputs outputs, int vocab, InferenceConfig config, List<int> previousTokens)
    {
        var span = outputs.GetLogitsSpan();
        var logitsShape = outputs.Logits.GetTensorTypeAndShape().Shape;
        
        Span<float> logitsForSampling;
        if (logitsShape.Length == 3)
        {
            var seqLen = (int)logitsShape[1];
            var vocabSize = (int)logitsShape[2];
            var lastTokenStart = (seqLen - 1) * vocabSize;
            logitsForSampling = span.Slice(lastTokenStart, vocabSize);
        }
        else if (logitsShape.Length == 2)
        {
            var vocabSize = (int)logitsShape[1];
            logitsForSampling = span.Slice(0, vocabSize);
        }
        else
        {
            logitsForSampling = span;
        }
        
        var previousTokensSpan = previousTokens.Count > 0 ? previousTokens.ToArray().AsSpan() : ReadOnlySpan<int>.Empty;
        return Sampling.Sample(logitsForSampling, config, previousTokensSpan);
    }
}


