using Microsoft.ML.OnnxRuntime;
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
        bool isInitialized = false;
        
        System.Console.WriteLine("🤖 OrtForge.AI Chat - Llama 3.2 Agent with KV Cache Session Management");
        System.Console.WriteLine("💬 Enter your message (empty line to quit):");
        System.Console.WriteLine();
        
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
                if (!isInitialized)
                {
                    var retrieved = new List<string>();
                    await session.InitializeSystemPromptAsync(llama, retrieved, enableTools: false);
                    isInitialized = true;
                }
                
                await session.AddMessageAsync("user", user!, llama);
                
                var assistantStartTokens = tok.EncodeToIds("<|start_header_id|>assistant<|end_header_id|>\n\n");
                var currentKvState = session.GetCurrentKvState();
                
                var (answer, finalKvState) = await GenerateResponseWithSession(llama, tok, assistantStartTokens, currentKvState);
                
                if (!string.IsNullOrEmpty(answer))
                {
                    session.UpdateKvState(finalKvState);
                    await session.AddMessageAsync("assistant", answer, null);
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

    private static async Task<(string response, KvState finalKvState)> GenerateResponseWithSession(LlamaSession llama, TokenizerService tokenizer, int[] startTokens, KvState kvState)
    {
        if (startTokens == null || startTokens.Length == 0)
            return (string.Empty, kvState);
            
        var config = LlamaOptimizations.GetOptimalConfigForModel(llama.ModelName);
        var response = new System.Text.StringBuilder();
        var generatedTokens = new List<int>();
        var currentKvState = kvState;

        var idsArray = startTokens.Select(id => (long)id).ToArray();
        
        for (int step = 0; step < config.MaxTokens; step++)
        {
            var currentInput = step == 0 ? idsArray : new long[] { generatedTokens[^1] };
            
            var currentInputIds = OrtValue.CreateTensorValueFromMemory(currentInput, [1L, currentInput.Length ]);
            LlamaSession.StepOutputs outputs;
            using (var stepInputs = new LlamaSession.StepInputs(currentInputIds, currentKvState, null, null))
            {
                outputs = await llama.RunStepAsync(stepInputs);
            }

            var logitsShape = outputs.Logits.GetTensorTypeAndShape().Shape;
            var vocab = (int)logitsShape[^1];
            
            var nextId = GetNextToken(outputs, vocab, config, generatedTokens);
            generatedTokens.Add(nextId);

            var tokenText = tokenizer.DecodeFromIds(new[] { nextId });
            System.Console.Write(tokenText);
            response.Append(tokenText);

            bool shouldStop = config.StopTokenIds.Contains(nextId) || 
                             config.StopSequences.Any(seq => response.ToString().Contains(seq));

            currentKvState = outputs.KvCache;
            outputs.Dispose();
            
            if (shouldStop) break;
        }

        System.Console.WriteLine();
        return (response.ToString(), currentKvState);
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


