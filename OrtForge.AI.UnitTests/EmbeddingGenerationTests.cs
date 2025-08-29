using System.Numerics.Tensors;
using Microsoft.ML.OnnxRuntime.Tensors;
using OrtForge.AI.Models.Astractions;
using OrtForge.AI.Models.Models;
using OrtForge.AI.Models.Options;
using Xunit.Abstractions;

namespace OrtForge.AI.UnitTests;

public class EmbeddingGenerationTests
{
    private readonly BgeM3Model _model;
    public EmbeddingGenerationTests(ITestOutputHelper outputHelper) {
        var home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
        _model = new BgeM3Model(new BgeM3Options
        {
            TokenizerModelPath = Path.Combine(home, "LLM/bge_m3_onnx_gpu/sentencepiece.bpe.model"), 
            ModelPath = Path.Combine(home, "LLM/bge_m3_onnx_gpu/model.onnx"),
            TensorElementType = TensorElementType.Float16
        });
#if WINDOWS
        outputHelper.WriteLine("Running on DirectML.");
        _model.Initialize(providers: ExecutionProvider.DirectML | ExecutionProvider.CPU);
#elif ROCM
        outputHelper.WriteLine("Running on ROCm.");
        _model.Initialize(providers: ExecutionProvider.ROCm | ExecutionProvider.CPU);
#elif CUDA
        outputHelper.WriteLine("Running on CUDA.");
        _model.Initialize(providers: ExecutionProvider.CUDA | ExecutionProvider.CPU);
#else
        outputHelper.WriteLine("Running on CPU.");
        _model.Initialize();
#endif
    }
    
    [Fact]
    public async Task TestEmbeddingGeneration() {
        var generalSearch = "physics";
        var directSearchWithMissingContext = "Data Science and Analytics definition with explanation";
        var contextOnlySearch =
            "Field that combines several domains and expertise to extract insights from information.";
        var text = await File.ReadAllTextAsync("test_docs/data_science.txt");
        var embedding = await _model.CreateEmbeddingAsync(text);
        var generalSearchEmbedding = await _model.CreateEmbeddingAsync(generalSearch);
        var directSearchEmbedding = await _model.CreateEmbeddingAsync(directSearchWithMissingContext);
        var contextSearchEmbedding = await _model.CreateEmbeddingAsync(contextOnlySearch);
        Assert.Equal(1024, embedding.Length);
        Assert.Equal(1024, generalSearchEmbedding.Length);
        Assert.Equal(1024, directSearchEmbedding.Length);
        Assert.Equal(1024, contextSearchEmbedding.Length);
        var similarity1 = TensorPrimitives.CosineSimilarity(embedding, generalSearchEmbedding);
        var similarity2 = TensorPrimitives.CosineSimilarity(embedding, directSearchEmbedding);
        var similarity3 = TensorPrimitives.CosineSimilarity(embedding, contextSearchEmbedding);
        Assert.True(similarity1 > 0.5);
        Assert.True(similarity2 > similarity1);
        Assert.True(similarity2 > 0.6);
        Assert.True(similarity3 > similarity2);
        Assert.True(similarity3 > 0.7);
        
        var tcs = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        
        // Dispose the model in a separate thread to work around deadlocks
        new Thread(_ => {
            _model.Dispose();
            tcs.SetResult();
        }).Start();
        await tcs.Task;
    }
}