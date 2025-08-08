using System.Numerics.Tensors;
using OrtForge.AI.Models.Models;

namespace OrtForge.AI.UnitTests;

public class EmbeddingGenerationTests
{
    [Fact]
    public async Task TestEmbeddingGeneration() {
        var model = new BgeM3Model("../../../../bge_m3_onnx_gpu/sentencepiece.bpe.model", "../../../../bge_m3_onnx_gpu/model.onnx");
        var generalSearch = "physics";
        var directSearchWithMissingContext = "Data Science and Analytics definition with explanation";
        var contextOnlySearch =
            "Field that combines several domains and expertise to extract insights from information.";
        var text = await File.ReadAllTextAsync("test_docs/data_science.txt");
        var embedding = await model.CreateEmbeddingAsync(text);
        var generalSearchEmbedding = await model.CreateEmbeddingAsync(generalSearch);
        var directSearchEmbedding = await model.CreateEmbeddingAsync(directSearchWithMissingContext);
        var contextSearchEmbedding = await model.CreateEmbeddingAsync(contextOnlySearch);
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
            model.Dispose();
            tcs.SetResult();
        }).Start();
        await tcs.Task;
    }
}