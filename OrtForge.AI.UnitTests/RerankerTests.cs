using OrtForge.AI.Models.Models;

namespace OrtForge.AI.UnitTests;

public class RerankerTests : IAsyncLifetime
{
    private readonly BgeRerankerM3 _model;
    
    public RerankerTests() {
        _model = new BgeRerankerM3("../../../../reranker_m3_onnx_gpu/sentencepiece.bpe.model", "../../../../reranker_m3_onnx_gpu/model.onnx");
    }
    
    [Fact]
    public async Task TestRerankerComparisons() {
        var generalSearch = "physics";
        var directSearchWithMissingContext = "Data Science and Analytics definition with explanation";
        var contextOnlySearch =
            "Field that combines several domains and expertise to extract insights from information.";
        var directQuestion = "What is Data Science and Analytics?";
        var text = await File.ReadAllTextAsync("test_docs/data_science.txt");
        var relevance1 = await _model.GetRerankingScoreAsync(generalSearch, text);
        var relevance2 = await _model.GetRerankingScoreAsync(directSearchWithMissingContext, text);
        var relevance3 = await _model.GetRerankingScoreAsync(contextOnlySearch, text);
        var relevance4 = await _model.GetRerankingScoreAsync(directQuestion, text);
        Assert.True(relevance1 < 0.001);
        Assert.True(relevance2 > relevance1);
        Assert.True(relevance2 > 0.1);
        Assert.True(relevance3 > relevance2);
        Assert.True(relevance3 > 0.5);
        //Assert.True(relevance4 > 0.5);
    }
    
    [Fact]
    public async Task TestRerankerOrdering() {
        var contextSearch =
            "Field that combines several domains and expertise to extract insights from information.";
        var text1 = await File.ReadAllTextAsync("test_docs/data_science.txt");
        var text2 = await File.ReadAllTextAsync("test_docs/ml_overview.txt");
        var text3 = await File.ReadAllTextAsync("test_docs/software_dev.txt");
        var relevance1 = await _model.GetRerankingScoreAsync(contextSearch, text1);
        var relevance2 = await _model.GetRerankingScoreAsync(contextSearch, text2);
        var relevance3 = await _model.GetRerankingScoreAsync(contextSearch, text3);
        Assert.True(relevance1 > 0.5); //> 50%
        Assert.True(relevance2 < 0.001); //< 1%
        Assert.True(relevance3 > 0.002); //> 2%
    }

    public Task InitializeAsync() {
        return Task.CompletedTask;
    }
    
    public async Task DisposeAsync() {
        var tcs = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        
        // Dispose the model in a separate thread to work around deadlocks
        new Thread(_ => {
            _model.Dispose();
            tcs.SetResult();
        }).Start();
        await tcs.Task;
    }
}