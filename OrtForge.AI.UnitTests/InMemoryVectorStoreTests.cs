using OrtForge.AI.Agent.Rag;

namespace OrtForge.AI.UnitTests;

public class InMemoryVectorStoreTests
{
    [Fact]
    public void Upsert_AddsAndReplacesById()
    {
        var vs = new InMemoryVectorStore();
        vs.Upsert(new InMemoryVectorStore.Item("a", new float[] {1, 0}, "Doc A", null));
        vs.Upsert(new InMemoryVectorStore.Item("b", new float[] {0, 1}, "Doc B", null));
        var top = vs.TopK(new float[] {1, 0}, 2);
        Assert.Collection(top,
            item => Assert.Equal("a", item.Id),
            item => Assert.Equal("b", item.Id));
        vs.Upsert(new InMemoryVectorStore.Item("a", new float[] {0, 1}, "Doc A2", new Dictionary<string,string>{{"v","2"}}));
        top = vs.TopK(new float[] {1, 0}, 2);
        Assert.Equal(2, top.Count);
        var ids = top.Select(t => t.Id).ToHashSet();
        Assert.Contains("a", ids);
        Assert.Contains("b", ids);
        var a = top.First(t => t.Id == "a");
        Assert.Equal("Doc A2", a.Text);
        Assert.Equal("2", a.Metadata!["v"]);
    }

    [Fact]
    public void TopK_ReturnsOrderedByCosineSimilarity()
    {
        var vs = new InMemoryVectorStore();
        vs.Upsert(new InMemoryVectorStore.Item("x", new float[] {1, 0}, "X", null));
        vs.Upsert(new InMemoryVectorStore.Item("y", new float[] {0.7f, 0.7f}, "Y", null));
        vs.Upsert(new InMemoryVectorStore.Item("z", new float[] {0, 1}, "Z", null));
        var query = new float[] {0.9f, 0.1f};
        var top2 = vs.TopK(query, 2);
        Assert.Equal("x", top2[0].Id);
        Assert.Equal("y", top2[1].Id);
        var top3 = vs.TopK(query, 3);
        Assert.Equal(new[]{"x","y","z"}, new[]{top3[0].Id, top3[1].Id, top3[2].Id});
    }
}
