namespace OrtForge.AI.Agent.Rag;

public sealed class InMemoryVectorStore
{
    public sealed record Item(string Id, float[] Vector, string Text, IReadOnlyDictionary<string, string>? Metadata);

    private readonly List<Item> _items = new();

    public void Upsert(Item item)
    {
        var idx = _items.FindIndex(x => x.Id == item.Id);
        if (idx >= 0) _items[idx] = item; else _items.Add(item);
    }

    public IReadOnlyList<Item> TopK(float[] query, int k = 5)
    {
        var qn = Normalize(query);
        return _items
            .Select(x => (item: x, score: Cosine(qn, Normalize(x.Vector))))
            .OrderByDescending(x => x.score)
            .Take(k)
            .Select(x => x.item)
            .ToList();
    }

    private static float[] Normalize(float[] v)
    {
        double s = 0; for (int i = 0; i < v.Length; i++) s += (double)v[i] * v[i];
        var n = Math.Sqrt(Math.Max(s, 1e-9));
        var o = new float[v.Length];
        for (int i = 0; i < v.Length; i++) o[i] = (float)(v[i] / n);
        return o;
    }

    private static double Cosine(float[] a, float[] b)
    {
        double s = 0; for (int i = 0; i < a.Length; i++) s += (double)a[i] * b[i];
        return s;
    }
}


