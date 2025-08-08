namespace OrtForge.AI.PgSql.Database;

/// <summary>
/// Represents a document with its embedding stored in PostgreSQL
/// Updated to use float[] for consistency with the main entity model
/// </summary>
public class DocumentEmbedding
{
    public int Id { get; set; }
    public string FilePath { get; set; } = string.Empty;
    public string FileName { get; set; } = string.Empty;
    public string Content { get; set; } = string.Empty;
    public float[] Embedding { get; set; } = [];
    public DateTime CreatedAt { get; set; }
    public DateTime UpdatedAt { get; set; }
    public string FileHash { get; set; } = string.Empty; // To detect file changes
    public long FileSize { get; set; }
    public string? FileExtension { get; set; }
    public string? Tags { get; set; }
}

/// <summary>
/// Search result with similarity score
/// </summary>
public class SearchResult
{
    public DocumentEmbedding Document { get; set; } = new DocumentEmbedding();
    public double Similarity { get; set; }
}
