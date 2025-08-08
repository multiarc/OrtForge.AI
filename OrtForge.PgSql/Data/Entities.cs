namespace OrtForge.AI.PgSql.Data;

/// <summary>
/// Entity representing a document with its embedding in the database
/// </summary>
public class DocumentEmbedding
{
    public int Id { get; set; }
    public string FilePath { get; set; } = string.Empty;
    public string FileName { get; set; } = string.Empty;
    public string Content { get; set; } = string.Empty;
    public float[] Embedding { get; set; } = []; // Using float array instead of Vector
    public DateTime CreatedAt { get; set; }
    public DateTime UpdatedAt { get; set; }
    public string FileHash { get; set; } = string.Empty;
    public long FileSize { get; set; }
    public string? FileExtension { get; set; } = string.Empty;
    public string? Tags { get; set; } // JSON string for document tags/metadata
}

/// <summary>
/// Search result with similarity score for direct PostgreSQL queries
/// </summary>
public class DocumentSearchResult
{
    public DocumentEmbedding Document { get; set; } = new DocumentEmbedding();
    public double Similarity { get; set; }
}
