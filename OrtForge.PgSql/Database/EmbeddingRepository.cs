using System.Security.Cryptography;
using Npgsql;

namespace OrtForge.AI.PgSql.Database;

/// <summary>
/// Repository for managing document embeddings in PostgreSQL with pgvector
/// Updated to use float[] for consistency
/// </summary>
public class EmbeddingRepository : IDisposable
{
    private readonly string _connectionString;
    private readonly string _tableName;

    public EmbeddingRepository(string connectionString, string tableName = "document_embeddings")
    {
        _connectionString = connectionString;
        _tableName = tableName;
    }

    /// <summary>
    /// Initialize the database and create necessary tables and extensions
    /// </summary>
    public async Task InitializeAsync()
    {
        using var connection = new NpgsqlConnection(_connectionString);
        await connection.OpenAsync();
        
        // Create pgvector extension if it doesn't exist
        using var extensionCmd = new NpgsqlCommand("CREATE EXTENSION IF NOT EXISTS vector;", connection);
        await extensionCmd.ExecuteNonQueryAsync();
        
        // Create table if it doesn't exist
        var createTableSql = $@"
            CREATE TABLE IF NOT EXISTS {_tableName} (
                id SERIAL PRIMARY KEY,
                file_path TEXT NOT NULL UNIQUE,
                file_name TEXT NOT NULL,
                content TEXT NOT NULL,
                embedding vector(1024), -- BGE-M3 produces 1024-dimensional embeddings
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                file_hash TEXT NOT NULL,
                file_size BIGINT NOT NULL,
                file_extension TEXT,
                tags JSONB
            );
        ";
        
        using var createTableCmd = new NpgsqlCommand(createTableSql, connection);
        await createTableCmd.ExecuteNonQueryAsync();
        
        // Create indexes for better performance
        await CreateIndexesAsync(connection);
    }

    /// <summary>
    /// Insert or update a document embedding
    /// </summary>
    public async Task<int> UpsertDocumentAsync(DocumentEmbedding document)
    {
        using var connection = new NpgsqlConnection(_connectionString);
        await connection.OpenAsync();
        
        var upsertSql = $@"
            INSERT INTO {_tableName} (file_path, file_name, content, embedding, file_hash, file_size, file_extension, tags, created_at, updated_at)
            VALUES (@file_path, @file_name, @content, @embedding::vector, @file_hash, @file_size, @file_extension, @tags::jsonb, @created_at, @updated_at)
            ON CONFLICT (file_path) 
            DO UPDATE SET 
                file_name = EXCLUDED.file_name,
                content = EXCLUDED.content,
                embedding = EXCLUDED.embedding,
                file_hash = EXCLUDED.file_hash,
                file_size = EXCLUDED.file_size,
                file_extension = EXCLUDED.file_extension,
                tags = EXCLUDED.tags,
                updated_at = EXCLUDED.updated_at
            RETURNING id;";

        using var cmd = new NpgsqlCommand(upsertSql, connection);
        cmd.Parameters.AddWithValue("@file_path", document.FilePath);
        cmd.Parameters.AddWithValue("@file_name", document.FileName);
        cmd.Parameters.AddWithValue("@content", document.Content);
        cmd.Parameters.AddWithValue("@embedding", FormatFloatArrayAsVector(document.Embedding));
        cmd.Parameters.AddWithValue("@file_hash", document.FileHash);
        cmd.Parameters.AddWithValue("@file_size", document.FileSize);
        cmd.Parameters.AddWithValue("@file_extension", document.FileExtension ?? (object)DBNull.Value);
        cmd.Parameters.AddWithValue("@tags", document.Tags ?? (object)DBNull.Value);
        cmd.Parameters.AddWithValue("@created_at", document.CreatedAt);
        cmd.Parameters.AddWithValue("@updated_at", document.UpdatedAt);

        var result = await cmd.ExecuteScalarAsync();
        return result != null ? Convert.ToInt32(result) : 0;
    }

    /// <summary>
    /// Search for similar documents using cosine similarity
    /// </summary>
    public async Task<List<SearchResult>> SearchSimilarAsync(float[] queryEmbedding, int limit = 10, double threshold = 0.5)
    {
        using var connection = new NpgsqlConnection(_connectionString);
        await connection.OpenAsync();

        var searchSql = $@"
            SELECT 
                id, file_path, file_name, content, embedding, created_at, updated_at, 
                file_hash, file_size, file_extension, tags,
                (1 - (embedding <=> @query_embedding)) AS similarity
            FROM {_tableName}
            WHERE (1 - (embedding <=> @query_embedding)) > @threshold
            ORDER BY embedding <=> @query_embedding
            LIMIT @limit;";

        using var cmd = new NpgsqlCommand(searchSql, connection);
        cmd.Parameters.AddWithValue("@query_embedding", FormatFloatArrayAsVector(queryEmbedding));
        cmd.Parameters.AddWithValue("@threshold", threshold);
        cmd.Parameters.AddWithValue("@limit", limit);

        var results = new List<SearchResult>();
        using var reader = await cmd.ExecuteReaderAsync();

        while (await reader.ReadAsync())
        {
            var document = new DocumentEmbedding
            {
                Id = reader.GetInt32(reader.GetOrdinal("id")),
                FilePath = reader.GetString(reader.GetOrdinal("file_path")),
                FileName = reader.GetString(reader.GetOrdinal("file_name")),
                Content = reader.GetString(reader.GetOrdinal("content")),
                Embedding = ParseVectorToFloatArray(reader.GetString(reader.GetOrdinal("embedding"))),
                CreatedAt = reader.GetDateTime(reader.GetOrdinal("created_at")),
                UpdatedAt = reader.GetDateTime(reader.GetOrdinal("updated_at")),
                FileHash = reader.GetString(reader.GetOrdinal("file_hash")),
                FileSize = reader.GetInt64(reader.GetOrdinal("file_size")),
                FileExtension = reader.IsDBNull(reader.GetOrdinal("file_extension")) ? null : reader.GetString(reader.GetOrdinal("file_extension")),
                Tags = reader.IsDBNull(reader.GetOrdinal("tags")) ? null : reader.GetString(reader.GetOrdinal("tags"))
            };

            var similarity = reader.GetDouble(reader.GetOrdinal("similarity"));

            results.Add(new SearchResult
            {
                Document = document,
                Similarity = similarity
            });
        }

        return results;
    }

    /// <summary>
    /// Create database indexes for better performance
    /// </summary>
    private async Task CreateIndexesAsync(NpgsqlConnection connection)
    {
        var indexes = new[]
        {
            // Vector similarity index using IVFFlat
            $"CREATE INDEX IF NOT EXISTS idx_{_tableName}_embedding_ivfflat ON {_tableName} USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);",
            
            // Additional indexes for filtering
            $"CREATE INDEX IF NOT EXISTS idx_{_tableName}_file_extension ON {_tableName} (file_extension);",
            $"CREATE INDEX IF NOT EXISTS idx_{_tableName}_created_at ON {_tableName} (created_at);",
            $"CREATE INDEX IF NOT EXISTS idx_{_tableName}_file_hash ON {_tableName} (file_hash);",
            $"CREATE INDEX IF NOT EXISTS idx_{_tableName}_tags ON {_tableName} USING GIN (tags);"
        };

        foreach (var indexSql in indexes)
        {
            try
            {
                using var cmd = new NpgsqlCommand(indexSql, connection);
                await cmd.ExecuteNonQueryAsync();
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Warning: Could not create index: {ex.Message}");
                // Continue with other indexes
            }
        }
    }

    /// <summary>
    /// Format float array as PostgreSQL vector string
    /// </summary>
    private static string FormatFloatArrayAsVector(float[] embedding)
    {
        return "[" + string.Join(",", embedding.Select(f => f.ToString("G17", System.Globalization.CultureInfo.InvariantCulture))) + "]";
    }

    /// <summary>
    /// Parse PostgreSQL vector string back to float array
    /// </summary>
    private static float[] ParseVectorToFloatArray(string vectorString)
    {
        var cleaned = vectorString.Trim('[', ']');
        var parts = cleaned.Split(',');
        
        var result = new float[parts.Length];
        for (int i = 0; i < parts.Length; i++)
        {
            result[i] = float.Parse(parts[i].Trim(), System.Globalization.CultureInfo.InvariantCulture);
        }
        
        return result;
    }

    /// <summary>
    /// Calculate SHA256 hash of a file
    /// </summary>
    public static string CalculateFileHash(string filePath)
    {
        using var sha256 = SHA256.Create();
        using var fileStream = File.OpenRead(filePath);
        var hashBytes = sha256.ComputeHash(fileStream);
        return Convert.ToHexString(hashBytes);
    }

    /// <summary>
    /// Dispose of resources
    /// </summary>
    public void Dispose()
    {
        // Nothing to dispose in this implementation
    }
}
