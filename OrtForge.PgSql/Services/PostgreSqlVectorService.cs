using System.Security.Cryptography;
using Npgsql;
using OrtForge.AI.PgSql.Data;

namespace OrtForge.AI.PgSql.Services;

/// <summary>
/// Direct PostgreSQL service for pgvector operations without EF Core pgvector dependencies
/// </summary>
public class PostgreSqlVectorService : IDisposable
{
    private readonly string _connectionString;
    private readonly string _tableName;

    public PostgreSqlVectorService(string connectionString, string tableName = "document_embeddings")
    {
        _connectionString = connectionString;
        _tableName = tableName;
    }

    /// <summary>
    /// Initialize the database with pgvector extension and create tables
    /// </summary>
    public async Task InitializeDatabaseAsync()
    {
        // First, ensure the database exists by connecting to the default 'postgres' database
        var builder = new NpgsqlConnectionStringBuilder(_connectionString);
        var databaseName = builder.Database ?? throw new InvalidOperationException("Database name is not specified in connection string");
        builder.Database = "postgres"; // Connect to default database first
        var defaultConnectionString = builder.ToString();

        // Create the database if it doesn't exist
        try
        {
            using var defaultConnection = new NpgsqlConnection(defaultConnectionString);
            await defaultConnection.OpenAsync();

            // Check if database exists
            var checkDbSql = "SELECT 1 FROM pg_database WHERE datname = @database_name;";
            using var checkCmd = new NpgsqlCommand(checkDbSql, defaultConnection);
            checkCmd.Parameters.AddWithValue("@database_name", databaseName);
            
            var dbExists = await checkCmd.ExecuteScalarAsync();
            
            if (dbExists == null)
            {
                // Create the database
                var createDbSql = $"CREATE DATABASE \"{databaseName}\";";
                using var createDbCmd = new NpgsqlCommand(createDbSql, defaultConnection);
                await createDbCmd.ExecuteNonQueryAsync();
                Console.WriteLine($"Created database: {databaseName}");
            }
            else
            {
                Console.WriteLine($"Database {databaseName} already exists");
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Warning: Could not create database: {ex.Message}");
            Console.WriteLine("Please ensure PostgreSQL is running and you have proper permissions.");
        }

        // Now connect to the target database
        using var connection = new NpgsqlConnection(_connectionString);
        await connection.OpenAsync();

        // Create pgvector extension
        using var extensionCmd = new NpgsqlCommand("CREATE EXTENSION IF NOT EXISTS vector;", connection);
        await extensionCmd.ExecuteNonQueryAsync();

        // Create table with vector column
        var createTableSql = $@"
            CREATE TABLE IF NOT EXISTS {_tableName} (
                id SERIAL PRIMARY KEY,
                file_path TEXT NOT NULL UNIQUE,
                file_name TEXT NOT NULL,
                content TEXT NOT NULL,
                embedding vector(1024),
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                file_hash TEXT NOT NULL,
                file_size BIGINT NOT NULL,
                file_extension TEXT,
                tags JSONB
            );";

        using var createTableCmd = new NpgsqlCommand(createTableSql, connection);
        await createTableCmd.ExecuteNonQueryAsync();

        // Create indexes
        await CreateIndexesAsync(connection);

        Console.WriteLine("Database initialized with pgvector extension");
    }

    /// <summary>
    /// Insert or update a document with its embedding
    /// </summary>
    public async Task UpsertDocumentAsync(DocumentEmbedding document)
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
        if (result != null)
        {
            document.Id = Convert.ToInt32(result);
        }
    }

    /// <summary>
    /// Search for similar documents using pgvector cosine similarity
    /// </summary>
    public async Task<List<DocumentSearchResult>> SearchSimilarAsync(
        float[] queryEmbedding, 
        int limit = 10, 
        double threshold = 0.5,
        string? fileExtension = null,
        DateTime? fromDate = null,
        DateTime? toDate = null)
    {
        using var connection = new NpgsqlConnection(_connectionString);
        await connection.OpenAsync();

        var whereConditions = new List<string>();
        var parameters = new List<NpgsqlParameter>();

        // Add similarity threshold
        whereConditions.Add("(1 - (embedding <=> @query_embedding)) > @threshold");
        parameters.Add(new NpgsqlParameter("@query_embedding", FormatFloatArrayAsVector(queryEmbedding)));
        parameters.Add(new NpgsqlParameter("@threshold", threshold));

        // Add optional filters
        if (!string.IsNullOrEmpty(fileExtension))
        {
            whereConditions.Add("file_extension = @file_extension");
            parameters.Add(new NpgsqlParameter("@file_extension", fileExtension.ToLowerInvariant()));
        }

        if (fromDate.HasValue)
        {
            whereConditions.Add("created_at >= @from_date");
            parameters.Add(new NpgsqlParameter("@from_date", fromDate.Value));
        }

        if (toDate.HasValue)
        {
            whereConditions.Add("created_at <= @to_date");
            parameters.Add(new NpgsqlParameter("@to_date", toDate.Value));
        }

        var searchSql = $@"
            SELECT 
                id, file_path, file_name, content, created_at, updated_at, 
                file_hash, file_size, file_extension, tags,
                (1 - (embedding <=> @query_embedding::vector)) AS similarity
            FROM {_tableName}
            WHERE (1 - (embedding <=> @query_embedding::vector)) > @threshold
            ORDER BY embedding <=> @query_embedding::vector
            LIMIT @limit;";

        parameters.Add(new NpgsqlParameter("@limit", limit));

        using var cmd = new NpgsqlCommand(searchSql, connection);
        cmd.Parameters.AddRange(parameters.ToArray());

        var results = new List<DocumentSearchResult>();
        using var reader = await cmd.ExecuteReaderAsync();

        while (await reader.ReadAsync())
        {
            var document = new DocumentEmbedding
            {
                Id = reader.GetInt32(reader.GetOrdinal("id")),
                FilePath = reader.GetString(reader.GetOrdinal("file_path")),
                FileName = reader.GetString(reader.GetOrdinal("file_name")),
                Content = reader.GetString(reader.GetOrdinal("content")),
                Embedding = [], // Don't read the embedding for search results
                CreatedAt = reader.GetDateTime(reader.GetOrdinal("created_at")),
                UpdatedAt = reader.GetDateTime(reader.GetOrdinal("updated_at")),
                FileHash = reader.GetString(reader.GetOrdinal("file_hash")),
                FileSize = reader.GetInt64(reader.GetOrdinal("file_size")),
                FileExtension = reader.IsDBNull(reader.GetOrdinal("file_extension")) ? null : reader.GetString(reader.GetOrdinal("file_extension")),
                Tags = reader.IsDBNull(reader.GetOrdinal("tags")) ? null : reader.GetString(reader.GetOrdinal("tags"))
            };

            var similarity = reader.GetDouble(reader.GetOrdinal("similarity"));

            results.Add(new DocumentSearchResult
            {
                Document = document,
                Similarity = similarity
            });
        }

        return results;
    }

    /// <summary>
    /// Check if a document exists and if it has changed
    /// </summary>
    public async Task<(bool exists, bool changed)> CheckDocumentStatusAsync(string filePath, string fileHash)
    {
        using var connection = new NpgsqlConnection(_connectionString);
        await connection.OpenAsync();

        var checkSql = $"SELECT file_hash FROM {_tableName} WHERE file_path = @file_path;";
        using var cmd = new NpgsqlCommand(checkSql, connection);
        cmd.Parameters.AddWithValue("@file_path", filePath);

        var result = await cmd.ExecuteScalarAsync();
        
        if (result == null)
        {
            return (false, false); // Document doesn't exist
        }

        var existingHash = result.ToString();
        return (true, existingHash != fileHash); // Document exists, check if changed
    }

    /// <summary>
    /// Delete a document from the database
    /// </summary>
    public async Task<bool> DeleteDocumentAsync(string filePath)
    {
        using var connection = new NpgsqlConnection(_connectionString);
        await connection.OpenAsync();

        var deleteSql = $"DELETE FROM {_tableName} WHERE file_path = @file_path;";
        using var cmd = new NpgsqlCommand(deleteSql, connection);
        cmd.Parameters.AddWithValue("@file_path", filePath);

        var rowsAffected = await cmd.ExecuteNonQueryAsync();
        return rowsAffected > 0;
    }

    /// <summary>
    /// Get document count in the database
    /// </summary>
    public async Task<int> GetDocumentCountAsync()
    {
        using var connection = new NpgsqlConnection(_connectionString);
        await connection.OpenAsync();

        var countSql = $"SELECT COUNT(*) FROM {_tableName};";
        using var cmd = new NpgsqlCommand(countSql, connection);

        var result = await cmd.ExecuteScalarAsync();
        return Convert.ToInt32(result);
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
        // Remove brackets and split by comma
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
    /// Calculate file hash for change detection
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
        // Nothing to dispose for now, but keeping the interface
    }
}
