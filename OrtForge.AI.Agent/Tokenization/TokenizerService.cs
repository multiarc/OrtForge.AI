using Microsoft.ML.Tokenizers;
using HfTokenizer = Tokenizers.DotNet.Tokenizer;

namespace OrtForge.AI.Agent.Tokenization;

public sealed class TokenizerService
{
    private readonly Tokenizer _tokenizer;

    public TokenizerService(Tokenizer tokenizer)
    {
        _tokenizer = tokenizer;
    }

    public static TokenizerService FromPretrained(string pathOrDir)
    {
        if (Directory.Exists(pathOrDir))
        {
            var spmPath = Path.Combine(pathOrDir, "sentencepiece.bpe.model");
            using var fs = File.OpenRead(spmPath);
            var tk = SentencePieceTokenizer.Create(fs);
            return new TokenizerService(tk);
        }
        else
        {
            if (pathOrDir.EndsWith(".model", StringComparison.OrdinalIgnoreCase))
            {
                using var fs = File.OpenRead(pathOrDir);
                var tk = SentencePieceTokenizer.Create(fs);
                return new TokenizerService(tk);
            }
            throw new ArgumentException("Unsupported tokenizer format", nameof(pathOrDir));
        }
    }
    
    /// <summary>
    /// Creates a TikToken-based tokenizer from a tokenizer.json file.
    /// Note: This only works with OpenAI-compatible tokenizer formats, not Hugging Face BPE formats.
    /// </summary>
    public static TokenizerService FromTikToken(string filePath)
    {
        if (File.Exists(filePath))
        {
            using var fs = File.OpenRead(filePath);
            var tk = TiktokenTokenizer.Create(fs, null, null);
            return new TokenizerService(tk);
        }
        else
        {
            throw new ArgumentException("File not found", nameof(filePath));
        }
    }

    /// <summary>
    /// Creates a Hugging Face tokenizer from a tokenizer.json file.
    /// This supports BPE, WordPiece, and other Hugging Face tokenizer formats.
    /// </summary>
    public static TokenizerService FromHuggingFace(string tokenizerJsonPath)
    {
        if (!File.Exists(tokenizerJsonPath))
        {
            throw new ArgumentException("Tokenizer file not found", nameof(tokenizerJsonPath));
        }

        try
        {
            var hfTokenizer = new HfTokenizer(tokenizerJsonPath);
            var wrapper = new HuggingFaceTokenizerWrapper(hfTokenizer);
            return new TokenizerService(wrapper);
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException($"Failed to load Hugging Face tokenizer: {ex.Message}", ex);
        }
    }

    public int[] EncodeToIds(string text)
    {
        var tokens = _tokenizer.EncodeToTokens(text, out _);
        return tokens.Select(t => t.Id).ToArray();
    }

    public string DecodeFromIds(IReadOnlyList<int> ids)
    {
        return _tokenizer.Decode(ids.ToArray());
    }
}


