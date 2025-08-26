using Microsoft.ML.Tokenizers;

namespace OrtAgent.Core.Tokenization;

public sealed class TokenizerService
{
    private readonly Tokenizer _tokenizer;

    public TokenizerService(Tokenizer tokenizer)
    {
        _tokenizer = tokenizer;
    }

    public static TokenizerService FromPretrained(string pathOrDir)
    {
        if (System.IO.Directory.Exists(pathOrDir))
        {
            var spmPath = System.IO.Path.Combine(pathOrDir, "sentencepiece.bpe.model");
            using var fs = System.IO.File.OpenRead(spmPath);
            var tk = SentencePieceTokenizer.Create(fs);
            return new TokenizerService(tk);
        }
        else
        {
            if (pathOrDir.EndsWith(".model", StringComparison.OrdinalIgnoreCase))
            {
                using var fs = System.IO.File.OpenRead(pathOrDir);
                var tk = SentencePieceTokenizer.Create(fs);
                return new TokenizerService(tk);
            }
            throw new ArgumentException("Unsupported tokenizer format", nameof(pathOrDir));
        }
    }
    
    /// <summary>
    /// Creates a TikToken-based tokenizer from a tokenizer.json file.
    /// Notes for Llama 3.1/3.2:
    /// - The official tokenizer.json published with Meta Llama 3.x includes the regex pre-tokenization pattern (pat_str)
    ///   and special tokens. Microsoft.ML.Tokenizers.TiktokenTokenizer reads those from the JSON, so no explicit
    ///   pre-tokenizer or special tokens need to be supplied here.
    /// - Only if you have a non-standard or incomplete tokenizer.json (missing pat_str or special tokens) would you
    ///   need to construct and pass a RegexPreTokenizer or a special-tokens dictionary. This service keeps the API
    ///   minimal and relies on the canonical JSON. If such a need arises, extend this method to accept optional
    ///   overrides and pass them to TiktokenTokenizer.Create.
    /// </summary>
    public static TokenizerService FromJson(string pathOrDir)
    {
        if (System.IO.Directory.Exists(pathOrDir))
        {
            var spmPath = System.IO.Path.Combine(pathOrDir, "tokenizer.json");
            using var fs = System.IO.File.OpenRead(spmPath);
            var tk = TiktokenTokenizer.Create(fs, null, null);
            return new TokenizerService(tk);
        }
        else
        {
            if (pathOrDir.EndsWith(".json", StringComparison.OrdinalIgnoreCase))
            {
                using var fs = System.IO.File.OpenRead(pathOrDir);
                var tk = TiktokenTokenizer.Create(fs, null, null);
                return new TokenizerService(tk);
            }
            throw new ArgumentException("Unsupported tokenizer format", nameof(pathOrDir));
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


