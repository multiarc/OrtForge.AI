using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Tokenizers;

namespace OrtAgent.Core.Tokenization;

public sealed class TokenizerService
{
    private readonly Tokenizer _tokenizer;

    public TokenizerService(Tokenizer tokenizer)
    {
        _tokenizer = tokenizer;
    }

    public static TokenizerService FromModelFiles(string pathOrDir)
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


