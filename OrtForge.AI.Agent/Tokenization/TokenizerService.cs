using System;
using System.Buffers;
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

    public static TokenizerService FromModelFiles(string tokenizerJsonOrDir)
    {
        // Accept either a tokenizer.json or a directory containing it
        var tk = System.IO.Directory.Exists(tokenizerJsonOrDir)
            ? Tokenizer.FromFile(System.IO.Path.Combine(tokenizerJsonOrDir, "tokenizer.json"))
            : Tokenizer.FromFile(tokenizerJsonOrDir);
        return new TokenizerService(tk);
    }

    public int[] EncodeToIds(string text)
    {
        var enc = _tokenizer.Encode(text);
        return enc.Ids.ToArray();
    }

    public string DecodeFromIds(IReadOnlyList<int> ids)
    {
        return _tokenizer.Decode(ids.ToArray());
    }
}


