using System.Buffers;
using Microsoft.ML.Tokenizers;
using EncodedToken = Microsoft.ML.Tokenizers.EncodedToken;

namespace OrtForge.AI.Agent.Tokenization;

/// <summary>
/// A wrapper that adapts Hugging Face Tokenizers.DotNet to work with Microsoft.ML.Tokenizers interface
/// </summary>
public sealed class HuggingFaceTokenizerWrapper : Tokenizer
{
    private readonly Tokenizers.DotNet.Tokenizer _hfTokenizer;

    public HuggingFaceTokenizerWrapper(Tokenizers.DotNet.Tokenizer hfTokenizer)
    {
        _hfTokenizer = hfTokenizer ?? throw new ArgumentNullException(nameof(hfTokenizer));
    }

    //TODO: replace with Span able implementation
    protected override EncodeResults<EncodedToken> EncodeToTokens(string? text, ReadOnlySpan<char> textSpan,
        EncodeSettings settings)
    {
        try
        {
            uint[] tokenIds;
            if (text != null)
            {
                tokenIds = _hfTokenizer.Encode(text);
            }
            else
            {
                tokenIds = _hfTokenizer.Encode(new string(textSpan));
            }

            var encodedTokens = new List<EncodedToken>(tokenIds.Length);
            foreach (var tid in tokenIds)
            {
                encodedTokens.Add(new EncodedToken((int)tid, string.Empty, default));
            }

            return new EncodeResults<EncodedToken>
            {
                CharsConsumed = text?.Length ?? textSpan.Length,
                NormalizedText = null,
                Tokens = encodedTokens
            };
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException($"Failed to encode text: {ex.Message}", ex);
        }
    }

    //TODO: replace with proper implementation that works with ints
    public override OperationStatus Decode(IEnumerable<int> ids, Span<char> destination, out int idsConsumed,
        out int charsWritten)
    {
        try
        {
            var idArray = ids.Select(x => (uint)x).ToArray();
            var result = _hfTokenizer.Decode(idArray);
            if (result.Length > destination.Length)
            {
                idsConsumed = 0;
                charsWritten = 0;
                return OperationStatus.DestinationTooSmall;
            }

            idsConsumed = idArray.Length;
            charsWritten = result.Length;
            result.CopyTo(destination);
            return OperationStatus.Done;
        }
        catch
        {
            idsConsumed = 0;
            charsWritten = 0;
            return OperationStatus.InvalidData;
        }
    }
}