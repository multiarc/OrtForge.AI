using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;

namespace OrtForge.AI.Models.Extensions;

public static class VectorExtensions
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Magnitude(this Span<float> vector) {
        return Magnitude((ReadOnlySpan<float>)vector);
    }
    
    public static float Magnitude(this ReadOnlySpan<float> vector) {
        if (Vector512<float>.IsSupported) {
            return MagnitudeVec512(vector);
        }

        if (Vector256<float>.IsSupported) {
            return MagnitudeVec256(vector);
        }

        if (Vector<float>.IsSupported) {
            return MagnitudeVecGeneric(vector);
        }

        // Fallback to a simple loop for unsupported platforms
        var sum = 0f;
        foreach (var value in vector) {
            sum += value * value; // x * x
        }

        return (float) Math.Sqrt(sum);
    }
    
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Half Magnitude(this Span<Half> vector) {
        return Magnitude((ReadOnlySpan<Half>)vector);
    }
    
    public static Half Magnitude(this ReadOnlySpan<Half> vector) {
        if (Vector512<float>.IsSupported) {
            return MagnitudeVec512(vector);
        }

        if (Vector256<float>.IsSupported) {
            return MagnitudeVec256(vector);
        }

        if (Vector<float>.IsSupported) {
            return MagnitudeVecGeneric(vector);
        }

        // Fallback to a simple loop for unsupported platforms
        var sum = 0f;
        foreach (var value in vector) {
            sum += (float)value * (float)value; // x * x
        }

        return (Half) Math.Sqrt(sum);
    }
    
    public static Span<float> Normalize(this Span<float> vector) {
        var magnitude = vector.Magnitude();
        if (magnitude == 0) {
            throw new InvalidOperationException("Cannot normalize a zero vector.");
        }

        if (Vector512<float>.IsSupported) {
            NormalizeVec512(vector, magnitude);
        }
        else if (Vector256<float>.IsSupported) {
            NormalizeVec256(vector, magnitude);
        }
        else if (Vector<float>.IsSupported) {
            NormalizeVecGeneric(vector, magnitude);
        }
        else {
            for (var i = 0; i < vector.Length; i++) {
                vector[i] /= magnitude;
            }
        }

        return vector;
    }
    
    public static Span<Half> Normalize(this Span<Half> vector) {
        var magnitude = vector.Magnitude();
        if (magnitude == (Half)(float)0) {
            throw new InvalidOperationException("Cannot normalize a zero vector.");
        }

        if (Vector512<Half>.IsSupported) {
            NormalizeVec512(vector, magnitude);
        }
        else if (Vector256<Half>.IsSupported) {
            NormalizeVec256(vector, magnitude);
        }
        else if (Vector<Half>.IsSupported) {
            NormalizeVecGeneric(vector, magnitude);
        }
        else {
            for (var i = 0; i < vector.Length; i++) {
                vector[i] /= magnitude;
            }
        }

        return vector;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void NormalizeVecGeneric(Span<float> vector, float magnitude) {
        var vectorLength = Vector<float>.Count;
        var iterations = vector.Length / vectorLength;
        var offset = 0;
        while (iterations > 0) {
            var vec = new Vector<float>(vector);
            var result = vec / magnitude;
            result.CopyTo(vector[offset..]);
            vector = vector[vectorLength..];
            offset += vectorLength;
            iterations--;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void NormalizeVec256(Span<float> vector, float magnitude) {
        var vectorLength = Vector256<float>.Count;
        var iterations = vector.Length / vectorLength;
        var offset = 0;
        while (iterations > 0) {
            var vec = Vector256.Create<float>(vector);
            var result = vec / magnitude;
            result.CopyTo(vector[offset..]);
            vector = vector[vectorLength..];
            offset += vectorLength;
            iterations--;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void NormalizeVec512(Span<float> vector, float magnitude) {
        var vectorLength = Vector512<float>.Count;
        var iterations = vector.Length / vectorLength;
        var offset = 0;
        while (iterations > 0) {
            var vec = Vector512.Create<float>(vector);
            var result = vec / magnitude;
            result.CopyTo(vector[offset..]);
            vector = vector[vectorLength..];
            offset += vectorLength;
            iterations--;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float MagnitudeVec256(ReadOnlySpan<float> vector) {
        var vectorLength = Vector256<float>.Count;
        var iterations = vector.Length / vectorLength;
        var accumulator = Vector256<float>.Zero;
        while (iterations > 0) {
            var vec = Vector256.Create(vector);
            vec *= vec; // x * x
            vec += accumulator;
            accumulator = vec;
            vector = vector[vectorLength..];
            iterations--;
        }

        var magnitude = (float)Math.Sqrt(Vector256.Sum(accumulator));
        return magnitude;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float MagnitudeVec512(ReadOnlySpan<float> vector) {
        var vectorLength = Vector512<float>.Count;
        var iterations = vector.Length / vectorLength;
        var accumulator = Vector512<float>.Zero;
        while (iterations > 0) {
            var vec = Vector512.Create(vector);
            vec *= vec; // x * x
            vec += accumulator;
            accumulator = vec;
            vector = vector[vectorLength..];
            iterations--;
        }

        var magnitude = (float)Math.Sqrt(Vector512.Sum(accumulator));
        return magnitude;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float MagnitudeVecGeneric(ReadOnlySpan<float> vector) {
        
        var vectorLength = Vector<float>.Count;
        var iterations = vector.Length / vectorLength;
        var accumulator = Vector<float>.Zero;
        while (iterations > 0) {
            var vec = new Vector<float>(vector);
            vec *= vec; // x * x
            vec += accumulator;
            accumulator = vec;
            vector = vector[vectorLength..];
            iterations--;
        }

        var magnitude = (float) Math.Sqrt(Vector.Sum(accumulator));
        return magnitude;
    }
    
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void NormalizeVecGeneric(Span<Half> vector, Half magnitude) {
        var vectorLength = Vector<Half>.Count;
        var iterations = vector.Length / vectorLength;
        var offset = 0;
        while (iterations > 0) {
            var vec = new Vector<Half>(vector);
            var result = vec / magnitude;
            result.CopyTo(vector[offset..]);
            vector = vector[vectorLength..];
            offset += vectorLength;
            iterations--;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void NormalizeVec256(Span<Half> vector, Half magnitude) {
        var vectorLength = Vector256<Half>.Count;
        var iterations = vector.Length / vectorLength;
        var offset = 0;
        while (iterations > 0) {
            var vec = Vector256.Create<Half>(vector);
            var result = vec / magnitude;
            result.CopyTo(vector[offset..]);
            vector = vector[vectorLength..];
            offset += vectorLength;
            iterations--;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void NormalizeVec512(Span<Half> vector, Half magnitude) {
        var vectorLength = Vector512<Half>.Count;
        var iterations = vector.Length / vectorLength;
        var offset = 0;
        while (iterations > 0) {
            var vec = Vector512.Create<Half>(vector);
            var result = vec / magnitude;
            result.CopyTo(vector[offset..]);
            vector = vector[vectorLength..];
            offset += vectorLength;
            iterations--;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Half MagnitudeVec256(ReadOnlySpan<Half> vector) {
        var vectorLength = Vector256<Half>.Count;
        var iterations = vector.Length / vectorLength;
        var accumulator = Vector256<Half>.Zero;
        while (iterations > 0) {
            var vec = Vector256.Create(vector);
            vec *= vec; // x * x
            vec += accumulator;
            accumulator = vec;
            vector = vector[vectorLength..];
            iterations--;
        }

        var magnitude = (Half)Math.Sqrt((double)Vector256.Sum(accumulator));
        return magnitude;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Half MagnitudeVec512(ReadOnlySpan<Half> vector) {
        var vectorLength = Vector512<Half>.Count;
        var iterations = vector.Length / vectorLength;
        var accumulator = Vector512<Half>.Zero;
        while (iterations > 0) {
            var vec = Vector512.Create(vector);
            vec *= vec; // x * x
            vec += accumulator;
            accumulator = vec;
            vector = vector[vectorLength..];
            iterations--;
        }

        var magnitude = (Half)Math.Sqrt((double)Vector512.Sum(accumulator));
        return magnitude;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Half MagnitudeVecGeneric(ReadOnlySpan<Half> vector) {
        
        var vectorLength = Vector<Half>.Count;
        var iterations = vector.Length / vectorLength;
        var accumulator = Vector<Half>.Zero;
        while (iterations > 0) {
            var vec = new Vector<Half>(vector);
            vec *= vec; // x * x
            vec += accumulator;
            accumulator = vec;
            vector = vector[vectorLength..];
            iterations--;
        }

        var magnitude = (Half) Math.Sqrt((double)Vector.Sum(accumulator));
        return magnitude;
    }
}