using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using Microsoft.ML.OnnxRuntime;

namespace OrtForge.AI.Models.Astractions.Extensions;

public static class VectorExtensions
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static double Magnitude<T>(this Span<T> vector)
        where T : struct, IConvertible {
        return Magnitude((ReadOnlySpan<T>)vector);
    }
    
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static double Magnitude(this Span<Half> vector) {
        return Magnitude((ReadOnlySpan<Half>)vector);
    }
    
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static double Magnitude(this Span<BFloat16> vector) {
        return Magnitude((ReadOnlySpan<BFloat16>)vector);
    }

    public static double Magnitude<T>(this ReadOnlySpan<T> vector)
        where T : struct, IConvertible {
        if (Vector512<T>.IsSupported)
        {
            return MagnitudeVec512(vector);
        }

        if (Vector256<T>.IsSupported)
        {
            return MagnitudeVec256(vector);
        }

        if (Vector<T>.IsSupported)
        {
            return MagnitudeVecGeneric(vector);
        }

        // Fallback to a simple loop for unsupported platforms
        var sum = 0d;
        foreach (var value in vector)
        {
            sum += value.ToDouble(null) * value.ToDouble(null); // x * x
        }

        return Math.Sqrt(sum);
    }
    
    public static double Magnitude(this ReadOnlySpan<Half> vector) {
        if (Vector512<Half>.IsSupported) {
            return MagnitudeVec512(vector);
        }

        if (Vector256<Half>.IsSupported) {
            return MagnitudeVec256(vector);
        }

        if (Vector<Half>.IsSupported) {
            return MagnitudeVecGeneric(vector);
        }

        // Fallback to a simple loop for unsupported platforms
        var sum = 0d;
        foreach (var value in vector) {
            sum += (double)value * (double)value; // x * x
        }

        return Math.Sqrt(sum);
    }
    
    public static double Magnitude(this ReadOnlySpan<BFloat16> vector) {
        if (Vector512<BFloat16>.IsSupported) {
            return MagnitudeVec512(vector);
        }

        if (Vector256<BFloat16>.IsSupported) {
            return MagnitudeVec256(vector);
        }

        if (Vector<BFloat16>.IsSupported) {
            return MagnitudeVecGeneric(vector);
        }

        // Fallback to a simple loop for unsupported platforms
        var sum = 0d;
        foreach (var value in vector) {
            sum += (double)value * (double)value; // x * x
        }

        return Math.Sqrt(sum);
    }
    
    public static void Normalize<T>(this Span<T> vector)
        where T : struct, IConvertible, IEquatable<T>, IDivisionOperators<T, T, T> {
        var magnitude = (T)((IConvertible)vector.Magnitude()).ToType(typeof(T), null);
        if (magnitude.Equals(default)) {
            throw new InvalidOperationException("Cannot normalize a zero vector.");
        }

        if (Vector512<T>.IsSupported)
        {
            NormalizeVec512(vector, magnitude);
        }
        else if (Vector256<T>.IsSupported) {
            NormalizeVec256(vector, magnitude);
        }
        else if (Vector<T>.IsSupported) {
            NormalizeVecGeneric(vector, magnitude);
        }
        else {
            for (var i = 0; i < vector.Length; i++) {
                vector[i] /= magnitude;
            }
        }
    }
    
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void NormalizeVecGeneric<T>(Span<T> vector, T magnitude)
        where T : struct, IConvertible {
        var vectorLength = Vector<T>.Count;
        var iterations = vector.Length / vectorLength;
        var offset = 0;
        while (iterations > 0) {
            var vec = new Vector<T>(vector);
            var result = vec / magnitude;
            result.CopyTo(vector[offset..]);
            vector = vector[vectorLength..];
            offset += vectorLength;
            iterations--;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void NormalizeVec256<T>(Span<T> vector, T magnitude) 
        where T : struct, IConvertible {
        var vectorLength = Vector256<T>.Count;
        var iterations = vector.Length / vectorLength;
        var offset = 0;
        while (iterations > 0) {
            var vec = Vector256.Create<T>(vector);
            var result = vec / magnitude;
            result.CopyTo(vector[offset..]);
            vector = vector[vectorLength..];
            offset += vectorLength;
            iterations--;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void NormalizeVec512<T>(Span<T> vector, T magnitude) 
        where T : struct, IConvertible {
        var vectorLength = Vector512<T>.Count;
        var iterations = vector.Length / vectorLength;
        var offset = 0;
        while (iterations > 0) {
            var vec = Vector512.Create<T>(vector);
            var result = vec / magnitude;
            result.CopyTo(vector[offset..]);
            vector = vector[vectorLength..];
            offset += vectorLength;
            iterations--;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static double MagnitudeVec256<T>(ReadOnlySpan<T> vector)
        where T : struct, IConvertible
    {
        var vectorLength = Vector256<T>.Count;
        var iterations = vector.Length / vectorLength;
        var accumulator = Vector256<T>.Zero;
        while (iterations > 0)
        {
            var vec = Vector256.Create(vector);
            vec *= vec; // x * x
            vec += accumulator;
            accumulator = vec;
            vector = vector[vectorLength..];
            iterations--;
        }

        var magnitude = Math.Sqrt(Vector256.Sum(accumulator).ToDouble(null));
        return magnitude;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static double MagnitudeVec512<T>(ReadOnlySpan<T> vector) 
        where T: struct, IConvertible {
        var vectorLength = Vector512<T>.Count;
        var iterations = vector.Length / vectorLength;
        var accumulator = Vector512<T>.Zero;
        while (iterations > 0) {
            var vec = Vector512.Create(vector);
            vec *= vec; // x * x
            vec += accumulator;
            accumulator = vec;
            vector = vector[vectorLength..];
            iterations--;
        }

        var magnitude = Math.Sqrt(Vector512.Sum(accumulator).ToDouble(null));
        return magnitude;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static double MagnitudeVecGeneric<T>(ReadOnlySpan<T> vector)
        where T: struct, IConvertible {
        
        var vectorLength = Vector<T>.Count;
        var iterations = vector.Length / vectorLength;
        var accumulator = Vector<T>.Zero;
        while (iterations > 0) {
            var vec = new Vector<T>(vector);
            vec *= vec; // x * x
            vec += accumulator;
            accumulator = vec;
            vector = vector[vectorLength..];
            iterations--;
        }

        var magnitude = Math.Sqrt(Vector.Sum(accumulator).ToDouble(null));
        return magnitude;
    }
    
    public static void Normalize(this Span<BFloat16> vector) {
        var magnitude = vector.Magnitude();
        if (magnitude == 0) {
            throw new InvalidOperationException("Cannot normalize a zero vector.");
        }

        if (Vector512<BFloat16>.IsSupported) {
            NormalizeVec512(vector, (BFloat16)magnitude);
        }
        else if (Vector256<BFloat16>.IsSupported) {
            NormalizeVec256(vector, (BFloat16)magnitude);
        }
        else if (Vector<Half>.IsSupported) {
            NormalizeVecGeneric(vector, (BFloat16)magnitude);
        }
        else {
            var halfMagnitude = magnitude;
            for (var i = 0; i < vector.Length; i++) {
                vector[i] = (BFloat16)((float)vector[i] / halfMagnitude);
            }
        }
    }
    
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void NormalizeVecGeneric(Span<BFloat16> vector, BFloat16 magnitude) {
        var vectorLength = Vector<BFloat16>.Count;
        var iterations = vector.Length / vectorLength;
        var offset = 0;
        while (iterations > 0) {
            var vec = new Vector<BFloat16>(vector);
            var result = vec / magnitude;
            result.CopyTo(vector[offset..]);
            vector = vector[vectorLength..];
            offset += vectorLength;
            iterations--;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void NormalizeVec256(Span<BFloat16> vector, BFloat16 magnitude) {
        var vectorLength = Vector256<BFloat16>.Count;
        var iterations = vector.Length / vectorLength;
        var offset = 0;
        while (iterations > 0) {
            var vec = Vector256.Create<BFloat16>(vector);
            var result = vec / magnitude;
            result.CopyTo(vector[offset..]);
            vector = vector[vectorLength..];
            offset += vectorLength;
            iterations--;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void NormalizeVec512(Span<BFloat16> vector, BFloat16 magnitude) {
        var vectorLength = Vector512<BFloat16>.Count;
        var iterations = vector.Length / vectorLength;
        var offset = 0;
        while (iterations > 0) {
            var vec = Vector512.Create<BFloat16>(vector);
            var result = vec / magnitude;
            result.CopyTo(vector[offset..]);
            vector = vector[vectorLength..];
            offset += vectorLength;
            iterations--;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static double MagnitudeVec256(ReadOnlySpan<BFloat16> vector) {
        var vectorLength = Vector256<BFloat16>.Count;
        var iterations = vector.Length / vectorLength;
        var accumulator = Vector256<BFloat16>.Zero;
        while (iterations > 0) {
            var vec = Vector256.Create(vector);
            vec *= vec; // x * x
            vec += accumulator;
            accumulator = vec;
            vector = vector[vectorLength..];
            iterations--;
        }

        var magnitude = Math.Sqrt((double)Vector256.Sum(accumulator));
        return magnitude;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static double MagnitudeVec512(ReadOnlySpan<BFloat16> vector) {
        var vectorLength = Vector512<BFloat16>.Count;
        var iterations = vector.Length / vectorLength;
        var accumulator = Vector512<BFloat16>.Zero;
        while (iterations > 0) {
            var vec = Vector512.Create(vector);
            vec *= vec; // x * x
            vec += accumulator;
            accumulator = vec;
            vector = vector[vectorLength..];
            iterations--;
        }

        var magnitude = Math.Sqrt((double)Vector512.Sum(accumulator));
        return magnitude;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static double MagnitudeVecGeneric(ReadOnlySpan<BFloat16> vector) {
        
        var vectorLength = Vector<BFloat16>.Count;
        var iterations = vector.Length / vectorLength;
        var accumulator = Vector<BFloat16>.Zero;
        while (iterations > 0) {
            var vec = new Vector<BFloat16>(vector);
            vec *= vec; // x * x
            vec += accumulator;
            accumulator = vec;
            vector = vector[vectorLength..];
            iterations--;
        }

        var magnitude = Math.Sqrt((double)Vector.Sum(accumulator));
        return magnitude;
    }
    
    public static void Normalize(this Span<Half> vector) {
        var magnitude = vector.Magnitude();
        if (magnitude == 0) {
            throw new InvalidOperationException("Cannot normalize a zero vector.");
        }

        if (Vector512<Half>.IsSupported) {
            NormalizeVec512(vector, (Half)magnitude);
        }
        else if (Vector256<Half>.IsSupported) {
            NormalizeVec256(vector, (Half)magnitude);
        }
        else if (Vector<Half>.IsSupported) {
            NormalizeVecGeneric(vector, (Half)magnitude);
        }
        else {
            var halfMagnitude = magnitude;
            for (var i = 0; i < vector.Length; i++) {
                vector[i] = (Half)((float)vector[i] / halfMagnitude);
            }
        }
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
    private static double MagnitudeVec256(ReadOnlySpan<Half> vector) {
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

        var magnitude = Math.Sqrt((double)Vector256.Sum(accumulator));
        return magnitude;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static double MagnitudeVec512(ReadOnlySpan<Half> vector) {
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

        var magnitude = Math.Sqrt((double)Vector512.Sum(accumulator));
        return magnitude;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static double MagnitudeVecGeneric(ReadOnlySpan<Half> vector) {
        
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

        var magnitude = Math.Sqrt((double)Vector.Sum(accumulator));
        return magnitude;
    }
}