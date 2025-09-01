using BenchmarkDotNet.Attributes;
using System.Numerics;
using System.Runtime.Intrinsics;

namespace OrtForge.AI.MicroBenchmarks;

[MemoryDiagnoser]
public class VectorBenchmarks
{
    [Params(2048, 1024)]
    public int VectorSize { get; set; }
    
    [Params(1, 2)]
    public int RandSeed { get; set; }

    private float[] _vectorSource = null!;
    private float[] _vectorOutput = null!;

    [GlobalSetup]
    public void Initialize() {
        var rnd = new Random(RandSeed);
        var rnd2 = new Random(RandSeed + 1);
        _vectorSource = new float[VectorSize];
        _vectorOutput = new float[VectorSize];
        for (var i = 0; i < VectorSize; i++) {
            _vectorSource[i] = rnd.NextSingle() * rnd2.Next(1, 1000);
        }
    }

    [Benchmark]
    public float MagnitudeVectorT() {
        var vectorLength = Vector<float>.Count;
        var span = (ReadOnlySpan<float>)_vectorSource.AsSpan();
        var iterations = span.Length / vectorLength;
        var buffer = Vector<float>.Zero;
        while (iterations > 0) {
            var vec = new Vector<float>(span);
            vec *= vec; // x * x
            vec += buffer;
            buffer = vec;
            span = span[vectorLength..];
            iterations--;
        }

        var magnitude = (float) Math.Sqrt(Vector.Sum(buffer));
        return magnitude;
    }
    
    [Benchmark]
    public float MagnitudeVector512() {
        var vectorLength = Vector512<float>.Count;
        var span = (ReadOnlySpan<float>)_vectorSource.AsSpan();
        var iterations = span.Length / vectorLength;
        var buffer = Vector512<float>.Zero;
        while (iterations > 0) {
            var vec = Vector512.Create(span);
            vec *= vec; // x * x
            vec += buffer;
            buffer = vec;
            span = span[vectorLength..];
            iterations--;
        }

        var magnitude = (float)Math.Sqrt(Vector512.Sum(buffer));
        return magnitude;
    }
    
    [Benchmark]
    public float MagnitudeVector256() {
        var vectorLength = Vector256<float>.Count;
        var span = (ReadOnlySpan<float>)_vectorSource.AsSpan();
        var iterations = span.Length / vectorLength;
        var buffer = Vector256<float>.Zero;
        while (iterations > 0) {
            var vec = Vector256.Create(span);
            vec *= vec; // x * x
            vec += buffer;
            buffer = vec;
            span = span[vectorLength..];
            iterations--;
        }

        var magnitude = (float)Math.Sqrt(Vector256.Sum(buffer));
        return magnitude;
    }
    
    [Benchmark]
    public float MagnitudeVector128() {
        var vectorLength = Vector128<float>.Count;
        var span = (ReadOnlySpan<float>)_vectorSource.AsSpan();
        var iterations = span.Length / vectorLength;
        var buffer = Vector128<float>.Zero;
        while (iterations > 0) {
            var vec = Vector128.Create(span);
            vec *= vec; // x * x
            vec += buffer;
            buffer = vec;
            span = span[vectorLength..];
            iterations--;
        }

        var magnitude = (float)Math.Sqrt(Vector128.Sum(buffer));
        return magnitude;
    }
    
    [Benchmark]
    public void DivideVectorT() {
        var magnitude = MagnitudeVectorT();
        var vectorLength = Vector<float>.Count;
        var span = _vectorSource.AsSpan();
        var outputSpan = _vectorOutput.AsSpan();
        var iterations = span.Length / vectorLength;
        var offset = 0;
        while (iterations > 0) {
            var vec = new Vector<float>(span);
            var result = vec / magnitude;
            result.CopyTo(outputSpan[offset..]);
            span = span[vectorLength..];
            offset += vectorLength;
            iterations--;
        }
    }

    [Benchmark]
    public void DivideVector512() {
        var magnitude = MagnitudeVector512();
        var span = _vectorSource.AsSpan();
        var outputSpan = _vectorOutput.AsSpan();
        var vectorLength = Vector512<float>.Count;
        var iterations = span.Length / vectorLength;
        var offset = 0;
        while (iterations > 0) {
            var vec = Vector512.Create<float>(span);
            var result = vec / magnitude;
            result.CopyTo(outputSpan.Slice(offset));
            span = span.Slice(vectorLength);
            offset += vectorLength;
            iterations--;
        }
    }

    [Benchmark]
    public void DivideVector256() {
        var magnitude = MagnitudeVector256();
        var span = _vectorSource.AsSpan();
        var outputSpan = _vectorOutput.AsSpan();
        var vectorLength = Vector256<float>.Count;
        var iterations = span.Length / vectorLength;
        var offset = 0;
        while (iterations > 0) {
            var vec = Vector256.Create<float>(span);
            var result = vec / magnitude;
            result.CopyTo(outputSpan.Slice(offset));
            span = span.Slice(vectorLength);
            offset += vectorLength;
            iterations--;
        }
    }
    
    [Benchmark]
    public void DivideVector128() {
        var magnitude = MagnitudeVector128();
        var span = _vectorSource.AsSpan();
        var outputSpan = _vectorOutput.AsSpan();
        var vectorLength = Vector128<float>.Count;
        var iterations = span.Length / vectorLength;
        var offset = 0;
        while (iterations > 0) {
            var vec = Vector128.Create<float>(span);
            var result = vec / magnitude;
            result.CopyTo(outputSpan.Slice(offset));
            span = span.Slice(vectorLength);
            offset += vectorLength;
            iterations--;
        }
    }

    [Benchmark]
    public void DivideSimpleCycle() {
        var magnitude = (float)Math.Sqrt(_vectorSource.Sum(x => x * x));
        for (int i = 0; i < _vectorSource.Length; i++) {
            _vectorOutput[i] = _vectorSource[i] / magnitude;
        }
    }

}