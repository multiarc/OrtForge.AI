using BenchmarkDotNet.Running;

namespace OrtForge.AI.MicroBenchmarks;

class Program
{
    static void Main(string[] args) {
        BenchmarkRunner.Run<VectorBenchmarks>();
    }
}