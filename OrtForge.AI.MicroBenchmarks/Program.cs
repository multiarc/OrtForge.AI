using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Running;

namespace OrtForge.AI.MicroBenchmarks;

class Program
{
    static void Main(string[] args) {
        BenchmarkSwitcher.FromAssembly(typeof(Program).Assembly).Run(args);
    }
}