#!/bin/bash
cd OrtForge.AI.MicroBenchmarks
dotnet restore /property:OrtTarget=CUDA
dotnet build -c Release --no-restore /property:OrtTarget=CUDA
dotnet run -c Release --no-build --no-restore /property:OrtTarget=CUDA