#!/bin/bash
cd OrtForge.AI.MicroBenchmarks
dotnet restore /property:OrtTarget=CUDA
dotnet build --no-restore -c Release /property:OrtTarget=CUDA
dotnet run -c Release --no-build --no-restore /property:OrtTarget=CUDA