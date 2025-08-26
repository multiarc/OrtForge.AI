#!/bin/bash
cd OrtForge.AI.MicroBenchmarks
dotnet restore
dotnet build -c Release --no-restore
dotnet run -c Release --no-build --no-restore