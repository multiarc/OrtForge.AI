#!/bin/bash
dotnet restore /property:OrtTarget=CUDA
dotnet build --no-restore -c Release /property:OrtTarget=CUDA
dotnet test -c Release --no-build --no-restore --logger "console;verbosity=detailed"