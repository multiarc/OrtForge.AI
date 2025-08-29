#!/bin/bash
dotnet restore
dotnet build -c Release --no-restore
dotnet test -c Release --no-build --no-restore --logger "console;verbosity=detailed"