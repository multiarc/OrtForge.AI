#!/bin/bash
dotnet restore
dotnet build --no-restore -c Release
dotnet test --no-build --no-restore --logger "console;verbosity=detailed"