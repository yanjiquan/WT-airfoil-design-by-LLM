@echo off
echo.
echo Copyright 1987-2025 ANSYS, Inc. All Rights Reserved.

set PYTHONHOME=%FLUENT_INC%\..\commonfiles\CPython\3_10\winx64\Release\python
set PYTHONPATH=%FLUENT_INC%\..\commonfiles\CPython\3_10\winx64\Release\python

echo Compiler and linker: Clang (builtin)
set FLUENT_UDF_COMPILER=clang
set FLUENT_UDF_CLANG=builtin
"%PYTHONPATH%\Scripts\scons.exe" -s
