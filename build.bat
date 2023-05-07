@echo off
REM Run inside visual studio shell (e.g. x64 Native Tools Command Prompt for VS 2022)
pushd build
cl /Zi /W4 /EHsc ..\src\main.cpp
popd