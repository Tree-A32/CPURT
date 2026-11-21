#!/bin/bash
set -e
g++ -O2 -std=c++17 -mavx2 -mfma -fopenmp -D__AVX2__ -DNO_MAIN \
    game_win32.cpp cpurt16.cpp -o game_win32.exe \
    -lopengl32 -lgdi32 -luser32 \
    -mwindows \
    -static-libgcc -static-libstdc++ \
    -Wl,-Bstatic,--whole-archive -lgomp -Wl,-Bdynamic,--no-whole-archive
echo "Done: game_win32.exe"