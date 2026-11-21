#!/usr/bin/env bash
# Native Linux / MSYS2 MINGW64 build (SDL2 + GLEW), matching older CPURT builds.
set -euo pipefail

CXX="${CXX:-g++}"

if ! command -v "$CXX" >/dev/null 2>&1; then
    echo "ERROR: $CXX not found." >&2
    exit 1
fi

# MinGW needs opengl32 explicitly; Linux gets OpenGL from libGL via GLEW.
if [[ "$($CXX -dumpmachine)" == *mingw* ]]; then
    EXE=".exe"
    PLATFORM_LIBS=(-lopengl32)
    COMMON_EXTRA=(
        -static-libgcc -static-libstdc++
        -Wl,-Bstatic,--whole-archive -lgomp -Wl,-Bdynamic,--no-whole-archive
    )
    if [[ -d /mingw64/bin ]]; then
        export PATH="/mingw64/bin:$PATH"
    fi
else
    EXE=""
    PLATFORM_LIBS=()
    COMMON_EXTRA=()
fi

if ! command -v pkg-config >/dev/null 2>&1 || \
   ! pkg-config --exists sdl2 glew; then
    echo "ERROR: SDL2 and GLEW development packages are required." >&2
    echo "  Debian/Ubuntu: sudo apt install build-essential pkg-config libsdl2-dev libglew-dev" >&2
    exit 1
fi

# AVX512=1 forces AVX-512, AVX512=0 forces AVX2. By default, auto-detect.
AVX512="${AVX512:-auto}"
if [[ "$AVX512" == "1" ]] || \
   { [[ "$AVX512" == "auto" ]] && grep -q avx512f /proc/cpuinfo 2>/dev/null; }; then
    SIMD=(-mavx512f -mavx512dq -mavx512bw -mavx512vl -D__AVX512F__)
    echo "[AVX-512 mode]"
else
    SIMD=(-mavx2 -mfma -D__AVX2__)
    echo "[AVX2 mode]"
fi

COMMON=(-O3 -std=c++17 -fopenmp)

echo "Using compiler: $CXX"

# ── 1. Standalone CPU renderer ─────────────────────────────────────────────
echo
echo "=== Building cpurt17 (standalone renderer) ==="
"$CXX" "${COMMON[@]}" "${COMMON_EXTRA[@]}" "${SIMD[@]}" cpurt17.cpp -o "cpurt17$EXE" -lm
echo "Done: cpurt17$EXE"
echo "  Usage: ./cpurt17$EXE fast          200x113 preview"
echo "         ./cpurt17$EXE               400x225 still"
echo "         ./cpurt17$EXE hq            800x450 still"
echo "         ./cpurt17$EXE hd            1920x1080 still"

# ── 2. Realtime game ───────────────────────────────────────────────────────
echo
echo "=== Building game (realtime SDL2 + OpenGL window) ==="
"$CXX" "${COMMON[@]}" "${COMMON_EXTRA[@]}" "${SIMD[@]}" -DNO_MAIN \
    game.cpp cpurt17.cpp -o "game$EXE" \
    $(pkg-config --cflags --libs sdl2 glew) "${PLATFORM_LIBS[@]}" -lm
echo "Done: game$EXE"
echo "  Run from this folder: ./game$EXE"

# ── 3. SDF baker (optional) ────────────────────────────────────────────────
if [[ -f baker.cpp ]]; then
    echo
    echo "=== Building baker ==="
    "$CXX" -O3 -std=c++17 "${COMMON_EXTRA[@]}" baker.cpp -o "baker$EXE" -lm
    echo "Done: baker$EXE"
fi

echo
echo "All done!"
