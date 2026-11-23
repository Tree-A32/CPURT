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

# CPURT20 standardises on the AVX2/FMA path. All AVX-512 CPUs support AVX2,
# and the 8-wide batched path is the one carrying the new optimisations.
SIMD=(-mavx2 -mfma -D__AVX2__)
echo "[AVX2 mode]"

COMMON=(-O3 -std=c++17 -fopenmp)

echo "Using compiler: $CXX"

# ── 1. Standalone CPU renderer ─────────────────────────────────────────────
echo
echo "=== Building cpurt20 (standalone renderer) ==="
"$CXX" "${COMMON[@]}" "${COMMON_EXTRA[@]}" "${SIMD[@]}" cpurt20.cpp -o "cpurt20$EXE" -lm
echo "Done: cpurt20$EXE"
echo "  Usage: ./cpurt20$EXE fast          200x113 preview"
echo "         ./cpurt20$EXE               400x225 still"
echo "         ./cpurt20$EXE hq            800x450 still"
echo "         ./cpurt20$EXE hd            1920x1080 still"

# ── 2. Realtime game ───────────────────────────────────────────────────────
echo
echo "=== Building game (realtime SDL2 + OpenGL window) ==="
"$CXX" "${COMMON[@]}" "${COMMON_EXTRA[@]}" "${SIMD[@]}" -DNO_MAIN \
    game.cpp cpurt20.cpp -o "game$EXE" \
    $(pkg-config --cflags --libs sdl2 glew) "${PLATFORM_LIBS[@]}" -lm
echo "Done: game$EXE"
echo "  Run from this folder: ./game$EXE"

# ── 3. Headless CPU bridge benchmark ───────────────────────────────────────
echo
echo "=== Building bench20 (headless cpurt_render_buffer benchmark) ==="
"$CXX" "${COMMON[@]}" "${COMMON_EXTRA[@]}" "${SIMD[@]}" -DNO_MAIN \
    bench20.cpp cpurt20.cpp -o "bench20$EXE" -lm
echo "Done: bench20$EXE"
echo "  Usage: ./bench20$EXE W H FRAMES [move]"

# ── 4. SDF baker (optional) ────────────────────────────────────────────────
if [[ -f baker.cpp ]]; then
    echo
    echo "=== Building baker ==="
    "$CXX" -O3 -std=c++17 "${COMMON_EXTRA[@]}" baker.cpp -o "baker$EXE" -lm
    echo "Done: baker$EXE"
fi

echo
echo "All done!"
