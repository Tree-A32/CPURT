#!/bin/bash
# Linux only (g++ with OpenMP + SDL2/OpenGL for game)

echo "=== Building baker ==="
g++ -O3 -march=native -fopenmp -std=c++17 -o baker baker.cpp -lm

echo ""
echo "=== Building renderer (cpurt15) ==="
# Auto-detect AVX-512 support and build with best available ISA
if grep -q avx512f /proc/cpuinfo 2>/dev/null; then
    echo "  → AVX-512 detected, building with full AVX-512"
    g++ -O3 -mavx512f -mavx512dq -mavx512bw -mavx512vl -fopenmp -std=c++17 \
        -static-libstdc++ -static-libgcc -o cpurt15 cpurt15.cpp -lm
else
    echo "  → No AVX-512, building with AVX2"
    g++ -O3 -mavx2 -mfma -fopenmp -std=c++17 \
        -static-libstdc++ -static-libgcc -o cpurt15 cpurt15.cpp -lm
fi

echo ""
echo "=== Usage (cpurt15) ==="
echo "  ./cpurt15 fast              quick 200x113 preview"
echo "  ./cpurt15                   400x225 still"
echo "  ./cpurt15 hq                800x450 still"
echo "  ./cpurt15 hd                1920x1080 still"
echo ""
echo "  # Render video (48 frames):"
echo "  mkdir -p frames"
echo "  for i in \$(seq 0 47); do ./cpurt15 video \$i 48; done"
echo "  ffmpeg -framerate 24 -i frames/frame_%04d.ppm -vf scale=1280:720:flags=lanczos -c:v libx264 -crf 18 -pix_fmt yuv420p out.mp4"
echo ""
echo "  # Bake your own OBJ:"
echo "  ./baker mymodel.obj mymodel.sdf [resolution=64]"
echo "  # Then add: g_grids[g_ngrid++]=load_sdf(\"mymodel.sdf\",{x,y,z},scale,MAT_MESH_STONE);"
echo "  # inside render_frame() in cpurt15.cpp"
echo ""
echo "=== Building realtime game (GPU, standalone) ==="
# game.cpp is a self-contained GPU/OpenGL renderer — independent of cpurt10
# Linux: needs libsdl2-dev, libglew-dev
# Windows: adjust -lopengl32 -lglew32 for your GLEW install
if [[ "$OSTYPE" == "msys"* || "$OSTYPE" == "cygwin"* || "$OS" == "Windows_NT" ]]; then
    g++ -O3 -march=native -fopenmp -std=c++17 -DNO_MAIN -static-libgcc -static-libstdc++ -o game game.cpp cpurt15.cpp $(sdl2-config --cflags --libs) -lopengl32 -lglew32 -lm
else
    g++ -O3 -march=native -fopenmp -std=c++17 -DNO_MAIN -static-libgcc -static-libstdc++ -o game game.cpp cpurt15.cpp $(sdl2-config --cflags --libs) -lGL -lGLEW -lm
fi
echo "  ./game                     run realtime SDF raymarching FPS"
echo ""
echo "Note: game.cpp uses its own GLSL shaders (GPU path)."
echo "      cpurt10.cpp is the CPU offline renderer. They share the same scene"
echo "      design but are independent — no header dependency between them."
