#!/bin/bash
# Linux only (g++ with OpenMP)
echo "=== Building baker ==="
g++ -O3 -march=native -fopenmp -std=c++17 -o baker baker.cpp -lm

echo "=== Building renderer ==="
g++ -O3 -march=native -mavx2 -fopenmp -std=c++17 -static-libstdc++ -static-libgcc -o cpurt8 cpurt8.cpp -lm

echo ""
echo "=== Usage ==="
echo "  ./cpurt8 fast              quick 200x113 preview"
echo "  ./cpurt8                   400x225 still"
echo "  ./cpurt8 hq                800x450 still"
echo "  ./cpurt8 hd                1920x1080 still (~5min 2-core, ~40s 16-core)"
echo ""
echo "  # Render video (48 frames, ~3min on 2 cores, ~20s on 16-core):"
echo "  mkdir -p frames"
echo "  for i in \$(seq 0 47); do ./cpurt8 video \$i 48; done"
echo "  ffmpeg -framerate 24 -i frames/frame_%04d.ppm -vf scale=1280:720:flags=lanczos -c:v libx264 -crf 18 -pix_fmt yuv420p out.mp4"
echo ""
echo "  # Bake your own OBJ:"
echo "  ./baker mymodel.obj mymodel.sdf [resolution=64]"
echo "  # Then add: g_grids[g_ngrid++]=load_sdf(\"mymodel.sdf\",{x,y,z},scale,MAT_MESH_STONE);"
echo "  # inside render_frame() in cpurt8.cpp"
echo ""
echo "=== Building realtime game ==="
g++ -O2 -std=c++17 -o game game.cpp $(sdl2-config --cflags --libs) -lopengl32 -lglew32 -lm
echo "  ./game                     run realtime SDF raymarching FPS"
