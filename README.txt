Thank you for trying CPURT!


If you have any optimization ideas, bugs, etc. feel free to open an issue on GitHub.


Windows: build with MSYS2 MINGW64 (SDL2 + GLEW). Keep the .dll files next to game.exe.

Linux: build natively with g++, SDL2 and GLEW.

    cd "Release/CPURT20/source code"
    ./build.sh

Then run from the version folder so assets/*.sdf is found:

    cd ..
    "./source code/game"          # Linux
    "./source code/game.exe"      # MSYS2 MINGW64

The script also builds ./cpurt20, ./bench20 and ./baker. Dependencies on Debian/Ubuntu:
    sudo apt install build-essential pkg-config libsdl2-dev libglew-dev

CPURT20 performance (AVX2, 640x360 internal): ~3.8x faster than CPURT17 and ~4.2x faster than
CPURT15 on still renders; realtime CPU path ~5.7x faster moving and ~2.3x faster static than
CPURT17 at 8 threads (~82% and ~56% less frame time). Visual output is effectively identical
to CPURT17 (MAE 0.01/255).

Temporal reuse is disabled in the CPURT20 realtime path, so CPURT_NO_REUSE=1 has no effect on
current builds.


I also want to be transparent and tell you that this project is vibecoded and made with Claude Sonnet 4.6 as a fun project to test CPU limits.


Best regards,
A-32


LATEST RELEASE VERSION: CPURT20
LATEST BETA VERSION: CPURT17