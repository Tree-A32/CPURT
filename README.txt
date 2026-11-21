Thank you for trying CPURT!


If you have any optimization ideas, bugs, etc. feel free to open an issue on GitHub.


Windows: build with MSYS2 MINGW64 (SDL2 + GLEW). Keep SDL2.dll and glew32.dll next to game.exe.

Linux: build natively with g++, SDL2 and GLEW.

    cd "Beta/CPURT17/source code"
    ./build.sh

Then run from the version folder so assets/*.sdf is found:

    cd ..
    "./source code/game"          # Linux
    "./source code/game.exe"      # MSYS2 MINGW64

The script also builds ./cpurt17 and ./baker. Dependencies on Debian/Ubuntu:
    sudo apt install build-essential pkg-config libsdl2-dev libglew-dev


I also want to be transparent and tell you that this project is vibecoded and made with Claude Sonnet 4.6 as a fun project to test CPU limits.


Best regards,
A-32


LATEST RELEASE VERSION: CPURT15
LATEST BETA VERSION: CPURT17