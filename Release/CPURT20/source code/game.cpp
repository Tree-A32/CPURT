/*
 * game.cpp — Real-time SDF raymarching FPS, native Linux build
 *
 * Uses SDL2 + GLEW, matching the older CPURT game builds, while keeping the
 * CPURT17 renderer bridge and scene/shader optimizations.
 *
 * Build:
 *   ./build.sh
 *
 * Manual equivalent:
 *   g++ -O3 -std=c++17 -mavx2 -mfma -fopenmp -D__AVX2__ \
 *       game.cpp cpurt17.cpp -o game -DNO_MAIN \
 *       $(pkg-config --cflags --libs sdl2 glew) -lm
 *
 * Controls: WASD move | Mouse look | Shift sprint | Space jump
 *           F11 fullscreen | Esc back to menu | Click window to capture mouse
 */
#include <SDL2/SDL.h>
#include <GL/glew.h>
#include <GL/gl.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include <cstdint>

extern void cpurt_render_buffer(uint8_t *rgb_out, int W, int H,
                                float cx, float cy, float cz,
                                float fx, float fy, float fz,
                                float rx, float ry, float rz,
                                float ux, float uy, float uz,
                                float anim_t, int threads);

// OpenGL functions and constants come from GLEW on Linux.

// Forward declarations (UI::init calls these before they are defined below main)
static GLuint compileShader(GLenum type, const char *src);
static GLuint linkProgram(const char *vsrc, const char *fsrc);

// ─────────────────────────────────────────────────────────────────────────────
// SDL2 global window state
// ─────────────────────────────────────────────────────────────────────────────
static SDL_Window *g_window = nullptr;
static SDL_GLContext g_gl = nullptr;
static bool g_running = true;
static bool g_mouseCaptured = false;
static int g_mouseDX = 0, g_mouseDY = 0;
static int g_winW = 1280, g_winH = 720;
static bool g_sizeChanged = false;
static bool g_escPressed = false, g_f11Pressed = false;
static bool g_lbClick = false; // left button just clicked
static int g_lbX = 0, g_lbY = 0;
static SDL_Keycode g_menuKeyQueue[8];
static int g_menuKeyCount = 0;
static bool g_inputReady = false; // true after first rendered frame; ignores early injected keys

// ─────────────────────────────────────────────────────────────────────────────
// SDL2 + GLEW OpenGL 3.3 Core Profile bootstrap
// ─────────────────────────────────────────────────────────────────────────────
static bool createWindow(int w, int h)
{
    if (SDL_Init(SDL_INIT_VIDEO) != 0)
    {
        fprintf(stderr, "SDL_Init failed: %s\n", SDL_GetError());
        return false;
    }

    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);
    SDL_GL_SetAttribute(SDL_GL_ACCELERATED_VISUAL, 1);

    g_window = SDL_CreateWindow("cpurt realtime SDF",
                                SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                                w, h,
                                SDL_WINDOW_OPENGL | SDL_WINDOW_SHOWN | SDL_WINDOW_RESIZABLE);
    if (!g_window)
    {
        fprintf(stderr, "SDL_CreateWindow failed: %s\n", SDL_GetError());
        return false;
    }

    g_gl = SDL_GL_CreateContext(g_window);
    if (!g_gl)
    {
        fprintf(stderr, "SDL_GL_CreateContext failed: %s\n", SDL_GetError());
        return false;
    }
    SDL_GL_MakeCurrent(g_window, g_gl);
    SDL_GL_SetSwapInterval(1);

    glewExperimental = GL_TRUE;
    GLenum err = glewInit();
    if (err != GLEW_OK)
    {
        fprintf(stderr, "glewInit failed: %s\n",
                (const char *)glewGetErrorString(err));
        return false;
    }
    glGetError(); // clear GLEW's harmless core-profile enum error

    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// Game settings (populated by menu, consumed by game loop)
// ─────────────────────────────────────────────────────────────────────────────
struct Settings
{
    enum Renderer
    {
        GPU,
        CPU
    } renderer = GPU;
    int winW = 1280;
    int winH = 720;
    int renderScale = 2;
    int threads = 8;
    bool shadows = true;
};

static const uint8_t FONT[95][6] = {
    {0x00, 0x00, 0x00, 0x00, 0x00, 0x00}, // ' '
    {0x04, 0x04, 0x04, 0x04, 0x00, 0x04}, // '!'
    {0x0A, 0x0A, 0x00, 0x00, 0x00, 0x00}, // '"'
    {0x0A, 0x1F, 0x0A, 0x1F, 0x0A, 0x00}, // '#'
    {0x0E, 0x15, 0x0E, 0x15, 0x0E, 0x00}, // '$'  (simplified)
    {0x19, 0x1A, 0x04, 0x0D, 0x13, 0x00}, // '%'
    {0x0C, 0x12, 0x0C, 0x15, 0x12, 0x0D}, // '&'
    {0x04, 0x04, 0x00, 0x00, 0x00, 0x00}, // '\''
    {0x02, 0x04, 0x04, 0x04, 0x04, 0x02}, // '('
    {0x08, 0x04, 0x04, 0x04, 0x04, 0x08}, // ')'
    {0x00, 0x0A, 0x04, 0x0A, 0x00, 0x00}, // '*'
    {0x00, 0x04, 0x0E, 0x04, 0x00, 0x00}, // '+'
    {0x00, 0x00, 0x00, 0x00, 0x04, 0x08}, // ','
    {0x00, 0x00, 0x0E, 0x00, 0x00, 0x00}, // '-'
    {0x00, 0x00, 0x00, 0x00, 0x04, 0x00}, // '.'
    {0x01, 0x02, 0x04, 0x08, 0x10, 0x00}, // '/'
    {0x0E, 0x11, 0x13, 0x15, 0x19, 0x0E}, // '0'
    {0x04, 0x0C, 0x04, 0x04, 0x04, 0x0E}, // '1'
    {0x0E, 0x11, 0x06, 0x08, 0x10, 0x1F}, // '2'
    {0x1F, 0x02, 0x06, 0x01, 0x11, 0x0E}, // '3'
    {0x02, 0x06, 0x0A, 0x1F, 0x02, 0x02}, // '4'
    {0x1F, 0x10, 0x1E, 0x01, 0x11, 0x0E}, // '5'
    {0x06, 0x08, 0x1E, 0x11, 0x11, 0x0E}, // '6'
    {0x1F, 0x01, 0x02, 0x04, 0x08, 0x08}, // '7'
    {0x0E, 0x11, 0x0E, 0x11, 0x11, 0x0E}, // '8'
    {0x0E, 0x11, 0x0F, 0x01, 0x02, 0x0C}, // '9'
    {0x00, 0x04, 0x00, 0x04, 0x00, 0x00}, // ':'
    {0x00, 0x04, 0x00, 0x04, 0x04, 0x08}, // ';'
    {0x02, 0x04, 0x08, 0x04, 0x02, 0x00}, // '<'
    {0x00, 0x0E, 0x00, 0x0E, 0x00, 0x00}, // '='
    {0x08, 0x04, 0x02, 0x04, 0x08, 0x00}, // '>'
    {0x0E, 0x11, 0x02, 0x04, 0x00, 0x04}, // '?'
    {0x0E, 0x11, 0x17, 0x16, 0x10, 0x0E}, // '@'
    {0x0E, 0x11, 0x1F, 0x11, 0x11, 0x11}, // 'A'
    {0x1E, 0x11, 0x1E, 0x11, 0x11, 0x1E}, // 'B'
    {0x0E, 0x11, 0x10, 0x10, 0x11, 0x0E}, // 'C'
    {0x1C, 0x12, 0x11, 0x11, 0x12, 0x1C}, // 'D'
    {0x1F, 0x10, 0x1E, 0x10, 0x10, 0x1F}, // 'E'
    {0x1F, 0x10, 0x1E, 0x10, 0x10, 0x10}, // 'F'
    {0x0E, 0x11, 0x10, 0x17, 0x11, 0x0E}, // 'G'
    {0x11, 0x11, 0x1F, 0x11, 0x11, 0x11}, // 'H'
    {0x0E, 0x04, 0x04, 0x04, 0x04, 0x0E}, // 'I'
    {0x07, 0x02, 0x02, 0x02, 0x12, 0x0C}, // 'J'
    {0x11, 0x12, 0x1C, 0x12, 0x11, 0x11}, // 'K'
    {0x10, 0x10, 0x10, 0x10, 0x10, 0x1F}, // 'L'
    {0x11, 0x1B, 0x15, 0x11, 0x11, 0x11}, // 'M'
    {0x11, 0x19, 0x15, 0x13, 0x11, 0x11}, // 'N'
    {0x0E, 0x11, 0x11, 0x11, 0x11, 0x0E}, // 'O'
    {0x1E, 0x11, 0x1E, 0x10, 0x10, 0x10}, // 'P'
    {0x0E, 0x11, 0x11, 0x15, 0x12, 0x0D}, // 'Q'
    {0x1E, 0x11, 0x1E, 0x14, 0x12, 0x11}, // 'R'
    {0x0E, 0x11, 0x0E, 0x01, 0x11, 0x0E}, // 'S'
    {0x1F, 0x04, 0x04, 0x04, 0x04, 0x04}, // 'T'
    {0x11, 0x11, 0x11, 0x11, 0x11, 0x0E}, // 'U'
    {0x11, 0x11, 0x11, 0x11, 0x0A, 0x04}, // 'V'
    {0x11, 0x11, 0x15, 0x15, 0x1B, 0x11}, // 'W'
    {0x11, 0x0A, 0x04, 0x0A, 0x11, 0x11}, // 'X' (unused row removed)
    {0x11, 0x0A, 0x04, 0x04, 0x04, 0x04}, // 'Y'
    {0x1F, 0x02, 0x04, 0x08, 0x10, 0x1F}, // 'Z'
    {0x0E, 0x08, 0x08, 0x08, 0x08, 0x0E}, // '['
    {0x10, 0x08, 0x04, 0x02, 0x01, 0x00}, // '\\'
    {0x0E, 0x02, 0x02, 0x02, 0x02, 0x0E}, // ']'
    {0x04, 0x0A, 0x11, 0x00, 0x00, 0x00}, // '^'
    {0x00, 0x00, 0x00, 0x00, 0x00, 0x1F}, // '_'
    {0x08, 0x04, 0x00, 0x00, 0x00, 0x00}, // '`'
    {0x00, 0x0E, 0x01, 0x0F, 0x11, 0x0F}, // 'a'
    {0x10, 0x10, 0x1E, 0x11, 0x11, 0x1E}, // 'b'
    {0x00, 0x0E, 0x10, 0x10, 0x11, 0x0E}, // 'c'
    {0x01, 0x01, 0x0F, 0x11, 0x11, 0x0F}, // 'd'
    {0x00, 0x0E, 0x11, 0x1F, 0x10, 0x0E}, // 'e'
    {0x06, 0x08, 0x1E, 0x08, 0x08, 0x08}, // 'f'
    {0x00, 0x0F, 0x11, 0x0F, 0x01, 0x0E}, // 'g'
    {0x10, 0x10, 0x1E, 0x11, 0x11, 0x11}, // 'h'
    {0x04, 0x00, 0x04, 0x04, 0x04, 0x0E}, // 'i'
    {0x02, 0x00, 0x02, 0x02, 0x12, 0x0C}, // 'j'
    {0x10, 0x10, 0x12, 0x1C, 0x12, 0x11}, // 'k'
    {0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E}, // 'l'
    {0x00, 0x1A, 0x15, 0x15, 0x15, 0x15}, // 'm'
    {0x00, 0x1E, 0x11, 0x11, 0x11, 0x11}, // 'n'
    {0x00, 0x0E, 0x11, 0x11, 0x11, 0x0E}, // 'o'
    {0x00, 0x1E, 0x11, 0x1E, 0x10, 0x10}, // 'p'
    {0x00, 0x0F, 0x11, 0x0F, 0x01, 0x01}, // 'q'
    {0x00, 0x16, 0x19, 0x10, 0x10, 0x10}, // 'r'
    {0x00, 0x0E, 0x10, 0x0E, 0x01, 0x1E}, // 's'
    {0x08, 0x1C, 0x08, 0x08, 0x09, 0x06}, // 't'
    {0x00, 0x11, 0x11, 0x11, 0x13, 0x0D}, // 'u'
    {0x00, 0x11, 0x11, 0x11, 0x0A, 0x04}, // 'v'
    {0x00, 0x11, 0x15, 0x15, 0x15, 0x0A}, // 'w'
    {0x00, 0x11, 0x0A, 0x04, 0x0A, 0x11}, // 'x'
    {0x00, 0x11, 0x0A, 0x04, 0x08, 0x10}, // 'y' (descender simplified)
    {0x00, 0x1F, 0x02, 0x04, 0x08, 0x1F}, // 'z'
    {0x06, 0x04, 0x0C, 0x04, 0x04, 0x06}, // '{'
    {0x04, 0x04, 0x04, 0x04, 0x04, 0x04}, // '|'
    {0x0C, 0x04, 0x06, 0x04, 0x04, 0x0C}, // '}'
    {0x08, 0x15, 0x02, 0x00, 0x00, 0x00}, // '~'
};

// ─────────────────────────────────────────────────────────────────────────────
// UI renderer — draws colored rectangles and pixel-font text via immediate GL
// ─────────────────────────────────────────────────────────────────────────────
struct UI
{
    int W, H; // window dimensions (updated each frame)
    GLuint prog2D = 0;
    GLuint vao2D = 0, vbo2D = 0;
    GLuint fontTex = 0;

    static const char *VERT2D;
    static const char *FRAG2D;
    static const char *FRAG_TEX;

    GLuint progTex = 0;

    void init(int w, int h)
    {
        W = w;
        H = h;
        // 2D flat-colour shader
        prog2D = linkProgram(VERT2D, FRAG2D);
        progTex = linkProgram(VERT2D, FRAG_TEX);
        glGenVertexArrays(1, &vao2D);
        glGenBuffers(1, &vbo2D);
        // Build font texture: 95 glyphs × 6wide × 8tall, R8
        // We pack them into a 1D atlas: width=95*6, height=8
        const int GW = 6, GH = 8, NGLYPHS = 95;
        uint8_t atlas[NGLYPHS * GW * GH] = {};
        for (int g = 0; g < NGLYPHS; g++)
        {
            for (int row = 0; row < 6; row++)
            {
                for (int col = 0; col < GW - 1; col++)
                { // 5 data pixels + 1 gap
                    bool on = (FONT[g][row] >> (4 - col)) & 1;
                    atlas[g * GW + col + row * (NGLYPHS * GW)] = on ? 255 : 0;
                }
            }
        }
        glGenTextures(1, &fontTex);
        glBindTexture(GL_TEXTURE_2D, fontTex);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, NGLYPHS * GW, GH, 0, GL_RED, GL_UNSIGNED_BYTE, atlas);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    }

    void resize(int w, int h)
    {
        W = w;
        H = h;
    }

    // Convert pixel coords to NDC
    float px(float x) const { return x / W * 2.f - 1.f; }
    float py(float y) const { return -(y / H * 2.f - 1.f); }

    void drawRect(float x, float y, float w, float h,
                  float r, float g, float b, float a = 1.f) const
    {
        // 4 floats per vertex: xy + dummy uv (0,0) — matches VERT2D vec4 layout
        float verts[] = {
            px(x),
            py(y),
            0,
            0,
            px(x + w),
            py(y),
            0,
            0,
            px(x + w),
            py(y + h),
            0,
            0,
            px(x),
            py(y),
            0,
            0,
            px(x + w),
            py(y + h),
            0,
            0,
            px(x),
            py(y + h),
            0,
            0,
        };
        glUseProgram(prog2D);
        glUniform4f(glGetUniformLocation(prog2D, "uColor"), r, g, b, a);
        glBindVertexArray(vao2D);
        glBindBuffer(GL_ARRAY_BUFFER, vbo2D);
        glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_DYNAMIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, 0);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glDrawArrays(GL_TRIANGLES, 0, 6);
        glDisable(GL_BLEND);
    }

    // Draw a string of ASCII text at pixel position (x,y), scale=pixel size per font pixel
    void drawText(const char *s, float x, float y, float scale,
                  float r = 1, float g = 1, float b = 1) const
    {
        const int GW = 6, GH = 8, NGLYPHS = 95;
        glUseProgram(progTex);
        glBindTexture(GL_TEXTURE_2D, fontTex);
        glUniform1i(glGetUniformLocation(progTex, "uTex"), 0);
        glUniform4f(glGetUniformLocation(progTex, "uColor"), r, g, b, 1.f);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glBindVertexArray(vao2D);
        glBindBuffer(GL_ARRAY_BUFFER, vbo2D);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, 0); // xy + uv
        float cx = x;
        for (int ci = 0; s[ci]; ci++)
        {
            int c = (int)s[ci] - 32;
            if (c < 0 || c >= NGLYPHS)
            {
                cx += GW * scale;
                continue;
            }
            float u0 = (float)(c * GW) / (NGLYPHS * GW);
            float u1 = (float)(c * GW + GW - 1) / (NGLYPHS * GW);
            float v0 = 0.f, v1 = 1.f;
            float x0 = px(cx), x1 = px(cx + GW * scale);
            float y0 = py(y), y1 = py(y + GH * scale);
            float verts[] = {
                x0,
                y0,
                u0,
                v0,
                x1,
                y0,
                u1,
                v0,
                x1,
                y1,
                u1,
                v1,
                x0,
                y0,
                u0,
                v0,
                x1,
                y1,
                u1,
                v1,
                x0,
                y1,
                u0,
                v1,
            };
            glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_DYNAMIC_DRAW);
            glDrawArrays(GL_TRIANGLES, 0, 6);
            cx += GW * scale;
        }
        glDisable(GL_BLEND);
    }
};

const char *UI::VERT2D = R"GLSL(
#version 330 core
layout(location=0) in vec4 aPos;  // xy [+ uv]
out vec2 vUV;
void main(){ vUV=aPos.zw; gl_Position=vec4(aPos.xy,0,1); }
)GLSL";
const char *UI::FRAG2D = R"GLSL(
#version 330 core
uniform vec4 uColor;
out vec4 fragColor;
void main(){ fragColor=uColor; }
)GLSL";
const char *UI::FRAG_TEX = R"GLSL(
#version 330 core
in vec2 vUV;
uniform sampler2D uTex;
uniform vec4 uColor;
out vec4 fragColor;
void main(){
    float a=texture(uTex,vUV).r;
    fragColor=vec4(uColor.rgb, a*uColor.a);
}
)GLSL";

// ─────────────────────────────────────────────────────────────────────────────
// Menu state
// ─────────────────────────────────────────────────────────────────────────────
enum GameState
{
    STATE_MENU,
    STATE_GAME
};

struct MenuItem
{
    std::string label;
    std::vector<std::string> options;
    int selected = 0;
};

struct Menu
{
    std::vector<MenuItem> items;
    int focused = 0; // which row has keyboard focus
    bool startPressed = false;

    void init()
    {
        items.clear();
        items.push_back({"Renderer", {"GPU (GLSL)", "CPU (cpurt20)"}, 0});
        items.push_back({"Resolution", {"1280x720", "1920x1080", "2560x1440", "800x600"}, 0});
        items.push_back({"Render Scale", {"1x (full)", "2x (half)", "4x (quarter)"}, 1});
        items.push_back({"Shadows", {"On", "Off"}, 0});
        items.push_back({"CPU Threads", {"1", "2", "4", "8", "16"}, 3});
    }

    // Returns true if the CPU threads row should be shown
    bool showThreads() const { return items[0].selected == 1; }

    Settings toSettings(int winW, int winH) const
    {
        Settings s;
        s.renderer = (Settings::Renderer)items[0].selected;
        // Resolution
        const char *resOpts[] = {"1280x720", "1920x1080", "2560x1440", "800x600"};
        int ri = items[1].selected;
        if (ri == 0)
        {
            s.winW = 1280;
            s.winH = 720;
        }
        else if (ri == 1)
        {
            s.winW = 1920;
            s.winH = 1080;
        }
        else if (ri == 2)
        {
            s.winW = 2560;
            s.winH = 1440;
        }
        else
        {
            s.winW = 800;
            s.winH = 600;
        }
        // Render scale
        int sc = items[2].selected;
        s.renderScale = (sc == 0) ? 1 : (sc == 1) ? 2
                                                  : 4;
        // Shadows
        s.shadows = (items[3].selected == 0);
        // Threads
        const int thr[] = {1, 2, 4, 8, 16};
        s.threads = thr[items[4].selected];
        return s;
    }

    void handleKey(SDL_Keycode k)
    {
        int visRows = showThreads() ? (int)items.size() : (int)items.size() - 1;
        if (k == SDLK_UP)
            focused = (focused + visRows - 1) % visRows;
        if (k == SDLK_DOWN)
            focused = (focused + 1) % visRows;
        if (k == SDLK_LEFT || k == SDLK_a)
        {
            auto &it = items[focused];
            it.selected = (it.selected + (int)it.options.size() - 1) % it.options.size();
        }
        if (k == SDLK_RIGHT || k == SDLK_d)
        {
            auto &it = items[focused];
            it.selected = (it.selected + 1) % it.options.size();
        }
        if (k == SDLK_RETURN || k == SDLK_KP_ENTER || k == SDLK_SPACE)
            startPressed = true;
    }

    void draw(UI &ui) const
    {
        int W = ui.W, H = ui.H;
        // Dark semi-transparent background panel
        ui.drawRect(0, 0, W, H, 0.f, 0.f, 0.f, 0.85f);

        // Title
        float scale = std::max(1.f, W / 320.f);
        float titleScale = scale * 2.5f;
        const char *title = "CPURT8  REALTIME  SDF";
        float tw = strlen(title) * 6 * titleScale;
        ui.drawText(title, (W - tw) * 0.5f, H * 0.08f, titleScale, 0.4f, 0.8f, 1.f);

        const char *sub = "CPU SDF RAY TRACER  -  GPU GLSL PORT";
        float sw = strlen(sub) * 6 * scale;
        ui.drawText(sub, (W - sw) * 0.5f, H * 0.08f + titleScale * 10, scale, 0.5f, 0.6f, 0.7f);

        // Menu rows
        int visRows = showThreads() ? (int)items.size() : (int)items.size() - 1;
        float rowH = H * 0.09f;
        float startY = H * 0.28f;
        float colX = W * 0.28f;
        float valX = W * 0.55f;

        for (int i = 0; i < visRows; i++)
        {
            const auto &it = items[i];
            float y = startY + i * rowH;
            bool isFocused = (i == focused);

            // Highlight bar
            if (isFocused)
                ui.drawRect(W * 0.08f, y - scale * 2, W * 0.84f, rowH - scale * 2,
                            0.2f, 0.5f, 0.9f, 0.3f);

            // Label
            float lscale = scale * 1.5f;
            ui.drawText(it.label.c_str(), colX - it.label.size() * 6 * lscale, y,
                        lscale, 0.8f, 0.85f, 0.9f);

            // Left arrow
            if (isFocused)
                ui.drawText("<", valX - 8 * lscale, y, lscale, 0.4f, 0.8f, 1.f);

            // Current option value
            const std::string &val = it.options[it.selected];

            // CPU mode warning
            bool cpuWarn = (i == 0 && it.selected == 1);
            float vr = cpuWarn ? 1.f : 1.f;
            float vg = cpuWarn ? 0.6f : 1.f;
            float vb = cpuWarn ? 0.2f : 1.f;
            ui.drawText(val.c_str(), valX, y, lscale, vr, vg, vb);

            // Right arrow
            if (isFocused)
                ui.drawText(">", valX + val.size() * 6 * lscale + 4 * lscale, y,
                            lscale, 0.4f, 0.8f, 1.f);
        }

        // CPU warning note
        if (items[0].selected == 1)
        {
            float wy = startY + visRows * rowH + rowH * 0.3f;
            const char *warn = "CPU mode: requires cpurt*.cpp linked — falls back to GPU";
            float ww = strlen(warn) * 6 * scale;
            ui.drawText(warn, (W - ww) * 0.5f, wy, scale, 1.f, 0.5f, 0.2f);
        }

        // Start button
        float btnW = 180 * scale, btnH = rowH * 0.9f;
        float btnX = (W - btnW) * 0.5f;
        float btnY = H * 0.85f;
        ui.drawRect(btnX, btnY, btnW, btnH, 0.15f, 0.55f, 0.25f, 0.9f);
        ui.drawRect(btnX + 2 * scale, btnY + 2 * scale, btnW - 4 * scale, btnH - 4 * scale,
                    0.2f, 0.75f, 0.35f, 0.9f);
        const char *startLabel = "ENTER  /  START";
        float slw = strlen(startLabel) * 6 * scale * 1.5f;
        ui.drawText(startLabel, btnX + (btnW - slw) * 0.5f, btnY + (btnH - 8 * scale * 1.5f) * 0.5f,
                    scale * 1.5f, 1.f, 1.f, 1.f);

        // Controls hint
        const char *hint = "Arrow keys: navigate   Left/Right: change   Enter: start";
        float hw = strlen(hint) * 6 * scale;
        ui.drawText(hint, (W - hw) * 0.5f, H * 0.95f, scale, 0.4f, 0.5f, 0.6f);
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// GLSL fragment shader (inline)
// ─────────────────────────────────────────────────────────────────────────────
static const char *VERT_SRC = R"GLSL(
#version 330 core
layout(location=0) in vec2 aPos;
out vec2 vUV;
void main(){
    vUV = aPos*0.5+0.5;
    gl_Position = vec4(aPos,0,1);
}
)GLSL";

static const char *FRAG_SRC = R"GLSL(
#version 330 core
in  vec2 vUV;
out vec4 fragColor;

uniform vec3  uCamPos;
uniform vec3  uCamFwd;
uniform vec3  uCamRight;
uniform vec3  uCamUp;
uniform float uTime;
uniform vec2  uResolution;
uniform int   uShadows;    // 1=on, 0=off

// ── constants ────────────────────────────────────────────────────────────────
#define MAX_STEPS  128      // was 200 — horizon terrain masked by fog past ~150 anyway
#define MAX_DIST   150.0    // was 300 — fog opacity 83% at t=150, 95% at t=170
#define SURF_DIST  0.02
#define MAT_TERRAIN 0.0
#define MAT_ROCK    1.0
#define MAT_WATER   2.0
#define MAT_CRYSTAL 3.0
#define MAT_MESH    4.0

// ── math helpers ─────────────────────────────────────────────────────────────
float hash(vec2 p){
    p = fract(p*vec2(127.1,311.7));
    p += dot(p,p+19.3);
    return fract(p.x*p.y);
}
float noise(vec2 p){
    vec2 i=floor(p), f=fract(p);
    vec2 u=f*f*(3.0-2.0*f);
    return mix(mix(hash(i+vec2(0,0)),hash(i+vec2(1,0)),u.x),
               mix(hash(i+vec2(0,1)),hash(i+vec2(1,1)),u.x),u.y)*2.0-1.0;
}
float fbm(vec2 p){
    float v=0.0,a=0.5;
    mat2 m=mat2(1.6,1.2,-1.2,1.6);
    for(int i=0;i<6;i++){v+=a*noise(p);p=m*p;a*=0.5;}
    return v;
}

// ── SDF primitives ───────────────────────────────────────────────────────────
float sdSphere(vec3 p,float r){return length(p)-r;}
float sdBox(vec3 p,vec3 b){vec3 q=abs(p)-b;return length(max(q,0.))+min(max(q.x,max(q.y,q.z)),0.);}
float sdTorus(vec3 p,vec2 t){return length(vec2(length(p.xz)-t.x,p.y))-t.y;}
float sdCapsule(vec3 p,vec3 a,vec3 b,float r){vec3 ab=b-a,ap=p-a;float t=clamp(dot(ap,ab)/dot(ab,ab),0.,1.);return length(ap-ab*t)-r;}
float smin(float a,float b,float k){float h=max(k-abs(a-b),0.)/k;return min(a,b)-h*h*k*.25;}

// ── terrain SDF ──────────────────────────────────────────────────────────────
float terrainH(vec2 xz){
    float h = fbm(xz*0.08)*8.0
            + fbm(xz*0.3 )*2.0;
    // High-frequency micro-detail (fbm*1.2, amplitude ±0.4) dropped for performance:
    // saves 6 noise evals per terrainH call (33% of terrain cost); invisible at game
    // distances and completely fogged at the horizon where it hurt FPS most.
    // crater bowl for the lake
    float r=length(xz-vec2(2,4));
    h -= smoothstep(8.,2.,r)*4.5;
    return h;
}
float sdTerrain(vec3 p){
    // Raw value is NOT Lipschitz-1: FBM gradient can reach ~3 on steep slopes,
    // so the marcher overshoots and lands inside the surface → black AO/shadow.
    // Dividing by 1.6 makes the step conservative — costs ~60% more terrain
    // iterations but completely eliminates the overshoot artifact.
    return (p.y - terrainH(p.xz)) / 1.6;
}

// ── water SDF ────────────────────────────────────────────────────────────────
float sdWater(vec3 p){
    float base = p.y + 1.2;
    // gentle animated ripple
    float ripple = sin(p.x*1.5+uTime*1.2)*0.04
                 + sin(p.z*1.3+uTime*0.9)*0.04
                 + sin((p.x+p.z)*1.1+uTime*1.5)*0.02;
    return base + ripple;
}

// ── analytic stand-ins for the baked SDF objects ─────────────────────────────
// cpurt8 loads arch.sdf at (8,-3.8,0), scale=2.5
// We approximate it as a U-shape of 3 boxes
float sdArch(vec3 p){
    vec3 lp=p-vec3(8,-3.8,0);
    // two pillars + lintel
    float l=sdBox(lp-vec3(-1.1,1.5,0),vec3(0.35,2.5,0.35));
    float r=sdBox(lp-vec3( 1.1,1.5,0),vec3(0.35,2.5,0.35));
    float t=sdBox(lp-vec3(0,3.5,0),  vec3(1.5,0.4,0.35));
    return min(min(l,r),t);
}
// cpurt8 loads torus.sdf at (0,5,0), scale=2.0
float sdTorusObj(vec3 p){
    vec3 lp=p-vec3(0,5,0);
    return sdTorus(lp,vec2(1.6,0.45));
}
// cpurt8 loads sphere.sdf at (-10,2.6,15), scale=1.5
float sdSphereObj(vec3 p){
    return sdSphere(p-vec3(-10,2.6,15),1.5);
}

// ── scene: returns vec2(dist, matID) ─────────────────────────────────────────
vec2 sdScene(vec3 p){
    float dW = sdWater(p);
    float dT = sdTerrain(p);

    // Zebra-stripe fix: always return the TRUE minimum distance (preserves SDF
    // contract, no overshooting), but bias material to WATER whenever both
    // surfaces compete within 0.5 units of the water plane.
    // Terrain submerged at the waterline renders as water anyway.
    float dMin = min(dT, dW);
    float mat;
    if(dW < 0.5 && dT < dW + 0.4){
        mat = MAT_WATER;   // near waterline: water always wins
    } else {
        mat = (dT < dW) ? MAT_TERRAIN : MAT_WATER;
    }
    vec2 res = vec2(dMin, mat);

    // mesh objects (analytic)
    float dA = sdArch(p);
    if(dA < res.x) res = vec2(dA, MAT_MESH);

    float dTo = sdTorusObj(p);
    if(dTo < res.x) res = vec2(dTo, MAT_CRYSTAL);

    float dSp = sdSphereObj(p);
    if(dSp < res.x) res = vec2(dSp, MAT_ROCK);

    return res;
}

// ── normal ───────────────────────────────────────────────────────────────────
vec3 calcNormal(vec3 p){
    const float e=0.02;
    return normalize(vec3(
        sdScene(p+vec3(e,0,0)).x - sdScene(p-vec3(e,0,0)).x,
        sdScene(p+vec3(0,e,0)).x - sdScene(p-vec3(0,e,0)).x,
        sdScene(p+vec3(0,0,e)).x - sdScene(p-vec3(0,0,e)).x
    ));
}

// ── water normal with animated waves ─────────────────────────────────────────
vec3 waterNormal(vec3 p){
    const float e=0.05;
    float h00 = sin(p.x*1.5+uTime*1.2)*0.04+sin(p.z*1.3+uTime*0.9)*0.04;
    float hx  = sin((p.x+e)*1.5+uTime*1.2)*0.04+sin(p.z*1.3+uTime*0.9)*0.04;
    float hz  = sin(p.x*1.5+uTime*1.2)*0.04+sin((p.z+e)*1.3+uTime*0.9)*0.04;
    return normalize(vec3(h00-hx, e, h00-hz));
}

// ── soft shadow ───────────────────────────────────────────────────────────────
float softShadow(vec3 ro, vec3 rd, float tmin, float tmax, float k){
    float res=1.0, t=tmin, ph=1e10;
    for(int i=0;i<48;i++){
        vec3 p=ro+rd*t;
        float h=sdScene(p).x;
        if(h<0.001) return 0.0;
        float y=h*h/(2.0*ph);
        float d=sqrt(h*h-y*y);
        res=min(res, k*d/max(0.0001,t-y));
        ph=h;
        t+=max(h, 0.001);   // clamp: negative h (inside surface) must still advance
        if(t>tmax) break;
    }
    return clamp(res,0.0,1.0);
}

// ── ambient occlusion (4-step) ────────────────────────────────────────────────
float calcAO(vec3 p, vec3 n){
    float occ=0.0, sca=1.0;
    for(int i=1;i<=4;i++){
        float h=0.1*float(i);
        float d=sdScene(p+n*h).x;
        occ+=(h-d)*sca;
        sca*=0.75;
    }
    return clamp(1.0-2.0*occ,0.0,1.0);
}

// ── atmospheric sky ───────────────────────────────────────────────────────────
vec3 skyColor(vec3 rd){
    vec3 sunDir = normalize(vec3(0.6,0.5,-0.4));
    // Rayleigh
    float mu = dot(rd, sunDir);
    float rayleigh = 3.0/(16.0*3.14159)*(1.0+mu*mu);
    vec3 betaR = vec3(5.8e-3,1.35e-2,3.31e-2);
    float hr = exp(-max(rd.y,0.0)*8.0);
    vec3 sky = rayleigh*betaR*hr*8.0;
    // Mie (haze near horizon)
    float mie = pow(max(0.0,mu),8.0)*0.05;
    sky += vec3(mie*0.9,mie*0.85,mie*0.7)*exp(-max(rd.y,0.0)*4.0);
    // horizon gradient
    float h = clamp(rd.y*0.5+0.5,0.0,1.0);
    vec3 horizCol = mix(vec3(0.9,0.75,0.5), vec3(0.4,0.6,0.9), h);
    sky = mix(horizCol*0.5, sky+vec3(0.1,0.15,0.4), h*h);
    // sun disk
    float sunD = max(0.0,dot(rd,sunDir));
    sky += vec3(1.4,1.1,0.7)*pow(sunD,500.0)*8.0;
    sky += vec3(1.0,0.7,0.3)*pow(sunD,8.0)*0.4;
    return sky;
}

// ── material albedo ───────────────────────────────────────────────────────────
vec3 matColor(float matID, vec3 p, vec3 n){
    if(matID==MAT_WATER)   return vec3(0.05,0.3,0.6);
    if(matID==MAT_CRYSTAL) return vec3(0.3,0.7,0.9);
    if(matID==MAT_ROCK)    return vec3(0.4,0.35,0.3);
    if(matID==MAT_MESH)    return vec3(0.5,0.48,0.44);
    // terrain: grass vs rock by normal+height
    float rocky = smoothstep(0.6,0.3,n.y) + smoothstep(4.0,7.0,p.y);
    vec3 grass = mix(vec3(0.25,0.45,0.15), vec3(0.6,0.65,0.4), fbm(p.xz*0.4)*0.5+0.5);
    vec3 rock  = mix(vec3(0.35,0.3,0.25), vec3(0.5,0.45,0.4), fbm(p.xz*1.0)*0.5+0.5);
    vec3 snow  = vec3(0.88,0.9,0.95);
    vec3 col   = mix(grass, rock, clamp(rocky,0.0,1.0));
    col        = mix(col,   snow, smoothstep(5.5,8.0,p.y+n.y*0.5));
    return col;
}

// ── main ──────────────────────────────────────────────────────────────────────
void main(){
    vec2 uv = (vUV*2.0-1.0);
    uv.x *= uResolution.x/uResolution.y;
    float fov = 0.85;

    vec3 ro = uCamPos;
    vec3 rd = normalize(uCamFwd*fov + uCamRight*uv.x + uCamUp*uv.y);

    // ── ray march ────────────────────────────────────────────────────────────
    float t=0.05;
    float matID=-1.0;
    bool hit=false;
    for(int i=0;i<MAX_STEPS;i++){
        vec3 p=ro+rd*t;
        vec2 res=sdScene(p);
        if(res.x<SURF_DIST){matID=res.y;hit=true;break;}
        if(t>MAX_DIST){break;}
        t+=max(res.x, 0.001);
    }

    vec3 col;
    if(!hit){
        col = skyColor(rd);
        // fog blend at horizon
        float fogAmt=exp(-max(rd.y,0.0)*2.0)*0.3;
        col=mix(col,vec3(0.75,0.8,0.85),fogAmt);
    } else {
        vec3 p=ro+rd*t;
        vec3 sunDir=normalize(vec3(0.6,0.5,-0.4));

        // normal (use animated water normal for water)
        vec3 n;
        if(matID==MAT_WATER) n=waterNormal(p);
        else n=calcNormal(p);

        vec3 albedo = matColor(matID, p, n);

        // lighting
        float diff = max(0.0, dot(n,sunDir));
        float sha  = (matID==MAT_WATER||uShadows==0) ? 1.0
                   : softShadow(p+n*0.08, sunDir, 0.05, 60.0, 14.0);
        // Water: AO is pointless (reflection dominates, shadow already skipped)
        // — avoids 4 heavy sdScene calls per water pixel
        float ao   = (matID==MAT_WATER) ? 1.0 : calcAO(p,n);
        float skyAmb = 0.5+0.5*n.y;

        vec3 sunCol = vec3(1.4,1.15,0.8);
        vec3 ambCol = vec3(0.15,0.2,0.35);
        col = albedo*(sunCol*diff*sha + ambCol*skyAmb*ao);

        // specular
        vec3 h2=normalize(sunDir-rd);
        float spec=pow(max(0.0,dot(n,h2)),32.0)*0.4*sha;
        if(matID==MAT_MESH||matID==MAT_ROCK||matID==MAT_CRYSTAL)
            col+=sunCol*spec*(matID==MAT_CRYSTAL?1.5:0.6);

        // water reflection
        if(matID==MAT_WATER){
            vec3 refl=reflect(rd,n);
            vec3 refCol=skyColor(refl);
            float fresnel=pow(1.0-abs(dot(-rd,n)),4.0);
            fresnel=mix(0.04,1.0,fresnel);
            col=mix(col,refCol,fresnel*0.85);
            // caustic shimmer
            col+=vec3(0.1,0.2,0.3)*pow(max(0.0,sin(p.x*3.0+uTime)*sin(p.z*3.0+uTime*0.7)),3.0)*sha;
        }

        // fog
        float fog=1.0-exp(-t*t*0.00008);
        vec3 fogCol=skyColor(rd)*0.6+vec3(0.6,0.65,0.7)*0.4;
        col=mix(col,fogCol,fog);
    }

    // tonemap (ACES approx)
    col*=1.1;
    col=(col*(2.51*col+0.03))/(col*(2.43*col+0.59)+0.14);
    col=pow(clamp(col,0.0,1.0),vec3(1.0/2.2));

    fragColor=vec4(col,1.0);
}
)GLSL";

// ─────────────────────────────────────────────────────────────────────────────
// C++ Vec3 math helpers
// ─────────────────────────────────────────────────────────────────────────────
struct Vec3
{
    float x, y, z;
};
static Vec3 add(Vec3 a, Vec3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
static Vec3 scale(Vec3 a, float s) { return {a.x * s, a.y * s, a.z * s}; }
static Vec3 norm(Vec3 a)
{
    float l = sqrtf(a.x * a.x + a.y * a.y + a.z * a.z) + 1e-9f;
    return {a.x / l, a.y / l, a.z / l};
}
static Vec3 cross(Vec3 a, Vec3 b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }

// ─────────────────────────────────────────────────────────────────────────────
// C++ terrain height (must match GLSL terrainH for collision to align)
// ─────────────────────────────────────────────────────────────────────────────
static float hashf(float x, float y)
{
    float px = x * 127.1f + y * 311.7f;
    float py = x * 311.7f + y * 127.1f;
    px = px - floorf(px);
    py = py - floorf(py);
    px += px * px + 19.3f * py;
    py += py * py + 19.3f * px;
    float v = px * py;
    v = v - floorf(v);
    return v * 2.0f - 1.0f;
}
static float noiseC(float x, float z)
{
    float ix = floorf(x), iz = floorf(z);
    float fx = x - ix, fz = z - iz;
    float ux = fx * fx * (3 - 2 * fx), uz = fz * fz * (3 - 2 * fz);
    return (hashf(ix, iz) * (1 - ux) + hashf(ix + 1, iz) * ux) * (1 - uz) + (hashf(ix, iz + 1) * (1 - ux) + hashf(ix + 1, iz + 1) * ux) * uz;
}
static float fbmC(float x, float z)
{
    float v = 0, a = 0.5f;
    float m[4] = {1.6f, 1.2f, -1.2f, 1.6f};
    for (int i = 0; i < 6; i++)
    {
        v += a * noiseC(x, z);
        float nx = m[0] * x + m[1] * z, nz = m[2] * x + m[3] * z;
        x = nx;
        z = nz;
        a *= 0.5f;
    }
    return v;
}
static float smoothstepC(float e0, float e1, float x)
{
    float t = std::max(0.f, std::min(1.f, (x - e0) / (e1 - e0)));
    return t * t * (3.f - 2.f * t);
}
static float terrainH(float x, float z)
{
    float h = fbmC(x * 0.08f, z * 0.08f) * 8.0f + fbmC(x * 0.3f, z * 0.3f) * 2.0f;
    float r = sqrtf((x - 2.f) * (x - 2.f) + (z - 4.f) * (z - 4.f));
    h -= smoothstepC(8.f, 2.f, r) * 4.5f;
    return h;
}

static GLuint compileShader(GLenum type, const char *src)
{
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);
    GLint ok;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok)
    {
        char buf[4096];
        glGetShaderInfoLog(s, sizeof(buf), nullptr, buf);
        fprintf(stderr, "Shader compile error:\n%s\n", buf);
        exit(1);
    }
    return s;
}

static GLuint linkProgram(const char *vsrc, const char *fsrc)
{
    GLuint vs = compileShader(GL_VERTEX_SHADER, vsrc);
    GLuint fs = compileShader(GL_FRAGMENT_SHADER, fsrc);
    GLuint p = glCreateProgram();
    glAttachShader(p, vs);
    glAttachShader(p, fs);
    glLinkProgram(p);
    GLint ok;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok)
    {
        char buf[1024];
        glGetProgramInfoLog(p, sizeof(buf), nullptr, buf);
        fprintf(stderr, "Shader link error:\n%s\n", buf);
        exit(1);
    }
    glDeleteShader(vs);
    glDeleteShader(fs);
    return p;
}

// ─────────────────────────────────────────────────────────────────────────────
// Resize FBO helper
// ─────────────────────────────────────────────────────────────────────────────
static void resizeFBO(GLuint fboTex, int rw, int rh)
{
    glBindTexture(GL_TEXTURE_2D, fboTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, rw, rh, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);
}

// ─────────────────────────────────────────────────────────────────────────────
// SDL2 main
// ─────────────────────────────────────────────────────────────────────────────
int main(int, char **)
{
    int WIN_W = 1280, WIN_H = 720;
    if (!createWindow(WIN_W, WIN_H))
        return 1;

    {
        const char *ver = (const char *)glGetString(GL_VERSION);
        const char *ren = (const char *)glGetString(GL_RENDERER);
        char buf[256];
        snprintf(buf, sizeof(buf), "cpurt realtime SDF  |  GL %s  |  %s",
                 ver ? ver : "?", ren ? ren : "?");
        SDL_SetWindowTitle(g_window, buf);
    }

    // ── Shaders ───────────────────────────────────────────────────────────────
    GLuint prog = linkProgram(VERT_SRC, FRAG_SRC);
    static const char *BLIT_FRAG = R"GLSL(
#version 330 core
in vec2 vUV;
out vec4 fragColor;
uniform sampler2D uTex;
void main(){ fragColor=texture(uTex,vUV); }
)GLSL";
    GLuint blitProg = linkProgram(VERT_SRC, BLIT_FRAG);

    // ── Fullscreen quad ───────────────────────────────────────────────────────
    float quad[] = {-1, -1, 1, -1, -1, 1, 1, 1};
    GLuint vao, vbo;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quad), quad, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);

    // ── FBO ───────────────────────────────────────────────────────────────────
    GLuint fbo = 0, fboTex = 0;
    int RW = WIN_W / 2, RH = WIN_H / 2;
    glGenFramebuffers(1, &fbo);
    glGenTextures(1, &fboTex);
    glBindTexture(GL_TEXTURE_2D, fboTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, RW, RH, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, fboTex, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // ── Uniform locations ─────────────────────────────────────────────────────
    GLint uCamPos_loc = glGetUniformLocation(prog, "uCamPos");
    GLint uCamFwd_loc = glGetUniformLocation(prog, "uCamFwd");
    GLint uCamRight_loc = glGetUniformLocation(prog, "uCamRight");
    GLint uCamUp_loc = glGetUniformLocation(prog, "uCamUp");
    GLint uTime_loc = glGetUniformLocation(prog, "uTime");
    GLint uRes_loc = glGetUniformLocation(prog, "uResolution");
    GLint uShadows_loc = glGetUniformLocation(prog, "uShadows");

    // ── UI + Menu ─────────────────────────────────────────────────────────────
    UI ui;
    ui.init(WIN_W, WIN_H);
    Menu menu;
    menu.init();

    GameState state = STATE_MENU;
    Settings cfg;
    Uint32 gameStart = 0;

    // ── Camera / game state ───────────────────────────────────────────────────
    Vec3 camPos = {4.0f, 6.0f, -10.0f};
    float yaw = 0.0f, pitch = -0.3f, velY = 0.0f;
    bool fullscreen = false;
    int savedWinW = WIN_W, savedWinH = WIN_H;
    int savedWinX = 0, savedWinY = 0;

    // ── Timing ────────────────────────────────────────────────────────────────
    Uint64 perfFreq = SDL_GetPerformanceFrequency();
    Uint64 perfStart = SDL_GetPerformanceCounter();
    auto nowMs = [&]() -> Uint32
    {
        return (Uint32)((SDL_GetPerformanceCounter() - perfStart) * 1000 / perfFreq);
    };
    Uint32 lastTick = nowMs();
    Uint32 fpsTick = lastTick;
    int fpsFrames = 0;

    while (g_running)
    {
        // ── Pump SDL events ────────────────────────────────────────────────────
        SDL_Event ev;
        while (SDL_PollEvent(&ev))
        {
            switch (ev.type)
            {
            case SDL_QUIT:
                g_running = false;
                break;
            case SDL_WINDOWEVENT:
                if (ev.window.event == SDL_WINDOWEVENT_SIZE_CHANGED ||
                    ev.window.event == SDL_WINDOWEVENT_RESIZED)
                {
                    if (ev.window.data1 > 0 && ev.window.data2 > 0)
                    {
                        g_winW = ev.window.data1;
                        g_winH = ev.window.data2;
                        g_sizeChanged = true;
                    }
                }
                else if (ev.window.event == SDL_WINDOWEVENT_FOCUS_GAINED)
                {
                    // Ignore stale edge state after the window regains focus.
                    g_escPressed = false;
                    g_f11Pressed = false;
                    g_mouseDX = g_mouseDY = 0;
                }
                break;
            case SDL_KEYDOWN:
                if (ev.key.repeat)
                    break;
                if (ev.key.keysym.sym == SDLK_ESCAPE)
                    g_escPressed = true;
                else if (ev.key.keysym.sym == SDLK_F11)
                    g_f11Pressed = true;
                if (g_inputReady && g_menuKeyCount < 8)
                    g_menuKeyQueue[g_menuKeyCount++] = ev.key.keysym.sym;
                break;
            case SDL_MOUSEBUTTONDOWN:
                if (ev.button.button == SDL_BUTTON_LEFT)
                {
                    g_lbClick = true;
                    g_lbX = ev.button.x;
                    g_lbY = ev.button.y;
                    if (state == STATE_GAME && !g_mouseCaptured)
                    {
                        g_mouseCaptured = true;
                        SDL_SetRelativeMouseMode(SDL_TRUE);
                        SDL_ShowCursor(SDL_DISABLE);
                        g_mouseDX = g_mouseDY = 0;
                    }
                }
                break;
            case SDL_MOUSEMOTION:
                if (g_mouseCaptured && state == STATE_GAME)
                {
                    g_mouseDX += ev.motion.xrel;
                    g_mouseDY += ev.motion.yrel;
                }
                break;
            default:
                break;
            }
        }

        // ── Timing ────────────────────────────────────────────────────────────
        Uint32 now = nowMs();
        float dt = std::min((now - lastTick) * 0.001f, 0.05f);
        lastTick = now;

        // ── Handle window resize ───────────────────────────────────────────────
        if (g_sizeChanged)
        {
            g_sizeChanged = false;
            WIN_W = std::max(1, g_winW);
            WIN_H = std::max(1, g_winH);
            ui.resize(WIN_W, WIN_H);
            if (state == STATE_GAME)
            {
                RW = std::max(1, WIN_W / cfg.renderScale);
                RH = std::max(1, WIN_H / cfg.renderScale);
                resizeFBO(fboTex, RW, RH);
            }
        }

        // ── State: MENU ───────────────────────────────────────────────────────
        if (state == STATE_MENU)
        {
            if (g_escPressed)
                g_running = false;

            // Dispatch queued key events to menu
            for (int ki = 0; ki < g_menuKeyCount; ki++)
                menu.handleKey(g_menuKeyQueue[ki]);
            g_menuKeyCount = 0;

            // Check start button click (left-button down on button area)
            if (g_lbClick)
            {
                float hs = std::max(1.f, WIN_W / 320.f);
                float btnX = (WIN_W - 180.f * hs) * 0.5f;
                float btnY = WIN_H * 0.85f;
                float btnW = 180.f * hs;
                float btnH = WIN_H * 0.09f * 0.9f;
                if (g_lbX >= (int)btnX && g_lbX <= (int)(btnX + btnW) &&
                    g_lbY >= (int)btnY && g_lbY <= (int)(btnY + btnH))
                    menu.startPressed = true;
            }
            g_lbClick = false;

            if (menu.startPressed)
            {
                menu.startPressed = false;
                cfg = menu.toSettings(WIN_W, WIN_H);
                SDL_SetWindowSize(g_window, cfg.winW, cfg.winH);
                SDL_SetWindowPosition(g_window, SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED);
                WIN_W = cfg.winW;
                WIN_H = cfg.winH;
                g_winW = WIN_W;
                g_winH = WIN_H;
                RW = std::max(1, WIN_W / cfg.renderScale);
                RH = std::max(1, WIN_H / cfg.renderScale);
                resizeFBO(fboTex, RW, RH);
                ui.resize(WIN_W, WIN_H);
                camPos = {4.0f, 6.0f, -10.0f};
                yaw = 0;
                pitch = -0.3f;
                velY = 0;
                state = STATE_GAME;
                gameStart = now;
            }
        }
        else
        { // STATE_GAME
            g_lbClick = false;
            g_menuKeyCount = 0;

            // Escape: release mouse capture or return to menu
            if (g_escPressed)
            {
                if (g_mouseCaptured)
                {
                    g_mouseCaptured = false;
                    SDL_SetRelativeMouseMode(SDL_FALSE);
                    SDL_ShowCursor(SDL_ENABLE);
                    g_mouseDX = g_mouseDY = 0;
                }
                else
                {
                    state = STATE_MENU;
                    menu.startPressed = false;
                }
            }

            // F11 fullscreen toggle
            if (g_f11Pressed)
            {
                fullscreen = !fullscreen;
                if (fullscreen)
                {
                    savedWinW = WIN_W;
                    savedWinH = WIN_H;
                    SDL_GetWindowPosition(g_window, &savedWinX, &savedWinY);
                    SDL_SetWindowFullscreen(g_window, SDL_WINDOW_FULLSCREEN_DESKTOP);
                }
                else
                {
                    SDL_SetWindowFullscreen(g_window, 0);
                    SDL_SetWindowSize(g_window, savedWinW, savedWinH);
                    SDL_SetWindowPosition(g_window, savedWinX, savedWinY);
                    WIN_W = savedWinW;
                    WIN_H = savedWinH;
                    g_winW = WIN_W;
                    g_winH = WIN_H;
                    ui.resize(WIN_W, WIN_H);
                    RW = std::max(1, WIN_W / cfg.renderScale);
                    RH = std::max(1, WIN_H / cfg.renderScale);
                    resizeFBO(fboTex, RW, RH);
                }
            }

            // Apply mouse look from relative motion
            if (g_mouseCaptured && (g_mouseDX || g_mouseDY))
            {
                yaw -= g_mouseDX * 0.002f;
                pitch -= g_mouseDY * 0.002f;
                pitch = std::max(-1.4f, std::min(1.4f, pitch));
                g_mouseDX = g_mouseDY = 0;
            }

            // ── Movement ──────────────────────────────────────────────────────
            Vec3 fwd = {sinf(yaw) * cosf(pitch), sinf(pitch), cosf(yaw) * cosf(pitch)};
            Vec3 fwd_flat = {sinf(yaw), 0.f, cosf(yaw)};
            Vec3 right_flat = {cosf(yaw), 0.f, -sinf(yaw)};
            Vec3 up = {0, 1, 0};
            Vec3 right = norm(cross(fwd, up));
            up = norm(cross(right, fwd));

            const Uint8 *keys = SDL_GetKeyboardState(nullptr);
            float speed = (keys[SDL_SCANCODE_LSHIFT] || keys[SDL_SCANCODE_RSHIFT])
                              ? 12.f
                              : 5.f;
            Vec3 mv = {0, 0, 0};
            if (keys[SDL_SCANCODE_W])
                mv = add(mv, fwd_flat);
            if (keys[SDL_SCANCODE_S])
                mv = add(mv, scale(fwd_flat, -1));
            if (keys[SDL_SCANCODE_A])
                mv = add(mv, right_flat);
            if (keys[SDL_SCANCODE_D])
                mv = add(mv, scale(right_flat, -1));
            float mlen = sqrtf(mv.x * mv.x + mv.z * mv.z);
            if (mlen > 0.001f)
            {
                mv.x /= mlen;
                mv.z /= mlen;
                camPos.x += mv.x * speed * dt;
                camPos.z += mv.z * speed * dt;
            }

            // Jump + gravity
            float groundH = terrainH(camPos.x, camPos.z) + 1.75f;
            float lakeR = sqrtf((camPos.x - 2.f) * (camPos.x - 2.f) + (camPos.z - 4.f) * (camPos.z - 4.f));
            float waterH = (lakeR < 60.f) ? (-1.2f + 1.75f) : -1e9f;
            float floorH = std::max(groundH, waterH);
            if (keys[SDL_SCANCODE_SPACE] && camPos.y <= floorH + 0.05f)
                velY = 6.f;
            velY -= 18.f * dt;
            camPos.y += velY * dt;
            if (camPos.y < floorH)
            {
                camPos.y = floorH;
                velY = 0;
            }

            // ── Render scene ──────────────────────────────────────────────────
            static std::vector<uint8_t> cpuFramebuffer;
            if (cfg.renderer == Settings::CPU)
            {
                cpuFramebuffer.resize((size_t)RW * RH * 3);
                cpurt_render_buffer(cpuFramebuffer.data(), RW, RH,
                                    camPos.x, camPos.y, camPos.z,
                                    fwd.x, fwd.y, fwd.z,
                                    right.x, right.y, right.z,
                                    up.x, up.y, up.z,
                                    now * 0.001f, cfg.threads);
                glBindTexture(GL_TEXTURE_2D, fboTex);
                glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
                glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, RW, RH, GL_RGB, GL_UNSIGNED_BYTE, cpuFramebuffer.data());
            }
            else
            {
                glBindFramebuffer(GL_FRAMEBUFFER, fbo);
                glViewport(0, 0, RW, RH);
                glUseProgram(prog);
                glUniform3f(uCamPos_loc, camPos.x, camPos.y, camPos.z);
                glUniform3f(uCamFwd_loc, fwd.x, fwd.y, fwd.z);
                glUniform3f(uCamRight_loc, right.x, right.y, right.z);
                glUniform3f(uCamUp_loc, up.x, up.y, up.z);
                glUniform1f(uTime_loc, now * 0.001f);
                glUniform2f(uRes_loc, (float)RW, (float)RH);
                glUniform1i(uShadows_loc, cfg.shadows ? 1 : 0);
                glBindVertexArray(vao);
                glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
            }

            // Blit to window
            glBindFramebuffer(GL_FRAMEBUFFER, 0);
            glViewport(0, 0, WIN_W, WIN_H);
            glClearColor(0, 0, 0, 1);
            glClear(GL_COLOR_BUFFER_BIT);
            glUseProgram(blitProg);
            glBindVertexArray(vao);
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, fboTex);
            glUniform1i(glGetUniformLocation(blitProg, "uTex"), 0);
            glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

            // HUD
            if (now - gameStart < 5000)
            {
                float hs = std::max(1.f, WIN_W / 320.f);
                ui.drawText("WASD move  Mouse look  Shift sprint  Space jump  Esc menu",
                            10, WIN_H - 12 * hs, hs, 0.7f, 0.8f, 0.9f);
                if (!cfg.shadows)
                    ui.drawText("SHADOWS OFF", 10, 10, hs * 1.5f, 1.f, 0.5f, 0.2f);
                const char *rmode = (cfg.renderer == Settings::CPU) ? "CPU" : "GPU";
                char info[64];
                snprintf(info, sizeof(info), "Renderer: %s  Scale: 1/%d", rmode, cfg.renderScale);
                ui.drawText(info, 10, 10 + 14 * hs, hs, 0.5f, 0.6f, 0.7f);
            }

            // FPS counter in title bar
            fpsFrames++;
            if (now - fpsTick >= 500)
            {
                float fps = fpsFrames * 1000.f / (now - fpsTick);
                char title[128];
                snprintf(title, sizeof(title), "cpurt realtime SDF  |  %.0f fps  |  pos %.1f %.1f %.1f",
                         fps, camPos.x, camPos.y, camPos.z);
                SDL_SetWindowTitle(g_window, title);
                fpsTick = now;
                fpsFrames = 0;
            }
        }

        // ── Menu render ───────────────────────────────────────────────────────
        if (state == STATE_MENU)
        {
            static Vec3 menuCam = {4.f, 8.f, -10.f};
            static float menuYaw = 0.f;
            menuYaw += dt * 0.04f;
            menuCam.x = sinf(menuYaw) * 15.f;
            menuCam.z = cosf(menuYaw) * 15.f;
            menuCam.y = terrainH(menuCam.x, menuCam.z) + 8.f;
            int bw = std::max(1, WIN_W / 4);
            int bh = std::max(1, WIN_H / 4);
            resizeFBO(fboTex, bw, bh);
            glBindFramebuffer(GL_FRAMEBUFFER, fbo);
            glViewport(0, 0, bw, bh);
            glUseProgram(prog);
            Vec3 mfwd = {sinf(menuYaw) * cosf(-0.2f), sinf(-0.2f), cosf(menuYaw) * cosf(-0.2f)};
            Vec3 mup = {0, 1, 0};
            Vec3 mright = norm(cross(mfwd, mup));
            mup = norm(cross(mright, mfwd));
            glUniform3f(uCamPos_loc, menuCam.x, menuCam.y, menuCam.z);
            glUniform3f(uCamFwd_loc, mfwd.x, mfwd.y, mfwd.z);
            glUniform3f(uCamRight_loc, mright.x, mright.y, mright.z);
            glUniform3f(uCamUp_loc, mup.x, mup.y, mup.z);
            glUniform1f(uTime_loc, now * 0.001f);
            glUniform2f(uRes_loc, (float)bw, (float)bh);
            glUniform1i(uShadows_loc, 0);
            glBindVertexArray(vao);
            glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

            glBindFramebuffer(GL_FRAMEBUFFER, 0);
            glViewport(0, 0, WIN_W, WIN_H);
            glClearColor(0, 0, 0, 1);
            glClear(GL_COLOR_BUFFER_BIT);
            glUseProgram(blitProg);
            glBindVertexArray(vao);
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, fboTex);
            glUniform1i(glGetUniformLocation(blitProg, "uTex"), 0);
            glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
            menu.draw(ui);
        }

        SDL_GL_SwapWindow(g_window);
        g_inputReady = true;  // ignore any OS-injected keys before first rendered frame
        g_escPressed = false;
        g_f11Pressed = false;
    }

    SDL_SetRelativeMouseMode(SDL_FALSE);
    SDL_GL_DeleteContext(g_gl);
    SDL_DestroyWindow(g_window);
    SDL_Quit();
    return 0;
}
