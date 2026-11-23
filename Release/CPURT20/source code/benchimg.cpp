// Headless image capture for cpurt_render_buffer(). Usage:
//   ./benchimg W H FRAMES out.ppm [move]
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

extern void cpurt_render_buffer(uint8_t* rgb_out, int W, int H,
                                float cx, float cy, float cz,
                                float fx, float fy, float fz,
                                float rx, float ry, float rz,
                                float ux, float uy, float uz,
                                float anim_t, int threads);

static void norm3(float& x, float& y, float& z) {
    float l = sqrtf(x*x + y*y + z*z);
    if (l > 1e-9f) { x /= l; y /= l; z /= l; }
}

int main(int argc, char** argv) {
    int W = argc > 1 ? atoi(argv[1]) : 640;
    int H = argc > 2 ? atoi(argv[2]) : 360;
    int frames = argc > 3 ? atoi(argv[3]) : 8;
    const char* path = argc > 4 ? argv[4] : "out.ppm";
    bool moving = argc > 5;

    std::vector<uint8_t> fb((size_t)W * H * 3);
    for (int f = -2; f < frames; f++) {
        float t = 0.01f * (float)f;
        float angle = 3.8f + t * 0.18f;
        float cx = cosf(angle) * 38.f;
        float cz = sinf(angle) * 38.f;
        float cy = 11.f;
        float fx = -cx, fy = 2.f - cy, fz = -cz;
        norm3(fx, fy, fz);
        float rx = fz, ry = 0.f, rz = -fx;
        norm3(rx, ry, rz);
        float ux = ry*fz - rz*fy;
        float uy = rz*fx - rx*fz;
        float uz = rx*fy - ry*fx;

        if (!moving) { cx = 4.f; cy = 6.f; cz = -10.f;
                       fx = 0.f; fy = 0.f; fz = 1.f;
                       rx = 1.f; ry = 0.f; rz = 0.f;
                       ux = 0.f; uy = 1.f; uz = 0.f; }

        cpurt_render_buffer(fb.data(), W, H,
                            cx, cy, cz, fx, fy, fz,
                            rx, ry, rz, ux, uy, uz,
                            t, 8);
    }

    FILE* fp = fopen(path, "wb");
    fprintf(fp, "P6\n%d %d\n255\n", W, H);
    fwrite(fb.data(), 1, fb.size(), fp);
    fclose(fp);
    printf("wrote %s (%dx%d)\n", path, W, H);
    return 0;
}
