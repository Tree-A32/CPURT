// Headless benchmark for cpurt_render_buffer(). Usage:
//   ./bench20 W H FRAMES [move]
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

extern void cpurt_render_buffer(uint8_t* rgb_out, int W, int H,
                                float cx, float cy, float cz,
                                float fx, float fy, float fz,
                                float rx, float ry, float rz,
                                float ux, float uy, float uz,
                                float anim_t, int threads);
extern void cpurt_reuse_stats(long long* attempts, long long* hits, long long* sky);

static void norm3(float& x, float& y, float& z) {
    float l = sqrtf(x*x + y*y + z*z);
    if (l > 1e-9f) { x /= l; y /= l; z /= l; }
}

int main(int argc, char** argv) {
    int W = argc > 1 ? atoi(argv[1]) : 200;
    int H = argc > 2 ? atoi(argv[2]) : 150;
    int frames = argc > 3 ? atoi(argv[3]) : 10;
    bool moving = argc > 4;
    int threads = 8;
    if (const char* env = getenv("OMP_NUM_THREADS")) {
        int t = atoi(env);
        if (t > 0) threads = t;
    }

    std::vector<uint8_t> fb((size_t)W * H * 3);
    std::vector<double> times;

    for (int f = -2; f < frames; f++) {
        float t = 0.01f * (float)f;
        float angle = 3.8f + t * 0.18f;
        float cx = cosf(angle) * 38.f;
        float cz = sinf(angle) * 38.f;
        float cy = 11.f;
        float fx = -cx, fy = 2.f - cy, fz = -cz;
        norm3(fx, fy, fz);
        // right = normalize(cross(fwd, up)), up = cross(right, fwd)
        float rx = fz, ry = 0.f, rz = -fx;
        norm3(rx, ry, rz);
        float ux = ry*fz - rz*fy;
        float uy = rz*fx - rx*fz;
        float uz = rx*fy - ry*fx;

        if (!moving) { cx = 4.f; cy = 6.f; cz = -10.f;
                       fx = 0.f; fy = 0.f; fz = 1.f;
                       rx = 1.f; ry = 0.f; rz = 0.f;
                       ux = 0.f; uy = 1.f; uz = 0.f; }

        auto t0 = std::chrono::steady_clock::now();
        cpurt_render_buffer(fb.data(), W, H,
                            cx, cy, cz, fx, fy, fz,
                            rx, ry, rz, ux, uy, uz,
                            t, threads);
        auto t1 = std::chrono::steady_clock::now();
        if (f >= 0) {
            times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
    }

    std::sort(times.begin(), times.end());
    double med = times[times.size() / 2];
    long long tries = 0, hits = 0, skyhits = 0;
    cpurt_reuse_stats(&tries, &hits, &skyhits);
    printf("bench %dx%d %s threads=%d median=%.2f ms (%.1f fps) reuse=%lld/%lld sky=%lld\n",
           W, H, moving ? "moving" : "static", threads, med, 1000.0 / med,
           hits + skyhits, tries, skyhits);
    return 0;
}
