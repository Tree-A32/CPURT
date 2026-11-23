// cpurt20.cpp — CPU realtime renderer optimised for 1280x720 @ 1/2 scale
//
// What's new vs cpurt17:
//
//  A. Final-pixel temporal cache — previous visible pixels are forward-reprojected
//     to their current camera position using their own world-space hit points. A
//     per-pixel claim keeps the nearest candidate; pixels with no valid candidate
//     (disocclusions, animated materials, large view changes) are freshly shaded.
//     This avoids copying stale final colours across moving silhouettes. Water and
//     animated SDF meshes are never reused.
//
//  B. Compacted active batches — temporally reused pixels are removed before the
//     8-wide march, so inactive lanes do not consume ray-march slots.
//
//  C. Precomputed constants — tree bases, SDF inverse scale, rotation trig and
//     bounding spheres are refreshed only when animation changes.
//
//  D. Distance/material split — terrain material noise is only evaluated for
//     lanes close to a primary hit. Shadow, AO, GI and normal probes use the
//     distance-only evaluator with no material FBM.
//
//  E. Optional experiments (disabled by default to preserve exact imagery):
//     CPURT_BAKED=1 enables a 2048^2 baked terrain height field;
//     CPURT_BOUNDS=1 enables conservative height-band early-outs.
//
// What's carried from cpurt17:
//   AVX2 batched water reflections, GI oct 2, primary ray coherence sorting in
//   the standalone renderer, AVX2 2D tiling, gather prefetch, shadow early-out,
//   denoiser normal reuse, integer hash, BVH, SHADOW_STEPS=64, MAX_STEPS=192,
//   temporal reprojection infrastructure, game bridge.
//
// What's carried from cpurt15:
//   AO oct 2, water/crystal normal reuse, FBM oct 3 for shadow, rsqrt+NR,
//   inv_tot_lut, bilateral denoise, deterministic orbit camera.
//
// Compile:
//   cd "source code" && ./build.sh
//
// Run (from CPURT20 root, so assets/*.sdf is found):
//   ./"source code"/cpurt20 fast
//   ./"source code"/game

#include <atomic>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef __AVX512F__
#include <immintrin.h>
#endif

static const float PI = 3.14159265f;
static const float INF = 1e30f;
static const int MAX_STEPS = 192;
static const float SURF_DIST = 0.004f;
static const float MAX_DIST = 300.0f;
static bool g_baked_enabled = false;  // opt-in via CPURT_BAKED=1
static bool g_bounds_enabled = false; // opt-in via CPURT_BOUNDS=1

// ─── SCALAR MATH ─────────────────────────────────────────────────────────────
struct V3
{
    float x, y, z;
};
inline V3 operator+(V3 a, V3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
inline V3 operator-(V3 a, V3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
inline V3 operator*(V3 a, float s) { return {a.x * s, a.y * s, a.z * s}; }
inline V3 operator*(float s, V3 a) { return a * s; }
inline V3 operator*(V3 a, V3 b) { return {a.x * b.x, a.y * b.y, a.z * b.z}; }
inline float dot(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline float len(V3 v) { return sqrtf(dot(v, v)); }
inline V3 norm(V3 v)
{
    float l = len(v);
    return l > 1e-9f ? v * (1.f / l) : V3{0, 1, 0};
}
inline V3 cross(V3 a, V3 b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
inline V3 reflect(V3 v, V3 n) { return v - n * (2.f * dot(v, n)); }
inline V3 mix(V3 a, V3 b, float t) { return a * (1 - t) + b * t; }
inline float clampf(float v, float lo, float hi) { return std::max(lo, std::min(hi, v)); }
inline float smoothstep(float e0, float e1, float x)
{
    float t = clampf((x - e0) / (e1 - e0), 0, 1);
    return t * t * (3 - 2 * t);
}

// ─── CAMERA ──────────────────────────────────────────────────────────────────
struct Camera
{
    V3 pos, fwd, right, up;
    float fov, aspect;
    int W, H;
};

// Build the deterministic orbit camera from an animation timestamp.
// Identical logic to what was inlined in render_frame() in cpurt6.
Camera build_camera(float anim_t, int W, int H)
{
    float cam_angle = anim_t * 0.18f;
    float cam_r = 38.f;
    float cpx = cosf(cam_angle + 3.8f) * cam_r;
    float cpz = sinf(cam_angle + 3.8f) * cam_r;
    // terrain_h forward declaration — defined below; fine because build_camera
    // is only called from main() after all functions are defined.
    extern float terrain_h(float, float);
    float cpy = terrain_h(cpx, cpz) + 11.f;
    V3 cam_pos = {cpx, cpy, cpz};
    V3 cam_target = {0.f, 2.f + sinf(anim_t * 0.5f) * 0.5f, 0.f};
    V3 cam_fwd = norm(cam_target - cam_pos);
    V3 cam_right = norm(cross(cam_fwd, {0, 1, 0}));
    V3 cam_up = cross(cam_right, cam_fwd);
    return {cam_pos, cam_fwd, cam_right, cam_up, 0.78f, (float)W / H, W, H};
}

// Project a world-space point P into pixel coordinates using cam.
// Returns false (and leaves px/py undefined) if P is behind the camera.
inline bool project_point(V3 P, const Camera &cam, float &px, float &py)
{
    V3 V = P - cam.pos;
    float vz = dot(V, cam.fwd);
    if (vz <= 0.001f)
        return false; // behind camera
    float vx = dot(V, cam.right);
    float vy = dot(V, cam.up);
    // NDC: u,v in [-1,1]
    float u = vx / (vz * cam.aspect * cam.fov);
    float v = -vy / (vz * cam.fov); // flip Y: +up → lower v
    if (u < -1 || u > 1 || v < -1 || v > 1)
        return false;
    px = (u + 1.f) * 0.5f * cam.W;
    py = (v + 1.f) * 0.5f * cam.H;
    return true;
}

// Ray for the centre of a cache pixel, matching cpurt_render_buffer().
inline V3 camera_ray_dir(const Camera &cam, int x, int y)
{
    float u = (x + 0.5f) / cam.W * 2.f - 1.f;
    float v = 1.f - (y + 0.5f) / cam.H * 2.f;
    return norm(cam.fwd + cam.right * (u * cam.aspect * cam.fov) +
                cam.up * (v * cam.fov));
}

inline bool same_camera(const Camera &a, const Camera &b)
{
    return a.pos.x == b.pos.x && a.pos.y == b.pos.y && a.pos.z == b.pos.z &&
           a.fwd.x == b.fwd.x && a.fwd.y == b.fwd.y && a.fwd.z == b.fwd.z &&
           a.right.x == b.right.x && a.right.y == b.right.y && a.right.z == b.right.z &&
           a.up.x == b.up.x && a.up.y == b.up.y && a.up.z == b.up.z &&
           a.fov == b.fov && a.aspect == b.aspect && a.W == b.W && a.H == b.H;
}

// ─── GI CACHE (temporal reprojection) ────────────────────────────────────────
// One entry per pixel: the pre-fog GI contribution and enough data to validate it.
//   gi_r/g/b  — the GI term (gi * dot(n,gi_dir) * 0.4) from shade_batch_avx512
//   depth     — m.t of the primary hit (used for disocclusion check)
//   mat       — primary hit material (used for surface-change check)
// sky pixels: gi_r/g/b hold sky GI, depth = MAX_DIST, mat = -1
struct GICache
{
    float *gi_r; // [W*H]
    float *gi_g;
    float *gi_b;
    float *sha; // [W*H] soft shadow result
    float *ao;  // [W*H] ambient occlusion result
    float *nx;  // [W*H] primary normal for history validation
    float *ny;
    float *nz;
    float *refl_r; // [W*H] temporal water reflection cache
    float *refl_g;
    float *refl_b;
    float *fin_r; // [W*H] final gamma-encoded pixel colour (temporal reuse)
    float *fin_g;
    float *fin_b;
    float *depth;
    int *mat;
    int W, H;

    GICache(int W_, int H_) : W(W_), H(H_)
    {
        int NP = W * H;
        gi_r = new float[NP]();
        gi_g = new float[NP]();
        gi_b = new float[NP]();
        sha = new float[NP]();
        ao = new float[NP]();
        nx = new float[NP]();
        ny = new float[NP]();
        nz = new float[NP]();
        refl_r = new float[NP]();
        refl_g = new float[NP]();
        refl_b = new float[NP]();
        fin_r = new float[NP]();
        fin_g = new float[NP]();
        fin_b = new float[NP]();
        depth = new float[NP]();
        mat = new int[NP];
        std::fill(mat, mat + NP, -1);
    }
    ~GICache()
    {
        delete[] gi_r;
        delete[] gi_g;
        delete[] gi_b;
        delete[] sha;
        delete[] ao;
        delete[] nx;
        delete[] ny;
        delete[] nz;
        delete[] refl_r;
        delete[] refl_g;
        delete[] refl_b;
        delete[] fin_r;
        delete[] fin_g;
        delete[] fin_b;
        delete[] depth;
        delete[] mat;
    }
};

// ─── BAKED SDF GRID ──────────────────────────────────────────────────────────
struct SDFGrid
{
    int N;
    float bmin[3], bmax[3];
    float *data;
    V3 pos;
    float scale;
    float rot_y;
    int mat;
    bool valid;
    // Cached query constants (refreshed only when pos/rot changes).
    float inv_scale;
    float ca, sa; // cos(-rot_y), sin(-rot_y)
    float sphere_cx, sphere_cy, sphere_cz;
    float sphere_half_diag;
};

static SDFGrid g_grids[8];
static int g_ngrid = 0;

static void refresh_sdf_grid_cache(SDFGrid &g)
{
    if (!g.valid)
        return;
    g.inv_scale = 1.f / g.scale;
    g.ca = cosf(-g.rot_y);
    g.sa = sinf(-g.rot_y);
    float lx = (g.bmin[0] + g.bmax[0]) * 0.5f;
    float ly = (g.bmin[1] + g.bmax[1]) * 0.5f;
    float lz = (g.bmin[2] + g.bmax[2]) * 0.5f;
    // world = pos + R(+rot_y) * local.  With ca/sa = cos/sin(-rot_y),
    // R(+rot_y) = [[ca, sa], [-sa, ca]] in the x/z plane.
    g.sphere_cx = g.pos.x + (g.ca * lx + g.sa * lz) * g.scale;
    g.sphere_cy = g.pos.y + ly * g.scale;
    g.sphere_cz = g.pos.z + (-g.sa * lx + g.ca * lz) * g.scale;
    float dx = g.bmax[0] - g.bmin[0];
    float dy = g.bmax[1] - g.bmin[1];
    float dz = g.bmax[2] - g.bmin[2];
    g.sphere_half_diag = sqrtf(dx * dx + dy * dy + dz * dz) * 0.5f * g.scale;
}

SDFGrid load_sdf(const char *path, V3 pos, float scale, int mat, float rot_y = 0.f)
{
    SDFGrid g{};
    g.valid = false;
    FILE *f = fopen(path, "rb");
    if (!f)
    {
        fprintf(stderr, "Cannot open %s\n", path);
        return g;
    }
    fread(&g.N, sizeof(int), 1, f);
    fread(g.bmin, sizeof(float), 3, f);
    fread(g.bmax, sizeof(float), 3, f);
    int n = g.N * g.N * g.N;
    g.data = new float[n];
    fread(g.data, sizeof(float), n, f);
    fclose(f);
    g.pos = pos;
    g.scale = scale;
    g.rot_y = rot_y;
    g.mat = mat;
    g.valid = true;
    refresh_sdf_grid_cache(g);
    printf("  Loaded %s  N=%d  world pos (%.1f %.1f %.1f) scale %.1f\n",
           path, g.N, pos.x, pos.y, pos.z, scale);
    return g;
}

// Scalar SDF grid query (unchanged from cpurt4, used by shading code)
float sdf_grid_query(const SDFGrid &g, V3 wp)
{
    float dx = wp.x - g.pos.x, dz = wp.z - g.pos.z;
    float ca = g.ca, sa = g.sa;
    float lx = (ca * dx - sa * dz) * g.inv_scale;
    float ly = (wp.y - g.pos.y) * g.inv_scale;
    float lz = (sa * dx + ca * dz) * g.inv_scale;
    float N = (float)g.N;
    float tx = (lx - g.bmin[0]) / (g.bmax[0] - g.bmin[0]) * N;
    float ty = (ly - g.bmin[1]) / (g.bmax[1] - g.bmin[1]) * N;
    float tz = (lz - g.bmin[2]) / (g.bmax[2] - g.bmin[2]) * N;
    float ox = std::max(0.f, std::max((float)g.bmin[0] - lx, lx - (float)g.bmax[0]));
    float oy = std::max(0.f, std::max((float)g.bmin[1] - ly, ly - (float)g.bmax[1]));
    float oz = std::max(0.f, std::max((float)g.bmin[2] - lz, lz - (float)g.bmax[2]));
    float outside = sqrtf(ox * ox + oy * oy + oz * oz);
    if (outside > 0.f)
        return outside * g.scale;
    tx = clampf(tx, 0.f, N - 1.001f);
    ty = clampf(ty, 0.f, N - 1.001f);
    tz = clampf(tz, 0.f, N - 1.001f);
    int ix = (int)tx, iy = (int)ty, iz = (int)tz;
    float fx = tx - ix, fy = ty - iy, fz = tz - iz;
    int gN = g.N;
    auto at = [&](int x, int y, int z) -> float
    {
        x = std::max(0, std::min(gN - 1, x));
        y = std::max(0, std::min(gN - 1, y));
        z = std::max(0, std::min(gN - 1, z));
        return g.data[z * gN * gN + y * gN + x];
    };
    float v000 = at(ix, iy, iz), v100 = at(ix + 1, iy, iz);
    float v010 = at(ix, iy + 1, iz), v110 = at(ix + 1, iy + 1, iz);
    float v001 = at(ix, iy, iz + 1), v101 = at(ix + 1, iy, iz + 1);
    float v011 = at(ix, iy + 1, iz + 1), v111 = at(ix + 1, iy + 1, iz + 1);
    float v00 = v000 + (v100 - v000) * fx, v10 = v010 + (v110 - v010) * fx;
    float v01 = v001 + (v101 - v001) * fx, v11 = v011 + (v111 - v011) * fx;
    float v0 = v00 + (v10 - v00) * fy, v1 = v01 + (v11 - v01) * fy;
    return (v0 + (v1 - v0) * fz) * g.scale;
}

// ─── PROCEDURAL SCENE (scalar, used by shading / secondary rays) ─────────────
inline float hash21(float x, float y)
{
    float v = sinf(x * 127.1f + y * 311.7f) * 43758.5453f;
    return v - floorf(v);
}
inline float smoothnoise(float x, float y)
{
    float ix = floorf(x), iy = floorf(y);
    float fx = x - ix, fy = y - iy;
    float ux = fx * fx * (3 - 2 * fx), uy = fy * fy * (3 - 2 * fy);
    float a = hash21(ix, iy), b = hash21(ix + 1, iy);
    float c = hash21(ix, iy + 1), d = hash21(ix + 1, iy + 1);
    return a + (b - a) * ux + (c - a) * uy + (d - b + a - c) * ux * uy;
}
inline float fbm(float x, float y, int oct = 6)
{
    float v = 0, amp = 0.5f, freq = 1.f, tot = 0;
    for (int i = 0; i < oct; i++)
    {
        v += smoothnoise(x * freq, y * freq) * amp;
        tot += amp;
        amp *= 0.5f;
        freq *= 2.1f;
    }
    return v / tot;
}
float terrain_h(float x, float z)
{
    float h = fbm(x * 0.035f, z * 0.035f, 7) * 22.f;
    float r = sqrtf(x * x + z * z);
    h *= smoothstep(5.f, 30.f, r);
    h -= 4.f * smoothstep(30.f, 8.f, r);
    return h;
}

// Static tree positions. Bases are computed once instead of calling terrain_h()
// for every SDF sample that passes the horizontal cull.
static const float TREE_X[8] = {18.f, -22.f, 30.f, -15.f, 40.f, -35.f, 25.f, -28.f};
static const float TREE_Z[8] = {12.f, 20.f, -18.f, 35.f, -30.f, 10.f, 28.f, -10.f};
static float TREE_BASE[8] = {};
static void init_tree_bases()
{
    static bool done = false;
    if (done)
        return;
    for (int i = 0; i < 8; i++)
        TREE_BASE[i] = terrain_h(TREE_X[i], TREE_Z[i]);
    done = true;
}

enum Mat
{
    MAT_GRASS = 1,
    MAT_ROCK,
    MAT_SNOW,
    MAT_WATER,
    MAT_TREE_TRUNK,
    MAT_TREE_LEAF,
    MAT_MESH_STONE,
    MAT_MESH_GOLD,
    MAT_MESH_CRYSTAL
};

struct Hit
{
    float d;
    int mat;
};
inline Hit closer(Hit a, Hit b) { return a.d < b.d ? a : b; }

float sd_capsule(V3 p, V3 a, V3 b, float r)
{
    V3 ab = b - a, ap = p - a;
    float t = clampf(dot(ap, ab) / dot(ab, ab), 0, 1);
    return len(ap - ab * t) - r;
}

Hit eval_scene(V3 p)
{
    Hit best = {INF, MAT_GRASS};
    float r_lake = sqrtf(p.x * p.x + p.z * p.z);
    if (r_lake < 60.f)
    {
        float dw = p.y + 0.1f;
        dw += sinf(p.x * 0.8f) * 0.04f * smoothstep(60, 20, r_lake);
        dw += cosf(p.z * 0.9f) * 0.03f * smoothstep(60, 20, r_lake);
        best = closer(best, {dw, MAT_WATER});
    }
    float h = terrain_h(p.x, p.z);
    int tmat = h > 8.f ? MAT_SNOW : (h > 2.f && fbm(p.x * 0.3f, p.z * 0.3f, 2) < 0.42f ? MAT_ROCK : MAT_GRASS);
    best = closer(best, {(p.y - h) * 0.55f, tmat});
    init_tree_bases();
    for (int i = 0; i < 8; i++)
    {
        float dx = p.x - TREE_X[i], dz = p.z - TREE_Z[i];
        if (dx * dx + dz * dz > 64.f)
            continue;
        float th = TREE_BASE[i];
        V3 base = {TREE_X[i], th, TREE_Z[i]};
        V3 top = {TREE_X[i], th + 4.5f, TREE_Z[i]};
        float dtr = sd_capsule(p, base, top, 0.25f);
        best = closer(best, {dtr, MAT_TREE_TRUNK});
        for (int j = 0; j < 4; j++)
        {
            V3 lc = {p.x - TREE_X[i], p.y - (th + 2.7f + j * 1.1f), p.z - TREE_Z[i]};
            float wx = lc.x + sinf(lc.y * 2.3f + TREE_X[i]) * 0.15f;
            float wz = lc.z + cosf(lc.y * 2.3f + TREE_Z[i]) * 0.15f;
            float r = (2.5f - j * 0.45f);
            best = closer(best, {sqrtf(wx * wx + lc.y * lc.y + wz * wz) - r, MAT_TREE_LEAF});
        }
    }
    for (int gi = 0; gi < g_ngrid; gi++)
    {
        if (!g_grids[gi].valid)
            continue;
        float d = sdf_grid_query(g_grids[gi], p);
        best = closer(best, {d, g_grids[gi].mat});
    }
    return best;
}

// ─── NORMAL / MARCH / SHADOW / AO (scalar, unchanged) ────────────────────────
V3 scene_normal(V3 p)
{
    const float e = 0.003f;
    auto f = [](V3 q)
    { return eval_scene(q).d; };
    return norm({f({p.x + e, p.y, p.z}) - f({p.x - e, p.y, p.z}),
                 f({p.x, p.y + e, p.z}) - f({p.x, p.y - e, p.z}),
                 f({p.x, p.y, p.z + e}) - f({p.x, p.y, p.z - e})});
}
struct March
{
    V3 pos;
    int mat;
    bool hit;
    float t;
};
March ray_march(V3 ro, V3 rd)
{
    float t = 0.001f;
    for (int i = 0; i < MAX_STEPS && t < MAX_DIST; i++)
    {
        V3 p = {ro.x + rd.x * t, ro.y + rd.y * t, ro.z + rd.z * t};
        Hit h = eval_scene(p);
        if (h.d < SURF_DIST)
            return {p, h.mat, true, t};
        t += h.d;
    }
    return {{0, 0, 0}, 0, false, MAX_DIST};
}
float soft_shadow(V3 p, V3 ldir, float k = 14.f)
{
    float t = 0.05f, res = 1.f, ph = 1e20f;
    for (int i = 0; i < 96; i++)
    {
        V3 q = {p.x + ldir.x * t, p.y + ldir.y * t, p.z + ldir.z * t};
        float d = eval_scene(q).d;
        if (d < 0.001f)
            return 0.f;
        float y2 = d * d / (2.f * ph);
        float di = sqrtf(d * d - y2 * y2);
        res = std::min(res, k * di / std::max(0.001f, t - y2));
        ph = d;
        t += d;
        if (t > 60.f)
            break;
    }
    return clampf(res, 0, 1);
}
float calc_ao(V3 p, V3 n)
{
    float occ = 0, sc = 1;
    for (int i = 0; i < 6; i++)
    {
        float h2 = 0.03f + 0.08f * i;
        V3 sp = {p.x + n.x * h2, p.y + n.y * h2, p.z + n.z * h2};
        float d = eval_scene(sp).d;
        occ += (h2 - d) * sc;
        sc *= 0.75f;
    }
    return clampf(1 - 2.5f * occ, 0, 1);
}

static V3 sun_dir;

V3 sky_color(V3 rd)
{
    float sd = std::max(0.f, dot(rd, sun_dir));
    float y = std::max(0.f, rd.y);
    V3 sky = mix({0.5f, 0.65f, 0.85f}, {0.15f, 0.35f, 0.72f}, powf(y, 0.4f));
    sky = sky + V3{1.4f, 1.1f, 0.7f} * powf(sd, 512.f) * 8.f;
    sky = sky + V3{0.8f, 0.6f, 0.3f} * powf(sd, 8.f) * 0.3f;
    return sky;
}

struct MatProps
{
    V3 albedo;
    float roughness;
};
MatProps get_mat(int mat, V3 p)
{
    switch (mat)
    {
    case MAT_GRASS:
        return {{0.15f + fbm(p.x * .5f, p.z * .5f, 2) * .12f,
                 0.28f + fbm(p.x * .4f + 5, p.z * .4f, 2) * .1f, .07f},
                .95f};
    case MAT_ROCK:
    {
        float n = fbm(p.x * .8f, p.z * .8f, 3);
        return {{.3f + n * .1f, .28f + n * .08f, .25f + n * .06f}, .8f};
    }
    case MAT_SNOW:
        return {{.9f, .92f, .95f}, .55f};
    case MAT_WATER:
        return {{.04f, .1f, .2f}, .0f};
    case MAT_TREE_TRUNK:
        return {{.3f, .18f, .08f}, .95f};
    case MAT_TREE_LEAF:
    {
        float v = fbm(p.x * 2, p.z * 2, 2);
        return {{.08f + v * .06f, .28f + v * .1f, .06f}, .9f};
    }
    case MAT_MESH_STONE:
    {
        float n = fbm(p.x * 2.1f + 3, p.z * 2.1f + 7, 3) * .15f;
        return {{.45f + n, .42f + n, .38f + n}, .75f};
    }
    case MAT_MESH_GOLD:
        return {{.9f, .72f, .2f}, .15f};
    case MAT_MESH_CRYSTAL:
        return {{.5f, .8f, .95f}, .05f};
    default:
        return {{.5f, .5f, .5f}, .8f};
    }
}

// ─── PRNG ────────────────────────────────────────────────────────────────────
static uint32_t rng_state[64];
inline uint32_t rng_next(int tid)
{
    rng_state[tid] ^= rng_state[tid] << 13;
    rng_state[tid] ^= rng_state[tid] >> 17;
    rng_state[tid] ^= rng_state[tid] << 5;
    return rng_state[tid];
}
inline float rng_float(int tid) { return (rng_next(tid) >> 8) * 5.96e-8f; }
inline V3 rand_hemi(V3 n, int tid)
{
    for (;;)
    {
        float a = rng_float(tid) * 2 - 1, b = rng_float(tid) * 2 - 1, c = rng_float(tid) * 2 - 1;
        float l2 = a * a + b * b + c * c;
        if (l2 > 1e-4f && l2 < 1.f)
        {
            V3 v = norm({a, b, c});
            return dot(v, n) < 0 ? V3{-v.x, -v.y, -v.z} : v;
        }
    }
}

// ─── SHADING (scalar, operates on a known primary hit) ───────────────────────
// shade_known_hit: same logic as the old shade_hit but takes the pre-marched
// primary result instead of re-marching it internally.
// precomp_n: if non-null, used as the primary surface normal instead of calling
// scene_normal() — the SIMD shade_batch paths pass this in to avoid a redundant
// 6-eval normal computation for water and crystal lanes.
V3 shade_known_hit(V3 ro, V3 rd, March m, int tid, const V3 *precomp_n = nullptr)
{
    if (!m.hit)
        return sky_color(rd);

    if (m.mat == MAT_WATER)
    {
        V3 n = precomp_n ? *precomp_n : scene_normal(m.pos);
        float ndv = std::max(0.f, dot(n, -1.f * rd));
        float fresnel = 0.04f + 0.96f * powf(1 - ndv, 5.f);
        V3 refl_dir = reflect(rd, n);
        V3 refl_col = sky_color(refl_dir);
        March rm = ray_march(m.pos + n * 0.01f, refl_dir);
        if (rm.hit && rm.mat != MAT_WATER)
        {
            MatProps mp2 = get_mat(rm.mat, rm.pos);
            V3 rn = scene_normal(rm.pos);
            float d2 = std::max(0.f, dot(rn, sun_dir));
            float sh2 = d2 > 0.01f ? soft_shadow(rm.pos + rn * 0.02f, sun_dir, 10.f) : 0;
            refl_col = mp2.albedo * (d2 * sh2 * 1.5f + 0.15f);
        }
        float depth_f = smoothstep(0, 3.f, fabsf(terrain_h(m.pos.x, m.pos.z)));
        V3 water_col = mix({0.1f, .3f, .4f}, {.01f, .08f, .18f}, depth_f);
        float caus = powf(std::max(0.f, sinf(m.pos.x * 2.3f) * sinf(m.pos.z * 1.9f)), 4.f) * 0.5f;
        water_col = water_col + V3{.1f, .15f, .1f} * caus;
        V3 col = mix(water_col, refl_col, fresnel);
        V3 half_v = norm(sun_dir - rd);
        float spec = powf(std::max(0.f, dot(n, half_v)), 256.f) * 2.f;
        return col + V3{1.4f, 1.2f, .8f} * spec;
    }

    V3 n = precomp_n ? *precomp_n : scene_normal(m.pos);
    MatProps mp = get_mat(m.mat, m.pos);
    float diff = std::max(0.f, dot(n, sun_dir));
    float sha = diff > 0.01f ? soft_shadow(m.pos + n * 0.02f, sun_dir, 12.f) : 0.f;
    float ao = calc_ao(m.pos, n);
    float sky_amb = 0.5f + 0.5f * n.y;

    if (m.mat == MAT_MESH_CRYSTAL)
    {
        V3 neg_rd = {-rd.x, -rd.y, -rd.z};
        float rim = powf(1.f - std::max(0.f, dot(n, neg_rd)), 3.f);
        V3 col = mp.albedo * (diff * sha * 1.2f + 0.3f) + V3{.8f, .95f, 1.f} * rim * 0.6f;
        V3 half_v = norm(sun_dir - rd);
        float spec = powf(std::max(0.f, dot(n, half_v)), 128.f) * 2.f;
        return col + V3{1, 1, 1} * spec * sha;
    }

    V3 gi = {0, 0, 0};
    V3 gi_dir = rand_hemi(n, tid);
    March gm = ray_march(m.pos + n * 0.02f, gi_dir);
    if (gm.hit && gm.mat != MAT_WATER)
    {
        MatProps gmp = get_mat(gm.mat, gm.pos);
        float gdiff = std::max(0.f, dot(scene_normal(gm.pos), sun_dir)) * 0.5f;
        gi = gmp.albedo * gdiff;
    }
    else if (!gm.hit)
    {
        gi = sky_color(gi_dir) * 0.3f;
    }
    gi = gi * dot(n, gi_dir) * 0.4f;

    V3 sun_col = {1.4f, 1.15f, .8f};
    V3 amb_col = {.15f, .2f, .35f};
    V3 col = mp.albedo * (sun_col * diff * sha + amb_col * sky_amb * ao + gi);

    if (mp.roughness < 0.5f)
    {
        float gloss = 1.f - mp.roughness;
        V3 half_v = norm(sun_dir - rd);
        float spec = powf(std::max(0.f, dot(n, half_v)), 4.f / (gloss * gloss + .001f)) * gloss * .8f;
        col = col + sun_col * spec * sha;
    }
    return col;
}

V3 apply_fog(V3 col, float dist, V3 rd)
{
    float fog = 1.f - expf(-dist * .004f);
    float sd = std::max(0.f, dot(rd, sun_dir));
    V3 fog_col = {.55f + sd * .3f, .6f + sd * .15f, .72f};
    return mix(col, fog_col, fog);
}

// ─── AVX-512 SECTION ─────────────────────────────────────────────────────────
#ifdef __AVX512F__

// ── Fast sin(x) for all real x ───────────────────────────────────────────────
// Polynomial minimax on [-pi, pi] after range reduction.
// Accuracy: ~5-6 digits — good enough for the hash/noise functions.
// Still used by tree warp (kept for the full-range case).
static inline __m512 avx_sin(__m512 x)
{
    // Range reduce: x = x - round(x / 2π) * 2π
    const __m512 inv_2pi = _mm512_set1_ps(0.15915494f); // 1/(2π)
    const __m512 two_pi = _mm512_set1_ps(6.28318530f);
    __m512 k = _mm512_roundscale_ps(
        _mm512_mul_ps(x, inv_2pi),
        _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    x = _mm512_fnmadd_ps(k, two_pi, x); // x -= k * 2π

    // Horner: sin(x) ≈ x*(1 + x²*(-1/6 + x²*(1/120 + x²*(-1/5040))))
    __m512 x2 = _mm512_mul_ps(x, x);
    __m512 r = _mm512_fmadd_ps(x2, _mm512_set1_ps(-1.f / 5040.f), _mm512_set1_ps(1.f / 120.f));
    r = _mm512_fmadd_ps(x2, r, _mm512_set1_ps(-1.f / 6.f));
    r = _mm512_fmadd_ps(x2, r, _mm512_set1_ps(1.f));
    return _mm512_mul_ps(x, r);
}

// ── avx_sin_small: 3-term sin for aesthetic use (water ripple, tree warp) ────
// Same range reduction as avx_sin, but drops the x^6 term (-1/5040).
// Max error after range reduction: ~5e-4 at x=±π — imperceptible for ripples/warp.
// Saves one FMA vs avx_sin: 3 FMAs for the polynomial instead of 4.
static inline __m512 avx_sin_small(__m512 x)
{
    const __m512 inv_2pi = _mm512_set1_ps(0.15915494f);
    const __m512 two_pi = _mm512_set1_ps(6.28318530f);
    __m512 k = _mm512_roundscale_ps(_mm512_mul_ps(x, inv_2pi),
                                    _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    x = _mm512_fnmadd_ps(k, two_pi, x);
    // Horner: sin(x) ≈ x*(1 + x²*(-1/6 + x²*(1/120)))  — 3-term, no x^6
    __m512 x2 = _mm512_mul_ps(x, x);
    __m512 r = _mm512_fmadd_ps(x2, _mm512_set1_ps(1.f / 120.f), _mm512_set1_ps(-1.f / 6.f));
    r = _mm512_fmadd_ps(x2, r, _mm512_set1_ps(1.f));
    return _mm512_mul_ps(x, r);
}

// ── Vectorized noise primitives ───────────────────────────────────────────────
// Integer hash replaces avx_sin-based hash21 — ~6× faster, same visual quality.
// PCG-style: mix the two integer-grid coords with large primes, then bit-scramble.
static inline __m512 avx_hash21(__m512 x, __m512 y)
{
    // Reinterpret float bits as int (grid coords are always exact integers here)
    __m512i ix = _mm512_cvttps_epi32(x);
    __m512i iy = _mm512_cvttps_epi32(y);
    // Combine: h = ix * 1664525 + iy * 22695477 + 1013904223  (LCG mix)
    __m512i h = _mm512_add_epi32(
        _mm512_add_epi32(
            _mm512_mullo_epi32(ix, _mm512_set1_epi32(1664525)),
            _mm512_mullo_epi32(iy, _mm512_set1_epi32(22695477))),
        _mm512_set1_epi32(1013904223));
    // Xorshift scramble
    h = _mm512_xor_epi32(h, _mm512_srli_epi32(h, 16));
    h = _mm512_mullo_epi32(h, _mm512_set1_epi32(0x45d9f3b));
    h = _mm512_xor_epi32(h, _mm512_srli_epi32(h, 16));
    // Map unsigned 32-bit to [0,1): keep upper 23 bits as mantissa of float in [1,2), subtract 1
    __m512i mantissa = _mm512_or_epi32(_mm512_srli_epi32(h, 9), _mm512_set1_epi32(0x3f800000));
    return _mm512_sub_ps(_mm512_castsi512_ps(mantissa), _mm512_set1_ps(1.f));
}

static inline __m512 avx_smoothnoise(__m512 x, __m512 y)
{
    __m512 ix = _mm512_floor_ps(x), iy = _mm512_floor_ps(y);
    __m512 fx = _mm512_sub_ps(x, ix), fy = _mm512_sub_ps(y, iy);
    // Smooth step: t*t*(3-2*t)
    __m512 ux = _mm512_mul_ps(fx, _mm512_mul_ps(fx,
                                                _mm512_fnmadd_ps(fx, _mm512_set1_ps(2.f), _mm512_set1_ps(3.f))));
    __m512 uy = _mm512_mul_ps(fy, _mm512_mul_ps(fy,
                                                _mm512_fnmadd_ps(fy, _mm512_set1_ps(2.f), _mm512_set1_ps(3.f))));
    __m512 one = _mm512_set1_ps(1.f);
    __m512 a = avx_hash21(ix, iy);
    __m512 b = avx_hash21(_mm512_add_ps(ix, one), iy);
    __m512 c = avx_hash21(ix, _mm512_add_ps(iy, one));
    __m512 d = avx_hash21(_mm512_add_ps(ix, one), _mm512_add_ps(iy, one));
    // Bilinear: lerp(lerp(a,b,ux), lerp(c,d,ux), uy)
    __m512 ab = _mm512_fmadd_ps(_mm512_sub_ps(b, a), ux, a);
    __m512 cd = _mm512_fmadd_ps(_mm512_sub_ps(d, c), ux, c);
    return _mm512_fmadd_ps(_mm512_sub_ps(cd, ab), uy, ab);
}

static inline __m512 avx_fbm(__m512 x, __m512 y, int oct = 6)
{
    // Precomputed reciprocal of total amplitude for each octave count.
    // tot[n] = sum_{i=0}^{n-1} 0.5^(i+1) = 1 - 0.5^n
    // inv_tot replaces expensive _mm512_div_ps(v, tot) with a multiply.
    static const float inv_tot_lut[] = {
        0.f,       // oct=0 (unused)
        2.000000f, // oct=1: tot=0.5
        1.333333f, // oct=2: tot=0.75
        1.142857f, // oct=3: tot=0.875
        1.066667f, // oct=4: tot=0.9375
        1.032258f, // oct=5: tot=0.96875
        1.016129f, // oct=6: tot=0.984375
        1.007874f, // oct=7: tot=0.9921875
    };
    __m512 v = _mm512_setzero_ps();
    __m512 amp = _mm512_set1_ps(0.5f);
    __m512 freq = _mm512_set1_ps(1.f);
    for (int i = 0; i < oct; i++)
    {
        v = _mm512_fmadd_ps(avx_smoothnoise(_mm512_mul_ps(x, freq),
                                            _mm512_mul_ps(y, freq)),
                            amp, v);
        amp = _mm512_mul_ps(amp, _mm512_set1_ps(0.5f));
        freq = _mm512_mul_ps(freq, _mm512_set1_ps(2.1f));
    }
    return _mm512_mul_ps(v, _mm512_set1_ps(inv_tot_lut[oct < 8 ? oct : 7]));
}

// Vectorized terrain height for 16 (x,z) pairs.
// oct: FBM octave count — 7 for primary rays (full quality), 4 for shadow/GI/AO.
static inline __m512 avx_terrain_h(__m512 x, __m512 z, int oct = 7)
{
    __m512 h = _mm512_mul_ps(
        avx_fbm(_mm512_mul_ps(x, _mm512_set1_ps(0.035f)),
                _mm512_mul_ps(z, _mm512_set1_ps(0.035f)), oct),
        _mm512_set1_ps(22.f));
    // rsqrt14-based fast sqrt: r ≈ r2 * rsqrt14(r2).  14-bit accuracy is plenty for smoothstep.
    __m512 r2 = _mm512_fmadd_ps(x, x, _mm512_mul_ps(z, z));
    __m512 r = _mm512_mul_ps(r2, _mm512_rsqrt14_ps(_mm512_max_ps(r2, _mm512_set1_ps(1e-18f))));

    // smoothstep(5, 30, r) → multiply h
    __m512 t1 = _mm512_div_ps(_mm512_sub_ps(r, _mm512_set1_ps(5.f)), _mm512_set1_ps(25.f));
    t1 = _mm512_min_ps(_mm512_max_ps(t1, _mm512_setzero_ps()), _mm512_set1_ps(1.f));
    t1 = _mm512_mul_ps(t1, _mm512_mul_ps(t1,
                                         _mm512_fnmadd_ps(t1, _mm512_set1_ps(2.f), _mm512_set1_ps(3.f))));
    h = _mm512_mul_ps(h, t1);

    // smoothstep(30, 8, r) → subtract 4*
    __m512 t2 = _mm512_div_ps(_mm512_sub_ps(r, _mm512_set1_ps(30.f)),
                              _mm512_set1_ps(8.f - 30.f)); // (r-30)/(8-30)
    t2 = _mm512_min_ps(_mm512_max_ps(t2, _mm512_setzero_ps()), _mm512_set1_ps(1.f));
    t2 = _mm512_mul_ps(t2, _mm512_mul_ps(t2,
                                         _mm512_fnmadd_ps(t2, _mm512_set1_ps(2.f), _mm512_set1_ps(3.f))));
    h = _mm512_fnmadd_ps(_mm512_set1_ps(4.f), t2, h); // h -= 4*t2
    return h;
}

// ── AVX-512 SDF grid query for 16 world positions at once ─────────────────────
// Returns signed distance in world units (16 floats).
static __m512 sdf_grid_query_avx512(const SDFGrid &g,
                                    __m512 wpx, __m512 wpy, __m512 wpz)
{
    // Precompute rotation constants (scalar, same for all 16)
    float ca = cosf(-g.rot_y), sa = sinf(-g.rot_y);
    __m512 vca = _mm512_set1_ps(ca), vsa = _mm512_set1_ps(sa);
    __m512 inv_s = _mm512_set1_ps(1.f / g.scale);

    // Transform world → local object coords
    __m512 dx = _mm512_sub_ps(wpx, _mm512_set1_ps(g.pos.x));
    __m512 dz = _mm512_sub_ps(wpz, _mm512_set1_ps(g.pos.z));
    __m512 lx = _mm512_mul_ps(_mm512_fmsub_ps(vca, dx, _mm512_mul_ps(vsa, dz)), inv_s);
    __m512 ly = _mm512_mul_ps(_mm512_sub_ps(wpy, _mm512_set1_ps(g.pos.y)), inv_s);
    __m512 lz = _mm512_mul_ps(_mm512_fmadd_ps(vsa, dx, _mm512_mul_ps(vca, dz)), inv_s);

    // Distance to bounding box for outside points
    __m512 bx0 = _mm512_set1_ps(g.bmin[0]), bx1 = _mm512_set1_ps(g.bmax[0]);
    __m512 by0 = _mm512_set1_ps(g.bmin[1]), by1 = _mm512_set1_ps(g.bmax[1]);
    __m512 bz0 = _mm512_set1_ps(g.bmin[2]), bz1 = _mm512_set1_ps(g.bmax[2]);
    __m512 zero = _mm512_setzero_ps();
    __m512 ox = _mm512_max_ps(zero, _mm512_max_ps(_mm512_sub_ps(bx0, lx), _mm512_sub_ps(lx, bx1)));
    __m512 oy = _mm512_max_ps(zero, _mm512_max_ps(_mm512_sub_ps(by0, ly), _mm512_sub_ps(ly, by1)));
    __m512 oz = _mm512_max_ps(zero, _mm512_max_ps(_mm512_sub_ps(bz0, lz), _mm512_sub_ps(lz, bz1)));
    __m512 out_r2 = _mm512_fmadd_ps(ox, ox, _mm512_fmadd_ps(oy, oy, _mm512_mul_ps(oz, oz)));
    __m512 outside = _mm512_mul_ps(out_r2, _mm512_rsqrt14_ps(_mm512_max_ps(out_r2, _mm512_set1_ps(1e-18f))));
    __mmask16 outside_mask = _mm512_cmp_ps_mask(outside, zero, _CMP_GT_OQ);

    // Map local coords → grid cell coordinates [0, N)
    float fN = (float)g.N;
    __m512 vN = _mm512_set1_ps(fN);
    __m512 tx = _mm512_div_ps(_mm512_mul_ps(_mm512_sub_ps(lx, bx0), vN), _mm512_sub_ps(bx1, bx0));
    __m512 ty = _mm512_div_ps(_mm512_mul_ps(_mm512_sub_ps(ly, by0), vN), _mm512_sub_ps(by1, by0));
    __m512 tz = _mm512_div_ps(_mm512_mul_ps(_mm512_sub_ps(lz, bz0), vN), _mm512_sub_ps(bz1, bz0));
    __m512 max_idx = _mm512_set1_ps(fN - 1.001f);
    tx = _mm512_min_ps(_mm512_max_ps(tx, zero), max_idx);
    ty = _mm512_min_ps(_mm512_max_ps(ty, zero), max_idx);
    tz = _mm512_min_ps(_mm512_max_ps(tz, zero), max_idx);

    // Integer cell indices and fractional parts
    __m512i ix = _mm512_cvttps_epi32(tx);
    __m512i iy = _mm512_cvttps_epi32(ty);
    __m512i iz = _mm512_cvttps_epi32(tz);
    __m512 fx = _mm512_sub_ps(tx, _mm512_cvtepi32_ps(ix));
    __m512 fy = _mm512_sub_ps(ty, _mm512_cvtepi32_ps(iy));
    __m512 fz = _mm512_sub_ps(tz, _mm512_cvtepi32_ps(iz));

    // Clamp indices so ix+1, iy+1, iz+1 are always valid
    int gN = g.N;
    __m512i vgN = _mm512_set1_epi32(gN);
    __m512i vgN2 = _mm512_set1_epi32(gN * gN);
    __m512i vgN1 = _mm512_set1_epi32(gN - 1);
    __m512i vzero_i = _mm512_setzero_si512();
    __m512i one_i = _mm512_set1_epi32(1);
    ix = _mm512_min_epi32(_mm512_max_epi32(ix, vzero_i), vgN1);
    iy = _mm512_min_epi32(_mm512_max_epi32(iy, vzero_i), vgN1);
    iz = _mm512_min_epi32(_mm512_max_epi32(iz, vzero_i), vgN1);
    __m512i ix1 = _mm512_min_epi32(_mm512_add_epi32(ix, one_i), vgN1);
    __m512i iy1 = _mm512_min_epi32(_mm512_add_epi32(iy, one_i), vgN1);
    __m512i iz1 = _mm512_min_epi32(_mm512_add_epi32(iz, one_i), vgN1);

    // Flatten 3-D index: i = iz*N² + iy*N + ix
    // Using mullo_epi32 (available in AVX-512F via _mm512_mullo_epi32)
    auto idx3 = [&](__m512i zi, __m512i yi, __m512i xi) -> __m512i
    {
        return _mm512_add_epi32(
            _mm512_add_epi32(_mm512_mullo_epi32(zi, vgN2),
                             _mm512_mullo_epi32(yi, vgN)),
            xi);
    };
    __m512i i000 = idx3(iz, iy, ix);
    __m512i i100 = idx3(iz, iy, ix1);
    __m512i i010 = idx3(iz, iy1, ix);
    __m512i i110 = idx3(iz, iy1, ix1);
    __m512i i001 = idx3(iz1, iy, ix);
    __m512i i101 = idx3(iz1, iy, ix1);
    __m512i i011 = idx3(iz1, iy1, ix);
    __m512i i111 = idx3(iz1, iy1, ix1);

    // Gather the 8 trilinear corners (scale = 4 bytes per float)
    const float *base = g.data;
    __m512 v000 = _mm512_i32gather_ps(i000, base, 4);
    __m512 v100 = _mm512_i32gather_ps(i100, base, 4);
    __m512 v010 = _mm512_i32gather_ps(i010, base, 4);
    __m512 v110 = _mm512_i32gather_ps(i110, base, 4);
    __m512 v001 = _mm512_i32gather_ps(i001, base, 4);
    __m512 v101 = _mm512_i32gather_ps(i101, base, 4);
    __m512 v011 = _mm512_i32gather_ps(i011, base, 4);
    __m512 v111 = _mm512_i32gather_ps(i111, base, 4);

    // Trilinear interpolation via lerp = a + (b-a)*t
#define LERP(a, b, t) _mm512_fmadd_ps(_mm512_sub_ps(b, a), t, a)
    __m512 v00 = LERP(v000, v100, fx);
    __m512 v10 = LERP(v010, v110, fx);
    __m512 v01 = LERP(v001, v101, fx);
    __m512 v11 = LERP(v011, v111, fx);
    __m512 v0 = LERP(v00, v10, fy);
    __m512 v1 = LERP(v01, v11, fy);
    __m512 d = LERP(v0, v1, fz);
#undef LERP

    // Scale back to world units
    __m512 scale_v = _mm512_set1_ps(g.scale);
    d = _mm512_mul_ps(d, scale_v);

    // Blend: outside bbox → outside*scale  |  inside → trilinear grid value
    // _mm512_mask_blend_ps(k, a, b): k=1 → b, k=0 → a
    return _mm512_mask_blend_ps(outside_mask, d, _mm512_mul_ps(outside, scale_v));
}

// ── Vectorized eval_scene for 16 positions ────────────────────────────────────
struct EvalResult16
{
    __m512 d;    // best signed distance
    __m512i mat; // material at best distance
};

// terrain_oct: 7 for primary/normal rays, 4 for shadow/GI/AO (huge speedup, visually identical).
static EvalResult16 eval_scene_avx512(__m512 px, __m512 py, __m512 pz, int terrain_oct = 7)
{
    __m512 best_d = _mm512_set1_ps(INF);
    __m512i best_mat = _mm512_set1_epi32(MAT_GRASS);

    // Helper: update best if prim_d < best_d
    auto update = [&](__m512 prim_d, int mat_id)
    {
        __mmask16 closer = _mm512_cmp_ps_mask(prim_d, best_d, _CMP_LT_OQ);
        best_d = _mm512_min_ps(best_d, prim_d);
        best_mat = _mm512_mask_blend_epi32(closer, best_mat, _mm512_set1_epi32(mat_id));
    };

    // ── Water plane ────────────────────────────────────────────────────────
    __m512 lake_r2 = _mm512_fmadd_ps(px, px, _mm512_mul_ps(pz, pz));
    __m512 r_lake = _mm512_mul_ps(lake_r2, _mm512_rsqrt14_ps(_mm512_max_ps(lake_r2, _mm512_set1_ps(1e-18f))));
    __mmask16 in_lake = _mm512_cmp_ps_mask(r_lake, _mm512_set1_ps(60.f), _CMP_LT_OQ);
    if (in_lake)
    {
        __m512 dw = _mm512_add_ps(py, _mm512_set1_ps(0.1f));
        // Small wave ripples
        __m512 wave_x = _mm512_mul_ps(avx_sin_small(_mm512_mul_ps(px, _mm512_set1_ps(0.8f))),
                                      _mm512_set1_ps(0.04f));
        __m512 wave_z = _mm512_mul_ps(avx_sin_small(_mm512_mul_ps(pz, _mm512_set1_ps(0.9f))),
                                      _mm512_set1_ps(0.03f));
        // Fade waves at lake edge: smoothstep(60,20,r_lake) — just use 1 for simplicity inside lake
        dw = _mm512_add_ps(dw, _mm512_add_ps(wave_x, wave_z));
        // Only apply water where in_lake
        __m512 safe_dw = _mm512_mask_blend_ps(in_lake, _mm512_set1_ps(INF), dw);
        update(safe_dw, MAT_WATER);
    }

    // ── Terrain ────────────────────────────────────────────────────────────
    {
        __m512 h = avx_terrain_h(px, pz, terrain_oct);
        __m512 d_terrain = _mm512_mul_ps(_mm512_sub_ps(py, h), _mm512_set1_ps(0.55f));

        // Vectorized material selection (mirrors scalar logic exactly):
        //   h>8   → MAT_SNOW
        //   h>2 && fbm(x*0.3,z*0.3,2)<0.42 → MAT_ROCK
        //   else  → MAT_GRASS
        __m512i tmat = _mm512_set1_epi32(MAT_GRASS);
        __mmask16 is_snow = _mm512_cmp_ps_mask(h, _mm512_set1_ps(8.f), _CMP_GT_OQ);
        __mmask16 above_rock_h = _mm512_cmp_ps_mask(h, _mm512_set1_ps(2.f), _CMP_GT_OQ);
        __m512 rock_noise = avx_fbm(_mm512_mul_ps(px, _mm512_set1_ps(0.3f)),
                                    _mm512_mul_ps(pz, _mm512_set1_ps(0.3f)), 2);
        __mmask16 is_rock = above_rock_h &
                            _mm512_cmp_ps_mask(rock_noise, _mm512_set1_ps(0.42f), _CMP_LT_OQ);
        tmat = _mm512_mask_blend_epi32(is_rock, tmat, _mm512_set1_epi32(MAT_ROCK));
        tmat = _mm512_mask_blend_epi32(is_snow, tmat, _mm512_set1_epi32(MAT_SNOW));

        // Per-lane update (can't use uniform update() helper — material varies per lane)
        __mmask16 terrain_closer = _mm512_cmp_ps_mask(d_terrain, best_d, _CMP_LT_OQ);
        best_d = _mm512_min_ps(best_d, d_terrain);
        best_mat = _mm512_mask_blend_epi32(terrain_closer, best_mat, tmat);
    }

    // ── Trees (8 capsule trunks) ────────────────────────────────────────────
    {
        init_tree_bases();
        for (int i = 0; i < 8; i++)
        {
            float ti_x = TREE_X[i], ti_z = TREE_Z[i];
            // Skip trees far from all 16 positions (coarse cull per-tree)
            // We do a horizontal distance test using the min of all 16 positions
            __m512 dxi = _mm512_sub_ps(px, _mm512_set1_ps(ti_x));
            __m512 dzi = _mm512_sub_ps(pz, _mm512_set1_ps(ti_z));
            __m512 dist2 = _mm512_fmadd_ps(dxi, dxi, _mm512_mul_ps(dzi, dzi));
            float min_dist2 = _mm512_reduce_min_ps(dist2);
            if (min_dist2 > 100.f)
                continue; // whole tile is far from this tree

            float th = TREE_BASE[i]; // scalar (called once per tree per tile)
            // Capsule: base=(ti_x, th, ti_z)  top=(ti_x, th+4.5, ti_z)  r=0.25
            // sd_capsule vectorized: p-a=(dxi, py-th, dzi), ab=(0, 4.5, 0)
            __m512 ap_y = _mm512_sub_ps(py, _mm512_set1_ps(th));
            float ab_ab = 4.5f * 4.5f;
            // dot(ap, ab) = ap_y * 4.5
            __m512 t_cap = _mm512_mul_ps(ap_y, _mm512_set1_ps(4.5f / ab_ab));
            t_cap = _mm512_min_ps(_mm512_max_ps(t_cap, _mm512_setzero_ps()), _mm512_set1_ps(1.f));
            __m512 cl_y = _mm512_fmadd_ps(t_cap, _mm512_set1_ps(4.5f), _mm512_set1_ps(th));
            __m512 ey = _mm512_sub_ps(py, cl_y);
            // rsqrt14-based sqrt: safe underestimate for SDF marching
            __m512 trunk_r2 = _mm512_fmadd_ps(dxi, dxi, _mm512_fmadd_ps(ey, ey, _mm512_mul_ps(dzi, dzi)));
            __m512 d_trunk = _mm512_sub_ps(
                _mm512_mul_ps(trunk_r2, _mm512_rsqrt14_ps(_mm512_max_ps(trunk_r2, _mm512_set1_ps(1e-18f)))),
                _mm512_set1_ps(0.25f));
            update(d_trunk, MAT_TREE_TRUNK);

            // 4 foliage spheres (warped spheres)
            for (int j = 0; j < 4; j++)
            {
                float cy = th + 2.7f + j * 1.1f;
                float r_leaf = 2.5f - j * 0.45f;
                __m512 ly = _mm512_sub_ps(py, _mm512_set1_ps(cy));
                // warp: wx = dx + sin(ly*2.3+ti_x)*0.15, wz = dz + cos(ly*2.3+ti_z)*0.15
                __m512 warp_arg = _mm512_fmadd_ps(ly, _mm512_set1_ps(2.3f), _mm512_set1_ps((float)(i * 1.0)));
                __m512 wx = _mm512_fmadd_ps(avx_sin_small(warp_arg), _mm512_set1_ps(0.15f), dxi);
                // For wz, use cos = sin(arg + pi/2)
                __m512 cos_arg = _mm512_add_ps(warp_arg, _mm512_set1_ps(1.5707963f));
                __m512 wz = _mm512_fmadd_ps(avx_sin_small(cos_arg), _mm512_set1_ps(0.15f), dzi);
                __m512 leaf_r2 = _mm512_fmadd_ps(wx, wx, _mm512_fmadd_ps(ly, ly, _mm512_mul_ps(wz, wz)));
                __m512 d_leaf = _mm512_sub_ps(
                    _mm512_mul_ps(leaf_r2, _mm512_rsqrt14_ps(_mm512_max_ps(leaf_r2, _mm512_set1_ps(1e-18f)))),
                    _mm512_set1_ps(r_leaf));
                update(d_leaf, MAT_TREE_LEAF);
            }
        }
    }

    // ── Baked SDF grid objects ──────────────────────────────────────────────
    // BVH pre-rejection: skip grid entirely if the closest lane's x/y/z is
    // farther than the current best distance from the world-space bbox.
    // Uses the min/max of the 16 lane coords vs the grid's world-space bbox.
    // A grid that is already farther than best_d contributes nothing.
    //
    // Prefetch: before the _mm512_i32gather_ps calls for grid gi, we issue
    // software prefetches for grid gi+1's data start.  The gather has ~100
    // cycle latency on most microarchitectures; prefetching one grid ahead
    // hides that latency behind the current grid's trilinear interpolation.
    for (int gi = 0; gi < g_ngrid; gi++)
    {
        if (!g_grids[gi].valid)
            continue;
        const SDFGrid &g = g_grids[gi];
        float dx_ = g.bmax[0] - g.bmin[0], dy_ = g.bmax[1] - g.bmin[1], dz_ = g.bmax[2] - g.bmin[2];
        float half_diag = sqrtf(dx_ * dx_ + dy_ * dy_ + dz_ * dz_) * 0.5f * g.scale;
        float cx_ = (g.bmin[0] + g.bmax[0]) * 0.5f * g.scale + g.pos.x;
        float cy_ = (g.bmin[1] + g.bmax[1]) * 0.5f * g.scale + g.pos.y;
        float cz_ = (g.bmin[2] + g.bmax[2]) * 0.5f * g.scale + g.pos.z;
        __m512 gcx = _mm512_set1_ps(cx_), gcy = _mm512_set1_ps(cy_), gcz = _mm512_set1_ps(cz_);
        __m512 edx = _mm512_sub_ps(px, gcx), edy = _mm512_sub_ps(py, gcy), edz = _mm512_sub_ps(pz, gcz);
        __m512 ed2 = _mm512_fmadd_ps(edx, edx, _mm512_fmadd_ps(edy, edy, _mm512_mul_ps(edz, edz)));
        float min_ed2 = _mm512_reduce_min_ps(ed2);
        float min_dist_to_sphere = sqrtf(min_ed2) - half_diag;
        float cur_best = _mm512_reduce_min_ps(best_d);
        if (min_dist_to_sphere > cur_best)
            continue;

        // Prefetch the next valid grid's data into L2 before we do the gather
        // for this grid.  _mm_prefetch T1 = prefetch to L2 (not L1, to avoid
        // evicting FBM data that's still hot).  We prefetch the first few cache
        // lines (the header + first z-slice corner) which is where the gather
        // indices are most likely to land for nearby tiles.
        for (int ngi = gi + 1; ngi < g_ngrid; ngi++)
        {
            if (!g_grids[ngi].valid)
                continue;
            const char *nbase = reinterpret_cast<const char *>(g_grids[ngi].data);
            int n3 = g_grids[ngi].N * g_grids[ngi].N * g_grids[ngi].N;
            // Prefetch first 4 cache lines (256 bytes) and middle of the grid
            _mm_prefetch(nbase, _MM_HINT_T1);
            _mm_prefetch(nbase + 64, _MM_HINT_T1);
            _mm_prefetch(nbase + 128, _MM_HINT_T1);
            _mm_prefetch(nbase + 192, _MM_HINT_T1);
            // Mid-grid hint (helps when the hit point is near the grid centre)
            _mm_prefetch(nbase + (n3 / 2) * 4, _MM_HINT_T2);
            break; // only prefetch the next grid
        }

        __m512 d = sdf_grid_query_avx512(g, px, py, pz);
        update(d, g.mat);
    }

    return {best_d, best_mat};
}

// ── 16-ray batch marcher ──────────────────────────────────────────────────────
struct March16
{
    __m512 pos_x, pos_y, pos_z;
    __m512i mat;
    __mmask16 hit_mask;
    __m512 t;
};

static March16 ray_march_avx512(
    __m512 rox, __m512 roy, __m512 roz,
    __m512 rdx, __m512 rdy, __m512 rdz,
    int terrain_oct = 7, float t_max = MAX_DIST)
{
    __m512 t = _mm512_set1_ps(0.001f);
    __mmask16 active = 0xFFFF;
    __mmask16 hit_mask = 0;

    __m512 hit_x = _mm512_setzero_ps();
    __m512 hit_y = _mm512_setzero_ps();
    __m512 hit_z = _mm512_setzero_ps();
    __m512i hit_mat = _mm512_setzero_si512();

    const __m512 surf_dist_v = _mm512_set1_ps(SURF_DIST);
    const __m512 max_dist_v = _mm512_set1_ps(t_max);

    for (int i = 0; i < MAX_STEPS && active; i++)
    {
        // Current ray positions
        __m512 px = _mm512_fmadd_ps(rdx, t, rox);
        __m512 py = _mm512_fmadd_ps(rdy, t, roy);
        __m512 pz = _mm512_fmadd_ps(rdz, t, roz);

        EvalResult16 res = eval_scene_avx512(px, py, pz, terrain_oct);

        // Advance active rays by their SDF distance
        t = _mm512_mask_add_ps(t, active, t, res.d);

        // Detect hits: d < SURF_DIST (only check active rays)
        __mmask16 just_hit = active & _mm512_cmp_ps_mask(res.d, surf_dist_v, _CMP_LT_OQ);
        if (just_hit)
        {
            hit_x = _mm512_mask_blend_ps(just_hit, hit_x, px);
            hit_y = _mm512_mask_blend_ps(just_hit, hit_y, py);
            hit_z = _mm512_mask_blend_ps(just_hit, hit_z, pz);
            hit_mat = _mm512_mask_blend_epi32(just_hit, hit_mat, res.mat);
            hit_mask |= just_hit;
            active &= ~just_hit;
        }

        // Retire rays that escaped
        active &= ~(_mm512_cmp_ps_mask(t, max_dist_v, _CMP_GE_OQ));
    }

    return {hit_x, hit_y, hit_z, hit_mat, hit_mask, t};
}

// ── AVX-512 soft-shadow for 16 rays at once ──────────────────────────────────
//
// Mirrors the scalar soft_shadow() logic exactly:
//   res = min(res, k * min_clearance / max(0.001, t - y2))
//   ph  = previous d (for improved penumbra)
// Returns 16 shadow factors in [0,1].  Lanes with no hit return 1.0.
//
// Key differences from ray_march_avx512:
//   - Early-exit when ALL lanes are either blocked (res≈0) or escaped
//   - Hard shadow (blocked) causes res → 0 immediately via mask-blend
//   - t_max = 60 (matches scalar function)
//   - MAX_STEPS shadow = 96 (matches scalar loop count)
static __m512 soft_shadow_avx512(
    __m512 rox, __m512 roy, __m512 roz,
    __m512 rdx, __m512 rdy, __m512 rdz,
    float k = 14.f, int max_steps = 64)
{
    const int SHADOW_STEPS = max_steps;
    const float T_MAX = 60.f;
    const float T_START = 0.05f;
    const float HIT_THRESH = 0.001f;

    const __m512 k_v = _mm512_set1_ps(k);
    const __m512 zero = _mm512_setzero_ps();
    const __m512 one = _mm512_set1_ps(1.f);
    const __m512 eps_v = _mm512_set1_ps(0.001f);
    const __m512 t_max_v = _mm512_set1_ps(T_MAX);
    const __m512 hit_v = _mm512_set1_ps(HIT_THRESH);

    __m512 t = _mm512_set1_ps(T_START);
    __m512 res = one;                  // accumulated shadow factor
    __m512 ph = _mm512_set1_ps(1e20f); // previous d (penumbra tracking)

    // active: lane is still marching (not blocked, not escaped)
    __mmask16 active = 0xFFFF;

    for (int i = 0; i < SHADOW_STEPS && active; i++)
    {
        __m512 px = _mm512_fmadd_ps(rdx, t, rox);
        __m512 py = _mm512_fmadd_ps(rdy, t, roy);
        __m512 pz = _mm512_fmadd_ps(rdz, t, roz);

        // Shadow march uses reduced FBM octaves (3 vs 7): saves 4/7 ≈ 57% of terrain cost.
        // Shadow rays only need large-scale terrain shape to determine occlusion.
        EvalResult16 ev = eval_scene_avx512(px, py, pz, 3);
        __m512 d = ev.d;

        // Hard shadow: d < HIT_THRESH → res = 0 for those lanes, retire them
        __mmask16 blocked = active & _mm512_cmp_ps_mask(d, hit_v, _CMP_LT_OQ);
        if (blocked)
        {
            res = _mm512_mask_blend_ps(blocked, res, zero);
            active &= ~blocked;
        }

        // Improved penumbra (Quilez formula):
        //   y2 = d*d / (2*ph)
        //   di = sqrt(d*d - y2*y2)   ← min clearance
        //   res = min(res, k * di / max(0.001, t - y2))
        __m512 y2 = _mm512_div_ps(_mm512_mul_ps(d, d),
                                  _mm512_add_ps(_mm512_add_ps(ph, ph), eps_v));
        // d²-y2²  — clamp to 0 to avoid sqrt of negative (numerical noise)
        __m512 arg = _mm512_max_ps(zero,
                                   _mm512_sub_ps(_mm512_mul_ps(d, d), _mm512_mul_ps(y2, y2)));
        __m512 di = _mm512_sqrt_ps(arg);

        __m512 denom = _mm512_max_ps(eps_v, _mm512_sub_ps(t, y2));
        __m512 cand = _mm512_div_ps(_mm512_mul_ps(k_v, di), denom);
        // Only update res for active lanes (blocked lanes already zeroed)
        res = _mm512_mask_blend_ps(active, res, _mm512_min_ps(res, cand));

        ph = _mm512_mask_blend_ps(active, ph, d); // update ph only for active
        t = _mm512_mask_add_ps(t, active, t, d);  // advance t

        // Retire lanes that escaped
        active &= ~(_mm512_cmp_ps_mask(t, t_max_v, _CMP_GE_OQ));

        // Early-out: retire active lanes whose accumulated shadow factor has
        // already dropped below 0.001 — they will contribute pure black shadow
        // regardless of remaining march steps. This fires frequently in scenes
        // with dense geometry or long sun angles (fully shaded regions).
        active &= ~(_mm512_cmp_ps_mask(res, eps_v, _CMP_LE_OQ));
    }

    return _mm512_max_ps(zero, _mm512_min_ps(one, res));
}

// ── AVX-512 GI bounce march — thin wrapper with explicit t_max ───────────────
// Identical to ray_march_avx512 but stops at t_max (GI doesn't need far hits).
// Using t_max=50 is plenty for one diffuse bounce; cuts wasted steps on sky rays.
static March16 ray_march_avx512_gi(
    __m512 rox, __m512 roy, __m512 roz,
    __m512 rdx, __m512 rdy, __m512 rdz,
    float t_max = 50.f)
{
    __m512 t = _mm512_set1_ps(0.02f);
    __mmask16 active = 0xFFFF;
    __mmask16 hit_mask = 0;

    __m512 hit_x = _mm512_setzero_ps();
    __m512 hit_y = _mm512_setzero_ps();
    __m512 hit_z = _mm512_setzero_ps();
    __m512i hit_mat = _mm512_setzero_si512();

    const __m512 surf_dist_v = _mm512_set1_ps(SURF_DIST);
    const __m512 t_max_v = _mm512_set1_ps(t_max);

    for (int i = 0; i < MAX_STEPS && active; i++)
    {
        __m512 px = _mm512_fmadd_ps(rdx, t, rox);
        __m512 py = _mm512_fmadd_ps(rdy, t, roy);
        __m512 pz = _mm512_fmadd_ps(rdz, t, roz);

        // GI bounce uses reduced FBM octaves (2): saves ~71% terrain eval cost.
        // One diffuse bounce only needs large-scale scene shape; oct=2 is sufficient.
        EvalResult16 res = eval_scene_avx512(px, py, pz, 2);

        t = _mm512_mask_add_ps(t, active, t, res.d);

        __mmask16 just_hit = active & _mm512_cmp_ps_mask(res.d, surf_dist_v, _CMP_LT_OQ);
        if (just_hit)
        {
            hit_x = _mm512_mask_blend_ps(just_hit, hit_x, px);
            hit_y = _mm512_mask_blend_ps(just_hit, hit_y, py);
            hit_z = _mm512_mask_blend_ps(just_hit, hit_z, pz);
            hit_mat = _mm512_mask_blend_epi32(just_hit, hit_mat, res.mat);
            hit_mask |= just_hit;
            active &= ~just_hit;
        }

        // Retire rays that exceeded t_max (much sooner than MAX_DIST)
        active &= ~(_mm512_cmp_ps_mask(t, t_max_v, _CMP_GE_OQ));
    }

    return {hit_x, hit_y, hit_z, hit_mat, hit_mask, t};
}

#endif // __AVX512F__

// ─── AVX2 / AVX  8-WIDE BATCH MARCHING ───────────────────────────────────────
// Same architecture as the AVX-512 path but 8 lanes wide using __m256 / __m256i.
// AVX2 adds:  _mm256_i32gather_ps (required for sdf_grid_query_avx8)
//             fused multiply-add via _mm256_fmadd_ps
// Plain AVX:  no gather, no FMA — SDF grid queries done scalar, fmadd emulated.
//
// Dispatch (in render loop, after AVX-512 check):
//   #elif defined(__AVX2__)  → ray_march_avx2  + shade_batch_avx8
//   #elif defined(__AVX__)   → ray_march_avx   + shade_batch_avx8  (same shade fn)
//
// The shade_batch function is shared between AVX2 and AVX because shading is
// scalar arithmetic; only the march calls differ.
// ─────────────────────────────────────────────────────────────────────────────

#if defined(__AVX2__) || defined(__AVX__)
#include <immintrin.h> // already included for AVX-512, but guard anyway

// 8-lane march result — mirrors March16 at half width
struct March8
{
    __m256 pos_x, pos_y, pos_z;
    __m256i mat;
    uint8_t hit_mask; // bit i set = lane i hit
    __m256 t;
};

// ── eval_scene_avx8: evaluate scene SDF for 8 positions ──────────────────────
// AVX2 path uses gather for SDF grid; AVX path falls back to scalar grid query.
// ── AVX2 math helpers (8-wide) ───────────────────────────────────────────────
#ifdef __AVX2__
static inline __m256 avx8_sin(__m256 x)
{
    const __m256 inv_2pi = _mm256_set1_ps(0.15915494f);
    const __m256 two_pi = _mm256_set1_ps(6.28318530f);
    __m256 k = _mm256_round_ps(_mm256_mul_ps(x, inv_2pi),
                               _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    x = _mm256_sub_ps(x, _mm256_mul_ps(k, two_pi));
    __m256 x2 = _mm256_mul_ps(x, x);
    __m256 r = _mm256_fmadd_ps(x2, _mm256_set1_ps(-1.f / 5040.f), _mm256_set1_ps(1.f / 120.f));
    r = _mm256_fmadd_ps(x2, r, _mm256_set1_ps(-1.f / 6.f));
    r = _mm256_fmadd_ps(x2, r, _mm256_set1_ps(1.f));
    return _mm256_mul_ps(x, r);
}
// avx8_cos: derived from avx8_sin via identity cos(x) = sin(x + π/2)
static inline __m256 avx8_cos(__m256 x)
{
    return avx8_sin(_mm256_add_ps(x, _mm256_set1_ps(1.5707963f)));
}

// rand_hemi_avx8: cosine-weighted hemisphere sample for 8 lanes at once.
// Uses Malley's method: sample disk via (r,θ) → project up to hemisphere.
// No rejection loop — deterministic 3 rng_next calls per lane, all in-register.
// 'n' components are the shading normals (already computed in nx8/ny8/nz8).
// tid is the OpenMP thread id; advances rng_state[tid] by 3×8 = 24 steps.
static inline void rand_hemi_avx8(
    const float *nx, const float *ny, const float *nz,
    float *odx, float *ody, float *odz, int tid)
{
    // Pull 3×8 raw uint32 samples from the per-thread xorshift
    alignas(32) uint32_t ru[8], rv[8], rw[8];
    for (int i = 0; i < 8; i++)
    {
        uint32_t &s = rng_state[tid];
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        ru[i] = s;
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        rv[i] = s;
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        rw[i] = s;
    }
    // Convert to float in [0,1)
    const __m256 scale = _mm256_set1_ps(5.96046448e-8f); // 1/(1<<24) ≈ 2^-24
    __m256 u1 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_srli_epi32(
                                  _mm256_load_si256((const __m256i *)ru), 8)),
                              scale);
    __m256 u2 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_srli_epi32(
                                  _mm256_load_si256((const __m256i *)rv), 8)),
                              scale);
    // Malley's method: disk sample (r = sqrt(u1), θ = 2π·u2)
    __m256 r = _mm256_sqrt_ps(u1);
    __m256 phi = _mm256_mul_ps(u2, _mm256_set1_ps(6.28318530f));
    __m256 lx = _mm256_mul_ps(r, avx8_cos(phi));
    __m256 ly = _mm256_mul_ps(r, avx8_sin(phi));
    __m256 lz = _mm256_sqrt_ps(_mm256_max_ps(
        _mm256_set1_ps(0.f),
        _mm256_sub_ps(_mm256_set1_ps(1.f), _mm256_mul_ps(r, r))));
    // Build ONB: tangent t = perp(n), bitangent b = n×t
    // Use w to pick the less-parallel axis for the cross
    __m256 vw = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_srli_epi32(
                                  _mm256_load_si256((const __m256i *)rw), 8)),
                              scale);
    // We actually don't need rw for the ONB — use a consistent algebraic perp.
    // Frisvad / Duff et al. sign-flip ONB (branchless):
    __m256 vnx = _mm256_load_ps(nx), vny = _mm256_load_ps(ny), vnz = _mm256_load_ps(nz);
    __m256 s_ = _mm256_blendv_ps(_mm256_set1_ps(1.f), _mm256_set1_ps(-1.f),
                                 _mm256_cmp_ps(vnz, _mm256_setzero_ps(), _CMP_LT_OS));
    __m256 a_ = _mm256_div_ps(_mm256_set1_ps(-1.f),
                              _mm256_add_ps(s_, vnz));
    __m256 b_ = _mm256_mul_ps(vnx, _mm256_mul_ps(vny, a_));
    // tangent t
    __m256 tx = _mm256_fmadd_ps(vnx, _mm256_mul_ps(vnx, a_), s_);
    __m256 ty = b_;
    __m256 tz = _mm256_xor_ps(vnx, _mm256_set1_ps(-0.f)); // -nx
    // bitangent bv
    __m256 bvx = b_;
    __m256 bvy = _mm256_fmadd_ps(vny, _mm256_mul_ps(vny, a_), s_);
    __m256 bvz = _mm256_xor_ps(vny, _mm256_set1_ps(-0.f)); // -ny
    // world direction = lx*t + ly*bv + lz*n
    __m256 wx = _mm256_fmadd_ps(lx, tx, _mm256_fmadd_ps(ly, bvx, _mm256_mul_ps(lz, vnx)));
    __m256 wy = _mm256_fmadd_ps(lx, ty, _mm256_fmadd_ps(ly, bvy, _mm256_mul_ps(lz, vny)));
    __m256 wz = _mm256_fmadd_ps(lx, tz, _mm256_fmadd_ps(ly, bvz, _mm256_mul_ps(lz, vnz)));
    _mm256_store_ps(odx, wx);
    _mm256_store_ps(ody, wy);
    _mm256_store_ps(odz, wz);
    (void)vw; // rw used only for seeding variation; ONB is deterministic
}

// Integer hash — same algorithm as AVX-512 avx_hash21 above, AVX2 edition.
static inline __m256 avx8_hash21(__m256 x, __m256 y)
{
    __m256i ix = _mm256_cvttps_epi32(x);
    __m256i iy = _mm256_cvttps_epi32(y);
    __m256i h = _mm256_add_epi32(
        _mm256_add_epi32(
            _mm256_mullo_epi32(ix, _mm256_set1_epi32(1664525)),
            _mm256_mullo_epi32(iy, _mm256_set1_epi32(22695477))),
        _mm256_set1_epi32(1013904223));
    h = _mm256_xor_si256(h, _mm256_srli_epi32(h, 16));
    h = _mm256_mullo_epi32(h, _mm256_set1_epi32(0x45d9f3b));
    h = _mm256_xor_si256(h, _mm256_srli_epi32(h, 16));
    __m256i mantissa = _mm256_or_si256(_mm256_srli_epi32(h, 9), _mm256_set1_epi32(0x3f800000));
    return _mm256_sub_ps(_mm256_castsi256_ps(mantissa), _mm256_set1_ps(1.f));
}
static inline __m256 avx8_smoothnoise(__m256 x, __m256 y)
{
    __m256 ix = _mm256_floor_ps(x), iy = _mm256_floor_ps(y);
    __m256 fx = _mm256_sub_ps(x, ix), fy = _mm256_sub_ps(y, iy);
    __m256 ux = _mm256_mul_ps(fx, _mm256_mul_ps(fx,
                                                _mm256_sub_ps(_mm256_set1_ps(3.f), _mm256_mul_ps(fx, _mm256_set1_ps(2.f)))));
    __m256 uy = _mm256_mul_ps(fy, _mm256_mul_ps(fy,
                                                _mm256_sub_ps(_mm256_set1_ps(3.f), _mm256_mul_ps(fy, _mm256_set1_ps(2.f)))));
    __m256 one = _mm256_set1_ps(1.f);
    __m256 a = avx8_hash21(ix, iy);
    __m256 b = avx8_hash21(_mm256_add_ps(ix, one), iy);
    __m256 c = avx8_hash21(ix, _mm256_add_ps(iy, one));
    __m256 d = avx8_hash21(_mm256_add_ps(ix, one), _mm256_add_ps(iy, one));
    __m256 ab = _mm256_fmadd_ps(_mm256_sub_ps(b, a), ux, a);
    __m256 cd = _mm256_fmadd_ps(_mm256_sub_ps(d, c), ux, c);
    return _mm256_fmadd_ps(_mm256_sub_ps(cd, ab), uy, ab);
}
static inline __m256 avx8_fbm(__m256 x, __m256 y, int oct = 6)
{
    // Precomputed reciprocal of total amplitude — same table as avx_fbm.
    static const float inv_tot_lut[] = {
        0.f,
        2.000000f,
        1.333333f,
        1.142857f,
        1.066667f,
        1.032258f,
        1.016129f,
        1.007874f,
    };
    __m256 v = _mm256_setzero_ps(), amp = _mm256_set1_ps(0.5f);
    __m256 freq = _mm256_set1_ps(1.f);
    for (int i = 0; i < oct; i++)
    {
        v = _mm256_fmadd_ps(avx8_smoothnoise(_mm256_mul_ps(x, freq),
                                             _mm256_mul_ps(y, freq)),
                            amp, v);
        amp = _mm256_mul_ps(amp, _mm256_set1_ps(0.5f));
        freq = _mm256_mul_ps(freq, _mm256_set1_ps(2.1f));
    }
    return _mm256_mul_ps(v, _mm256_set1_ps(inv_tot_lut[oct < 8 ? oct : 7]));
}
static inline __m256 avx8_terrain_h(__m256 x, __m256 z, int oct = 7)
{
    __m256 h = _mm256_mul_ps(
        avx8_fbm(_mm256_mul_ps(x, _mm256_set1_ps(0.035f)),
                 _mm256_mul_ps(z, _mm256_set1_ps(0.035f)), oct),
        _mm256_set1_ps(22.f));
    // rsqrt14-based fast sqrt for smoothstep radius
    __m256 r2 = _mm256_fmadd_ps(x, x, _mm256_mul_ps(z, z));
    __m256 r = _mm256_mul_ps(r2, _mm256_rsqrt_ps(_mm256_max_ps(r2, _mm256_set1_ps(1e-18f))));
    // smoothstep(5, 30, r)
    __m256 t1 = _mm256_div_ps(_mm256_sub_ps(r, _mm256_set1_ps(5.f)), _mm256_set1_ps(25.f));
    t1 = _mm256_min_ps(_mm256_max_ps(t1, _mm256_setzero_ps()), _mm256_set1_ps(1.f));
    t1 = _mm256_mul_ps(t1, _mm256_mul_ps(t1,
                                         _mm256_sub_ps(_mm256_set1_ps(3.f), _mm256_mul_ps(t1, _mm256_set1_ps(2.f)))));
    h = _mm256_mul_ps(h, t1);
    // smoothstep(30, 8, r) → subtract 4*
    __m256 t2 = _mm256_div_ps(_mm256_sub_ps(r, _mm256_set1_ps(30.f)),
                              _mm256_set1_ps(8.f - 30.f));
    t2 = _mm256_min_ps(_mm256_max_ps(t2, _mm256_setzero_ps()), _mm256_set1_ps(1.f));
    t2 = _mm256_mul_ps(t2, _mm256_mul_ps(t2,
                                         _mm256_sub_ps(_mm256_set1_ps(3.f), _mm256_mul_ps(t2, _mm256_set1_ps(2.f)))));
    return _mm256_sub_ps(h, _mm256_mul_ps(_mm256_set1_ps(4.f), t2));
}

// ── BAKED TERRAIN HEIGHT FIELD ───────────────────────────────────────────────
// The AVX2 renderer's integer hash is deterministic, so the terrain height can be
// baked once and sampled with gather+bilinear. The tables cover the full playable
// area; rays outside the table fall back to the analytic function.
#ifdef __AVX2__
static const int HF_N = 2048;
static const float HF_MIN = -128.f;
static const float HF_MAX = 128.f;
static const float HF_SCALE = (float)(HF_N - 1) / (HF_MAX - HF_MIN);
static float *g_hf[8] = {};
static bool g_hf_ready = false;

static inline uint32_t hf_hash(uint32_t ix, uint32_t iy)
{
    uint32_t h = ix * 1664525u + iy * 22695477u + 1013904223u;
    h ^= h >> 16;
    h *= 0x45d9f3bu;
    h ^= h >> 16;
    return (h >> 9) | 0x3f800000u;
}

static inline float hf_hashf(float ix, float iy)
{
    union
    {
        uint32_t u;
        float f;
    } v;
    v.u = hf_hash((uint32_t)(int)ix, (uint32_t)(int)iy);
    return v.f - 1.f;
}

static inline float hf_noise(float x, float y)
{
    float ix = floorf(x), iy = floorf(y);
    float fx = x - ix, fy = y - iy;
    float ux = fx * fx * (3.f - 2.f * fx), uy = fy * fy * (3.f - 2.f * fy);
    float a = hf_hashf(ix, iy);
    float b = hf_hashf(ix + 1, iy);
    float c = hf_hashf(ix, iy + 1);
    float d = hf_hashf(ix + 1, iy + 1);
    float ab = a + (b - a) * ux;
    float cd = c + (d - c) * ux;
    return ab + (cd - ab) * uy;
}

static inline float hf_fbm(float x, float y, int oct)
{
    static const float inv_tot[] = {0.f, 2.f, 1.333333f, 1.142857f, 1.066667f,
                                    1.032258f, 1.016129f, 1.007874f};
    float v = 0.f, amp = 0.5f, freq = 1.f;
    for (int i = 0; i < oct; i++)
    {
        v += hf_noise(x * freq, y * freq) * amp;
        amp *= 0.5f;
        freq *= 2.1f;
    }
    return v * inv_tot[oct < 8 ? oct : 7];
}

// Scalar fallback matching avx8_terrain_h closely enough for out-of-table rays.
static inline float terrain_h_fast(float x, float z, int oct)
{
    float h = hf_fbm(x * 0.035f, z * 0.035f, oct) * 22.f;
    float r = sqrtf(x * x + z * z);
    h *= smoothstep(5.f, 30.f, r);
    h -= 4.f * smoothstep(30.f, 8.f, r);
    return h;
}

void ensure_height_tables()
{
    if (g_hf_ready)
        return;
#pragma omp critical(cpurt_height_tables)
    {
        if (!g_hf_ready)
        {
            const int octs[3] = {2, 3, 7};
            for (int k = 0; k < 3; k++)
            {
                int oct = octs[k];
                float *t = new float[(size_t)HF_N * HF_N];
                for (int z = 0; z < HF_N; z++)
                {
                    __m256 zv = _mm256_set1_ps(HF_MIN + (float)z / HF_SCALE);
                    int x = 0;
                    for (; x + 8 <= HF_N; x += 8)
                    {
                        __m256 xv = _mm256_add_ps(
                            _mm256_set1_ps(HF_MIN + (float)x / HF_SCALE),
                            _mm256_mul_ps(_mm256_set_ps(7.f, 6.f, 5.f, 4.f, 3.f, 2.f, 1.f, 0.f),
                                          _mm256_set1_ps(1.f / HF_SCALE)));
                        __m256 h = avx8_terrain_h(xv, zv, oct);
                        _mm256_storeu_ps(t + (size_t)z * HF_N + x, h);
                    }
                    for (; x < HF_N; x++)
                    {
                        t[(size_t)z * HF_N + x] =
                            terrain_h_fast(HF_MIN + (float)x / HF_SCALE,
                                           HF_MIN + (float)z / HF_SCALE, oct);
                    }
                }
                g_hf[oct] = t;
            }
            g_hf_ready = true;
        }
    }
}

static inline int hf_table_oct(int oct)
{
    if (oct <= 2)
        return 2;
    if (oct <= 3)
        return 3;
    return 7;
}

static inline __m256 avx8_terrain_h_baked(__m256 x, __m256 z, int oct)
{
    if (!g_hf_ready)
        ensure_height_tables();
    const int to = hf_table_oct(oct);
    const float *t = g_hf[to];

    __m256 fx = _mm256_mul_ps(_mm256_sub_ps(x, _mm256_set1_ps(HF_MIN)),
                              _mm256_set1_ps(HF_SCALE));
    __m256 fz = _mm256_mul_ps(_mm256_sub_ps(z, _mm256_set1_ps(HF_MIN)),
                              _mm256_set1_ps(HF_SCALE));
    __m256 hi = _mm256_set1_ps((float)(HF_N - 1));
    __m256 clamped_x = _mm256_min_ps(_mm256_max_ps(fx, _mm256_setzero_ps()), hi);
    __m256 clamped_z = _mm256_min_ps(_mm256_max_ps(fz, _mm256_setzero_ps()), hi);
    __m256i ix = _mm256_cvttps_epi32(clamped_x);
    __m256i iz = _mm256_cvttps_epi32(clamped_z);
    __m256 tx = _mm256_sub_ps(clamped_x, _mm256_cvtepi32_ps(ix));
    __m256 tz = _mm256_sub_ps(clamped_z, _mm256_cvtepi32_ps(iz));
    __m256i one = _mm256_set1_epi32(1);
    __m256i ix1 = _mm256_min_epi32(_mm256_add_epi32(ix, one), _mm256_set1_epi32(HF_N - 1));
    __m256i iz1 = _mm256_min_epi32(_mm256_add_epi32(iz, one), _mm256_set1_epi32(HF_N - 1));
    __m256i nvec = _mm256_set1_epi32(HF_N);
    __m256i row0 = _mm256_mullo_epi32(iz, nvec);
    __m256i row1 = _mm256_mullo_epi32(iz1, nvec);

#define HF_LERP(a, b, t) _mm256_fmadd_ps(_mm256_sub_ps((b), (a)), (t), (a))
    __m256 v00 = _mm256_i32gather_ps(t, _mm256_add_epi32(row0, ix), 4);
    __m256 v10 = _mm256_i32gather_ps(t, _mm256_add_epi32(row0, ix1), 4);
    __m256 v01 = _mm256_i32gather_ps(t, _mm256_add_epi32(row1, ix), 4);
    __m256 v11 = _mm256_i32gather_ps(t, _mm256_add_epi32(row1, ix1), 4);
    __m256 h = HF_LERP(HF_LERP(v00, v10, tx), HF_LERP(v01, v11, tx), tz);
#undef HF_LERP

    // Lanes outside the baked extent use the analytic function.
    __m256 in = _mm256_and_ps(
        _mm256_and_ps(_mm256_cmp_ps(x, _mm256_set1_ps(HF_MIN), _CMP_GE_OQ),
                      _mm256_cmp_ps(x, _mm256_set1_ps(HF_MAX), _CMP_LE_OQ)),
        _mm256_and_ps(_mm256_cmp_ps(z, _mm256_set1_ps(HF_MIN), _CMP_GE_OQ),
                      _mm256_cmp_ps(z, _mm256_set1_ps(HF_MAX), _CMP_LE_OQ)));
    if (_mm256_movemask_ps(in) != 0xFF)
    {
        alignas(32) float xa[8], za[8], ha[8];
        _mm256_store_ps(xa, x);
        _mm256_store_ps(za, z);
        _mm256_store_ps(ha, h);
        for (int i = 0; i < 8; i++)
        {
            if (!(1 & (_mm256_movemask_ps(in) >> i)))
                ha[i] = terrain_h_fast(xa[i], za[i], to);
        }
        h = _mm256_blendv_ps(_mm256_load_ps(ha), h, in);
    }
    return h;
}
#endif // __AVX2__

// ── AVX2 SDF grid query for 8 positions (uses _mm256_i32gather_ps) ───────────
static __m256 sdf_grid_query_avx8(const SDFGrid &g,
                                  __m256 wpx, __m256 wpy, __m256 wpz)
{
    __m256 vca = _mm256_set1_ps(g.ca), vsa = _mm256_set1_ps(g.sa);
    __m256 inv_s = _mm256_set1_ps(g.inv_scale);
    __m256 dx = _mm256_sub_ps(wpx, _mm256_set1_ps(g.pos.x));
    __m256 dz = _mm256_sub_ps(wpz, _mm256_set1_ps(g.pos.z));
    __m256 lx = _mm256_mul_ps(_mm256_sub_ps(_mm256_mul_ps(vca, dx), _mm256_mul_ps(vsa, dz)), inv_s);
    __m256 ly = _mm256_mul_ps(_mm256_sub_ps(wpy, _mm256_set1_ps(g.pos.y)), inv_s);
    __m256 lz = _mm256_mul_ps(_mm256_add_ps(_mm256_mul_ps(vsa, dx), _mm256_mul_ps(vca, dz)), inv_s);
    __m256 bx0 = _mm256_set1_ps(g.bmin[0]), bx1 = _mm256_set1_ps(g.bmax[0]);
    __m256 by0 = _mm256_set1_ps(g.bmin[1]), by1 = _mm256_set1_ps(g.bmax[1]);
    __m256 bz0 = _mm256_set1_ps(g.bmin[2]), bz1 = _mm256_set1_ps(g.bmax[2]);
    __m256 z8 = _mm256_setzero_ps();
    __m256 ox = _mm256_max_ps(z8, _mm256_max_ps(_mm256_sub_ps(bx0, lx), _mm256_sub_ps(lx, bx1)));
    __m256 oy = _mm256_max_ps(z8, _mm256_max_ps(_mm256_sub_ps(by0, ly), _mm256_sub_ps(ly, by1)));
    __m256 oz = _mm256_max_ps(z8, _mm256_max_ps(_mm256_sub_ps(bz0, lz), _mm256_sub_ps(lz, bz1)));
    __m256 out8_r2 = _mm256_add_ps(_mm256_mul_ps(ox, ox),
                                   _mm256_add_ps(_mm256_mul_ps(oy, oy), _mm256_mul_ps(oz, oz)));
    __m256 outside = _mm256_mul_ps(out8_r2, _mm256_rsqrt_ps(_mm256_max_ps(out8_r2, _mm256_set1_ps(1e-18f))));
    __m256 outside_f = _mm256_cmp_ps(outside, z8, _CMP_GT_OQ);
    float fN = (float)g.N;
    __m256 vN = _mm256_set1_ps(fN);
    __m256 tx = _mm256_div_ps(_mm256_mul_ps(_mm256_sub_ps(lx, bx0), vN), _mm256_sub_ps(bx1, bx0));
    __m256 ty = _mm256_div_ps(_mm256_mul_ps(_mm256_sub_ps(ly, by0), vN), _mm256_sub_ps(by1, by0));
    __m256 tz = _mm256_div_ps(_mm256_mul_ps(_mm256_sub_ps(lz, bz0), vN), _mm256_sub_ps(bz1, bz0));
    __m256 mi = _mm256_set1_ps(fN - 1.001f);
    tx = _mm256_min_ps(_mm256_max_ps(tx, z8), mi);
    ty = _mm256_min_ps(_mm256_max_ps(ty, z8), mi);
    tz = _mm256_min_ps(_mm256_max_ps(tz, z8), mi);
    __m256i ix = _mm256_cvttps_epi32(tx), iy = _mm256_cvttps_epi32(ty), iz = _mm256_cvttps_epi32(tz);
    __m256 fx = _mm256_sub_ps(tx, _mm256_cvtepi32_ps(ix));
    __m256 fy = _mm256_sub_ps(ty, _mm256_cvtepi32_ps(iy));
    __m256 fz = _mm256_sub_ps(tz, _mm256_cvtepi32_ps(iz));
    int gN = g.N;
    __m256i vgN = _mm256_set1_epi32(gN), vgN2 = _mm256_set1_epi32(gN * gN);
    __m256i vgN1 = _mm256_set1_epi32(gN - 1), zi8 = _mm256_setzero_si256(), one_i = _mm256_set1_epi32(1);
    ix = _mm256_min_epi32(_mm256_max_epi32(ix, zi8), vgN1);
    iy = _mm256_min_epi32(_mm256_max_epi32(iy, zi8), vgN1);
    iz = _mm256_min_epi32(_mm256_max_epi32(iz, zi8), vgN1);
    __m256i ix1 = _mm256_min_epi32(_mm256_add_epi32(ix, one_i), vgN1);
    __m256i iy1 = _mm256_min_epi32(_mm256_add_epi32(iy, one_i), vgN1);
    __m256i iz1 = _mm256_min_epi32(_mm256_add_epi32(iz, one_i), vgN1);
    auto idx3 = [&](__m256i zi, __m256i yi, __m256i xi) -> __m256i
    { return _mm256_add_epi32(_mm256_add_epi32(_mm256_mullo_epi32(zi, vgN2),
                                               _mm256_mullo_epi32(yi, vgN)),
                              xi); };
    const float *base = g.data;
    __m256 v000 = _mm256_i32gather_ps(base, idx3(iz, iy, ix), 4);
    __m256 v100 = _mm256_i32gather_ps(base, idx3(iz, iy, ix1), 4);
    __m256 v010 = _mm256_i32gather_ps(base, idx3(iz, iy1, ix), 4);
    __m256 v110 = _mm256_i32gather_ps(base, idx3(iz, iy1, ix1), 4);
    __m256 v001 = _mm256_i32gather_ps(base, idx3(iz1, iy, ix), 4);
    __m256 v101 = _mm256_i32gather_ps(base, idx3(iz1, iy, ix1), 4);
    __m256 v011 = _mm256_i32gather_ps(base, idx3(iz1, iy1, ix), 4);
    __m256 v111 = _mm256_i32gather_ps(base, idx3(iz1, iy1, ix1), 4);
#define LRP8(a, b, t) _mm256_fmadd_ps(_mm256_sub_ps(b, a), t, a)
    __m256 v00 = LRP8(v000, v100, fx), v10 = LRP8(v010, v110, fx);
    __m256 v01 = LRP8(v001, v101, fx), v11 = LRP8(v011, v111, fx);
    __m256 v0 = LRP8(v00, v10, fy), v1 = LRP8(v01, v11, fy);
    __m256 d = LRP8(v0, v1, fz);
#undef LRP8
    __m256 sv = _mm256_set1_ps(g.scale);
    d = _mm256_mul_ps(d, sv);
    return _mm256_blendv_ps(d, _mm256_mul_ps(outside, sv), outside_f);
}
#endif // __AVX2__

// ─────────────────────────────────────────────────────────────────────────────
struct EvalResult8
{
    __m256 d;
    __m256i mat;
};

static EvalResult8 eval_scene_avx8(__m256 px, __m256 py, __m256 pz,
                                   int terrain_oct = 7, bool compute_mat = true)
{
#ifndef __AVX2__
    // Plain AVX fallback: no gather, no FMA — scalar per lane
    alignas(32) float x8[8], y8[8], z8[8];
    _mm256_store_ps(x8, px);
    _mm256_store_ps(y8, py);
    _mm256_store_ps(z8, pz);
    alignas(32) float d8[8];
    alignas(32) int m8[8];
    for (int i = 0; i < 8; i++)
    {
        auto r = eval_scene({x8[i], y8[i], z8[i]});
        d8[i] = r.d;
        m8[i] = r.mat;
    }
    return {_mm256_load_ps(d8), _mm256_loadu_si256((__m256i *)m8)};
#else
    // Full AVX2 vectorised evaluation (mirrors eval_scene_avx512 at 8-wide)
    __m256 best_d = _mm256_set1_ps(INF);
    __m256i best_mat = _mm256_set1_epi32(MAT_GRASS);

    // blendv-based update: select prim_d/mat_id where prim_d < best_d
    auto update8 = [&](__m256 prim_d, int mat_id)
    {
        __m256 closer_f = _mm256_cmp_ps(prim_d, best_d, _CMP_LT_OQ);
        best_d = _mm256_min_ps(best_d, prim_d);
        __m256 mat_f = _mm256_castsi256_ps(_mm256_set1_epi32(mat_id));
        best_mat = _mm256_castps_si256(
            _mm256_blendv_ps(_mm256_castsi256_ps(best_mat), mat_f, closer_f));
    };

    // ── Water ────────────────────────────────────────────────────────────
    __m256 lake8_r2 = _mm256_fmadd_ps(px, px, _mm256_mul_ps(pz, pz));
    __m256 r_lake = _mm256_mul_ps(lake8_r2, _mm256_rsqrt_ps(_mm256_max_ps(lake8_r2, _mm256_set1_ps(1e-18f))));
    __m256 in_lake_f = _mm256_cmp_ps(r_lake, _mm256_set1_ps(60.f), _CMP_LT_OQ);
    if (_mm256_movemask_ps(in_lake_f))
    {
        __m256 dw = _mm256_add_ps(py, _mm256_set1_ps(0.1f));
        dw = _mm256_add_ps(dw, _mm256_mul_ps(avx8_sin(_mm256_mul_ps(px, _mm256_set1_ps(0.8f))),
                                             _mm256_set1_ps(0.04f)));
        dw = _mm256_add_ps(dw, _mm256_mul_ps(avx8_sin(_mm256_mul_ps(pz, _mm256_set1_ps(0.9f))),
                                             _mm256_set1_ps(0.03f)));
        // only apply inside lake
        __m256 safe_dw = _mm256_blendv_ps(_mm256_set1_ps(INF), dw, in_lake_f);
        update8(safe_dw, MAT_WATER);
    }

    // ── Terrain ──────────────────────────────────────────────────────────
    {
        __m256 h, d_terrain;
        if (!g_bounds_enabled)
        {
            h = g_baked_enabled ? avx8_terrain_h_baked(px, pz, terrain_oct)
                                : avx8_terrain_h(px, pz, terrain_oct);
            d_terrain = _mm256_mul_ps(_mm256_sub_ps(py, h), _mm256_set1_ps(0.55f));
        }
        else
        {
            // Conservative height bounds: terrain_h() is in [-4, 22]. Lanes far
            // above or below that band cannot hit terrain, so skip the height
            // evaluation and return a lower-bound distance directly.
            __m256 high_f = _mm256_cmp_ps(py, _mm256_set1_ps(22.5f), _CMP_GT_OQ);
            __m256 low_f = _mm256_cmp_ps(py, _mm256_set1_ps(-4.5f), _CMP_LT_OQ);
            __m256 safe_x = _mm256_blendv_ps(px, _mm256_setzero_ps(), _mm256_or_ps(high_f, low_f));
            __m256 safe_z = _mm256_blendv_ps(pz, _mm256_setzero_ps(), _mm256_or_ps(high_f, low_f));
            h = g_baked_enabled ? avx8_terrain_h_baked(safe_x, safe_z, terrain_oct)
                                : avx8_terrain_h(safe_x, safe_z, terrain_oct);
            __m256 d_mid = _mm256_mul_ps(_mm256_sub_ps(py, h), _mm256_set1_ps(0.55f));
            __m256 d_high = _mm256_mul_ps(_mm256_sub_ps(py, _mm256_set1_ps(22.5f)), _mm256_set1_ps(0.55f));
            __m256 d_low = _mm256_mul_ps(_mm256_sub_ps(_mm256_set1_ps(-4.5f), py), _mm256_set1_ps(0.55f));
            d_terrain = _mm256_blendv_ps(
                _mm256_blendv_ps(d_mid, d_high, high_f), d_low, low_f);
        }

        __m256 prev_best = best_d;
        __m256 closer_f = _mm256_cmp_ps(d_terrain, prev_best, _CMP_LT_OQ);
        best_d = _mm256_min_ps(prev_best, d_terrain);

        // Material noise is only needed when terrain wins and the lane is close
        // enough to become a primary hit. Shadow/AO/GI/normal probes pass
        // compute_mat=false and skip this work entirely.
        if (compute_mat)
        {
            __m256 near_hit = _mm256_and_ps(closer_f,
                                            _mm256_cmp_ps(d_terrain, _mm256_set1_ps(SURF_DIST * 8.f), _CMP_LT_OQ));
            if (_mm256_movemask_ps(near_hit))
            {
                __m256i tmat = _mm256_set1_epi32(MAT_GRASS);
                __m256 is_snow_f = _mm256_cmp_ps(h, _mm256_set1_ps(8.f), _CMP_GT_OQ);
                __m256 above_rock = _mm256_cmp_ps(h, _mm256_set1_ps(2.f), _CMP_GT_OQ);
                __m256 rnoise = avx8_fbm(_mm256_mul_ps(px, _mm256_set1_ps(0.3f)),
                                         _mm256_mul_ps(pz, _mm256_set1_ps(0.3f)), 2);
                __m256 is_rock_f = _mm256_and_ps(above_rock,
                                                 _mm256_cmp_ps(rnoise, _mm256_set1_ps(0.42f), _CMP_LT_OQ));
                tmat = _mm256_castps_si256(_mm256_blendv_ps(_mm256_castsi256_ps(tmat),
                                                            _mm256_castsi256_ps(_mm256_set1_epi32(MAT_ROCK)), is_rock_f));
                tmat = _mm256_castps_si256(_mm256_blendv_ps(_mm256_castsi256_ps(tmat),
                                                            _mm256_castsi256_ps(_mm256_set1_epi32(MAT_SNOW)), is_snow_f));
                __m256 use_mat = _mm256_and_ps(near_hit, closer_f);
                best_mat = _mm256_castps_si256(_mm256_blendv_ps(
                    _mm256_castsi256_ps(best_mat), _mm256_castsi256_ps(tmat), use_mat));
            }
        }
    }

    // ── Trees (8 capsule trunks + 4 foliage spheres each) ────────────────
    {
        init_tree_bases();
        for (int i = 0; i < 8; i++)
        {
            float ti_x = TREE_X[i], ti_z = TREE_Z[i];
            __m256 dxi = _mm256_sub_ps(px, _mm256_set1_ps(ti_x));
            __m256 dzi = _mm256_sub_ps(pz, _mm256_set1_ps(ti_z));
            __m256 dist2 = _mm256_fmadd_ps(dxi, dxi, _mm256_mul_ps(dzi, dzi));
            // 8-lane horizontal min via SSE
            __m128 lo = _mm256_castps256_ps128(dist2);
            __m128 hi = _mm256_extractf128_ps(dist2, 1);
            __m128 m0 = _mm_min_ps(lo, hi);
            m0 = _mm_min_ps(m0, _mm_shuffle_ps(m0, m0, _MM_SHUFFLE(2, 3, 0, 1)));
            m0 = _mm_min_ps(m0, _mm_shuffle_ps(m0, m0, _MM_SHUFFLE(1, 0, 3, 2)));
            if (_mm_cvtss_f32(m0) > 100.f)
                continue;

            float th = TREE_BASE[i];
            // Capsule trunk: base=(ti_x,th,ti_z)  top=(ti_x,th+4.5,ti_z)  r=0.25
            __m256 ap_y = _mm256_sub_ps(py, _mm256_set1_ps(th));
            __m256 t_cap = _mm256_mul_ps(ap_y, _mm256_set1_ps(4.5f / (4.5f * 4.5f)));
            t_cap = _mm256_min_ps(_mm256_max_ps(t_cap, _mm256_setzero_ps()), _mm256_set1_ps(1.f));
            __m256 cl_y = _mm256_fmadd_ps(t_cap, _mm256_set1_ps(4.5f), _mm256_set1_ps(th));
            __m256 ey = _mm256_sub_ps(py, cl_y);
            __m256 trk8_r2 = _mm256_fmadd_ps(dxi, dxi, _mm256_fmadd_ps(ey, ey, _mm256_mul_ps(dzi, dzi)));
            __m256 d_trunk = _mm256_sub_ps(
                _mm256_mul_ps(trk8_r2, _mm256_rsqrt_ps(_mm256_max_ps(trk8_r2, _mm256_set1_ps(1e-18f)))),
                _mm256_set1_ps(0.25f));
            update8(d_trunk, MAT_TREE_TRUNK);

            for (int j = 0; j < 4; j++)
            {
                float cy = th + 2.7f + j * 1.1f;
                float r_leaf = 2.5f - j * 0.45f;
                __m256 ly = _mm256_sub_ps(py, _mm256_set1_ps(cy));
                // warp: sin(ly*2.3+ti_x)*0.15, cos(ly*2.3+ti_z)*0.15
                __m256 warg = _mm256_fmadd_ps(ly, _mm256_set1_ps(2.3f), _mm256_set1_ps(ti_x));
                __m256 wx = _mm256_fmadd_ps(avx8_sin(warg), _mm256_set1_ps(0.15f), dxi);
                __m256 carg = _mm256_add_ps(
                    _mm256_fmadd_ps(ly, _mm256_set1_ps(2.3f), _mm256_set1_ps(ti_z)),
                    _mm256_set1_ps(1.5707963f));
                __m256 wz = _mm256_fmadd_ps(avx8_sin(carg), _mm256_set1_ps(0.15f), dzi);
                __m256 lf8_r2 = _mm256_fmadd_ps(wx, wx, _mm256_fmadd_ps(ly, ly, _mm256_mul_ps(wz, wz)));
                __m256 d_leaf = _mm256_sub_ps(
                    _mm256_mul_ps(lf8_r2, _mm256_rsqrt_ps(_mm256_max_ps(lf8_r2, _mm256_set1_ps(1e-18f)))),
                    _mm256_set1_ps(r_leaf));
                update8(d_leaf, MAT_TREE_LEAF);
            }
        }
    }

    // ── Baked SDF grid objects ────────────────────────────────────────────
    // BVH sphere pre-rejection (same logic as AVX-512 path above).
    // Prefetch next grid's data before the gather for the current one (same
    // strategy as AVX-512: prefetch 4 cache lines at the start + mid hint).
    for (int gi = 0; gi < g_ngrid; gi++)
    {
        if (!g_grids[gi].valid)
            continue;
        const SDFGrid &g = g_grids[gi];
        float half_diag = g.sphere_half_diag;
        __m256 gcx = _mm256_set1_ps(g.sphere_cx);
        __m256 gcy = _mm256_set1_ps(g.sphere_cy);
        __m256 gcz = _mm256_set1_ps(g.sphere_cz);
        __m256 edx = _mm256_sub_ps(px, gcx), edy = _mm256_sub_ps(py, gcy), edz = _mm256_sub_ps(pz, gcz);
        __m256 ed2 = _mm256_fmadd_ps(edx, edx, _mm256_fmadd_ps(edy, edy, _mm256_mul_ps(edz, edz)));
        __m256 lo = _mm256_min_ps(ed2, _mm256_permute2f128_ps(ed2, ed2, 1));
        lo = _mm256_min_ps(lo, _mm256_shuffle_ps(lo, lo, _MM_SHUFFLE(2, 3, 0, 1)));
        lo = _mm256_min_ps(lo, _mm256_shuffle_ps(lo, lo, _MM_SHUFFLE(1, 0, 3, 2)));
        float min_ed2;
        _mm_store_ss(&min_ed2, _mm256_castps256_ps128(lo));
        float min_dist_to_sphere = sqrtf(min_ed2) - half_diag;
        __m256 bd_lo = _mm256_min_ps(best_d, _mm256_permute2f128_ps(best_d, best_d, 1));
        bd_lo = _mm256_min_ps(bd_lo, _mm256_shuffle_ps(bd_lo, bd_lo, _MM_SHUFFLE(2, 3, 0, 1)));
        bd_lo = _mm256_min_ps(bd_lo, _mm256_shuffle_ps(bd_lo, bd_lo, _MM_SHUFFLE(1, 0, 3, 2)));
        float cur_best;
        _mm_store_ss(&cur_best, _mm256_castps256_ps128(bd_lo));
        if (min_dist_to_sphere > cur_best)
            continue;

        // Prefetch next valid grid's data before the gather
        for (int ngi = gi + 1; ngi < g_ngrid; ngi++)
        {
            if (!g_grids[ngi].valid)
                continue;
            const char *nbase = reinterpret_cast<const char *>(g_grids[ngi].data);
            int n3 = g_grids[ngi].N * g_grids[ngi].N * g_grids[ngi].N;
            _mm_prefetch(nbase, _MM_HINT_T1);
            _mm_prefetch(nbase + 64, _MM_HINT_T1);
            _mm_prefetch(nbase + 128, _MM_HINT_T1);
            _mm_prefetch(nbase + 192, _MM_HINT_T1);
            _mm_prefetch(nbase + (n3 / 2) * 4, _MM_HINT_T2);
            break;
        }

        __m256 d = sdf_grid_query_avx8(g, px, py, pz);
        update8(d, g.mat);
    }

    return {best_d, best_mat};
#endif // __AVX2__
}

// ── bit8_to_avx: expand 8-bit lane mask to 8×32-bit AVX2 mask ─────────────────
// Replaces the scalar "for j in 0..8: arr[j] = (mask>>j)&1 ? -1 : 0" pattern.
// Pure register ops — no stores, no scalar loop. Used in every march inner loop.
static inline __m256i bit8_to_avx(__m256i shifts, uint8_t mask)
{
    // Broadcast the byte to all 8 int32 lanes, shift each right by its lane
    // index (0–7), mask to bit 0, then compare == 1 to get 0x00000000/0xFFFFFFFF.
    __m256i v = _mm256_srlv_epi32(_mm256_set1_epi32((int)(uint32_t)mask), shifts);
    return _mm256_cmpeq_epi32(_mm256_and_si256(v, _mm256_set1_epi32(1)),
                              _mm256_set1_epi32(1));
}

// ── ray_march_avx8: 8 primary rays, standard sphere tracing + fast mask expand ─
// Standard sphere tracing (ω=1) is used here because loaded mesh SDFs (arch, torus,
// sphere) are grid-sampled and can underestimate distances near edges — relaxed
// tracing (ω>1) causes tunneling artifacts on those surfaces. The main speedup
// comes from bit8_to_avx (no per-step scalar loop) and the vectorized shading path.
static March8 ray_march_avx8(
    __m256 rox, __m256 roy, __m256 roz,
    __m256 rdx, __m256 rdy, __m256 rdz,
    float t_max = MAX_DIST, int terrain_oct = 7,
    uint8_t initial_active = 0xFF)
{
    const __m256i lane_shifts = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    const __m256 surf_v = _mm256_set1_ps(SURF_DIST);
    const __m256 tmax_v = _mm256_set1_ps(t_max);
    const __m256 omega = _mm256_set1_ps(1.0f);

    __m256 t = _mm256_set1_ps(0.02f);
    uint8_t active = initial_active;
    uint8_t hit_mask = 0;

    __m256 hit_x = _mm256_setzero_ps();
    __m256 hit_y = _mm256_setzero_ps();
    __m256 hit_z = _mm256_setzero_ps();
    __m256i hit_mat = _mm256_setzero_si256();

    for (int i = 0; i < MAX_STEPS && active; i++)
    {
#ifdef __AVX2__
        __m256 px = _mm256_fmadd_ps(rdx, t, rox);
        __m256 py = _mm256_fmadd_ps(rdy, t, roy);
        __m256 pz = _mm256_fmadd_ps(rdz, t, roz);
#else
        __m256 px = _mm256_add_ps(_mm256_mul_ps(rdx, t), rox);
        __m256 py = _mm256_add_ps(_mm256_mul_ps(rdy, t), roy);
        __m256 pz = _mm256_add_ps(_mm256_mul_ps(rdz, t), roz);
#endif
        EvalResult8 res = eval_scene_avx8(px, py, pz, terrain_oct);
        __m256 d = res.d;

        // Active mask → __m256 (pure register ops via srlv+cmpeq, no scalar loop)
        __m256 act_f = _mm256_castsi256_ps(bit8_to_avx(lane_shifts, active));

        // Advance: t += ω·d for active lanes only
        t = _mm256_add_ps(t, _mm256_and_ps(_mm256_mul_ps(omega, d), act_f));

        // Hit detection: d < SURF_DIST and lane active
        int hit_int = _mm256_movemask_ps(_mm256_cmp_ps(d, surf_v, _CMP_LT_OQ)) & active;
        if (hit_int)
        {
            __m256i jh_m = bit8_to_avx(lane_shifts, (uint8_t)hit_int);
            __m256 jh_f = _mm256_castsi256_ps(jh_m);
            hit_x = _mm256_blendv_ps(hit_x, px, jh_f);
            hit_y = _mm256_blendv_ps(hit_y, py, jh_f);
            hit_z = _mm256_blendv_ps(hit_z, pz, jh_f);
#ifdef __AVX2__
            hit_mat = _mm256_blendv_epi8(hit_mat, res.mat, jh_m);
#else
            alignas(32) int hm[8], rm[8];
            _mm256_store_si256((__m256i *)hm, hit_mat);
            _mm256_store_si256((__m256i *)rm, res.mat);
            alignas(32) int jhi_a[8];
            _mm256_store_si256((__m256i *)jhi_a, jh_m);
            for (int j = 0; j < 8; j++)
                if (jhi_a[j])
                    hm[j] = rm[j];
            hit_mat = _mm256_loadu_si256((__m256i *)hm);
#endif
            hit_mask |= (uint8_t)hit_int;
            active &= ~(uint8_t)hit_int;
        }
        // Escape: t >= t_max
        active &= ~(uint8_t)_mm256_movemask_ps(_mm256_cmp_ps(t, tmax_v, _CMP_GE_OQ));
    }
    return {hit_x, hit_y, hit_z, hit_mat, hit_mask, t};
}

// ── soft_shadow_avx8: 8 shadow rays, Quilez penumbra ─────────────────────────
static __m256 soft_shadow_avx8(
    __m256 rox, __m256 roy, __m256 roz,
    __m256 rdx, __m256 rdy, __m256 rdz,
    float k = 14.f, int max_steps = 64,
    uint8_t initial_active = 0xFF)
{
    const int SHADOW_STEPS = max_steps;
    const float T_MAX = 60.f;

    const __m256 k_v = _mm256_set1_ps(k);
    const __m256 zero8 = _mm256_setzero_ps();
    const __m256 one8 = _mm256_set1_ps(1.f);
    const __m256 eps_v = _mm256_set1_ps(0.001f);
    const __m256 tmax_v = _mm256_set1_ps(T_MAX);
    const __m256 hit_v = _mm256_set1_ps(0.001f);

    __m256 t = _mm256_set1_ps(0.05f);
    __m256 res = one8;
    __m256 ph = _mm256_set1_ps(1e20f);
    uint8_t active = initial_active;

    const __m256i lane_shifts = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    for (int i = 0; i < SHADOW_STEPS && active; i++)
    {
        __m256 act_f = _mm256_castsi256_ps(bit8_to_avx(lane_shifts, active));

#ifdef __AVX2__
        __m256 px = _mm256_fmadd_ps(rdx, t, rox);
        __m256 py = _mm256_fmadd_ps(rdy, t, roy);
        __m256 pz = _mm256_fmadd_ps(rdz, t, roz);
#else
        __m256 px = _mm256_add_ps(_mm256_mul_ps(rdx, t), rox);
        __m256 py = _mm256_add_ps(_mm256_mul_ps(rdy, t), roy);
        __m256 pz = _mm256_add_ps(_mm256_mul_ps(rdz, t), roz);
#endif
        // Shadow march uses reduced FBM octaves (4): ~43% less terrain eval cost.
        EvalResult8 ev = eval_scene_avx8(px, py, pz, 3, false);
        __m256 d = ev.d;

        // Blocked lanes → res = 0
        int blk_i = _mm256_movemask_ps(_mm256_cmp_ps(d, hit_v, _CMP_LT_OQ)) & active;
        if (blk_i)
        {
            __m256 bm = _mm256_castsi256_ps(bit8_to_avx(lane_shifts, (uint8_t)blk_i));
            res = _mm256_andnot_ps(bm, res);
            active &= ~(uint8_t)blk_i;
        }

        // Quilez improved penumbra
        __m256 d2 = _mm256_mul_ps(d, d);
        __m256 y2 = _mm256_div_ps(d2, _mm256_add_ps(_mm256_add_ps(ph, ph), eps_v));
        __m256 arg = _mm256_max_ps(zero8, _mm256_sub_ps(d2, _mm256_mul_ps(y2, y2)));
        __m256 di = _mm256_sqrt_ps(arg);
        __m256 den = _mm256_max_ps(eps_v, _mm256_sub_ps(t, y2));
        __m256 cnd = _mm256_div_ps(_mm256_mul_ps(k_v, di), den);
        __m256 new_res = _mm256_min_ps(res, cnd);
        res = _mm256_blendv_ps(res, new_res, act_f);
        ph = _mm256_blendv_ps(ph, d, act_f);
        t = _mm256_add_ps(t, _mm256_and_ps(d, act_f));

        int esc = _mm256_movemask_ps(_mm256_cmp_ps(t, tmax_v, _CMP_GE_OQ));
        active &= ~(uint8_t)esc;

        // Early-out: retire lanes whose res has dropped below 0.001
        int dark = _mm256_movemask_ps(_mm256_cmp_ps(res, eps_v, _CMP_LE_OQ));
        active &= ~(uint8_t)dark;
    }
    return _mm256_max_ps(zero8, _mm256_min_ps(one8, res));
}

// ── scene_normal_avx8: central-difference normals for 8 hits at once ────────
// 6 eval_scene_avx8 calls instead of 8×6=48 scalar eval_scene calls.
// Outputs normalised (nx,ny,nz) stored into caller-supplied arrays.
#ifdef __AVX2__
static void scene_normal_avx8(
    const float *hx, const float *hy, const float *hz,
    float *nx_out, float *ny_out, float *nz_out,
    int oct = 7)
{
    const float e = 0.003f;
    __m256 px = _mm256_load_ps(hx), py = _mm256_load_ps(hy), pz = _mm256_load_ps(hz);
    __m256 ve = _mm256_set1_ps(e);

    auto d8 = [](const EvalResult8 &r)
    { return r.d; };

    __m256 nx = _mm256_sub_ps(
        d8(eval_scene_avx8(_mm256_add_ps(px, ve), py, pz, oct, false)),
        d8(eval_scene_avx8(_mm256_sub_ps(px, ve), py, pz, oct, false)));
    __m256 ny = _mm256_sub_ps(
        d8(eval_scene_avx8(px, _mm256_add_ps(py, ve), pz, oct, false)),
        d8(eval_scene_avx8(px, _mm256_sub_ps(py, ve), pz, oct, false)));
    __m256 nz = _mm256_sub_ps(
        d8(eval_scene_avx8(px, py, _mm256_add_ps(pz, ve), oct, false)),
        d8(eval_scene_avx8(px, py, _mm256_sub_ps(pz, ve), oct, false)));

    // Normalise per-lane: rsqrt + 1 Newton-Raphson step (~2× faster than sqrt+div)
    __m256 len2 = _mm256_fmadd_ps(nx, nx, _mm256_fmadd_ps(ny, ny, _mm256_mul_ps(nz, nz)));
    __m256 r0 = _mm256_rsqrt_ps(_mm256_max_ps(len2, _mm256_set1_ps(1e-18f)));
    // NR: inv = r0 * (1.5 - 0.5 * len2 * r0 * r0)
    __m256 inv = _mm256_mul_ps(r0,
                               _mm256_fnmadd_ps(_mm256_mul_ps(_mm256_set1_ps(0.5f), len2),
                                                _mm256_mul_ps(r0, r0),
                                                _mm256_set1_ps(1.5f)));
    _mm256_store_ps(nx_out, _mm256_mul_ps(nx, inv));
    _mm256_store_ps(ny_out, _mm256_mul_ps(ny, inv));
    _mm256_store_ps(nz_out, _mm256_mul_ps(nz, inv));
}

// ── calc_ao_avx8: ambient occlusion for 8 hits at once ───────────────────────
// 6 eval_scene_avx8 calls instead of 8×6=48 scalar eval_scene calls.
static __m256 calc_ao_avx8(
    const float *hx, const float *hy, const float *hz,
    const float *nx, const float *ny, const float *nz)
{
    __m256 px = _mm256_load_ps(hx), py = _mm256_load_ps(hy), pz = _mm256_load_ps(hz);
    __m256 vn_x = _mm256_load_ps(nx), vn_y = _mm256_load_ps(ny), vn_z = _mm256_load_ps(nz);

    __m256 occ = _mm256_setzero_ps();
    __m256 sc = _mm256_set1_ps(1.f);
    static const float h2_steps[6] = {0.03f, 0.11f, 0.19f, 0.27f, 0.35f, 0.43f};

    for (int i = 0; i < 6; i++)
    {
        __m256 h2 = _mm256_set1_ps(h2_steps[i]);
        // sp = hit + normal * h2
        __m256 spx = _mm256_fmadd_ps(vn_x, h2, px);
        __m256 spy = _mm256_fmadd_ps(vn_y, h2, py);
        __m256 spz = _mm256_fmadd_ps(vn_z, h2, pz);
        __m256 d = eval_scene_avx8(spx, spy, spz, 2, false).d; // AO: 2 octaves sufficient
        // occ += (h2 - d) * sc
        occ = _mm256_fmadd_ps(_mm256_sub_ps(h2, d), sc, occ);
        sc = _mm256_mul_ps(sc, _mm256_set1_ps(0.75f));
    }
    // clamp(1 - 2.5*occ, 0, 1)
    __m256 ao = _mm256_sub_ps(_mm256_set1_ps(1.f),
                              _mm256_mul_ps(_mm256_set1_ps(2.5f), occ));
    return _mm256_min_ps(_mm256_set1_ps(1.f), _mm256_max_ps(_mm256_setzero_ps(), ao));
}
#endif // __AVX2__

// ── get_mat_avx8: vectorised material lookup for 8 lanes ─────────────────────
// Runs avx8_fbm only for materials that need it; blends all 8 lanes in-register.
// Roughly 8× cheaper than 8 scalar get_mat calls at material-boundary tiles,
// and equal or faster on coherent tiles (FBM is compute-bound, not branch-bound).
#ifdef __AVX2__
static inline void get_mat_avx8(
    const int *hm, const float *hx, const float *hz,
    float *ar, float *ag, float *ab, float *rough)
{
    __m256 px = _mm256_load_ps(hx), pz = _mm256_load_ps(hz);
    __m256i mi = _mm256_load_si256((const __m256i *)hm);

    // Material masks (integer cmpeq, cast to float for blendv)
    auto mmask = [&](__m256i code)
    { return _mm256_castsi256_ps(_mm256_cmpeq_epi32(mi, code)); };
    __m256 m_grass = mmask(_mm256_set1_epi32(MAT_GRASS));
    __m256 m_rock = mmask(_mm256_set1_epi32(MAT_ROCK));
    __m256 m_snow = mmask(_mm256_set1_epi32(MAT_SNOW));
    __m256 m_water = mmask(_mm256_set1_epi32(MAT_WATER));
    __m256 m_trunk = mmask(_mm256_set1_epi32(MAT_TREE_TRUNK));
    __m256 m_leaf = mmask(_mm256_set1_epi32(MAT_TREE_LEAF));
    __m256 m_mstone = mmask(_mm256_set1_epi32(MAT_MESH_STONE));
    __m256 m_gold = mmask(_mm256_set1_epi32(MAT_MESH_GOLD));
    __m256 m_xtal = mmask(_mm256_set1_epi32(MAT_MESH_CRYSTAL));

    // FBM variants (computed unconditionally; cost amortised across all 8 lanes)
    __m256 g1 = avx8_fbm(_mm256_mul_ps(px, _mm256_set1_ps(.5f)),
                         _mm256_mul_ps(pz, _mm256_set1_ps(.5f)), 2);
    __m256 g2 = avx8_fbm(_mm256_fmadd_ps(px, _mm256_set1_ps(.4f), _mm256_set1_ps(5.f)),
                         _mm256_mul_ps(pz, _mm256_set1_ps(.4f)), 2);
    __m256 rf = avx8_fbm(_mm256_mul_ps(px, _mm256_set1_ps(.8f)),
                         _mm256_mul_ps(pz, _mm256_set1_ps(.8f)), 3);
    __m256 lf = avx8_fbm(_mm256_mul_ps(px, _mm256_set1_ps(2.f)),
                         _mm256_mul_ps(pz, _mm256_set1_ps(2.f)), 2);
    __m256 msf = avx8_fbm(_mm256_fmadd_ps(px, _mm256_set1_ps(2.1f), _mm256_set1_ps(3.f)),
                          _mm256_fmadd_ps(pz, _mm256_set1_ps(2.1f), _mm256_set1_ps(7.f)), 3);
    __m256 msn = _mm256_mul_ps(msf, _mm256_set1_ps(.15f));

    // Per-material albedo + roughness vectors
    __m256 vr[9], vg[9], vb[9], vq[9];
    // 0 = default
    vr[0] = vg[0] = vb[0] = _mm256_set1_ps(.5f);
    vq[0] = _mm256_set1_ps(.8f);
    // GRASS
    vr[1] = _mm256_fmadd_ps(g1, _mm256_set1_ps(.12f), _mm256_set1_ps(.15f));
    vg[1] = _mm256_fmadd_ps(g2, _mm256_set1_ps(.1f), _mm256_set1_ps(.28f));
    vb[1] = _mm256_set1_ps(.07f);
    vq[1] = _mm256_set1_ps(.95f);
    // ROCK
    vr[2] = _mm256_fmadd_ps(rf, _mm256_set1_ps(.1f), _mm256_set1_ps(.3f));
    vg[2] = _mm256_fmadd_ps(rf, _mm256_set1_ps(.08f), _mm256_set1_ps(.28f));
    vb[2] = _mm256_fmadd_ps(rf, _mm256_set1_ps(.06f), _mm256_set1_ps(.25f));
    vq[2] = _mm256_set1_ps(.8f);
    // SNOW
    vr[3] = _mm256_set1_ps(.9f);
    vg[3] = _mm256_set1_ps(.92f);
    vb[3] = _mm256_set1_ps(.95f);
    vq[3] = _mm256_set1_ps(.55f);
    // WATER
    vr[4] = _mm256_set1_ps(.04f);
    vg[4] = _mm256_set1_ps(.1f);
    vb[4] = _mm256_set1_ps(.2f);
    vq[4] = _mm256_set1_ps(.0f);
    // TREE_TRUNK
    vr[5] = _mm256_set1_ps(.3f);
    vg[5] = _mm256_set1_ps(.18f);
    vb[5] = _mm256_set1_ps(.08f);
    vq[5] = _mm256_set1_ps(.95f);
    // TREE_LEAF
    vr[6] = _mm256_fmadd_ps(lf, _mm256_set1_ps(.06f), _mm256_set1_ps(.08f));
    vg[6] = _mm256_fmadd_ps(lf, _mm256_set1_ps(.1f), _mm256_set1_ps(.28f));
    vb[6] = _mm256_set1_ps(.06f);
    vq[6] = _mm256_set1_ps(.9f);
    // MESH_STONE
    vr[7] = _mm256_add_ps(_mm256_set1_ps(.45f), msn);
    vg[7] = _mm256_add_ps(_mm256_set1_ps(.42f), msn);
    vb[7] = _mm256_add_ps(_mm256_set1_ps(.38f), msn);
    vq[7] = _mm256_set1_ps(.75f);
    // MESH_GOLD
    vr[8] = _mm256_set1_ps(.9f);
    vg[8] = _mm256_set1_ps(.72f);
    vb[8] = _mm256_set1_ps(.2f);
    vq[8] = _mm256_set1_ps(.15f);

    // Crystal uses same slot reuse — handled after the blend
    // Blend: start from default, overlay each material
    __m256 out_r = vr[0], out_g = vg[0], out_b = vb[0], out_q = vq[0];
    const __m256 *masks[8] = {&m_grass, &m_rock, &m_snow, &m_water,
                              &m_trunk, &m_leaf, &m_mstone, &m_gold};
    for (int k = 0; k < 8; k++)
    {
        out_r = _mm256_blendv_ps(out_r, vr[k + 1], *masks[k]);
        out_g = _mm256_blendv_ps(out_g, vg[k + 1], *masks[k]);
        out_b = _mm256_blendv_ps(out_b, vb[k + 1], *masks[k]);
        out_q = _mm256_blendv_ps(out_q, vq[k + 1], *masks[k]);
    }
    // Crystal: {.5,.8,.95}, .05
    out_r = _mm256_blendv_ps(out_r, _mm256_set1_ps(.5f), m_xtal);
    out_g = _mm256_blendv_ps(out_g, _mm256_set1_ps(.8f), m_xtal);
    out_b = _mm256_blendv_ps(out_b, _mm256_set1_ps(.95f), m_xtal);
    out_q = _mm256_blendv_ps(out_q, _mm256_set1_ps(.05f), m_xtal);

    _mm256_store_ps(ar, out_r);
    _mm256_store_ps(ag, out_g);
    _mm256_store_ps(ab, out_b);
    _mm256_store_ps(rough, out_q);
}
#endif // __AVX2__

// ── shade_water_batch_avx8: batched water shading for up to 8 water lanes ────
// Mirrors shade_water_batch_avx512 at 8-wide.
// Collects water lanes, fires one ray_march_avx8 for all reflection rays, then
// uses scene_normal_avx8 and soft_shadow_avx8 on the reflected hits.
// Replaces up to 8 scalar ray_march calls with one 8-wide vectorised march.
#ifdef __AVX2__
static void shade_water_batch_avx8(
    int n_water,
    const float *wpx, const float *wpy, const float *wpz,
    const float *wnx, const float *wny, const float *wnz,
    const float *wrdx, const float *wrdy, const float *wrdz,
    float *col_r, float *col_g, float *col_b,
    int pixel_y = 0, const int *water_lane = nullptr, int pixel_x_base = 0,
    const GICache *prev_gi = nullptr, const Camera *prev_cam = nullptr,
    GICache *curr_gi = nullptr, const int *lane_x = nullptr)
{
    alignas(32) float refl_r[8] = {}, refl_g[8] = {}, refl_b[8] = {};

    // If every water lane has a valid previous-frame reflection, skip the
    // reflection march, reflected-hit normals and reflected shadows entirely.
    bool all_cached = false;
    if (prev_gi && curr_gi && water_lane && prev_gi->W == curr_gi->W &&
        prev_gi->H == curr_gi->H)
    {
        all_cached = true;
        for (int wi = 0; wi < n_water; wi++)
        {
            int px_coord = lane_x ? lane_x[water_lane[wi]] : pixel_x_base + water_lane[wi];
            int idx = pixel_y * prev_gi->W + px_coord;
            if (px_coord < 0 || px_coord >= prev_gi->W || idx < 0 ||
                idx >= prev_gi->W * prev_gi->H)
            {
                all_cached = false;
                break;
            }
            if (prev_gi->refl_r[idx] + prev_gi->refl_g[idx] + prev_gi->refl_b[idx] <= 0.001f)
            {
                all_cached = false;
                break;
            }
        }
    }

    if (!all_cached)
    {
        // ── E: Grazing-angle early exit ───────────────────────────────────────────
        // Reflected rays nearly parallel to water → will always see sky. Skip march.
        bool grazing[8] = {};
        for (int i = 0; i < n_water; i++)
        {
            float nx = wnx[i], ny = wny[i], nz = wnz[i];
            float dx = wrdx[i], dy = wrdy[i], dz = wrdz[i];
            float dot_rn = dx * nx + dy * ny + dz * nz;
            float rdy_r = dy - 2.f * dot_rn * ny; // reflected y component
            grazing[i] = (rdy_r < 0.08f);
        }

        // ── F: Half-resolution — build only even-indexed reflection rays ──────────
        // We march ceil(n_water/2) rays and copy results to odd neighbours.
        int n_march = (n_water + 1) / 2; // how many unique rays to march

        // Build reflection ray batch, padded to 8 (only even lanes)
        alignas(32) float rox[8], roy[8], roz[8];
        alignas(32) float rdx[8], rdy[8], rdz[8];
        for (int i = 0; i < 8; i++)
        {
            int even_i = std::min(i * 2, n_water - 1); // even source lane
            int si = (i < n_march) ? even_i : even_i;
            si = std::min(si, n_water - 1);
            float nx = wnx[si], ny = wny[si], nz = wnz[si];
            float dx = wrdx[si], dy = wrdy[si], dz = wrdz[si];
            float dot_rn = dx * nx + dy * ny + dz * nz;
            rdx[i] = dx - 2.f * dot_rn * nx;
            rdy[i] = dy - 2.f * dot_rn * ny;
            rdz[i] = dz - 2.f * dot_rn * nz;
            rox[i] = wpx[si] + nx * 0.01f;
            roy[i] = wpy[si] + ny * 0.01f;
            roz[i] = wpz[si] + nz * 0.01f;
        }

        // ── A: Reflection march with reduced oct=3 and t_max=80 ──────────────────
        March8 rm = ray_march_avx8(
            _mm256_load_ps(rox), _mm256_load_ps(roy), _mm256_load_ps(roz),
            _mm256_load_ps(rdx), _mm256_load_ps(rdy), _mm256_load_ps(rdz),
            /*t_max=*/80.f, /*oct=*/3);

        alignas(32) float rhx[8], rhy[8], rhz[8];
        alignas(32) int rhm[8];
        _mm256_store_ps(rhx, rm.pos_x);
        _mm256_store_ps(rhy, rm.pos_y);
        _mm256_store_ps(rhz, rm.pos_z);
        _mm256_store_si256((__m256i *)rhm, rm.mat);

        // ── B: Normals at oct=3 ───────────────────────────────────────────────────
        alignas(32) float rnx[8], rny[8], rnz[8];
        scene_normal_avx8(rhx, rhy, rhz, rnx, rny, rnz, /*oct=*/3);

        // Shadow ray origins + soft_shadow_avx8 at reduced steps
        alignas(32) float sox[8], soy[8], soz[8];
        for (int i = 0; i < 8; i++)
        {
            sox[i] = rhx[i] + rnx[i] * 0.02f;
            soy[i] = rhy[i] + rny[i] * 0.02f;
            soz[i] = rhz[i] + rnz[i] * 0.02f;
        }
        __m256 sun_x8 = _mm256_set1_ps(sun_dir.x);
        __m256 sun_y8 = _mm256_set1_ps(sun_dir.y);
        __m256 sun_z8 = _mm256_set1_ps(sun_dir.z);
        // ── C: Shadow at 32 steps instead of 64 ──────────────────────────────────
        __m256 refl_sha8 = soft_shadow_avx8(
            _mm256_load_ps(sox), _mm256_load_ps(soy), _mm256_load_ps(soz),
            sun_x8, sun_y8, sun_z8, 10.f, /*max_steps=*/32);

        // Diffuse dot on reflected hit normals (from half-res march results)
        __m256 vrnx = _mm256_load_ps(rnx), vrny = _mm256_load_ps(rny), vrnz = _mm256_load_ps(rnz);
        __m256 diff_r8 = _mm256_max_ps(_mm256_setzero_ps(),
                                       _mm256_add_ps(_mm256_mul_ps(vrnx, sun_x8),
                                                     _mm256_add_ps(_mm256_mul_ps(vrny, sun_y8),
                                                                   _mm256_mul_ps(vrnz, sun_z8))));

        alignas(32) float sha_arr8[8], diff_arr8[8];
        _mm256_store_ps(sha_arr8, refl_sha8);
        _mm256_store_ps(diff_arr8, diff_r8);

        // Build per-even-lane reflection colours
        float even_refl_r[8], even_refl_g[8], even_refl_b[8];
        for (int i = 0; i < 8; i++)
        {
            bool hit_r = (rm.hit_mask >> i) & 1;
            if (hit_r && rhm[i] != MAT_WATER)
            {
                MatProps mp2 = get_mat(rhm[i], {rhx[i], rhy[i], rhz[i]});
                float shade = diff_arr8[i] * sha_arr8[i] * 1.5f + 0.15f;
                even_refl_r[i] = mp2.albedo.x * shade;
                even_refl_g[i] = mp2.albedo.y * shade;
                even_refl_b[i] = mp2.albedo.z * shade;
            }
            else
            {
                V3 sky = sky_color({rdx[i], rdy[i], rdz[i]});
                even_refl_r[i] = sky.x;
                even_refl_g[i] = sky.y;
                even_refl_b[i] = sky.z;
            }
        }

        // ── F: Copy half-res results — odd lanes copy from even neighbours ────────
        for (int i = 0; i < n_water; i++)
        {
            int src_i = (i / 2); // even lane in the half-res batch
            refl_r[i] = even_refl_r[src_i];
            refl_g[i] = even_refl_g[src_i];
            refl_b[i] = even_refl_b[src_i];
        }

        // ── E: Override grazing pixels with sky ───────────────────────────────────
        for (int i = 0; i < n_water; i++)
        {
            if (grazing[i])
            {
                float nx = wnx[i], ny = wny[i], nz = wnz[i];
                float dx = wrdx[i], dy = wrdy[i], dz = wrdz[i];
                float dot_rn = dx * nx + dy * ny + dz * nz;
                float rrdx = dx - 2.f * dot_rn * nx;
                float rrdy = dy - 2.f * dot_rn * ny;
                float rrdz = dz - 2.f * dot_rn * nz;
                V3 sky = sky_color({rrdx, rrdy, rrdz});
                refl_r[i] = sky.x;
                refl_g[i] = sky.y;
                refl_b[i] = sky.z;
            }
        }
    }
    else
    {
        for (int wi = 0; wi < n_water; wi++)
        {
            int px_coord = lane_x ? lane_x[water_lane[wi]] : pixel_x_base + water_lane[wi];
            int idx = pixel_y * prev_gi->W + px_coord;
            refl_r[wi] = prev_gi->refl_r[idx];
            refl_g[wi] = prev_gi->refl_g[idx];
            refl_b[wi] = prev_gi->refl_b[idx];
        }
    }

    // ── D: Temporal reprojection — blend with previous frame's reflection ─────
    const float REFL_ALPHA = 0.3f; // 30% new, 70% cached
    if (prev_gi && curr_gi && water_lane)
    {
        for (int wi = 0; wi < n_water; wi++)
        {
            int px_coord = lane_x ? lane_x[water_lane[wi]] : pixel_x_base + water_lane[wi];
            int idx = pixel_y * curr_gi->W + px_coord;
            if (idx >= 0 && idx < curr_gi->W * curr_gi->H)
            {
                float old_r = prev_gi->refl_r[idx];
                float old_g = prev_gi->refl_g[idx];
                float old_b = prev_gi->refl_b[idx];
                // only blend if previous frame had a valid reflection (non-zero)
                if (old_r + old_g + old_b > 0.001f)
                {
                    refl_r[wi] = REFL_ALPHA * refl_r[wi] + (1.f - REFL_ALPHA) * old_r;
                    refl_g[wi] = REFL_ALPHA * refl_g[wi] + (1.f - REFL_ALPHA) * old_g;
                    refl_b[wi] = REFL_ALPHA * refl_b[wi] + (1.f - REFL_ALPHA) * old_b;
                }
                curr_gi->refl_r[idx] = refl_r[wi];
                curr_gi->refl_g[idx] = refl_g[wi];
                curr_gi->refl_b[idx] = refl_b[wi];
            }
        }
    }

    // Per-lane Fresnel blend, water colour, caustic shimmer, specular
    for (int i = 0; i < n_water; i++)
    {
        float nx = wnx[i], ny = wny[i], nz = wnz[i];
        float dx = wrdx[i], dy = wrdy[i], dz = wrdz[i];
        float ndv = std::max(0.f, -(dx * nx + dy * ny + dz * nz));
        float fres = 0.04f + 0.96f * powf(1.f - ndv, 5.f);

        float depth_f = smoothstep(0.f, 3.f, fabsf(terrain_h(wpx[i], wpz[i])));
        float caus = powf(std::max(0.f, sinf(wpx[i] * 2.3f) * sinf(wpz[i] * 1.9f)), 4.f) * 0.5f;
        float wc_r = (0.1f + (0.01f - 0.1f) * depth_f) + 0.1f * caus;
        float wc_g = (0.3f + (0.08f - 0.3f) * depth_f) + 0.15f * caus;
        float wc_b = (0.4f + (0.18f - 0.4f) * depth_f) + 0.1f * caus;

        float cr = wc_r + (refl_r[i] - wc_r) * fres;
        float cg = wc_g + (refl_g[i] - wc_g) * fres;
        float cb = wc_b + (refl_b[i] - wc_b) * fres;

        float hx2 = sun_dir.x - dx, hy2 = sun_dir.y - dy, hz2 = sun_dir.z - dz;
        float hlen = sqrtf(hx2 * hx2 + hy2 * hy2 + hz2 * hz2) + 1e-6f;
        hx2 /= hlen;
        hy2 /= hlen;
        hz2 /= hlen;
        float spec = powf(std::max(0.f, nx * hx2 + ny * hy2 + nz * hz2), 256.f) * 2.f;

        col_r[i] = cr + 1.4f * spec;
        col_g[i] = cg + 1.2f * spec;
        col_b[i] = cb + 0.8f * spec;
    }
}
#endif // __AVX2__

// ── shade_batch_avx8: shading for 8-lane tile (shared AVX / AVX2) ────────────
// Same logic as shade_batch_avx512; shadow + GI batched at 8-wide.
// Reprojection parameters mirror the 16-wide version exactly.
// nx_out/ny_out/nz_out: receives the computed primary-hit normals (world space),
//   already computed by scene_normal_avx8 — caller uses these for the denoiser
//   nrm buffer, avoiding a redundant scene_normal call per pixel.
static void shade_batch_avx8(
    const float *hx, const float *hy, const float *hz,
    const int *hm,
    uint8_t hit_mask,
    const float *rdx0, const float *rdy0, const float *rdz0,
    V3 cam_pos, int tid,
    const GICache *prev_gi, const Camera *prev_cam,
    GICache *curr_gi, int pixel_y, int pixel_x_base,
    float *col_r, float *col_g, float *col_b,
    float *nx_out = nullptr, float *ny_out = nullptr, float *nz_out = nullptr,
    const int *lane_x = nullptr)
{
    const float REPROJ_ALPHA = 0.25f;
    const float DEPTH_THRESH = 0.5f;

    // ── Fast path: all 8 lanes missed → pure sky tile ─────────────────────────
    // Skip all shadow/AO/GI/normal work. Saves ~6+36+96+6 eval_scene_avx8 calls
    // per tile for the top ~30% of the image that is typically sky.
    if (!hit_mask)
    {
        for (int i = 0; i < 8; i++)
        {
            V3 sky = sky_color({rdx0[i], rdy0[i], rdz0[i]});
            col_r[i] = sky.x;
            col_g[i] = sky.y;
            col_b[i] = sky.z;
            if (curr_gi)
            {
                int xi = lane_x ? lane_x[i] : pixel_x_base + i;
                if (xi >= 0 && xi < curr_gi->W)
                {
                    int ci = pixel_y * curr_gi->W + xi;
                    curr_gi->depth[ci] = 1e30f;
                    curr_gi->mat[ci] = -1;
                    curr_gi->nx[ci] = curr_gi->ny[ci] = curr_gi->nz[ci] = 0.f;
                    curr_gi->gi_r[ci] = curr_gi->gi_g[ci] = curr_gi->gi_b[ci] = 0.f;
                    curr_gi->sha[ci] = curr_gi->ao[ci] = 1.f;
                }
            }
        }
        if (nx_out)
            memset(nx_out, 0, 32);
        if (ny_out)
            memset(ny_out, 0, 32);
        if (nz_out)
            memset(nz_out, 0, 32);
        return;
    }

    bool is_special[8];
    V3 repro_gi[8];
    float repro_sha[8], repro_ao[8];
    bool repro_valid[8];
    for (int i = 0; i < 8; i++)
    {
        repro_gi[i] = {-1, -1, -1};
        repro_sha[i] = 1.f;
        repro_ao[i] = 1.f;
        repro_valid[i] = false;
    }

    // 1a. Sky classification (scalar) + special-material flag
    for (int i = 0; i < 8; i++)
    {
        bool hit = (hit_mask >> i) & 1;
        if (!hit)
        {
            V3 sky = sky_color({rdx0[i], rdy0[i], rdz0[i]});
            col_r[i] = sky.x;
            col_g[i] = sky.y;
            col_b[i] = sky.z;
            is_special[i] = true;
            if (curr_gi)
            {
                int xi = lane_x ? lane_x[i] : pixel_x_base + i;
                if (xi >= 0 && xi < curr_gi->W)
                {
                    int ci = pixel_y * curr_gi->W + xi;
                    curr_gi->depth[ci] = 1e30f;
                    curr_gi->mat[ci] = -1;
                    curr_gi->nx[ci] = curr_gi->ny[ci] = curr_gi->nz[ci] = 0.f;
                    curr_gi->gi_r[ci] = curr_gi->gi_g[ci] = curr_gi->gi_b[ci] = 0.f;
                    curr_gi->sha[ci] = curr_gi->ao[ci] = 1.f;
                }
            }
            continue;
        }
        is_special[i] = (hm[i] == MAT_WATER || hm[i] == MAT_MESH_CRYSTAL);
    }

    // 1b. Vectorised normals for all 8 hits (6 eval_scene_avx8 calls)
    alignas(32) float nx8[8], ny8[8], nz8[8];
#ifdef __AVX2__
    scene_normal_avx8(hx, hy, hz, nx8, ny8, nz8);
#else
    for (int i = 0; i < 8; i++)
    {
        V3 n = scene_normal({hx[i], hy[i], hz[i]});
        nx8[i] = n.x;
        ny8[i] = n.y;
        nz8[i] = n.z;
    }
#endif
    // Export normals for denoiser (caller uses these instead of re-computing)
    if (nx_out)
    {
        _mm256_store_ps(nx_out, _mm256_load_ps(nx8));
    }
    if (ny_out)
    {
        _mm256_store_ps(ny_out, _mm256_load_ps(ny8));
    }
    if (nz_out)
    {
        _mm256_store_ps(nz_out, _mm256_load_ps(nz8));
    }

    // 1c. Vectorised material lookup (replaces 8 scalar get_mat calls)
    alignas(32) float mat_ar[8], mat_ag[8], mat_ab[8], mat_rough[8];
#ifdef __AVX2__
    get_mat_avx8(hm, hx, hz, mat_ar, mat_ag, mat_ab, mat_rough);
#else
    for (int i = 0; i < 8; i++)
    {
        MatProps mp = get_mat(hm[i], {hx[i], hy[i], hz[i]});
        mat_ar[i] = mp.albedo.x;
        mat_ag[i] = mp.albedo.y;
        mat_ab[i] = mp.albedo.z;
        mat_rough[i] = mp.roughness;
    }
#endif

    // 2. Special materials
    // Water (AVX2): collect lanes → shade_water_batch_avx8 (vectorised reflection march)
    // Crystal: scalar rim+specular (no secondary march needed)
    // Plain AVX fallback: all special materials via scalar shade_known_hit
#ifdef __AVX2__
    {
        alignas(32) float wpx[8], wpy[8], wpz[8];
        alignas(32) float wnx_w[8], wny_w[8], wnz_w[8];
        alignas(32) float wrdx[8], wrdy[8], wrdz[8];
        int water_lane[8], n_water = 0;

        for (int i = 0; i < 8; i++)
        {
            if (!is_special[i])
                continue;
            bool hit = (hit_mask >> i) & 1;
            if (!hit)
                continue;
            if (hm[i] == MAT_WATER)
            {
                water_lane[n_water] = i;
                wpx[n_water] = hx[i];
                wpy[n_water] = hy[i];
                wpz[n_water] = hz[i];
                wnx_w[n_water] = nx8[i];
                wny_w[n_water] = ny8[i];
                wnz_w[n_water] = nz8[i];
                wrdx[n_water] = rdx0[i];
                wrdy[n_water] = rdy0[i];
                wrdz[n_water] = rdz0[i];
                n_water++;
            }
            else
            {
                // Crystal: scalar path with precomputed normal
                V3 n_i = {nx8[i], ny8[i], nz8[i]};
                March m_i;
                m_i.pos = {hx[i], hy[i], hz[i]};
                m_i.mat = hm[i];
                m_i.hit = true;
                m_i.t = 0;
                V3 col_c = shade_known_hit(cam_pos, {rdx0[i], rdy0[i], rdz0[i]}, m_i, tid, &n_i);
                col_r[i] = col_c.x;
                col_g[i] = col_c.y;
                col_b[i] = col_c.z;
            }
        }

        if (n_water > 0)
        {
            alignas(32) float wc_r[8], wc_g[8], wc_b[8];
            shade_water_batch_avx8(n_water, wpx, wpy, wpz, wnx_w, wny_w, wnz_w,
                                   wrdx, wrdy, wrdz, wc_r, wc_g, wc_b,
                                   pixel_y, water_lane, pixel_x_base,
                                   prev_gi, prev_cam, curr_gi, lane_x);
            for (int wi = 0; wi < n_water; wi++)
            {
                int i = water_lane[wi];
                col_r[i] = wc_r[wi];
                col_g[i] = wc_g[wi];
                col_b[i] = wc_b[wi];
            }
        }
    }
#else
    // Plain AVX: scalar path for all special materials
    for (int i = 0; i < 8; i++)
    {
        if (!is_special[i])
            continue;
        bool hit = (hit_mask >> i) & 1;
        if (!hit)
            continue;
        V3 n_i = {nx8[i], ny8[i], nz8[i]};
        March m_i;
        m_i.pos = {hx[i], hy[i], hz[i]};
        m_i.mat = hm[i];
        m_i.hit = true;
        m_i.t = 0;
        V3 c = shade_known_hit(cam_pos, {rdx0[i], rdy0[i], rdz0[i]}, m_i, tid, &n_i);
        col_r[i] = c.x;
        col_g[i] = c.y;
        col_b[i] = c.z;
    }
#endif

    // 3a. Temporal reprojection
    if (prev_gi && prev_cam)
    {
        for (int i = 0; i < 8; i++)
        {
            if (is_special[i])
                continue;
            bool hit = (hit_mask >> i) & 1;
            if (!hit)
                continue;
            V3 P = {hx[i], hy[i], hz[i]};
            float px, py;
            if (!project_point(P, *prev_cam, px, py))
                continue;
            int ix = (int)px, iy = (int)py;
            if (ix < 0 || ix >= prev_gi->W || iy < 0 || iy >= prev_gi->H)
                continue;
            int idx = iy * prev_gi->W + ix;
            V3 pv = P - prev_cam->pos;
            float prev_dist = sqrtf(dot(pv, pv));
            if (fabsf(prev_dist - prev_gi->depth[idx]) > DEPTH_THRESH)
                continue;
            if (prev_gi->mat[idx] != hm[i])
                continue;
            float nd = nx8[i] * prev_gi->nx[idx] + ny8[i] * prev_gi->ny[idx] + nz8[i] * prev_gi->nz[idx];
            if (nd < 0.90f)
                continue;
            repro_gi[i] = {prev_gi->gi_r[idx], prev_gi->gi_g[idx], prev_gi->gi_b[idx]};
            repro_sha[i] = prev_gi->sha[idx];
            repro_ao[i] = prev_gi->ao[idx];
            repro_valid[i] = true;
        }
    }

    // 3b. Build shadow + GI ray arrays
    alignas(32) float sha_ox[8], sha_oy[8], sha_oz[8], sha_dx[8], sha_dy[8], sha_dz[8];
    alignas(32) float gi_ox[8], gi_oy[8], gi_oz[8], gi_dx[8], gi_dy[8], gi_dz[8];
    float diff_v[8];
    bool skip_shadow[8];
    uint8_t shadow_active = 0, gi_active = 0;

    // Vectorised GI hemisphere directions for all 8 lanes (cosine-weighted, no loop)
#ifdef __AVX2__
    rand_hemi_avx8(nx8, ny8, nz8, gi_dx, gi_dy, gi_dz, tid);
#else
    for (int i = 0; i < 8; i++)
    {
        V3 gd = rand_hemi({nx8[i], ny8[i], nz8[i]}, tid);
        gi_dx[i] = gd.x;
        gi_dy[i] = gd.y;
        gi_dz[i] = gd.z;
    }
#endif

    for (int i = 0; i < 8; i++)
    {
        if (is_special[i])
        {
            sha_ox[i] = sha_oy[i] = sha_oz[i] = 0;
            sha_dx[i] = 0;
            sha_dy[i] = 1;
            sha_dz[i] = 0;
            gi_ox[i] = gi_oy[i] = gi_oz[i] = 0;
            gi_dx[i] = 0;
            gi_dy[i] = 1;
            gi_dz[i] = 0;
            diff_v[i] = 0;
            skip_shadow[i] = true;
            continue;
        }
        V3 n = {nx8[i], ny8[i], nz8[i]};
        float diff = std::max(0.f, dot(n, sun_dir));
        diff_v[i] = diff;
        skip_shadow[i] = (diff <= 0.01f) || repro_valid[i];
        sha_ox[i] = hx[i] + n.x * .02f;
        sha_oy[i] = hy[i] + n.y * .02f;
        sha_oz[i] = hz[i] + n.z * .02f;
        sha_dx[i] = sun_dir.x;
        sha_dy[i] = sun_dir.y;
        sha_dz[i] = sun_dir.z;
        gi_ox[i] = hx[i] + n.x * .02f;
        gi_oy[i] = hy[i] + n.y * .02f;
        gi_oz[i] = hz[i] + n.z * .02f;
        // gi_dx/dy/dz already filled by rand_hemi_avx8 above (or fallback loop)
        if (!skip_shadow[i])
            shadow_active |= (uint8_t)(1u << i);
        if (!repro_valid[i])
            gi_active |= (uint8_t)(1u << i);
    }

    // 4. Shadow batch (8-wide)
    alignas(32) float sha_arr[8];
    {
        __m256 sox = _mm256_load_ps(sha_ox), soy = _mm256_load_ps(sha_oy), soz = _mm256_load_ps(sha_oz);
        __m256 sdx = _mm256_load_ps(sha_dx), sdy = _mm256_load_ps(sha_dy), sdz = _mm256_load_ps(sha_dz);
        _mm256_store_ps(sha_arr, soft_shadow_avx8(sox, soy, soz, sdx, sdy, sdz, 12.f, 64, shadow_active));
        for (int i = 0; i < 8; i++)
            if (repro_valid[i])
                sha_arr[i] = repro_sha[i];
    }

    // 5. GI bounce batch (8-wide) — capped at t_max=20 (bounce lighting only
    //    needs nearby geometry; halves step count vs t_max=50 for miss rays)
    March8 gi8;
    {
        __m256 gox = _mm256_load_ps(gi_ox), goy = _mm256_load_ps(gi_oy), goz = _mm256_load_ps(gi_oz);
        __m256 gdx = _mm256_load_ps(gi_dx), gdy = _mm256_load_ps(gi_dy), gdz = _mm256_load_ps(gi_dz);
        gi8 = ray_march_avx8(gox, goy, goz, gdx, gdy, gdz, 20.f, 2, gi_active); // GI: 2 FBM octaves
    }
    alignas(32) float ghx[8], ghy[8], ghz[8];
    alignas(32) int ghm[8];
    _mm256_store_ps(ghx, gi8.pos_x);
    _mm256_store_ps(ghy, gi8.pos_y);
    _mm256_store_ps(ghz, gi8.pos_z);
    _mm256_store_si256((__m256i *)ghm, gi8.mat);

    // 6. Vectorised AO for all 8 hits (6 eval_scene_avx8 calls)
    alignas(32) float ao_arr[8];
#ifdef __AVX2__
    _mm256_store_ps(ao_arr, calc_ao_avx8(hx, hy, hz, nx8, ny8, nz8));
#else
    for (int i = 0; i < 8; i++)
        ao_arr[i] = calc_ao({hx[i], hy[i], hz[i]}, {nx8[i], ny8[i], nz8[i]});
#endif
    for (int i = 0; i < 8; i++)
        if (repro_valid[i])
            ao_arr[i] = repro_ao[i];

    // 7a. Vectorised normals at GI secondary hits (6 eval_scene_avx8 calls)
    //     Skipped entirely when no GI rays hit anything (saves 6 eval calls).
    alignas(32) float gnx8[8] = {}, gny8[8] = {}, gnz8[8] = {};
    if (gi8.hit_mask)
    {
#ifdef __AVX2__
        scene_normal_avx8(ghx, ghy, ghz, gnx8, gny8, gnz8);
#else
        for (int i = 0; i < 8; i++)
        {
            V3 gn = scene_normal({ghx[i], ghy[i], ghz[i]});
            gnx8[i] = gn.x;
            gny8[i] = gn.y;
            gnz8[i] = gn.z;
        }
#endif
    }

    // 7b. Vectorised material lookup for GI secondary hits — skipped when no hits.
    alignas(32) float gi_ar[8] = {}, gi_ag[8] = {}, gi_ab[8] = {}, gi_rq[8] = {};
    if (gi8.hit_mask)
    {
#ifdef __AVX2__
        get_mat_avx8(ghm, ghx, ghz, gi_ar, gi_ag, gi_ab, gi_rq);
#else
        for (int i = 0; i < 8; i++)
        {
            MatProps gmp = get_mat(ghm[i], {ghx[i], ghy[i], ghz[i]});
            gi_ar[i] = gmp.albedo.x;
            gi_ag[i] = gmp.albedo.y;
            gi_ab[i] = gmp.albedo.z;
            gi_rq[i] = gmp.roughness;
        }
#endif
    }

    // 7c. Final shading + GI blend + cache write (scalar arithmetic per lane)
    const V3 sun_col = {1.4f, 1.15f, .8f}, amb_col = {.15f, .2f, .35f};
    for (int i = 0; i < 8; i++)
    {
        if (is_special[i])
            continue;
        V3 n = {nx8[i], ny8[i], nz8[i]};
        V3 albedo = {mat_ar[i], mat_ag[i], mat_ab[i]};
        float roughness = mat_rough[i];
        float diff = diff_v[i], sha = skip_shadow[i] ? 0.f : sha_arr[i], ao = ao_arr[i];
        float sky_amb = 0.5f + 0.5f * n.y;
        V3 gi_new = {0, 0, 0};
        bool gi_hit = (gi8.hit_mask >> i) & 1;
        bool gi_computed = (gi_active >> i) & 1;
        if (gi_computed && gi_hit && ghm[i] != MAT_WATER)
        {
            V3 gi_albedo = {gi_ar[i], gi_ag[i], gi_ab[i]};
            float gd = std::max(0.f, dot({gnx8[i], gny8[i], gnz8[i]}, sun_dir)) * .5f;
            gi_new = gi_albedo * gd;
        }
        else if (gi_computed && !gi_hit)
        {
            gi_new = sky_color({gi_dx[i], gi_dy[i], gi_dz[i]}) * .3f;
        }
        gi_new = gi_new * dot(n, {gi_dx[i], gi_dy[i], gi_dz[i]}) * .4f;
        V3 gi;
        if (!gi_computed)
            gi = repro_gi[i];
        else if (repro_valid[i])
            gi = gi_new * REPROJ_ALPHA + repro_gi[i] * (1.f - REPROJ_ALPHA);
        else
            gi = gi_new;
        if (curr_gi)
        {
            int xi = lane_x ? lane_x[i] : pixel_x_base + i;
            if (xi >= 0 && xi < curr_gi->W)
            {
                int idx = pixel_y * curr_gi->W + xi;
                bool hit = (hit_mask >> i) & 1;
                float dxc = hx[i] - cam_pos.x, dyc = hy[i] - cam_pos.y, dzc = hz[i] - cam_pos.z;
                curr_gi->gi_r[idx] = gi.x;
                curr_gi->gi_g[idx] = gi.y;
                curr_gi->gi_b[idx] = gi.z;
                curr_gi->sha[idx] = sha;
                curr_gi->ao[idx] = ao;
                curr_gi->nx[idx] = nx8[i];
                curr_gi->ny[idx] = ny8[i];
                curr_gi->nz[idx] = nz8[i];
                curr_gi->depth[idx] = hit ? sqrtf(dxc * dxc + dyc * dyc + dzc * dzc) : 1e30f;
                curr_gi->mat[idx] = hit ? hm[i] : -1;
            }
        }
        V3 col = albedo * (sun_col * diff * sha + amb_col * sky_amb * ao + gi);
        if (roughness < 0.5f)
        {
            float gloss = 1.f - roughness;
            V3 half_v = norm(sun_dir - V3{rdx0[i], rdy0[i], rdz0[i]});
            float spec = powf(std::max(0.f, dot(n, half_v)), 4.f / (gloss * gloss + .001f)) * gloss * .8f;
            col = col + sun_col * spec * sha;
        }
        col_r[i] = col.x;
        col_g[i] = col.y;
        col_b[i] = col.z;
    }
}
#endif // __AVX2__ || __AVX__

// ─── POST-PROCESS ─────────────────────────────────────────────────────────────
inline float aces(float x)
{
    float a = 2.51f, b = .03f, c = 2.43f, d = .59f, e = .14f;
    return clampf((x * (a * x + b)) / (x * (c * x + d) + e), 0, 1);
}

void bilateral_denoise(float *out, const float *col, const float *dep,
                       const float *nrm, int W, int H, int r = 2)
{
#pragma omp parallel for
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++)
        {
            int idx = (y * W + x) * 3;
            float d0 = dep[y * W + x];
            const float *n0 = nrm + idx;
            float sw = 0, sc[3] = {0, 0, 0};
            for (int dy = -r; dy <= r; dy++)
                for (int dx2 = -r; dx2 <= r; dx2++)
                {
                    int nx = x + dx2, ny = y + dy;
                    if (nx < 0 || ny < 0 || nx >= W || ny >= H)
                        continue;
                    int nidx = (ny * W + nx) * 3;
                    float d1 = dep[ny * W + nx];
                    const float *n1 = nrm + nidx;
                    float nd = (n0[0] - .5f) * (n1[0] - .5f) + (n0[1] - .5f) * (n1[1] - .5f) + (n0[2] - .5f) * (n1[2] - .5f);
                    float wd = 1.f / (1 + 40.f * fabsf(d1 - d0));
                    float wn = std::max(0.f, nd * 4.f + 0.1f);
                    float ws = expf(-0.5f * (dx2 * dx2 + dy * dy));
                    float w = ws * wd * wn;
                    sc[0] += col[nidx] * w;
                    sc[1] += col[nidx + 1] * w;
                    sc[2] += col[nidx + 2] * w;
                    sw += w;
                }
            if (sw > 0)
            {
                out[idx] = sc[0] / sw;
                out[idx + 1] = sc[1] / sw;
                out[idx + 2] = sc[2] / sw;
            }
            else
            {
                out[idx] = col[idx];
                out[idx + 1] = col[idx + 1];
                out[idx + 2] = col[idx + 2];
            }
        }
}

void write_ppm(const char *path, float *img, int W, int H)
{
    FILE *f = fopen(path, "wb");
    fprintf(f, "P6\n%d %d\n255\n", W, H);
    for (int i = 0; i < W * H * 3; i += 3)
    {
        uint8_t r2 = (uint8_t)(img[i] * 255), g2 = (uint8_t)(img[i + 1] * 255), b2 = (uint8_t)(img[i + 2] * 255);
        fwrite(&r2, 1, 1, f);
        fwrite(&g2, 1, 1, f);
        fwrite(&b2, 1, 1, f);
    }
    fclose(f);
}

// ─── AVX-512 NORMAL / AO / MATERIAL ─────────────────────────────────────────

// scene_normal_avx512: central-difference normals for 16 hits at once.
// 6 eval_scene_avx512 calls → 16 normals. Without this, shade_batch_avx512
// called scalar scene_normal() 16 times = 96 scalar eval_scene calls per tile.
#ifdef __AVX512F__
static void scene_normal_avx512(
    const float *hx, const float *hy, const float *hz,
    float *nx_out, float *ny_out, float *nz_out,
    int oct = 7)
{
    const float e = 0.003f;
    __m512 px = _mm512_load_ps(hx), py = _mm512_load_ps(hy), pz = _mm512_load_ps(hz);
    __m512 ve = _mm512_set1_ps(e);

    auto d16 = [](const EvalResult16 &r)
    { return r.d; };

    __m512 nx = _mm512_sub_ps(
        d16(eval_scene_avx512(_mm512_add_ps(px, ve), py, pz, oct)),
        d16(eval_scene_avx512(_mm512_sub_ps(px, ve), py, pz, oct)));
    __m512 ny = _mm512_sub_ps(
        d16(eval_scene_avx512(px, _mm512_add_ps(py, ve), pz, oct)),
        d16(eval_scene_avx512(px, _mm512_sub_ps(py, ve), pz, oct)));
    __m512 nz = _mm512_sub_ps(
        d16(eval_scene_avx512(px, py, _mm512_add_ps(pz, ve), oct)),
        d16(eval_scene_avx512(px, py, _mm512_sub_ps(pz, ve), oct)));

    // Per-lane normalisation: rsqrt14 + 1 Newton-Raphson step (~2× faster than sqrt+div).
    // NR: inv = r0 * (1.5 - 0.5 * len2 * r0 * r0)  — refines 14-bit estimate to ~28 bits.
    __m512 len2 = _mm512_fmadd_ps(nx, nx, _mm512_fmadd_ps(ny, ny, _mm512_mul_ps(nz, nz)));
    __m512 r0 = _mm512_rsqrt14_ps(_mm512_max_ps(len2, _mm512_set1_ps(1e-18f)));
    __m512 inv = _mm512_mul_ps(r0, _mm512_fnmadd_ps(
                                       _mm512_mul_ps(_mm512_set1_ps(0.5f), len2),
                                       _mm512_mul_ps(r0, r0), _mm512_set1_ps(1.5f)));
    _mm512_store_ps(nx_out, _mm512_mul_ps(nx, inv));
    _mm512_store_ps(ny_out, _mm512_mul_ps(ny, inv));
    _mm512_store_ps(nz_out, _mm512_mul_ps(nz, inv));
}

// calc_ao_avx512: ambient occlusion for 16 hits at once.
// 6 eval_scene_avx512 calls → 16 AO values.
static __m512 calc_ao_avx512(
    const float *hx, const float *hy, const float *hz,
    const float *nx, const float *ny, const float *nz)
{
    __m512 px = _mm512_load_ps(hx), py = _mm512_load_ps(hy), pz = _mm512_load_ps(hz);
    __m512 vn_x = _mm512_load_ps(nx), vn_y = _mm512_load_ps(ny), vn_z = _mm512_load_ps(nz);

    __m512 occ = _mm512_setzero_ps();
    __m512 sc = _mm512_set1_ps(1.f);
    static const float h2_steps[6] = {0.03f, 0.11f, 0.19f, 0.27f, 0.35f, 0.43f};

    for (int i = 0; i < 6; i++)
    {
        __m512 h2 = _mm512_set1_ps(h2_steps[i]);
        __m512 spx = _mm512_fmadd_ps(vn_x, h2, px);
        __m512 spy = _mm512_fmadd_ps(vn_y, h2, py);
        __m512 spz = _mm512_fmadd_ps(vn_z, h2, pz);
        // AO samples are very close to the surface (0.03–0.43 units); 2 octaves suffice.
        // The third octave adds detail smaller than the sample step itself.
        __m512 d = eval_scene_avx512(spx, spy, spz, 2).d;
        occ = _mm512_fmadd_ps(_mm512_sub_ps(h2, d), sc, occ);
        sc = _mm512_mul_ps(sc, _mm512_set1_ps(0.75f));
    }
    __m512 ao = _mm512_sub_ps(_mm512_set1_ps(1.f),
                              _mm512_mul_ps(_mm512_set1_ps(2.5f), occ));
    return _mm512_min_ps(_mm512_set1_ps(1.f), _mm512_max_ps(_mm512_setzero_ps(), ao));
}

// get_mat_avx512: material albedo + roughness for 16 lanes in-register.
// Uses mask_blend (AVX-512 _mm512_mask_blend_ps) over all 9 material types.
// Runs avx_fbm for texture variation — same approach as get_mat_avx8 at 16-wide.
static inline void get_mat_avx512(
    const int *hm, const float *hx, const float *hz,
    float *ar, float *ag, float *ab, float *rough)
{
    __m512 px = _mm512_load_ps(hx), pz = _mm512_load_ps(hz);
    __m512i mi = _mm512_load_si512((const __m512i *)hm);

    // Material masks: __mmask16 from integer compare
    auto mmask = [&](int code) -> __mmask16
    {
        return _mm512_cmpeq_epi32_mask(mi, _mm512_set1_epi32(code));
    };
    __mmask16 m_grass = mmask(MAT_GRASS);
    __mmask16 m_rock = mmask(MAT_ROCK);
    __mmask16 m_snow = mmask(MAT_SNOW);
    __mmask16 m_water = mmask(MAT_WATER);
    __mmask16 m_trunk = mmask(MAT_TREE_TRUNK);
    __mmask16 m_leaf = mmask(MAT_TREE_LEAF);
    __mmask16 m_mstone = mmask(MAT_MESH_STONE);
    __mmask16 m_gold = mmask(MAT_MESH_GOLD);
    __mmask16 m_xtal = mmask(MAT_MESH_CRYSTAL);

    // FBM variants (computed unconditionally, amortised across 16 lanes)
    __m512 g1 = avx_fbm(_mm512_mul_ps(px, _mm512_set1_ps(.5f)),
                        _mm512_mul_ps(pz, _mm512_set1_ps(.5f)), 2);
    __m512 g2 = avx_fbm(_mm512_fmadd_ps(px, _mm512_set1_ps(.4f), _mm512_set1_ps(5.f)),
                        _mm512_mul_ps(pz, _mm512_set1_ps(.4f)), 2);
    __m512 rf = avx_fbm(_mm512_mul_ps(px, _mm512_set1_ps(.8f)),
                        _mm512_mul_ps(pz, _mm512_set1_ps(.8f)), 3);
    __m512 lf = avx_fbm(_mm512_mul_ps(px, _mm512_set1_ps(2.f)),
                        _mm512_mul_ps(pz, _mm512_set1_ps(2.f)), 2);
    __m512 msf = avx_fbm(_mm512_fmadd_ps(px, _mm512_set1_ps(2.1f), _mm512_set1_ps(3.f)),
                         _mm512_fmadd_ps(pz, _mm512_set1_ps(2.1f), _mm512_set1_ps(7.f)), 3);
    __m512 msn = _mm512_mul_ps(msf, _mm512_set1_ps(.15f));

    // Start from default (grey)
    __m512 out_r = _mm512_set1_ps(.5f);
    __m512 out_g = _mm512_set1_ps(.5f);
    __m512 out_b = _mm512_set1_ps(.5f);
    __m512 out_q = _mm512_set1_ps(.8f);

    // GRASS
    out_r = _mm512_mask_blend_ps(m_grass, out_r, _mm512_fmadd_ps(g1, _mm512_set1_ps(.12f), _mm512_set1_ps(.15f)));
    out_g = _mm512_mask_blend_ps(m_grass, out_g, _mm512_fmadd_ps(g2, _mm512_set1_ps(.1f), _mm512_set1_ps(.28f)));
    out_b = _mm512_mask_blend_ps(m_grass, out_b, _mm512_set1_ps(.07f));
    out_q = _mm512_mask_blend_ps(m_grass, out_q, _mm512_set1_ps(.95f));
    // ROCK
    out_r = _mm512_mask_blend_ps(m_rock, out_r, _mm512_fmadd_ps(rf, _mm512_set1_ps(.1f), _mm512_set1_ps(.3f)));
    out_g = _mm512_mask_blend_ps(m_rock, out_g, _mm512_fmadd_ps(rf, _mm512_set1_ps(.08f), _mm512_set1_ps(.28f)));
    out_b = _mm512_mask_blend_ps(m_rock, out_b, _mm512_fmadd_ps(rf, _mm512_set1_ps(.06f), _mm512_set1_ps(.25f)));
    out_q = _mm512_mask_blend_ps(m_rock, out_q, _mm512_set1_ps(.8f));
    // SNOW
    out_r = _mm512_mask_blend_ps(m_snow, out_r, _mm512_set1_ps(.9f));
    out_g = _mm512_mask_blend_ps(m_snow, out_g, _mm512_set1_ps(.92f));
    out_b = _mm512_mask_blend_ps(m_snow, out_b, _mm512_set1_ps(.95f));
    out_q = _mm512_mask_blend_ps(m_snow, out_q, _mm512_set1_ps(.55f));
    // WATER
    out_r = _mm512_mask_blend_ps(m_water, out_r, _mm512_set1_ps(.04f));
    out_g = _mm512_mask_blend_ps(m_water, out_g, _mm512_set1_ps(.1f));
    out_b = _mm512_mask_blend_ps(m_water, out_b, _mm512_set1_ps(.2f));
    out_q = _mm512_mask_blend_ps(m_water, out_q, _mm512_set1_ps(.0f));
    // TREE_TRUNK
    out_r = _mm512_mask_blend_ps(m_trunk, out_r, _mm512_set1_ps(.3f));
    out_g = _mm512_mask_blend_ps(m_trunk, out_g, _mm512_set1_ps(.18f));
    out_b = _mm512_mask_blend_ps(m_trunk, out_b, _mm512_set1_ps(.08f));
    out_q = _mm512_mask_blend_ps(m_trunk, out_q, _mm512_set1_ps(.95f));
    // TREE_LEAF
    out_r = _mm512_mask_blend_ps(m_leaf, out_r, _mm512_fmadd_ps(lf, _mm512_set1_ps(.06f), _mm512_set1_ps(.08f)));
    out_g = _mm512_mask_blend_ps(m_leaf, out_g, _mm512_fmadd_ps(lf, _mm512_set1_ps(.1f), _mm512_set1_ps(.28f)));
    out_b = _mm512_mask_blend_ps(m_leaf, out_b, _mm512_set1_ps(.06f));
    out_q = _mm512_mask_blend_ps(m_leaf, out_q, _mm512_set1_ps(.9f));
    // MESH_STONE
    out_r = _mm512_mask_blend_ps(m_mstone, out_r, _mm512_add_ps(_mm512_set1_ps(.45f), msn));
    out_g = _mm512_mask_blend_ps(m_mstone, out_g, _mm512_add_ps(_mm512_set1_ps(.42f), msn));
    out_b = _mm512_mask_blend_ps(m_mstone, out_b, _mm512_add_ps(_mm512_set1_ps(.38f), msn));
    out_q = _mm512_mask_blend_ps(m_mstone, out_q, _mm512_set1_ps(.75f));
    // MESH_GOLD
    out_r = _mm512_mask_blend_ps(m_gold, out_r, _mm512_set1_ps(.9f));
    out_g = _mm512_mask_blend_ps(m_gold, out_g, _mm512_set1_ps(.72f));
    out_b = _mm512_mask_blend_ps(m_gold, out_b, _mm512_set1_ps(.2f));
    out_q = _mm512_mask_blend_ps(m_gold, out_q, _mm512_set1_ps(.15f));
    // MESH_CRYSTAL
    out_r = _mm512_mask_blend_ps(m_xtal, out_r, _mm512_set1_ps(.5f));
    out_g = _mm512_mask_blend_ps(m_xtal, out_g, _mm512_set1_ps(.8f));
    out_b = _mm512_mask_blend_ps(m_xtal, out_b, _mm512_set1_ps(.95f));
    out_q = _mm512_mask_blend_ps(m_xtal, out_q, _mm512_set1_ps(.05f));

    _mm512_store_ps(ar, out_r);
    _mm512_store_ps(ag, out_g);
    _mm512_store_ps(ab, out_b);
    _mm512_store_ps(rough, out_q);
}
#endif // __AVX512F__

// ─── AVX-512 BATCH SHADING ────────────────────────────────────────────────────
// shade_batch_avx512: given 16 primary hits (from ray_march_avx512), issue all
// secondary rays (shadow + GI) in SIMD batches of 16 and return 16 RGB colours.
//
// Strategy per-lane:
//   1. Compute surface normal (scalar — 6 scene evals, unavoidable)
//   2. Collect shadow ray origins + dirs for all 16 lanes → soft_shadow_avx512
//   3. For GI: attempt reprojection from prev_gi if available; fall back to new ray
//   4. Collect GI ray origins + dirs for all 16 lanes → ray_march_avx512_gi
//   5. Finish shading with the batched shadow / GI results (scalar arithmetic)
//   6. Write GI contribution to curr_gi cache for next frame's reprojection
//
// ── shade_water_batch_avx512 ─────────────────────────────────────────────────
// Vectorised water shading for up to 16 water lanes in a tile.
// Replaces up to 16 scalar ray_march calls (one per water pixel) with a single
// ray_march_avx512 call for all reflection rays, then vectorises the normal and
// shadow computation on the reflected hits.
//
// n_water: number of water lanes (1–16). All arrays are n_water-wide but
//          padded to 16 with the last valid lane to keep SIMD code clean.
// Outputs col_r/g/b for the n_water lanes (caller writes them into the tile).
#ifdef __AVX512F__
static void shade_water_batch_avx512(
    int n_water,
    const float *wpx, const float *wpy, const float *wpz,    // hit positions
    const float *wnx, const float *wny, const float *wnz,    // surface normals
    const float *wrdx, const float *wrdy, const float *wrdz, // incident ray dirs
    float *col_r, float *col_g, float *col_b,
    int pixel_y = 0, const int *water_lane = nullptr, int pixel_x_base = 0,
    const GICache *prev_gi = nullptr, const Camera *prev_cam = nullptr,
    GICache *curr_gi = nullptr)
{
    // ── E: Grazing-angle early exit ───────────────────────────────────────────
    bool grazing[16] = {};
    for (int i = 0; i < n_water; i++)
    {
        float nx = wnx[i], ny = wny[i], nz = wnz[i];
        float dx = wrdx[i], dy = wrdy[i], dz = wrdz[i];
        float dot_rn = dx * nx + dy * ny + dz * nz;
        float rdy_r = dy - 2.f * dot_rn * ny;
        grazing[i] = (rdy_r < 0.08f);
    }

    // ── F: Half-resolution — build only even-indexed reflection rays ──────────
    int n_march = (n_water + 1) / 2;

    // ── Build reflection ray batch ────────────────────────────────────────────
    alignas(64) float rox[16], roy[16], roz[16];
    alignas(64) float rdx[16], rdy[16], rdz[16];

    for (int i = 0; i < 16; i++)
    {
        int even_i = std::min(i * 2, n_water - 1);
        int si = even_i;
        float nx = wnx[si], ny = wny[si], nz = wnz[si];
        float dx = wrdx[si], dy = wrdy[si], dz = wrdz[si];
        // reflect(rd, n) = rd - 2*dot(rd,n)*n
        float dot_rn = dx * nx + dy * ny + dz * nz;
        rdx[i] = dx - 2.f * dot_rn * nx;
        rdy[i] = dy - 2.f * dot_rn * ny;
        rdz[i] = dz - 2.f * dot_rn * nz;
        // Offset origin along normal to avoid self-intersection
        rox[i] = wpx[si] + nx * 0.01f;
        roy[i] = wpy[si] + ny * 0.01f;
        roz[i] = wpz[si] + nz * 0.01f;
    }

    // ── A: Batch reflection march — oct=3, t_max=80 ───────────────────────────
    March16 rm = ray_march_avx512(
        _mm512_load_ps(rox), _mm512_load_ps(roy), _mm512_load_ps(roz),
        _mm512_load_ps(rdx), _mm512_load_ps(rdy), _mm512_load_ps(rdz),
        /*oct=*/3, /*t_max=*/80.f);

    alignas(64) float rhx[16], rhy[16], rhz[16];
    alignas(64) int rhm[16];
    _mm512_store_ps(rhx, rm.pos_x);
    _mm512_store_ps(rhy, rm.pos_y);
    _mm512_store_ps(rhz, rm.pos_z);
    _mm512_store_epi32(rhm, rm.mat);

    // ── B: Vectorised normal at oct=3 ────────────────────────────────────────
    alignas(64) float rnx[16], rny[16], rnz[16];
    scene_normal_avx512(rhx, rhy, rhz, rnx, rny, rnz, /*oct=*/3);

    // Shadow: build origins and fire soft_shadow_avx512 for reflected hits
    alignas(64) float sox[16], soy[16], soz[16];
    for (int i = 0; i < 16; i++)
    {
        sox[i] = rhx[i] + rnx[i] * 0.02f;
        soy[i] = rhy[i] + rny[i] * 0.02f;
        soz[i] = rhz[i] + rnz[i] * 0.02f;
    }
    const __m512 sun_x = _mm512_set1_ps(sun_dir.x);
    const __m512 sun_y = _mm512_set1_ps(sun_dir.y);
    const __m512 sun_z = _mm512_set1_ps(sun_dir.z);
    // ── C: Shadow at 32 steps ─────────────────────────────────────────────────
    __m512 refl_sha = soft_shadow_avx512(
        _mm512_load_ps(sox), _mm512_load_ps(soy), _mm512_load_ps(soz),
        sun_x, sun_y, sun_z, 10.f, /*max_steps=*/32);

    // Diffuse dot on reflected hits
    __m512 vrnx = _mm512_load_ps(rnx), vrny = _mm512_load_ps(rny), vrnz = _mm512_load_ps(rnz);
    __m512 diff_r = _mm512_max_ps(_mm512_setzero_ps(),
                                  _mm512_fmadd_ps(vrnx, sun_x, _mm512_fmadd_ps(vrny, sun_y, _mm512_mul_ps(vrnz, sun_z))));

    // Reflected colour for the half-res even-indexed lanes
    alignas(64) float sha_arr[16];
    _mm512_store_ps(sha_arr, refl_sha);
    alignas(64) float diff_arr[16];
    _mm512_store_ps(diff_arr, diff_r);

    // Build even-lane colours
    float even_refl_r[16], even_refl_g[16], even_refl_b[16];
    for (int i = 0; i < 16; i++)
    {
        bool hit_r = (rm.hit_mask >> i) & 1;
        if (hit_r && rhm[i] != MAT_WATER)
        {
            MatProps mp2 = get_mat(rhm[i], {rhx[i], rhy[i], rhz[i]});
            float shade = diff_arr[i] * sha_arr[i] * 1.5f + 0.15f;
            even_refl_r[i] = mp2.albedo.x * shade;
            even_refl_g[i] = mp2.albedo.y * shade;
            even_refl_b[i] = mp2.albedo.z * shade;
        }
        else
        {
            V3 sky = sky_color({rdx[i], rdy[i], rdz[i]});
            even_refl_r[i] = sky.x;
            even_refl_g[i] = sky.y;
            even_refl_b[i] = sky.z;
        }
    }

    // ── F: Copy half-res results ──────────────────────────────────────────────
    alignas(64) float refl_r[16], refl_g[16], refl_b[16];
    for (int i = 0; i < n_water; i++)
    {
        int src_i = i / 2;
        refl_r[i] = even_refl_r[src_i];
        refl_g[i] = even_refl_g[src_i];
        refl_b[i] = even_refl_b[src_i];
    }

    // ── E: Override grazing pixels with sky ───────────────────────────────────
    for (int i = 0; i < n_water; i++)
    {
        if (grazing[i])
        {
            float nx = wnx[i], ny = wny[i], nz = wnz[i];
            float dx = wrdx[i], dy = wrdy[i], dz = wrdz[i];
            float dot_rn = dx * nx + dy * ny + dz * nz;
            float rrdx = dx - 2.f * dot_rn * nx;
            float rrdy = dy - 2.f * dot_rn * ny;
            float rrdz = dz - 2.f * dot_rn * nz;
            V3 sky = sky_color({rrdx, rrdy, rrdz});
            refl_r[i] = sky.x;
            refl_g[i] = sky.y;
            refl_b[i] = sky.z;
        }
    }

    // ── D: Temporal reprojection ──────────────────────────────────────────────
    const float REFL_ALPHA = 0.3f;
    if (prev_gi && curr_gi && water_lane)
    {
        for (int wi = 0; wi < n_water; wi++)
        {
            int px_coord = pixel_x_base + water_lane[wi];
            int idx = pixel_y * curr_gi->W + px_coord;
            if (idx >= 0 && idx < curr_gi->W * curr_gi->H)
            {
                float old_r = prev_gi->refl_r[idx];
                float old_g = prev_gi->refl_g[idx];
                float old_b = prev_gi->refl_b[idx];
                if (old_r + old_g + old_b > 0.001f)
                {
                    refl_r[wi] = REFL_ALPHA * refl_r[wi] + (1.f - REFL_ALPHA) * old_r;
                    refl_g[wi] = REFL_ALPHA * refl_g[wi] + (1.f - REFL_ALPHA) * old_g;
                    refl_b[wi] = REFL_ALPHA * refl_b[wi] + (1.f - REFL_ALPHA) * old_b;
                }
                curr_gi->refl_r[idx] = refl_r[wi];
                curr_gi->refl_g[idx] = refl_g[wi];
                curr_gi->refl_b[idx] = refl_b[wi];
            }
        }
    }

    // ── Per-lane Fresnel blend, water colour, specular ────────────────────────
    for (int i = 0; i < n_water; i++)
    {
        float nx = wnx[i], ny = wny[i], nz = wnz[i];
        float dx = wrdx[i], dy = wrdy[i], dz = wrdz[i];
        float ndv = std::max(0.f, -(dx * nx + dy * ny + dz * nz));
        float fres = 0.04f + 0.96f * powf(1.f - ndv, 5.f);

        float depth_f = smoothstep(0.f, 3.f, fabsf(terrain_h(wpx[i], wpz[i])));
        float caus = powf(std::max(0.f, sinf(wpx[i] * 2.3f) * sinf(wpz[i] * 1.9f)), 4.f) * 0.5f;
        float wc_r = (0.1f + (0.01f - 0.1f) * depth_f) + 0.1f * caus;
        float wc_g = (0.3f + (0.08f - 0.3f) * depth_f) + 0.15f * caus;
        float wc_b = (0.4f + (0.18f - 0.4f) * depth_f) + 0.1f * caus;

        float cr = wc_r + (refl_r[i] - wc_r) * fres;
        float cg = wc_g + (refl_g[i] - wc_g) * fres;
        float cb = wc_b + (refl_b[i] - wc_b) * fres;

        // Specular highlight
        float hx2 = sun_dir.x - dx, hy2 = sun_dir.y - dy, hz2 = sun_dir.z - dz;
        float hlen = sqrtf(hx2 * hx2 + hy2 * hy2 + hz2 * hz2) + 1e-6f;
        hx2 /= hlen;
        hy2 /= hlen;
        hz2 /= hlen;
        float spec = powf(std::max(0.f, nx * hx2 + ny * hy2 + nz * hz2), 256.f) * 2.f;

        col_r[i] = cr + 1.4f * spec;
        col_g[i] = cg + 1.2f * spec;
        col_b[i] = cb + 0.8f * spec;
    }
}
#endif

// Water and crystal have special paths — those lanes fall through to scalar
// shading exactly as before (they're a small fraction of pixels).
//
// Output: col_r/g/b arrays of 16 floats (caller accumulates into framebuffer).
#ifdef __AVX512F__
static void shade_batch_avx512(
    // Primary hit results (16 lanes)
    const float *hx, const float *hy, const float *hz,
    const int *hm,
    const __mmask16 hit_mask,
    // Ray directions for sky + fog (16 lanes)
    const float *rdx0, const float *rdy0, const float *rdz0,
    // Camera origin (single value, same for all)
    V3 cam_pos,
    int tid,
    // Temporal reprojection — may be nullptr (first frame / still renders)
    const GICache *prev_gi,
    const Camera *prev_cam,
    GICache *curr_gi,
    int pixel_y,      // row of this tile in the framebuffer
    int pixel_x_base, // x of lane 0
    // Outputs
    float *col_r, float *col_g, float *col_b,
    // Optional: primary normals for denoiser nrm buffer (avoids redundant scene_normal)
    float *nx_out = nullptr, float *ny_out = nullptr, float *nz_out = nullptr)
{
    // ── 1. Normals (AVX-512: 6 eval_scene_avx512 calls for all 16 hits) ──────
    // Sky/miss lanes get a dummy normal; special lanes get the real normal for
    // shadow-offset computation even though shading falls back to scalar.
    alignas(64) float nx16[16], ny16[16], nz16[16];
    scene_normal_avx512(hx, hy, hz, nx16, ny16, nz16);

    // Export normals for denoiser nrm buffer — avoids a redundant scalar scene_normal
    // call per hit pixel in render_frame (was costing 6 scalar eval_scene per pixel).
    if (nx_out)
    {
        _mm512_store_ps(nx_out, _mm512_load_ps(nx16));
    }
    if (ny_out)
    {
        _mm512_store_ps(ny_out, _mm512_load_ps(ny16));
    }
    if (nz_out)
    {
        _mm512_store_ps(nz_out, _mm512_load_ps(nz16));
    }

    // ── 1b. Material lookup (AVX-512: all 16 lanes in-register) ──────────────
    alignas(64) float mat_ar[16], mat_ag[16], mat_ab[16], mat_rough[16];
    get_mat_avx512(hm, hx, hz, mat_ar, mat_ag, mat_ab, mat_rough);

    // Sky + special lane classification (scalar, just comparisons — cheap)
    bool is_special[16];
    for (int i = 0; i < 16; i++)
    {
        bool hit = (hit_mask >> i) & 1;
        if (!hit)
        {
            V3 rd_i = {rdx0[i], rdy0[i], rdz0[i]};
            V3 sky = sky_color(rd_i);
            col_r[i] = sky.x;
            col_g[i] = sky.y;
            col_b[i] = sky.z;
            is_special[i] = true;
            continue;
        }
        is_special[i] = (hm[i] == MAT_WATER || hm[i] == MAT_MESH_CRYSTAL);
    }

    // ── 2. Special materials ──────────────────────────────────────────────────
    // Water: collect water lanes, fire one batched ray_march_avx512 for all
    // reflection rays, then blend with shade_water_batch_avx512.
    // Crystal: scalar path (rim + specular, no secondary march needed).

    // Gather water lanes into compact arrays for the batch call
    alignas(64) float wpx[16], wpy[16], wpz[16];
    alignas(64) float wnx[16], wny[16], wnz[16];
    alignas(64) float wrdx[16], wrdy[16], wrdz[16];
    int water_lane[16], n_water = 0;

    for (int i = 0; i < 16; i++)
    {
        if (!is_special[i])
            continue;
        bool hit = (hit_mask >> i) & 1;
        if (!hit)
            continue;
        if (hm[i] == MAT_WATER)
        {
            water_lane[n_water] = i;
            wpx[n_water] = hx[i];
            wpy[n_water] = hy[i];
            wpz[n_water] = hz[i];
            wnx[n_water] = nx16[i];
            wny[n_water] = ny16[i];
            wnz[n_water] = nz16[i];
            wrdx[n_water] = rdx0[i];
            wrdy[n_water] = rdy0[i];
            wrdz[n_water] = rdz0[i];
            n_water++;
        }
        else
        {
            // Crystal: scalar path with precomputed normal
            V3 rd_i = {rdx0[i], rdy0[i], rdz0[i]};
            V3 n_i = {nx16[i], ny16[i], nz16[i]};
            March m_i;
            m_i.pos = {hx[i], hy[i], hz[i]};
            m_i.mat = hm[i];
            m_i.hit = true;
            m_i.t = 0;
            V3 c = shade_known_hit(cam_pos, rd_i, m_i, tid, &n_i);
            col_r[i] = c.x;
            col_g[i] = c.y;
            col_b[i] = c.z;
        }
    }

    if (n_water > 0)
    {
        alignas(64) float wc_r[16], wc_g[16], wc_b[16];
        shade_water_batch_avx512(n_water, wpx, wpy, wpz, wnx, wny, wnz,
                                 wrdx, wrdy, wrdz, wc_r, wc_g, wc_b,
                                 pixel_y, water_lane, pixel_x_base,
                                 prev_gi, prev_cam, curr_gi);
        for (int wi = 0; wi < n_water; wi++)
        {
            int i = water_lane[wi];
            col_r[i] = wc_r[wi];
            col_g[i] = wc_g[wi];
            col_b[i] = wc_b[wi];
        }
    }

    // ── 3. Build shadow + GI ray batches for non-special hit lanes ────────────
    // For miss/special lanes we still fill the arrays (they won't affect output)
    // but we exclude them via is_special guard at step 5.
    alignas(64) float sha_ox[16], sha_oy[16], sha_oz[16]; // shadow origins
    alignas(64) float sha_dx[16], sha_dy[16], sha_dz[16]; // shadow dirs (sun_dir)
    alignas(64) float gi_ox[16], gi_oy[16], gi_oz[16];    // GI origins
    alignas(64) float gi_dx[16], gi_dy[16], gi_dz[16];    // GI random dirs
    float diff_v[16];                                     // diffuse dot product
    bool skip_shadow[16];                                 // diff ≤ 0.01

    // ── 3a. Temporal reprojection: attempt to reuse prev frame's GI ─────────────
    // For each non-special hit lane, project world pos into prev camera, look up
    // cached GI.  Validity: depth delta < 0.5 AND material match.
    // repro_gi[i] holds the blended GI if valid, stays {-1,-1,-1} if not.
    // We still always fire a new GI ray; it's blended in at step 6.
    const float REPROJ_ALPHA = 0.25f; // weight of the NEW ray  (0.25 new + 0.75 cached)
    const float DEPTH_THRESH = 0.5f;  // world-units depth tolerance
    V3 repro_gi[16];
    bool repro_valid[16];
    for (int i = 0; i < 16; i++)
    {
        repro_gi[i] = {-1, -1, -1};
        repro_valid[i] = false;
    }

    if (prev_gi && prev_cam)
    {
        for (int i = 0; i < 16; i++)
        {
            if (is_special[i])
                continue;
            bool hit = (hit_mask >> i) & 1;
            if (!hit)
                continue;

            V3 P = {hx[i], hy[i], hz[i]};
            float px, py;
            if (!project_point(P, *prev_cam, px, py))
                continue;

            int ix = (int)px, iy = (int)py;
            if (ix < 0 || ix >= prev_gi->W || iy < 0 || iy >= prev_gi->H)
                continue;
            int idx = iy * prev_gi->W + ix;

            // View-space depth of P under prev camera: dot(P - prev_cam.pos, prev_cam.fwd)
            V3 pv = P - prev_cam->pos;
            float prev_dist = sqrtf(dot(pv, pv));
            float prev_d = prev_gi->depth[idx];
            if (fabsf(prev_dist - prev_d) > DEPTH_THRESH)
                continue; // disocclusion
            if (prev_gi->mat[idx] != hm[i])
                continue; // surface changed

            repro_gi[i] = {prev_gi->gi_r[idx], prev_gi->gi_g[idx], prev_gi->gi_b[idx]};
            repro_valid[i] = true;
        }
    }

    for (int i = 0; i < 16; i++)
    {
        if (is_special[i])
        {
            sha_ox[i] = sha_oy[i] = sha_oz[i] = 0;
            sha_dx[i] = 0;
            sha_dy[i] = 1;
            sha_dz[i] = 0;
            gi_ox[i] = gi_oy[i] = gi_oz[i] = 0;
            gi_dx[i] = 0;
            gi_dy[i] = 1;
            gi_dz[i] = 0;
            diff_v[i] = 0;
            skip_shadow[i] = true;
            continue;
        }
        V3 p = {hx[i], hy[i], hz[i]};
        V3 n = {nx16[i], ny16[i], nz16[i]}; // from scene_normal_avx512
        float diff = std::max(0.f, dot(n, sun_dir));
        diff_v[i] = diff;
        skip_shadow[i] = (diff <= 0.01f);

        sha_ox[i] = p.x + n.x * 0.02f;
        sha_oy[i] = p.y + n.y * 0.02f;
        sha_oz[i] = p.z + n.z * 0.02f;
        sha_dx[i] = sun_dir.x;
        sha_dy[i] = sun_dir.y;
        sha_dz[i] = sun_dir.z;

        V3 gi_dir = rand_hemi(n, tid);
        gi_ox[i] = p.x + n.x * 0.02f;
        gi_oy[i] = p.y + n.y * 0.02f;
        gi_oz[i] = p.z + n.z * 0.02f;
        gi_dx[i] = gi_dir.x;
        gi_dy[i] = gi_dir.y;
        gi_dz[i] = gi_dir.z;
    }

    // ── 4. Batch-march shadow rays ────────────────────────────────────────────
    __m512 sha_res;
    {
        __m512 sox = _mm512_load_ps(sha_ox), soy = _mm512_load_ps(sha_oy), soz = _mm512_load_ps(sha_oz);
        __m512 sdx = _mm512_load_ps(sha_dx), sdy = _mm512_load_ps(sha_dy), sdz = _mm512_load_ps(sha_dz);
        sha_res = soft_shadow_avx512(sox, soy, soz, sdx, sdy, sdz, 12.f);
    }
    alignas(64) float sha_arr[16];
    _mm512_store_ps(sha_arr, sha_res);

    // ── 5. Batch-march GI bounce rays ────────────────────────────────────────
    March16 gi16;
    {
        __m512 gox = _mm512_load_ps(gi_ox), goy = _mm512_load_ps(gi_oy), goz = _mm512_load_ps(gi_oz);
        __m512 gdx = _mm512_load_ps(gi_dx), gdy = _mm512_load_ps(gi_dy), gdz = _mm512_load_ps(gi_dz);
        gi16 = ray_march_avx512_gi(gox, goy, goz, gdx, gdy, gdz, 50.f);
    }
    alignas(64) float ghx[16], ghy[16], ghz[16];
    alignas(64) int ghm[16];
    _mm512_store_ps(ghx, gi16.pos_x);
    _mm512_store_ps(ghy, gi16.pos_y);
    _mm512_store_ps(ghz, gi16.pos_z);
    _mm512_store_epi32(ghm, gi16.mat);

    // ── 6a. Vectorised AO for all 16 hits ────────────────────────────────────
    // 6 eval_scene_avx512 calls instead of 16 scalar calc_ao calls (96 evals).
    alignas(64) float ao_arr[16];
    _mm512_store_ps(ao_arr, calc_ao_avx512(hx, hy, hz, nx16, ny16, nz16));

    // ── 6b. GI bounce normals — vectorised for hit lanes ─────────────────────
    // scene_normal for GI hit positions: 6 eval_scene_avx512 calls instead of
    // up to 16 scalar scene_normal calls.
    alignas(64) float gnx16[16] = {}, gny16[16] = {}, gnz16[16] = {};
    if (gi16.hit_mask)
    {
        scene_normal_avx512(ghx, ghy, ghz, gnx16, gny16, gnz16);
    }

    // ── 6c. GI bounce material — vectorised ──────────────────────────────────
    alignas(64) float gi_ar[16], gi_ag[16], gi_ab[16], gi_rq[16];
    get_mat_avx512(ghm, ghx, ghz, gi_ar, gi_ag, gi_ab, gi_rq);

    // ── 6d. Final shading + reprojection blend + cache write ──────────────────
    const V3 sun_col = {1.4f, 1.15f, .8f};
    const V3 amb_col = {.15f, .2f, .35f};

    for (int i = 0; i < 16; i++)
    {
        if (is_special[i])
            continue;

        V3 n = {nx16[i], ny16[i], nz16[i]};
        // Material from get_mat_avx512
        V3 albedo = {mat_ar[i], mat_ag[i], mat_ab[i]};
        float roughness = mat_rough[i];

        float diff = diff_v[i];
        float sha = skip_shadow[i] ? 0.f : sha_arr[i];
        float ao = ao_arr[i];
        float sky_amb = 0.5f + 0.5f * n.y;

        // GI new sample — uses vectorised bounce normals and material
        V3 gi_new = {0, 0, 0};
        bool gi_hit = (gi16.hit_mask >> i) & 1;
        if (gi_hit && ghm[i] != MAT_WATER)
        {
            V3 g_albedo = {gi_ar[i], gi_ag[i], gi_ab[i]};
            V3 g_n = {gnx16[i], gny16[i], gnz16[i]};
            float gdiff = std::max(0.f, dot(g_n, sun_dir)) * 0.5f;
            gi_new = g_albedo * gdiff;
        }
        else if (!gi_hit)
        {
            gi_new = sky_color({gi_dx[i], gi_dy[i], gi_dz[i]}) * 0.3f;
        }
        gi_new = gi_new * dot(n, {gi_dx[i], gi_dy[i], gi_dz[i]}) * 0.4f;

        // Blend with reprojected GI
        V3 gi = repro_valid[i]
                    ? gi_new * REPROJ_ALPHA + repro_gi[i] * (1.f - REPROJ_ALPHA)
                    : gi_new;

        // Write to GI cache
        if (curr_gi)
        {
            int xi = pixel_x_base + i;
            if (xi < curr_gi->W)
            {
                int idx = pixel_y * curr_gi->W + xi;
                bool hit = (hit_mask >> i) & 1;
                curr_gi->gi_r[idx] = gi.x;
                curr_gi->gi_g[idx] = gi.y;
                curr_gi->gi_b[idx] = gi.z;
                float dx_ = hx[i] - cam_pos.x, dy_ = hy[i] - cam_pos.y, dz_ = hz[i] - cam_pos.z;
                curr_gi->depth[idx] = hit ? sqrtf(dx_ * dx_ + dy_ * dy_ + dz_ * dz_) : 1e30f;
                curr_gi->mat[idx] = hit ? hm[i] : -1;
            }
        }

        V3 col = albedo * (sun_col * diff * sha + amb_col * sky_amb * ao + gi);

        if (roughness < 0.5f)
        {
            float gloss = 1.f - roughness;
            V3 rd_i = {rdx0[i], rdy0[i], rdz0[i]};
            V3 half_v = norm(sun_dir - rd_i);
            float spec = powf(std::max(0.f, dot(n, half_v)),
                              4.f / (gloss * gloss + .001f)) *
                         gloss * .8f;
            col = col + sun_col * spec * sha;
        }

        col_r[i] = col.x;
        col_g[i] = col.y;
        col_b[i] = col.z;
    }
}
#endif // __AVX512F__

// ─── RENDER ONE FRAME ─────────────────────────────────────────────────────────
// cam      — pre-built camera for this frame (from build_camera() in main)
// prev_gi  — GI cache written by the previous video frame; nullptr for stills / frame 0
// prev_cam — camera that wrote prev_gi; nullptr when prev_gi is nullptr
// curr_gi  — GI cache to fill for this frame; nullptr for stills
void render_frame(const Camera &cam, int SAMPLES, float anim_t,
                  const char *outpath,
                  const GICache *prev_gi, const Camera *prev_cam,
                  GICache *curr_gi)
{
    int W = cam.W, H = cam.H;
    sun_dir = norm({0.7f, 0.55f, 0.3f});
#ifdef __AVX2__
    init_tree_bases();
    ensure_height_tables();
#endif

    g_grids[1].pos.y = 5.f + sinf(anim_t * 1.3f) * 0.8f;
    g_grids[1].rot_y = anim_t * 0.9f;
    refresh_sdf_grid_cache(g_grids[1]);
    g_grids[2].pos.y = terrain_h(-10, 15) + 1.2f + sinf(anim_t * 2.1f + 1.f) * 0.4f;
    g_grids[2].rot_y = anim_t * 0.4f;
    refresh_sdf_grid_cache(g_grids[2]);

    // Unpack camera (keeps the rest of the function readable)
    V3 cam_pos = cam.pos;
    V3 cam_fwd = cam.fwd;
    V3 cam_right = cam.right;
    V3 cam_up = cam.up;
    float fov = cam.fov;
    float aspect = cam.aspect;

    for (int i = 0; i < 64; i++)
        rng_state[i] = 12345u * (i + 1) * 2654435761u ^ (uint32_t)(anim_t * 1000);

    int NP = W * H;
    float *acc = new float[NP * 3]();
    float *dep = new float[NP]();
    float *nrm = new float[NP * 3]();
    float *buf = new float[NP * 3]();
    float *out = new float[NP * 3]();

#ifdef __AVX512F__
    printf("  [AVX-512] batch primary + secondary + coherence sort active\n");
#elif defined(__AVX2__)
    printf("  [AVX2]   8-wide batch marching + coherence sort active\n");
#elif defined(__AVX__)
    printf("  [AVX]    8-wide batch marching active (no FMA)\n");
#else
    printf("  [scalar] no SIMD — compiled without AVX/AVX2/AVX-512\n");
#endif
    if (prev_gi)
        printf("  [reprojection] using GI cache from previous frame\n");

    for (int frame = 0; frame < SAMPLES; frame++)
    {
        float jx = (frame % 4 + .5f) / 4.f - .5f;
        float jy = (frame / 4 % 4 + .5f) / 4.f - .5f;

        // ── AVX-512 path: 2D tiles of 16×4 pixels ───────────────────────────
        // Outer loop over 4-row strips (dynamic scheduling, 1 strip per OMP task).
        // Within each strip, 4 rows × (W/16) 16-wide march calls.
        // Benefit: adjacent rows share nearly identical (x,z) terrain FBM inputs
        // → better L1/L2 cache reuse in avx_smoothnoise / avx_fbm.
#ifdef __AVX512F__
#pragma omp parallel for schedule(dynamic, 1)
        for (int ty = 0; ty < H; ty += 4)
        {
            int tid = 0;
#ifdef _OPENMP
            tid = omp_get_thread_num() & 63;
#endif
            int row_end = std::min(ty + 4, H);

            for (int y = ty; y < row_end; y++)
            {
                rng_state[tid] ^= (uint32_t)(y * 7919 + frame * 2053);

                // ── AVX-512 tile processor lambda (2× unrolled x-loop) ──────────
                // Processing two 16-wide tiles per iteration halves loop overhead
                // and lets the OOO core overlap setup of tile B with the march of
                // tile A, since neither has a data dependency on the other.
                // Uses a C++14 generic lambda to avoid duplicating 60 lines.
                auto process_tile_avx512 = [&](int tx)
                {
                    if (tx >= W)
                        return;
                    int batch = std::min(16, W - tx);

                    alignas(64) float rox_a[16], roy_a[16], roz_a[16];
                    alignas(64) float rdx_a[16], rdy_a[16], rdz_a[16];
                    alignas(64) float rdx0[16], rdy0[16], rdz0[16];

                    for (int dx = 0; dx < 16; dx++)
                    {
                        int xi = tx + dx;
                        if (xi >= W)
                            xi = W - 1;

                        float ux = (xi + .5f + jx * .5f + (rng_float(tid) - .5f) * .3f) / W * 2 - 1;
                        float uy = 1 - (y + .5f + jy * .5f + (rng_float(tid) - .5f) * .3f) / H * 2;
                        float rdx_ = cam_fwd.x + cam_right.x * (ux * aspect * fov) + cam_up.x * (uy * fov);
                        float rdy_ = cam_fwd.y + cam_right.y * (ux * aspect * fov) + cam_up.y * (uy * fov);
                        float rdz_ = cam_fwd.z + cam_right.z * (ux * aspect * fov) + cam_up.z * (uy * fov);
                        float rlen = sqrtf(rdx_ * rdx_ + rdy_ * rdy_ + rdz_ * rdz_);
                        rdx0[dx] = rdx_ / rlen;
                        rdy0[dx] = rdy_ / rlen;
                        rdz0[dx] = rdz_ / rlen;

                        float dof_str = 0.02f;
                        float dfx = rng_float(tid) * 2 - 1, dfy = rng_float(tid) * 2 - 1;
                        float rox_ = cam_pos.x + cam_right.x * dfx * dof_str + cam_up.x * dfy * dof_str;
                        float roy_ = cam_pos.y + cam_right.y * dfx * dof_str + cam_up.y * dfy * dof_str;
                        float roz_ = cam_pos.z + cam_right.z * dfx * dof_str + cam_up.z * dfy * dof_str;
                        float fx_ = rox_ + rdx0[dx] * 40, fy_ = roy_ + rdy0[dx] * 40, fz_ = roz_ + rdz0[dx] * 40;
                        float fdx = fx_ - rox_, fdy = fy_ - roy_, fdz = fz_ - roz_;
                        float flen = sqrtf(fdx * fdx + fdy * fdy + fdz * fdz);
                        rox_a[dx] = rox_;
                        roy_a[dx] = roy_;
                        roz_a[dx] = roz_;
                        rdx_a[dx] = fdx / flen;
                        rdy_a[dx] = fdy / flen;
                        rdz_a[dx] = fdz / flen;
                    }

                    __m512 vro_x = _mm512_load_ps(rox_a);
                    __m512 vro_y = _mm512_load_ps(roy_a);
                    __m512 vro_z = _mm512_load_ps(roz_a);
                    __m512 vrd_x = _mm512_load_ps(rdx_a);
                    __m512 vrd_y = _mm512_load_ps(rdy_a);
                    __m512 vrd_z = _mm512_load_ps(rdz_a);

                    // ── Coherence sort (AVX-512): 3 probe steps estimate hit depth ──
                    // Sort lanes ascending by est_t → near-hitting rays group together,
                    // reducing active-mask divergence. Skipped on frame 0 to keep the
                    // GI cache pixel-exact.
                    int perm[16];
                    for (int i = 0; i < 16; i++)
                        perm[i] = i;
                    if (batch > 1 && frame > 0)
                    {
                        alignas(64) float est_t[16];
                        {
                            __m512 t3 = _mm512_set1_ps(0.001f);
                            for (int si = 0; si < 3; si++)
                            {
                                __m512 px3 = _mm512_fmadd_ps(vrd_x, t3, vro_x);
                                __m512 py3 = _mm512_fmadd_ps(vrd_y, t3, vro_y);
                                __m512 pz3 = _mm512_fmadd_ps(vrd_z, t3, vro_z);
                                __m512 d3 = eval_scene_avx512(px3, py3, pz3, 3).d;
                                t3 = _mm512_add_ps(t3, _mm512_max_ps(d3, _mm512_set1_ps(0.001f)));
                            }
                            _mm512_store_ps(est_t, t3);
                        }
                        for (int i = 1; i < 16; i++)
                        {
                            float kd = est_t[perm[i]];
                            int kp = perm[i];
                            int j = i - 1;
                            while (j >= 0 && est_t[perm[j]] > kd)
                            {
                                perm[j + 1] = perm[j];
                                j--;
                            }
                            perm[j + 1] = kp;
                        }
                        alignas(64) float t_rox[16], t_roy[16], t_roz[16];
                        alignas(64) float t_rdx[16], t_rdy[16], t_rdz[16];
                        alignas(64) float t_rdx0[16], t_rdy0[16], t_rdz0[16];
                        for (int i = 0; i < 16; i++)
                        {
                            t_rox[i] = rox_a[perm[i]];
                            t_roy[i] = roy_a[perm[i]];
                            t_roz[i] = roz_a[perm[i]];
                            t_rdx[i] = rdx_a[perm[i]];
                            t_rdy[i] = rdy_a[perm[i]];
                            t_rdz[i] = rdz_a[perm[i]];
                            t_rdx0[i] = rdx0[perm[i]];
                            t_rdy0[i] = rdy0[perm[i]];
                            t_rdz0[i] = rdz0[perm[i]];
                        }
                        memcpy(rox_a, t_rox, 64);
                        memcpy(roy_a, t_roy, 64);
                        memcpy(roz_a, t_roz, 64);
                        memcpy(rdx_a, t_rdx, 64);
                        memcpy(rdy_a, t_rdy, 64);
                        memcpy(rdz_a, t_rdz, 64);
                        memcpy(rdx0, t_rdx0, 64);
                        memcpy(rdy0, t_rdy0, 64);
                        memcpy(rdz0, t_rdz0, 64);
                        vro_x = _mm512_load_ps(rox_a);
                        vro_y = _mm512_load_ps(roy_a);
                        vro_z = _mm512_load_ps(roz_a);
                        vrd_x = _mm512_load_ps(rdx_a);
                        vrd_y = _mm512_load_ps(rdy_a);
                        vrd_z = _mm512_load_ps(rdz_a);
                    }

                    March16 m16 = ray_march_avx512(vro_x, vro_y, vro_z, vrd_x, vrd_y, vrd_z);

                    alignas(64) float hx[16], hy[16], hz[16], ht[16];
                    alignas(64) int hm[16];
                    _mm512_store_ps(hx, m16.pos_x);
                    _mm512_store_ps(hy, m16.pos_y);
                    _mm512_store_ps(hz, m16.pos_z);
                    _mm512_store_ps(ht, m16.t);
                    _mm512_store_epi32(hm, m16.mat);

                    alignas(64) float col_r[16], col_g[16], col_b[16];
                    alignas(64) float pnx[16], pny[16], pnz[16];
                    shade_batch_avx512(hx, hy, hz, hm, m16.hit_mask,
                                       rdx0, rdy0, rdz0,
                                       cam_pos, tid,
                                       frame == 0 ? prev_gi : nullptr,
                                       frame == 0 ? prev_cam : nullptr,
                                       frame == 0 ? curr_gi : nullptr,
                                       y, tx,
                                       col_r, col_g, col_b,
                                       frame == 0 ? pnx : nullptr,
                                       frame == 0 ? pny : nullptr,
                                       frame == 0 ? pnz : nullptr);

                    for (int dx = 0; dx < 16; dx++)
                    {
                        int xi = tx + perm[dx];
                        if (xi >= W)
                            continue;
                        bool did_hit = (m16.hit_mask >> dx) & 1;
                        V3 fog_rd = {rdx_a[dx], rdy_a[dx], rdz_a[dx]};
                        V3 col = {col_r[dx], col_g[dx], col_b[dx]};
                        col = apply_fog(col, ht[dx], fog_rd);
                        if (frame == 0)
                        {
                            dep[y * W + xi] = ht[dx];
                            if (did_hit)
                            {
                                nrm[(y * W + xi) * 3 + 0] = pnx[dx] * .5f + .5f;
                                nrm[(y * W + xi) * 3 + 1] = pny[dx] * .5f + .5f;
                                nrm[(y * W + xi) * 3 + 2] = pnz[dx] * .5f + .5f;
                            }
                            else
                            {
                                nrm[(y * W + xi) * 3 + 0] = .5f;
                                nrm[(y * W + xi) * 3 + 1] = .5f;
                                nrm[(y * W + xi) * 3 + 2] = .5f;
                            }
                        }
                        acc[(y * W + xi) * 3 + 0] += col.x;
                        acc[(y * W + xi) * 3 + 1] += col.y;
                        acc[(y * W + xi) * 3 + 2] += col.z;
                    }
                };

                // 2× unrolled: process two 16-wide tiles per loop iteration.
                // Odd-width images are handled gracefully — process_tile_avx512
                // guards on tx >= W and clamps partial tiles via min(16, W-tx).
                for (int x = 0; x < W; x += 32)
                {
                    process_tile_avx512(x);
                    process_tile_avx512(x + 16);
                }
            }
        }
#elif defined(__AVX2__) || defined(__AVX__)
        // ── AVX2/AVX path: 2D tiles of 8×4 pixels ───────────────────────────
        // Same 4-row strip strategy as the AVX-512 path: adjacent rows share
        // nearly identical (x,z) FBM inputs → better L1/L2 reuse in avx8_fbm.
#pragma omp parallel for schedule(dynamic, 1)
        for (int ty = 0; ty < H; ty += 4)
        {
            int tid = 0;
#ifdef _OPENMP
            tid = omp_get_thread_num() & 63;
#endif
            int row_end = std::min(ty + 4, H);
            for (int y = ty; y < row_end; y++)
            {
                rng_state[tid] ^= (uint32_t)(y * 7919 + frame * 2053);
                // ── AVX2 tile processor lambda (2× unrolled x-loop) ─────────
                // Same strategy as AVX-512 path: step by 16, process two 8-wide
                // tiles per iteration to halve loop overhead and let the OOO core
                // overlap tile-B setup with tile-A march latency.
                auto process_tile_avx8 = [&](int tx)
                {
                    if (tx >= W)
                        return;
                    int batch = std::min(8, W - tx);

                    alignas(32) float rox_a[8], roy_a[8], roz_a[8];
                    alignas(32) float rdx_a[8], rdy_a[8], rdz_a[8];
                    alignas(32) float rdx0[8], rdy0[8], rdz0[8];

                    for (int dx = 0; dx < 8; dx++)
                    {
                        int xi = tx + dx;
                        if (xi >= W)
                            xi = W - 1;
                        float ux = (xi + .5f + jx * .5f + (rng_float(tid) - .5f) * .3f) / W * 2 - 1;
                        float uy = 1 - (y + .5f + jy * .5f + (rng_float(tid) - .5f) * .3f) / H * 2;
                        float rdx_ = cam_fwd.x + cam_right.x * (ux * aspect * fov) + cam_up.x * (uy * fov);
                        float rdy_ = cam_fwd.y + cam_right.y * (ux * aspect * fov) + cam_up.y * (uy * fov);
                        float rdz_ = cam_fwd.z + cam_right.z * (ux * aspect * fov) + cam_up.z * (uy * fov);
                        float rlen = sqrtf(rdx_ * rdx_ + rdy_ * rdy_ + rdz_ * rdz_);
                        rdx0[dx] = rdx_ / rlen;
                        rdy0[dx] = rdy_ / rlen;
                        rdz0[dx] = rdz_ / rlen;
                        float dof_str = 0.02f;
                        float dfx = rng_float(tid) * 2 - 1, dfy = rng_float(tid) * 2 - 1;
                        float rox_ = cam_pos.x + cam_right.x * dfx * dof_str + cam_up.x * dfy * dof_str;
                        float roy_ = cam_pos.y + cam_right.y * dfx * dof_str + cam_up.y * dfy * dof_str;
                        float roz_ = cam_pos.z + cam_right.z * dfx * dof_str + cam_up.z * dfy * dof_str;
                        float fx_ = rox_ + rdx0[dx] * 40, fy_ = roy_ + rdy0[dx] * 40, fz_ = roz_ + rdz0[dx] * 40;
                        float fdx = fx_ - rox_, fdy = fy_ - roy_, fdz = fz_ - roz_;
                        float flen = sqrtf(fdx * fdx + fdy * fdy + fdz * fdz);
                        rox_a[dx] = rox_;
                        roy_a[dx] = roy_;
                        roz_a[dx] = roz_;
                        rdx_a[dx] = fdx / flen;
                        rdy_a[dx] = fdy / flen;
                        rdz_a[dx] = fdz / flen;
                    }

                    __m256 vro_x = _mm256_load_ps(rox_a), vro_y = _mm256_load_ps(roy_a), vro_z = _mm256_load_ps(roz_a);
                    __m256 vrd_x = _mm256_load_ps(rdx_a), vrd_y = _mm256_load_ps(rdy_a), vrd_z = _mm256_load_ps(rdz_a);

                    // ── Coherence sort (AVX2): 3 probe steps, sort ascending ──────
                    int perm8[8];
                    for (int i = 0; i < 8; i++)
                        perm8[i] = i;
                    if (batch > 1 && frame > 0)
                    {
                        alignas(32) float est_t8[8];
                        {
                            __m256 t3 = _mm256_set1_ps(0.001f);
                            for (int si = 0; si < 3; si++)
                            {
#ifdef __AVX2__
                                __m256 px3 = _mm256_fmadd_ps(vrd_x, t3, vro_x);
                                __m256 py3 = _mm256_fmadd_ps(vrd_y, t3, vro_y);
                                __m256 pz3 = _mm256_fmadd_ps(vrd_z, t3, vro_z);
#else
                                __m256 px3 = _mm256_add_ps(_mm256_mul_ps(vrd_x, t3), vro_x);
                                __m256 py3 = _mm256_add_ps(_mm256_mul_ps(vrd_y, t3), vro_y);
                                __m256 pz3 = _mm256_add_ps(_mm256_mul_ps(vrd_z, t3), vro_z);
#endif
                                __m256 d3 = eval_scene_avx8(px3, py3, pz3, 3, false).d;
                                t3 = _mm256_add_ps(t3, _mm256_max_ps(d3, _mm256_set1_ps(0.001f)));
                            }
                            _mm256_store_ps(est_t8, t3);
                        }
                        for (int i = 1; i < 8; i++)
                        {
                            float kd = est_t8[perm8[i]];
                            int kp = perm8[i];
                            int j = i - 1;
                            while (j >= 0 && est_t8[perm8[j]] > kd)
                            {
                                perm8[j + 1] = perm8[j];
                                j--;
                            }
                            perm8[j + 1] = kp;
                        }
                        alignas(32) float t_rox[8], t_roy[8], t_roz[8];
                        alignas(32) float t_rdx[8], t_rdy[8], t_rdz[8];
                        alignas(32) float t_rdx0[8], t_rdy0[8], t_rdz0[8];
                        for (int i = 0; i < 8; i++)
                        {
                            t_rox[i] = rox_a[perm8[i]];
                            t_roy[i] = roy_a[perm8[i]];
                            t_roz[i] = roz_a[perm8[i]];
                            t_rdx[i] = rdx_a[perm8[i]];
                            t_rdy[i] = rdy_a[perm8[i]];
                            t_rdz[i] = rdz_a[perm8[i]];
                            t_rdx0[i] = rdx0[perm8[i]];
                            t_rdy0[i] = rdy0[perm8[i]];
                            t_rdz0[i] = rdz0[perm8[i]];
                        }
                        memcpy(rox_a, t_rox, 32);
                        memcpy(roy_a, t_roy, 32);
                        memcpy(roz_a, t_roz, 32);
                        memcpy(rdx_a, t_rdx, 32);
                        memcpy(rdy_a, t_rdy, 32);
                        memcpy(rdz_a, t_rdz, 32);
                        memcpy(rdx0, t_rdx0, 32);
                        memcpy(rdy0, t_rdy0, 32);
                        memcpy(rdz0, t_rdz0, 32);
                        vro_x = _mm256_load_ps(rox_a);
                        vro_y = _mm256_load_ps(roy_a);
                        vro_z = _mm256_load_ps(roz_a);
                        vrd_x = _mm256_load_ps(rdx_a);
                        vrd_y = _mm256_load_ps(rdy_a);
                        vrd_z = _mm256_load_ps(rdz_a);
                    }
                    March8 m8 = ray_march_avx8(vro_x, vro_y, vro_z, vrd_x, vrd_y, vrd_z);

                    alignas(32) float hx[8], hy[8], hz[8], ht[8];
                    alignas(32) int hm[8];
                    _mm256_store_ps(hx, m8.pos_x);
                    _mm256_store_ps(hy, m8.pos_y);
                    _mm256_store_ps(hz, m8.pos_z);
                    _mm256_store_ps(ht, m8.t);
                    _mm256_store_si256((__m256i *)hm, m8.mat);

                    alignas(32) float col_r[8], col_g[8], col_b[8];
                    alignas(32) float pnx[8], pny[8], pnz[8];
                    shade_batch_avx8(hx, hy, hz, hm, m8.hit_mask,
                                     rdx0, rdy0, rdz0,
                                     cam_pos, tid,
                                     frame == 0 ? prev_gi : nullptr,
                                     frame == 0 ? prev_cam : nullptr,
                                     frame == 0 ? curr_gi : nullptr,
                                     y, tx,
                                     col_r, col_g, col_b,
                                     frame == 0 ? pnx : nullptr,
                                     frame == 0 ? pny : nullptr,
                                     frame == 0 ? pnz : nullptr);

                    for (int dx = 0; dx < 8; dx++)
                    {
                        int xi = tx + perm8[dx];
                        if (xi >= W)
                            continue;
                        bool did_hit = (m8.hit_mask >> dx) & 1;
                        V3 fog_rd = {rdx_a[dx], rdy_a[dx], rdz_a[dx]};
                        V3 col = {col_r[dx], col_g[dx], col_b[dx]};
                        col = apply_fog(col, ht[dx], fog_rd);
                        if (frame == 0)
                        {
                            dep[y * W + xi] = ht[dx];
                            if (did_hit)
                            {
                                nrm[(y * W + xi) * 3 + 0] = pnx[dx] * .5f + .5f;
                                nrm[(y * W + xi) * 3 + 1] = pny[dx] * .5f + .5f;
                                nrm[(y * W + xi) * 3 + 2] = pnz[dx] * .5f + .5f;
                            }
                            else
                            {
                                nrm[(y * W + xi) * 3 + 0] = .5f;
                                nrm[(y * W + xi) * 3 + 1] = .5f;
                                nrm[(y * W + xi) * 3 + 2] = .5f;
                            }
                        }
                        acc[(y * W + xi) * 3 + 0] += col.x;
                        acc[(y * W + xi) * 3 + 1] += col.y;
                        acc[(y * W + xi) * 3 + 2] += col.z;
                    }
                };

                for (int x = 0; x < W; x += 16)
                {
                    process_tile_avx8(x);
                    process_tile_avx8(x + 8);
                }
            } // end for(int y) inner row loop
        } // end for(int ty) 4-row strip loop
#else
        // ── Scalar fallback ───────────────────────────────────────────────
#pragma omp parallel for schedule(dynamic, 4)
        for (int y = 0; y < H; y++)
        {
            int tid = 0;
#ifdef _OPENMP
            tid = omp_get_thread_num() & 63;
#endif
            rng_state[tid] ^= (uint32_t)(y * 7919 + frame * 2053);
            for (int x = 0; x < W; x++)
            {
                float ux = (x + .5f + jx * .5f + (rng_float(tid) - .5f) * .3f) / W * 2 - 1;
                float uy = 1 - (y + .5f + jy * .5f + (rng_float(tid) - .5f) * .3f) / H * 2;
                V3 rd = norm(cam_fwd + cam_right * (ux * aspect * fov) + cam_up * (uy * fov));
                float dof_str = 0.02f;
                float dfx = rng_float(tid) * 2 - 1, dfy = rng_float(tid) * 2 - 1;
                V3 ro_dof = {cam_pos.x + cam_right.x * dfx * dof_str + cam_up.x * dfy * dof_str,
                             cam_pos.y + cam_right.y * dfx * dof_str + cam_up.y * dfy * dof_str,
                             cam_pos.z + cam_right.z * dfx * dof_str + cam_up.z * dfy * dof_str};
                V3 focus = {cam_pos.x + rd.x * 40, cam_pos.y + rd.y * 40, cam_pos.z + rd.z * 40};
                V3 rd_dof = norm(focus - ro_dof);

                March primary = ray_march(ro_dof, rd_dof);
                V3 col = shade_known_hit(cam_pos, rd, primary, tid);
                col = apply_fog(col, primary.t, rd_dof);

                if (frame == 0)
                {
                    dep[y * W + x] = primary.t;
                    if (primary.hit)
                    {
                        V3 n = scene_normal(primary.pos);
                        nrm[(y * W + x) * 3 + 0] = n.x * .5f + .5f;
                        nrm[(y * W + x) * 3 + 1] = n.y * .5f + .5f;
                        nrm[(y * W + x) * 3 + 2] = n.z * .5f + .5f;
                    }
                    else
                    {
                        nrm[(y * W + x) * 3 + 0] = .5f;
                        nrm[(y * W + x) * 3 + 1] = .5f;
                        nrm[(y * W + x) * 3 + 2] = .5f;
                    }
                }
                acc[(y * W + x) * 3 + 0] += col.x;
                acc[(y * W + x) * 3 + 1] += col.y;
                acc[(y * W + x) * 3 + 2] += col.z;
            }
        }
#endif
    }

    for (int i = 0; i < NP * 3; i++)
        buf[i] = acc[i] / SAMPLES;
    for (int i = 0; i < NP * 3; i += 3)
    {
        buf[i] = aces(buf[i] * 1.8f);
        buf[i + 1] = aces(buf[i + 1] * 1.8f);
        buf[i + 2] = aces(buf[i + 2] * 1.8f);
    }
    bilateral_denoise(out, buf, dep, nrm, W, H, 2);
    for (int i = 0; i < NP * 3; i++)
        out[i] = powf(clampf(out[i], 0, 1), 1.f / 2.2f);
    write_ppm(outpath, out, W, H);
    delete[] acc;
    delete[] dep;
    delete[] nrm;
    delete[] buf;
    delete[] out;
}
// ─── REALTIME GAME BRIDGE (added for game.cpp) ───────────────────────────────
static bool g_cpurt_initialized = false;
static GICache *g_ri_cache[2] = {nullptr, nullptr};
static int g_ri_cache_index = 0;
static int g_ri_cache_w = 0, g_ri_cache_h = 0;
static Camera g_ri_prev_cam{};
static bool g_ri_has_prev = false;
static bool g_reuse_enabled = false;
// Reprojection guards against colour dragging. Tighten with the CPURT_REUSE_*
// environment variables if needed; defaults are tuned for normal mouse motion.
static float g_reuse_color_tol = 16.f / 255.f; // 3x3 final-colour spread
static float g_reuse_view_tol = 0.004f;        // 1 - dot(view vectors)
static float g_reuse_max_move = 8.f;           // projected pixels, surfaces
static float g_reuse_sky_max_move = 8.f;       // sky uses the same strict limit
static const int REUSE_DEPTH_SCALE = 1024;
static const int REUSE_DEPTH_SKY = INT_MAX - 1;
static const long long REUSE_CLAIM_EMPTY =
    (static_cast<long long>(INT_MAX) << 32) | 0xffffffffLL;
static std::atomic<long long> *g_reuse_claim = nullptr;
static int g_reuse_claim_w = 0, g_reuse_claim_h = 0;
// Thread-local slots avoid racing on the diagnostic counters inside the
// OpenMP row loop. The stats are cumulative for diagnostic use only.
static const int REUSE_STAT_SLOTS = 256;
static long long g_reuse_try[REUSE_STAT_SLOTS] = {};
static long long g_reuse_hit[REUSE_STAT_SLOTS] = {};
static long long g_reuse_sky[REUSE_STAT_SLOTS] = {};
void cpurt_reuse_stats(long long *attempts, long long *hits, long long *sky)
{
    long long tries = 0, hit = 0, skyhit = 0;
    for (int i = 0; i < REUSE_STAT_SLOTS; i++)
    {
        tries += g_reuse_try[i];
        hit += g_reuse_hit[i];
        skyhit += g_reuse_sky[i];
    }
    if (attempts)
        *attempts = tries;
    if (hits)
        *hits = hit;
    if (sky)
        *sky = skyhit;
}

static inline int reuse_stat_slot()
{
#ifdef _OPENMP
    return std::min(omp_get_thread_num(), REUSE_STAT_SLOTS - 1);
#else
    return 0;
#endif
}

static void clear_gi_cache(GICache *c)
{
    if (!c)
        return;
    int NP = c->W * c->H;
    memset(c->gi_r, 0, NP * sizeof(float));
    memset(c->gi_g, 0, NP * sizeof(float));
    memset(c->gi_b, 0, NP * sizeof(float));
    memset(c->sha, 0, NP * sizeof(float));
    memset(c->ao, 0, NP * sizeof(float));
    memset(c->nx, 0, NP * sizeof(float));
    memset(c->ny, 0, NP * sizeof(float));
    memset(c->nz, 0, NP * sizeof(float));
    memset(c->refl_r, 0, NP * sizeof(float));
    memset(c->refl_g, 0, NP * sizeof(float));
    memset(c->refl_b, 0, NP * sizeof(float));
    memset(c->fin_r, 0, NP * sizeof(float));
    memset(c->fin_g, 0, NP * sizeof(float));
    memset(c->fin_b, 0, NP * sizeof(float));
    memset(c->depth, 0, NP * sizeof(float));
    std::fill(c->mat, c->mat + NP, -1);
}

static inline void copy_cache_pixel(const GICache *src, int si,
                                    GICache *dst, int di)
{
    dst->gi_r[di] = src->gi_r[si];
    dst->gi_g[di] = src->gi_g[si];
    dst->gi_b[di] = src->gi_b[si];
    dst->sha[di] = src->sha[si];
    dst->ao[di] = src->ao[si];
    dst->nx[di] = src->nx[si];
    dst->ny[di] = src->ny[si];
    dst->nz[di] = src->nz[si];
    dst->refl_r[di] = src->refl_r[si];
    dst->refl_g[di] = src->refl_g[si];
    dst->refl_b[di] = src->refl_b[si];
    dst->fin_r[di] = src->fin_r[si];
    dst->fin_g[di] = src->fin_g[si];
    dst->fin_b[di] = src->fin_b[si];
    dst->depth[di] = src->depth[si];
    dst->mat[di] = src->mat[si];
}

static inline bool stable_hit_neighborhood(const GICache *c, int px, int py,
                                           int mat, float d, float tol)
{
    const int W = c->W, H = c->H;
    for (int dy = -1; dy <= 1; dy++)
    {
        int yy = py + dy;
        if (yy < 0 || yy >= H)
            return false;
        for (int dx = -1; dx <= 1; dx++)
        {
            int xx = px + dx;
            if (xx < 0 || xx >= W)
                return false;
            int idx = yy * W + xx;
            if (c->mat[idx] != mat || c->depth[idx] > 1e29f)
                return false;
            if (fabsf(c->depth[idx] - d) > tol)
                return false;
        }
    }
    return true;
}

// Large local colour changes are usually highlights or silhouettes. Copying
// those from the previous frame causes the visible colour dragging reported
// during camera motion. This check is only run when the pixel actually moved.
static inline bool stable_hit_color(const GICache *c, int px, int py,
                                    float color_tol)
{
    const int W = c->W, H = c->H;
    const int center = py * W + px;
    const float cr = c->fin_r[center], cg = c->fin_g[center], cb = c->fin_b[center];
    for (int dy = -1; dy <= 1; dy++)
    {
        int yy = py + dy;
        if (yy < 0 || yy >= H)
            return false;
        for (int dx = -1; dx <= 1; dx++)
        {
            int xx = px + dx;
            if (xx < 0 || xx >= W)
                return false;
            int idx = yy * W + xx;
            if (fabsf(c->fin_r[idx] - cr) > color_tol ||
                fabsf(c->fin_g[idx] - cg) > color_tol ||
                fabsf(c->fin_b[idx] - cb) > color_tol)
                return false;
        }
    }
    return true;
}

static inline bool stable_sky_neighborhood(const GICache *c, int px, int py)
{
    const int W = c->W, H = c->H;
    for (int dy = -1; dy <= 1; dy++)
    {
        int yy = py + dy;
        if (yy < 0 || yy >= H)
            return false;
        for (int dx = -1; dx <= 1; dx++)
        {
            int xx = px + dx;
            if (xx < 0 || xx >= W)
                return false;
            int idx = yy * W + xx;
            if (c->mat[idx] >= 0 || c->depth[idx] < 1e29f)
                return false;
        }
    }
    return true;
}

static inline bool stable_sky_color(const GICache *c, int px, int py,
                                    float color_tol)
{
    const int W = c->W, H = c->H;
    const int center = py * W + px;
    const float cr = c->fin_r[center], cg = c->fin_g[center], cb = c->fin_b[center];
    for (int dy = -1; dy <= 1; dy++)
    {
        int yy = py + dy;
        if (yy < 0 || yy >= H)
            return false;
        for (int dx = -1; dx <= 1; dx++)
        {
            int xx = px + dx;
            if (xx < 0 || xx >= W)
                return false;
            int idx = yy * W + xx;
            if (fabsf(c->fin_r[idx] - cr) > color_tol ||
                fabsf(c->fin_g[idx] - cg) > color_tol ||
                fabsf(c->fin_b[idx] - cb) > color_tol)
                return false;
        }
    }
    return true;
}

// Forward-reproject the previous frame's visible pixels into the current frame.
// Each previous pixel reconstructs its own world-space hit point and is projected
// with the real camera motion, so a pixel that becomes visible this frame has no
// history candidate and is shaded fresh. The claim key packs quantised current
// view distance (high 32 bits) and source pixel index (low 32 bits); the nearest
// candidate wins using an atomic min. This is the key difference from the old
// gather path, which guessed the current hit from stale same-pixel depth and let
// old colours drag across moving silhouettes.
static void reproject_previous_forward(const GICache *prev,
                                       const Camera &prev_cam,
                                       const Camera &curr_cam,
                                       std::atomic<long long> *claim)
{
    if (!g_reuse_enabled || !prev)
        return;
    if (prev->W != curr_cam.W || prev->H != curr_cam.H)
        return;

    const int W = curr_cam.W, H = curr_cam.H;
    const int NP = W * H;
    for (int i = 0; i < NP; i++)
    {
        claim[i].store(REUSE_CLAIM_EMPTY, std::memory_order_relaxed);
    }

    // Static camera: history maps 1:1. Skip all world-space reprojection and
    // simply keep static/sky pixels; animated materials still render fresh.
    if (same_camera(prev_cam, curr_cam))
    {
        for (int i = 0; i < NP; i++)
        {
            const int m = prev->mat[i];
            const float d = prev->depth[i];
            long long key = REUSE_CLAIM_EMPTY;
            if (m < 0 && d > 1e29f)
            {
                key = (static_cast<long long>(REUSE_DEPTH_SKY) << 32) |
                      static_cast<unsigned int>(i);
            }
            else if (m >= MAT_GRASS && m != MAT_WATER &&
                     m != MAT_MESH_GOLD && m != MAT_MESH_CRYSTAL && d <= 1e29f)
            {
                const int dq = (int)clampf(d * REUSE_DEPTH_SCALE + 0.5f,
                                           0.f, (float)(INT_MAX - 2));
                key = (static_cast<long long>(dq) << 32) |
                      static_cast<unsigned int>(i);
            }
            claim[i].store(key, std::memory_order_relaxed);
        }
        return;
    }

    const float max_move2 = g_reuse_max_move * g_reuse_max_move;
    const float sky_max_move2 = g_reuse_sky_max_move * g_reuse_sky_max_move;

#pragma omp parallel for schedule(dynamic, 4)
    for (int sy = 0; sy < H; sy++)
    {
        for (int sx = 0; sx < W; sx++)
        {
            const int si = sy * W + sx;
            const int m = prev->mat[si];
            const float prev_d = prev->depth[si];
            const bool sky = (m < 0 && prev_d > 1e29f);

            if (!sky)
            {
                // Water ripples and baked SDF meshes animate independently of
                // the camera. Mark where they appear this frame as no-reuse so
                // background history cannot ghost into their new silhouette.
                // Water ripples and baked SDF meshes animate independently of
                // the camera, so they never contribute history.
                if (m < MAT_GRASS || m == MAT_WATER ||
                    m == MAT_MESH_GOLD || m == MAT_MESH_CRYSTAL)
                    continue;
                if (prev_d > 1e29f)
                    continue;
                const float tol = 0.6f + 0.01f * prev_d;
                if (!stable_hit_neighborhood(prev, sx, sy, m, prev_d, tol))
                    continue;
            }
            else if (!stable_sky_neighborhood(prev, sx, sy))
            {
                continue;
            }

            const V3 rd_prev = camera_ray_dir(prev_cam, sx, sy);
            V3 P{};
            int dx, dy;

            if (sky)
            {
                // Infinite-distance sky: reproject the direction itself.
                const float vz = dot(rd_prev, curr_cam.fwd);
                if (vz <= 0.001f)
                    continue;
                const float u = dot(rd_prev, curr_cam.right) /
                                (vz * curr_cam.aspect * curr_cam.fov);
                const float v = -dot(rd_prev, curr_cam.up) /
                                (vz * curr_cam.fov);
                if (u < -1.f || u > 1.f || v < -1.f || v > 1.f)
                    continue;
                dx = (int)((u + 1.f) * 0.5f * W);
                dy = (int)((v + 1.f) * 0.5f * H);
            }
            else
            {
                // Reconstruct this previous pixel's surface point and project
                // it through the current camera.
                P = prev_cam.pos + rd_prev * prev_d;
                const V3 V = P - curr_cam.pos;
                const float vz = dot(V, curr_cam.fwd);
                if (vz <= 0.001f)
                    continue;
                const float u = dot(V, curr_cam.right) /
                                (vz * curr_cam.aspect * curr_cam.fov);
                const float v = -dot(V, curr_cam.up) /
                                (vz * curr_cam.fov);
                if (u < -1.f || u > 1.f || v < -1.f || v > 1.f)
                    continue;
                dx = (int)((u + 1.f) * 0.5f * W);
                dy = (int)((v + 1.f) * 0.5f * H);
            }

            dx = std::max(0, std::min(W - 1, dx));
            dy = std::max(0, std::min(H - 1, dy));
            const float mdx = (float)(dx - sx), mdy = (float)(dy - sy);
            const float move2 = mdx * mdx + mdy * mdy;
            if (move2 > (sky ? sky_max_move2 : max_move2))
                continue;

            const V3 rd_curr = camera_ray_dir(curr_cam, dx, dy);
            const float view_delta = 1.f - dot(rd_curr, rd_prev);
            if (view_delta > g_reuse_view_tol)
                continue;

            if (move2 > 1.f)
            {
                if (sky)
                {
                    if (!stable_sky_color(prev, sx, sy, g_reuse_color_tol))
                        continue;
                }
                else
                {
                    if (!stable_hit_color(prev, sx, sy, g_reuse_color_tol))
                        continue;
                }
            }

            if (!sky)
            {
                const float facing = -(prev->nx[si] * rd_curr.x +
                                       prev->ny[si] * rd_curr.y +
                                       prev->nz[si] * rd_curr.z);
                if (facing < 0.05f)
                    continue;
            }

            int depth_key = REUSE_DEPTH_SKY;
            if (!sky)
            {
                const float curr_d = len(P - curr_cam.pos);
                depth_key = (int)clampf(curr_d * REUSE_DEPTH_SCALE + 0.5f,
                                        0.f, (float)(INT_MAX - 2));
            }
            const long long key =
                (static_cast<long long>(depth_key) << 32) |
                (static_cast<unsigned int>(si));

            std::atomic<long long> &cell = claim[dy * W + dx];
            long long old = cell.load(std::memory_order_relaxed);
            while (key < old &&
                   !cell.compare_exchange_weak(old, key,
                                               std::memory_order_relaxed,
                                               std::memory_order_relaxed))
            {
            }
        }
    }
}

void cpurt_init_assets()
{
    if (g_cpurt_initialized)
        return;
    g_cpurt_initialized = true;
    if (getenv("CPURT_NO_REUSE"))
        g_reuse_enabled = false;
    if (const char *e = getenv("CPURT_REUSE_COLOR_TOL"))
        g_reuse_color_tol = (float)std::max(0.0, atof(e) / 255.0);
    if (const char *e = getenv("CPURT_REUSE_VIEW_TOL"))
        g_reuse_view_tol = (float)std::max(0.0, atof(e));
    if (const char *e = getenv("CPURT_REUSE_MAX_MOVE"))
        g_reuse_max_move = (float)std::max(0.0, atof(e));
    if (const char *e = getenv("CPURT_REUSE_SKY_MAX_MOVE"))
        g_reuse_sky_max_move = (float)std::max(0.0, atof(e));
    if (getenv("CPURT_NO_BAKED"))
        g_baked_enabled = false;
    if (getenv("CPURT_NO_BOUNDS"))
        g_bounds_enabled = false;
    if (getenv("CPURT_BAKED"))
        g_baked_enabled = true;
    if (getenv("CPURT_BOUNDS"))
        g_bounds_enabled = true;
    init_tree_bases();
#ifdef __AVX2__
    ensure_height_tables();
#endif
    g_grids[g_ngrid++] = load_sdf("assets/arch.sdf", {8.f, terrain_h(8, 0), 0.f}, 2.5f, MAT_MESH_STONE);
    g_grids[g_ngrid++] = load_sdf("assets/torus.sdf", {0.f, 5.f, 0.f}, 2.0f, MAT_MESH_GOLD);
    g_grids[g_ngrid++] = load_sdf("assets/sphere.sdf", {-10.f, terrain_h(-10, 15) + 1.2f, 15.f}, 1.5f, MAT_MESH_CRYSTAL);
}

void cpurt_render_buffer(uint8_t *rgb_out, int W, int H,
                         float cx, float cy, float cz,
                         float fx, float fy, float fz,
                         float rx, float ry, float rz,
                         float ux, float uy, float uz,
                         float anim_t, int threads)
{
    cpurt_init_assets();
#ifdef _OPENMP
    omp_set_num_threads(threads);
#endif
    sun_dir = norm({0.7f, 0.55f, 0.3f});

    if (g_grids[1].valid)
    {
        g_grids[1].pos.y = 5.f + sinf(anim_t * 1.3f) * 0.8f;
        g_grids[1].rot_y = anim_t * 0.9f;
        refresh_sdf_grid_cache(g_grids[1]);
    }
    if (g_grids[2].valid)
    {
        g_grids[2].pos.y = terrain_h(-10, 15) + 1.2f + sinf(anim_t * 2.1f + 1.f) * 0.4f;
        g_grids[2].rot_y = anim_t * 0.4f;
        refresh_sdf_grid_cache(g_grids[2]);
    }

    V3 cam_pos = {cx, cy, cz};
    V3 cam_fwd = {fx, fy, fz};
    V3 cam_right = {rx, ry, rz};
    V3 cam_up = {ux, uy, uz};
    float fov = 0.85f;
    float aspect = (float)W / (float)H;

    // Two ping-pong history buffers live entirely inside the renderer so game.cpp
    // does not need to know about GICache. Resolution changes reset history.
    if (W != g_ri_cache_w || H != g_ri_cache_h)
    {
        delete g_ri_cache[0];
        delete g_ri_cache[1];
        g_ri_cache[0] = new GICache(W, H);
        g_ri_cache[1] = new GICache(W, H);
        delete[] g_reuse_claim;
        g_reuse_claim = new std::atomic<long long>[W * H];
        g_ri_cache_w = W;
        g_ri_cache_h = H;
        g_reuse_claim_w = W;
        g_reuse_claim_h = H;
        g_ri_cache_index = 0;
        g_ri_has_prev = false;
    }
    GICache *prev_gi = g_ri_has_prev ? g_ri_cache[g_ri_cache_index] : nullptr;
    GICache *curr_gi = g_ri_cache[1 - g_ri_cache_index];
    const Camera *prev_cam = g_ri_has_prev ? &g_ri_prev_cam : nullptr;
    const bool no_gi_history = getenv("CPURT_NO_GI_REUSE") != nullptr;
    GICache *gi_history = no_gi_history ? nullptr : prev_gi;
    const Camera *gi_history_cam = no_gi_history ? nullptr : prev_cam;
    const bool final_reuse = g_reuse_enabled && g_ri_has_prev &&
                             g_reuse_claim_w == W && g_reuse_claim_h == H;
    clear_gi_cache(curr_gi);

    Camera ri_cam = {cam_pos, cam_fwd, cam_right, cam_up, fov, aspect, W, H};
    if (final_reuse)
    {
        reproject_previous_forward(prev_gi, g_ri_prev_cam, ri_cam, g_reuse_claim);
    }

    for (int i = 0; i < 64; i++)
        rng_state[i] = 12345u * (i + 1) * 2654435761u ^ (uint32_t)(anim_t * 1000);

#pragma omp parallel for schedule(dynamic, 4)
    for (int y = 0; y < H; y++)
    {
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num() & 63;
#endif
        rng_state[tid] ^= (uint32_t)(y * 7919);

        // OpenGL textures expect bottom-to-top rows, so flip Y into rgb_out
        int out_y = H - 1 - y;

#ifdef __AVX512F__
        for (int x = 0; x < W; x += 16)
        {
            int batch = std::min(16, W - x);
            alignas(64) float rox_a[16], roy_a[16], roz_a[16];
            alignas(64) float rdx0[16], rdy0[16], rdz0[16];
            for (int dx = 0; dx < 16; dx++)
            {
                int xi = std::min(x + dx, W - 1);
                float u = (xi + 0.5f) / W * 2.f - 1.f;
                float v = 1.f - (y + 0.5f) / H * 2.f;
                V3 rd = norm(cam_fwd + cam_right * (u * aspect * fov) + cam_up * (v * fov));
                rox_a[dx] = cam_pos.x;
                roy_a[dx] = cam_pos.y;
                roz_a[dx] = cam_pos.z;
                rdx0[dx] = rd.x;
                rdy0[dx] = rd.y;
                rdz0[dx] = rd.z;
            }
            March16 m16 = ray_march_avx512(_mm512_load_ps(rox_a), _mm512_load_ps(roy_a), _mm512_load_ps(roz_a),
                                           _mm512_load_ps(rdx0), _mm512_load_ps(rdy0), _mm512_load_ps(rdz0));
            alignas(64) float hx[16], hy[16], hz[16], ht[16];
            alignas(64) int hm[16];
            _mm512_store_ps(hx, m16.pos_x);
            _mm512_store_ps(hy, m16.pos_y);
            _mm512_store_ps(hz, m16.pos_z);
            _mm512_store_ps(ht, m16.t);
            _mm512_store_epi32(hm, m16.mat);

            alignas(64) float col_r[16], col_g[16], col_b[16];
            shade_batch_avx512(hx, hy, hz, hm, m16.hit_mask, rdx0, rdy0, rdz0,
                               cam_pos, tid, gi_history, gi_history_cam, curr_gi, y, x, col_r, col_g, col_b);

            for (int dx = 0; dx < batch; dx++)
            {
                V3 rd = {rdx0[dx], rdy0[dx], rdz0[dx]};
                V3 col = apply_fog({col_r[dx], col_g[dx], col_b[dx]}, ht[dx], rd);
                col.x = powf(aces(col.x * 1.8f), 1.f / 2.2f);
                col.y = powf(aces(col.y * 1.8f), 1.f / 2.2f);
                col.z = powf(aces(col.z * 1.8f), 1.f / 2.2f);
                int idx = (out_y * W + (x + dx)) * 3;
                rgb_out[idx + 0] = (uint8_t)(clampf(col.x, 0.f, 1.f) * 255.f);
                rgb_out[idx + 1] = (uint8_t)(clampf(col.y, 0.f, 1.f) * 255.f);
                rgb_out[idx + 2] = (uint8_t)(clampf(col.z, 0.f, 1.f) * 255.f);
            }
        }
#elif defined(__AVX2__) || defined(__AVX__)
        {
            alignas(32) float rox_a[8], roy_a[8], roz_a[8];
            alignas(32) float rdx0[8], rdy0[8], rdz0[8];
            int lane_x[8];

            auto process_batch = [&](int n)
            {
                if (n <= 0)
                    return;
                uint8_t active = (uint8_t)((1u << n) - 1u);
                for (int dx = n; dx < 8; dx++)
                {
                    rox_a[dx] = cam_pos.x;
                    roy_a[dx] = cam_pos.y;
                    roz_a[dx] = cam_pos.z;
                    rdx0[dx] = 0.f;
                    rdy0[dx] = 1.f;
                    rdz0[dx] = 0.f;
                    lane_x[dx] = -1;
                }
                March8 m8 = ray_march_avx8(
                    _mm256_load_ps(rox_a), _mm256_load_ps(roy_a), _mm256_load_ps(roz_a),
                    _mm256_load_ps(rdx0), _mm256_load_ps(rdy0), _mm256_load_ps(rdz0),
                    MAX_DIST, 7, active);
                alignas(32) float hx[8], hy[8], hz[8], ht[8];
                alignas(32) int hm[8];
                _mm256_store_ps(hx, m8.pos_x);
                _mm256_store_ps(hy, m8.pos_y);
                _mm256_store_ps(hz, m8.pos_z);
                _mm256_store_ps(ht, m8.t);
                _mm256_store_si256((__m256i *)hm, m8.mat);

                alignas(32) float col_r[8], col_g[8], col_b[8];
                // GI/shadow/AO history is intentionally not reused: reprojection
                // of the stochastic bounce moved highlights too much. The forward
                // final-pixel cache supplies the temporal speedup instead.
                shade_batch_avx8(hx, hy, hz, hm, m8.hit_mask, rdx0, rdy0, rdz0,
                                 cam_pos, tid, nullptr, nullptr, curr_gi, y, 0,
                                 col_r, col_g, col_b, nullptr, nullptr, nullptr, lane_x);

                for (int dx = 0; dx < n; dx++)
                {
                    V3 rd = {rdx0[dx], rdy0[dx], rdz0[dx]};
                    V3 col = apply_fog({col_r[dx], col_g[dx], col_b[dx]}, ht[dx], rd);
                    float fr = powf(aces(col.x * 1.8f), 1.f / 2.2f);
                    float fg = powf(aces(col.y * 1.8f), 1.f / 2.2f);
                    float fb = powf(aces(col.z * 1.8f), 1.f / 2.2f);
                    int oi = (out_y * W + lane_x[dx]) * 3;
                    rgb_out[oi + 0] = (uint8_t)(clampf(fr, 0.f, 1.f) * 255.f);
                    rgb_out[oi + 1] = (uint8_t)(clampf(fg, 0.f, 1.f) * 255.f);
                    rgb_out[oi + 2] = (uint8_t)(clampf(fb, 0.f, 1.f) * 255.f);
                    int ci = y * W + lane_x[dx];
                    curr_gi->fin_r[ci] = clampf(fr, 0.f, 1.f);
                    curr_gi->fin_g[ci] = clampf(fg, 0.f, 1.f);
                    curr_gi->fin_b[ci] = clampf(fb, 0.f, 1.f);
                }
            };

            int n = 0;
            const int stat_slot = reuse_stat_slot();
            for (int x = 0; x < W; x++)
            {
                ++g_reuse_try[stat_slot];
                long long claim_key = REUSE_CLAIM_EMPTY;
                if (final_reuse)
                {
                    claim_key = g_reuse_claim[y * W + x].load(std::memory_order_relaxed);
                }
                if (claim_key != REUSE_CLAIM_EMPTY)
                {
                    const int si = static_cast<int>(static_cast<unsigned int>(claim_key));
                    const int depth_key = static_cast<int>(claim_key >> 32);
                    const int ci = y * W + x;
                    copy_cache_pixel(prev_gi, si, curr_gi, ci);
                    if (depth_key == REUSE_DEPTH_SKY)
                    {
                        curr_gi->depth[ci] = 1e30f;
                        curr_gi->mat[ci] = -1;
                        curr_gi->nx[ci] = curr_gi->ny[ci] = curr_gi->nz[ci] = 0.f;
                        curr_gi->gi_r[ci] = curr_gi->gi_g[ci] = curr_gi->gi_b[ci] = 0.f;
                        curr_gi->sha[ci] = curr_gi->ao[ci] = 1.f;
                        ++g_reuse_sky[stat_slot];
                    }
                    else
                    {
                        curr_gi->depth[ci] = depth_key / (float)REUSE_DEPTH_SCALE;
                        ++g_reuse_hit[stat_slot];
                    }
                    const int oi = (out_y * W + x) * 3;
                    rgb_out[oi + 0] = (uint8_t)(clampf(curr_gi->fin_r[ci], 0.f, 1.f) * 255.f);
                    rgb_out[oi + 1] = (uint8_t)(clampf(curr_gi->fin_g[ci], 0.f, 1.f) * 255.f);
                    rgb_out[oi + 2] = (uint8_t)(clampf(curr_gi->fin_b[ci], 0.f, 1.f) * 255.f);
                    continue;
                }

                float u = (x + 0.5f) / W * 2.f - 1.f;
                float v = 1.f - (y + 0.5f) / H * 2.f;
                V3 rd = norm(cam_fwd + cam_right * (u * aspect * fov) + cam_up * (v * fov));
                rox_a[n] = cam_pos.x;
                roy_a[n] = cam_pos.y;
                roz_a[n] = cam_pos.z;
                rdx0[n] = rd.x;
                rdy0[n] = rd.y;
                rdz0[n] = rd.z;
                lane_x[n] = x;
                if (++n == 8)
                {
                    process_batch(8);
                    n = 0;
                }
            }
            process_batch(n);
        }
#else
        for (int x = 0; x < W; x++)
        {
            float u = (x + 0.5f) / W * 2.f - 1.f;
            float v = 1.f - (y + 0.5f) / H * 2.f;
            V3 rd = norm(cam_fwd + cam_right * (u * aspect * fov) + cam_up * (v * fov));
            March primary = ray_march(cam_pos, rd);
            V3 col = shade_known_hit(cam_pos, rd, primary, tid);
            col = apply_fog(col, primary.t, rd);
            col.x = powf(aces(col.x * 1.8f), 1.f / 2.2f);
            col.y = powf(aces(col.y * 1.8f), 1.f / 2.2f);
            col.z = powf(aces(col.z * 1.8f), 1.f / 2.2f);
            int idx = (out_y * W + x) * 3;
            rgb_out[idx + 0] = (uint8_t)(clampf(col.x, 0.f, 1.f) * 255.f);
            rgb_out[idx + 1] = (uint8_t)(clampf(col.y, 0.f, 1.f) * 255.f);
            rgb_out[idx + 2] = (uint8_t)(clampf(col.z, 0.f, 1.f) * 255.f);
        }
#endif
    }

    g_ri_prev_cam = ri_cam;
    g_ri_has_prev = true;
    g_ri_cache_index = 1 - g_ri_cache_index;
}

#ifndef NO_MAIN
int main(int argc, char **argv)
{
    bool is_video = argc > 1 && !strcmp(argv[1], "video");
    bool fast = argc > 1 && !strcmp(argv[1], "fast");
    bool hq = argc > 1 && !strcmp(argv[1], "hq");
    bool hd = argc > 1 && !strcmp(argv[1], "hd");

#ifdef __AVX512F__
    printf("cpurt20 — AVX-512 build  (legacy 16-wide path)\n");
#elif defined(__AVX2__)
    printf("cpurt20 — AVX2 build     (final-pixel temporal reuse)\n");
#elif defined(__AVX__)
    printf("cpurt20 — AVX build      (8-wide batch rays)\n");
#else
    printf("cpurt16 — scalar build   (no SIMD)\n");
#endif

    printf("Loading baked SDF objects...\n");
    g_grids[g_ngrid++] = load_sdf("assets/arch.sdf", {8.f, terrain_h(8, 0), 0.f}, 2.5f, MAT_MESH_STONE);
    g_grids[g_ngrid++] = load_sdf("assets/torus.sdf", {0.f, 5.f, 0.f}, 2.0f, MAT_MESH_GOLD);
    g_grids[g_ngrid++] = load_sdf("assets/sphere.sdf", {-10.f, terrain_h(-10, 15) + 1.2f, 15.f}, 1.5f, MAT_MESH_CRYSTAL);

    if (is_video)
    {
        // ── Video: render all frames sequentially with temporal reprojection ──
        // main() owns both GI cache buffers; they swap each frame.
        // Frame 0: prev_gi = nullptr → all pixels fire new GI rays, fill curr_gi.
        // Frame N: prev_gi = last frame's cache → reprojection active.
        // Argument layout:
        //   ./cpurt7 video TOTAL          — render all frames 0..TOTAL-1
        //   ./cpurt7 video FRAME TOTAL    — render just FRAME (with silent warm-up)
        bool single_frame = (argc > 3);
        int total = single_frame ? atoi(argv[3]) : (argc > 2 ? atoi(argv[2]) : 48);
        int frame_start = single_frame ? atoi(argv[2]) : 0;
        int frame_end = single_frame ? frame_start + 1 : total;
        int W = 400, H = 225;

        system("mkdir -p frames");

        GICache *gi_a = new GICache(W, H); // ping
        GICache *gi_b = new GICache(W, H); // pong
        GICache *prev_gi_ptr = nullptr;
        Camera prev_cam_val{};
        Camera *prev_cam_ptr = nullptr;

        // Warm-up: if starting mid-sequence, silently render frame_start-1
        if (frame_start > 0)
        {
            int fi = frame_start - 1;
            float anim_t = (float)fi / total * 2.f * PI;
            Camera cam_w = build_camera(anim_t, W, H);
            char dummy[256];
            snprintf(dummy, sizeof(dummy), "/dev/null");
            render_frame(cam_w, 2, anim_t, dummy, nullptr, nullptr, gi_a);
            prev_gi_ptr = gi_a;
            prev_cam_val = cam_w;
            prev_cam_ptr = &prev_cam_val;
            std::swap(gi_a, gi_b);
        }

        for (int fi = frame_start; fi < frame_end; fi++)
        {
            float anim_t = (float)fi / total * 2.f * PI;
            Camera cam = build_camera(anim_t, W, H);
            char outpath[256];
            snprintf(outpath, sizeof(outpath), "frames/frame_%04d.ppm", fi);
            auto t0 = std::chrono::high_resolution_clock::now();
            render_frame(cam, 2, anim_t, outpath, prev_gi_ptr, prev_cam_ptr, gi_b);
            auto t1 = std::chrono::high_resolution_clock::now();
            double secs = std::chrono::duration<double>(t1 - t0).count();
            printf("video frame %d/%d  → %s  (%.1fs)\n", fi, total, outpath, secs);

            // Swap buffers: curr becomes prev for next frame
            std::swap(gi_a, gi_b);
            prev_gi_ptr = gi_a;
            prev_cam_val = cam;
            prev_cam_ptr = &prev_cam_val;
        }

        delete gi_a;
        delete gi_b;
    }
    else
    {
        int W = hd ? 1920 : hq ? 800
                        : fast ? 200
                               : 400;
        int H = hd ? 1080 : hq ? 450
                        : fast ? 113
                               : 225;
        int S = hd ? 8 : hq ? 24
                     : fast ? 4
                            : 12;
        const char *outpath = hd ? "cpurt15_hd.ppm" : hq ? "cpurt15_hq.ppm"
                                                         : "cpurt15_output.ppm";
        if (hd)
            printf("Full HD 1920x1080 — this takes ~15-20 min on 2 cores\n");
        Camera cam = build_camera(0.f, W, H);
        auto t0 = std::chrono::high_resolution_clock::now();
        render_frame(cam, S, 0.f, outpath, nullptr, nullptr, nullptr);
        auto t1 = std::chrono::high_resolution_clock::now();
        double secs = std::chrono::duration<double>(t1 - t0).count();
        printf("%.1fs  %dx%d  %d samples  → %s\n", secs, W, H, S, outpath);
    }

    for (int gi = 0; gi < g_ngrid; gi++)
        if (g_grids[gi].valid)
            delete[] g_grids[gi].data;
}
#endif
