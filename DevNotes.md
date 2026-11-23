# CPURT — CPU SDF Raytracer

A single-file, progressively-optimized CPU raytracer built on signed-distance-field (SDF)
raymarching. The goal has been to push real-world render throughput as far as possible using
AVX-512 and AVX2 SIMD, OpenMP multithreading, and algorithmic improvements — targeting ~3×
speedup over the cpurt8 baseline on both ISAs.

The companion `game.cpp` is a real-time GPU/CPU SDF viewer using the same scene. CPURT20
builds it with SDL2 + GLEW / OpenGL 3.3 core: GLSL shaders for GPU mode, and
`cpurt_render_buffer()` from `cpurt20.cpp` for CPU mode. `build.sh` supports both native
Linux and MSYS2 MINGW64 and standardises on the AVX2/FMA path.

---

## Files

Sources live under each version's `source code/` folder; prebuilt Windows binaries and DLLs
stay in the version root.

| File | Purpose |
|------|---------|
| `Release/CPURT15/source code/cpurt15.cpp` | Released renderer (stable baseline) |
| `Release/CPURT20/source code/cpurt20.cpp` | Current release: AVX2 CPU renderer with final-pixel temporal reuse for 1280×720 @ 1/2 scale |
| `Release/CPURT20/source code/game.cpp` | Current realtime viewer; defaults to 1280×720, 1/2 render scale, 8 CPU threads |
| `Release/CPURT20/source code/build.sh` | Current build script — native Linux or MSYS2 MINGW64, fixed AVX2/FMA mode |
| `Release/CPURT20/source code/bench20.cpp` | Headless `cpurt_render_buffer()` benchmark (`bench20 W H FRAMES [move]`) |
| `Beta/CPURT16/source code/cpurt16.cpp` | Beta renderer: AVX2 water reflections, GI oct 2, ray coherence sorting |
| `Beta/CPURT17/source code/cpurt17.cpp` | Previous beta renderer; carries cpurt16's optimizations plus cheaper/temporally cached water reflections |
| `Beta/CPURT17/source code/game.cpp` | Previous beta realtime viewer — SDL2/GLEW OpenGL 3.3 core, GPU GLSL or CPU path via `cpurt_render_buffer()` |
| `Beta/CPURT17/source code/build.sh` | Previous beta build script — native Linux or MSYS2 MINGW64, auto-detects AVX-512 vs AVX2 |
| `Beta/CPURT17/source code/baker.cpp` | Bakes an OBJ mesh into a `.sdf` voxel grid file (used to make `assets/*.sdf`) |
| `Beta/CPURT17/assets/*.sdf` | Baked SDF grids: arch (8, terrain_h(8,0), 0) scale 2.5; torus (0, 5, 0) scale 2.0 with animated rotation; sphere (-10, terrain_h(-10,15)+1.2, 15) scale 1.5 |
| `Beta/CPURT17/source code/game_win32.cpp` | Legacy standalone Win32/WGL viewer, kept for reference; not used by `build.sh` |

The `.sdf` files must be present in the working directory at runtime (for CPU mode). If
missing, those objects are silently skipped. Run the built binaries from `Release/CPURT20`,
or copy them there from `source code/`.

---

## Building

Current release (from the repository root):

```bash
cd "Release/CPURT20/source code"
./build.sh
```

This produces `cpurt20`, `game`, `bench20`, and `baker` on Linux, or the corresponding `.exe`
files under MSYS2 MINGW64. On Windows, keep `SDL2.dll` and `glew32.dll` next to `game.exe`;
`build.sh` adds `-lopengl32` automatically and statically links the OpenMP/runtime libs.

Dependencies:

```bash
# Debian/Ubuntu
sudo apt install build-essential pkg-config libsdl2-dev libglew-dev

# MSYS2 MINGW64
pacman -S mingw-w64-x86_64-gcc mingw-w64-x86_64-pkgconf \
         mingw-w64-x86_64-SDL2 mingw-w64-x86_64-glew
```

CPURT20 always builds the AVX2/FMA path; every AVX-512 CPU also supports AVX2, and the
8-wide batched path carries all current optimisations. Manual equivalents:

```bash
# CPURT20 CPU renderer — AVX2 (same as build.sh):
g++ -O3 -mavx2 -mfma -fopenmp -std=c++17 -D__AVX2__ \
    -o cpurt20 cpurt20.cpp -lm

# Game, Linux:
g++ -O3 -mavx2 -mfma -fopenmp -std=c++17 -D__AVX2__ -DNO_MAIN \
    -o game game.cpp cpurt20.cpp \
    $(pkg-config --cflags --libs sdl2 glew) -lm

# Game, MSYS2 MINGW64:
g++ -O3 -mavx2 -mfma -fopenmp -std=c++17 -D__AVX2__ -DNO_MAIN \
    -static-libgcc -static-libstdc++ \
    -o game.exe game.cpp cpurt20.cpp \
    $(pkg-config --cflags --libs sdl2 glew) -lopengl32 -lm
```

`-DNO_MAIN` suppresses `cpurt20.cpp`'s own `main()` so `game.cpp` can provide its own.
For older versions, use the same commands with the version's own source filenames.

---

## Running the renderer

Run from the version root so `assets/*.sdf` is found:

```bash
cd "Release/CPURT20"
"./source code/cpurt20" fast        # 200×113, 4 samples  — quick preview
"./source code/cpurt20"             # 400×225, 12 samples — default still
"./source code/cpurt20" hq          # 800×450, 24 samples — high quality
"./source code/cpurt20" hd          # 1920×1080, 8 samples — full HD
"./source code/cpurt20" video N     # all N frames (animated sequence, with temporal reprojection)
"./source code/cpurt20" video F N   # single frame F out of N (silent warm-up frame before it)
```

CPURT20 still writes `cpurt15_output.ppm` by default (legacy filename). Convert to PNG/MP4
with ImageMagick or ffmpeg:

```bash
# Still → PNG
convert cpurt15_output.ppm output.png

# Video sequence → MP4
mkdir -p frames
for i in $(seq 0 47); do "./source code/cpurt20" video $i 48; done
ffmpeg -framerate 24 -i frames/frame_%04d.ppm \
  -vf scale=1280:720:flags=lanczos -c:v libx264 -crf 18 -pix_fmt yuv420p out.mp4
```

---

## CPURT20 realtime CPU path

CPURT20 keeps CPURT17's scene, march limits and shading unchanged. The realtime bridge
(`cpurt_render_buffer()`) uses forward temporal reprojection: each previous visible pixel
reconstructs its world-space hit point from previous depth and is projected through the current
camera. A per-pixel atomic claim keeps the nearest candidate. Pixels with no valid candidate
(disocclusions, animated materials, large view changes) are freshly shaded; valid pixels skip
the primary march and all shadow/AO/GI/reflection work. Temporally reused lanes are compacted
before the 8-wide march so they do not occupy SIMD slots. When the camera is unchanged, history
maps 1:1 and the projection pass is skipped entirely.

This replaced the original gather-style cache lookup, which guessed the current hit from stale
same-pixel depth. That could accept a previous final colour across a moving silhouette and was
the source of the reported trailing smearing. Motion guards remain: reprojected pixels may not
move more than 8 screen pixels, view directions must stay within a small angular delta, and
high-contrast 3×3 final-colour neighbourhoods are recomputed. Water and animated SDF meshes are
never reused.

Measured on Ryzen 7 5700X (8 threads, AVX2) at 640×360 internal (1280×720 @ 1/2 scale):

| Workload | CPURT17 | CPURT20 (8 threads) | CPURT20 (12 threads) |
|----------|---------|---------------------|----------------------|
| Moving camera | 151.3 ms (6.6 fps) | 13.80 ms (72.5 fps) | 11.10 ms (90.1 fps) |
| Static camera | 73.5 ms (13.6 fps) | 18.27 ms (54.7 fps) | 13.99 ms (71.5 fps) |

Quality was compared against CPURT17 frames at 640×360. With reuse disabled the MAE is 2.34
(moving) and 6.46 (static); after a 3×3 blur those drop to ~0.85 and ~2.3, confirming the base
difference is stochastic shadow/GI noise rather than scene structure. Against a CPURT20 no-reuse
frame, the final forward-reuse path has MAE 4.29 (normal motion) and 2.66 (fast 0.03 rad/frame
pan); channels differing by more than 8 are 12.67% and 6.87% respectively. Residual difference
is stochastic shadow/GI noise from one-frame history, not geometry dragging. Very fast turns
intentionally reject most reuse, so frame rate drops during those moments; normal motion stays
above target. GI/shadow/AO history reuse was tested but disabled because it moved stochastic
highlights too much; final-pixel reuse alone preserves the image much better.

### Debug switches

`bench20` prints temporal-reuse counts. Environment variables (for experiments only):

- `CPURT_NO_REUSE=1` — disable final-pixel reuse.
- `CPURT_NO_GI_REUSE=1` — disable GI/shadow/AO history (already off by default in the bridge).
- `CPURT_REUSE_COLOR_TOL` — 3×3 final-colour spread in 8-bit units (default 16).
- `CPURT_REUSE_VIEW_TOL` — maximum view-direction change (default 0.004).
- `CPURT_REUSE_MAX_MOVE` / `CPURT_REUSE_SKY_MAX_MOVE` — maximum reprojection distance in pixels
  for surfaces / sky (default 8 each).
- `CPURT_BAKED=1` / `CPURT_BOUNDS=1` — enable optional baked-height / height-bound experiments.
  Both are disabled by default because they measurably change terrain edges.

---

## Scene

The scene is a procedural outdoor landscape rendered entirely with SDFs — no polygons.

**Terrain** — fractal Brownian motion (FBM) heightfield. `terrain_h(x, z)` returns the
surface height; the SDF is `(p.y - terrain_h(p.xz)) / 1.6` (the 1.6 guard prevents
overshoot on steep slopes). The FBM uses 7 octaves for primary/normal rays and 3 octaves for
shadow/GI/AO rays (a ~57% saving with no visible quality loss).

**Lake** — a smoothstepped bowl carved into the terrain around world-space (2, 4), radius ~8.
Water surface is an animated ripple SDF (`p.y + 1.2 + ripple`). Water gets special shading:
refraction approximation, Fresnel, caustic shimmer. The ripple displacement uses
`avx_sin_small` (3-term polynomial, see cpurt12 notes).

**Trees** — a small procedural forest (positions stored in `g_trees[]`). Each tree is a
capsule SDF (trunk) + sphere SDF (leaf canopy). Trees far from the current tile of rays are
culled with a coarse per-tree distance check. Foliage warp also uses `avx_sin_small`.

**Baked SDF grid objects** — arch, torus, sphere — loaded from `.sdf` files at startup.
Stored as 3D float arrays; trilinearly interpolated at query time using SIMD gather
instructions. World-space bounding-sphere pre-rejection skips the gather for tiles far from
each object (see BVH section below). Next-grid data is software-prefetched before each
gather to hide the ~100-cycle gather latency.

**Materials** (enum `Mat`):
- `MAT_GRASS`, `MAT_ROCK`, `MAT_SNOW` — terrain variants, chosen by height + FBM texture
- `MAT_WATER`
- `MAT_TREE_TRUNK`, `MAT_TREE_LEAF`
- `MAT_MESH_STONE`, `MAT_MESH_GOLD`, `MAT_MESH_CRYSTAL` — the three baked objects

**Lighting** — single directional sun at `normalize(0.6, 0.5, -0.4)`. Per-pixel:
diffuse · shadow · AO + ambient + one-bounce GI + specular (metals/crystal). Atmospheric sky
(Rayleigh + Mie scattering), exponential fog.

**Key constants (CPU offline renderer):**
- `MAX_STEPS = 192` — primary ray march iteration cap
- `SHADOW_STEPS = 64` — shadow ray iteration cap
- `SURF_DIST = 0.004` — hit threshold
- `MAX_DIST = 300.0` — miss threshold

The realtime GPU shader in `game.cpp` uses tighter constants: `MAX_STEPS = 128`,
`MAX_DIST = 150.0`, 48-step soft shadows with a minimum step clamp, and 4-step AO. The scene
is otherwise the same, so the GPU image remains close to the CPU renderer while running much
faster.

---

## Rendering pipeline

Each frame is processed in 32×4-pixel unrolled tiles (AVX-512, two 16-wide calls per loop iteration) or 16×4-pixel unrolled tiles (AVX2, two 8-wide calls per loop iteration).

```
render_frame()
  └─ for each 4-row strip (AVX-512 and AVX2):
       for each 16- (or 8-) pixel column tile:
         ray_march_avx512()          primary march, terrain_oct=7
           └─ eval_scene_avx512()    SDF evaluation (terrain + water + trees + grids)
         shade_batch_avx512()
           ├─ scene_normal_avx512()  central-difference normals (6 eval calls → 16 normals)
           │    └─ normals forwarded to denoiser nrm buffer (no redundant scene_normal call)
           ├─ calc_ao_avx512()       ambient occlusion (6 eval calls, oct=4)
           ├─ temporal reprojection  reuse prev-frame GI (REPROJ_ALPHA=0.25 new + 0.75 cached)
           ├─ rand_hemi_avx512()     cosine-weighted hemisphere direction for GI bounce
           ├─ soft_shadow_avx512()   64-step shadow march, oct=4
           └─ ray_march_avx512_gi()  GI bounce march, t_max=50, oct=4
       bilateral_denoise()           edge-aware spatial denoiser on the final image
```

The cache (`struct GICache`) stores per-pixel GI, shadow, AO, normals, reflection colour,
final gamma-encoded colour, depth and material. The standalone renderer still reprojects GI
(75% cached, 25% new ray). CPURT20's realtime bridge uses the same buffers for final-pixel
reprojection instead; GI history is deliberately not reused there because it visibly moved
stochastic highlights.

---

## SIMD architecture

The file contains three independent implementations that compile together and are selected
at runtime by `#ifdef __AVX512F__` / `#ifdef __AVX2__`:

### AVX-512 path (16-wide)
- **`eval_scene_avx512(px, py, pz, terrain_oct=7)`** — evaluates the full scene SDF for 16
  positions simultaneously. Returns `{__m512 d, __m512i mat}`.
- **`ray_march_avx512()`** — primary marcher, 192 steps max.
- **`ray_march_avx512_gi()`** — GI bounce marcher, 192 steps max, t_max=50, oct=4.
- **`soft_shadow_avx512()`** — 64-step shadow march, oct=4.
- **`scene_normal_avx512()`** — central-difference normals, 6 eval calls → 16 normals.
- **`calc_ao_avx512()`** — 6-step AO, oct=4.
- **`shade_batch_avx512()`** — full shading pipeline for a 16-pixel tile. Accepts optional
  `nx_out/ny_out/nz_out` to forward computed normals to the denoiser buffer.

### AVX2 path (8-wide)
Identical logic to AVX-512 but using 8-wide `__m256` / `__m256i`. Key differences:
- Uses `_mm256_rsqrt_ps` (11-bit) instead of `_mm512_rsqrt14_ps` (14-bit).
- Uses `_mm256_i32gather_ps` for SDF grid trilinear interpolation.
- Shadow and GI are also 8-wide; `shade_batch_avx8()` processes one 8-pixel tile.

### Scalar fallback
Used for pixels that don't fit a full 16/8-wide tile, and for some special materials (water
reflections). Functions: `eval_scene()`, `ray_march()`, `soft_shadow()`, `calc_ao()`,
`scene_normal()`.

---

## Optimization history

### cpurt1–4 — baseline scalar
Single-threaded scalar SDF raymarcher. Established the scene, materials, baked SDF grid
objects, and bilateral denoiser.

### cpurt5 — first SIMD (primary march)
AVX-512 and AVX2 primary ray march (`ray_march_avx512`, `ray_march_avx8`). Shadow, AO, GI,
normals, and shading remained scalar. Gave the first meaningful speedup.

### cpurt6 — full SIMD march
Shadow and GI bounce marches also vectorized. `shade_batch_avx512` / `shade_batch_avx8`
introduced. GI cache / temporal reprojection added.

### cpurt7 — AVX2 parity
Full AVX2 8-wide path matching the AVX-512 feature set. Scalar fallback tightened.

### cpurt8 — baseline for speedup measurement (~1×)
Cleaned up, all paths present. This is the reference against which speedups are measured.

### cpurt9 — vectorized normals, AO, materials (~1.49× over cpurt8)
- `scene_normal_avx512/8` — 6 vectorized eval calls per tile instead of 96 scalar calls.
- `calc_ao_avx512/8` — 6 vectorized eval calls per tile instead of 96 scalar calls.
- Material lookup vectorized (`get_mat_avx512/8`).
- GI secondary-hit normals vectorized.

### cpurt10 — rsqrt + FBM oct reduction + precomputed inv_tot
Key wins:
- **FBM octaves 7→4 for secondary rays** (shadow/GI/AO use `terrain_oct=4`). Saves ~43% of
  terrain eval cost for ~60–70% of all eval calls.
- **rsqrt14 + 1 Newton-Raphson step** replaces `sqrt + div` for all SIMD normalizations
  (~2× faster per normalization).
- **Precomputed `inv_tot_lut[]`** in `avx_fbm` / `avx8_fbm` — eliminates `tot`
  accumulation and the final SIMD division entirely.
- rsqrt applied to lake radius, tree capsule, leaf sphere, bbox outside distances.

### cpurt11 — integer hash + BVH + step count reductions
- **Integer hash replaces `avx_sin`-based `avx_hash21/avx8_hash21`** (~6× fewer
  instructions per hash). Strategy: `cvttps_epi32` → LCG mix (1664525 · x + 22695477 · y +
  1013904223) → xorshift scramble → mantissa-bit trick to convert to `[0,1)`. Every FBM
  octave makes 4 hash calls, so this directly accelerates every terrain eval.
- **BVH sphere pre-rejection for SDF grids**: before calling the full trilinear gather,
  compute the minimum distance from the tile's 16/8 ray positions to each grid's bounding
  sphere. If farther than the current best distance, skip the grid entirely.
- **`SHADOW_STEPS` 96 → 64** (~33% fewer shadow iterations).
- **`MAX_STEPS` 256 → 192** (~25% fewer primary iterations).

### cpurt12 — avx_sin_small + 2D tiling + gather prefetch + denoiser normal reuse
- **`avx_sin_small()`** — 3-term polynomial replacing the 4-term `avx_sin` for the two
  remaining aesthetic uses: water ripple displacement (`px*0.8`, `pz*0.9`) and foliage warp
  (`ly*2.3 + offset`). Drops the `-x^6/5040` term; same range reduction, one fewer FMA per
  call. Max error <5×10⁻⁴ — imperceptible in ripples or warp. `avx_sin` is kept for
  `rand_hemi_avx8` disk sampling where accuracy still matters.
- **2D tiling (16×4) in AVX-512 render_frame** — the OpenMP loop now iterates over 4-row
  strips (`schedule(dynamic,1)`) instead of single scanlines. Within each strip, the
  existing 16-wide march loop runs per row as before. Adjacent rows sample nearly identical
  `(x*0.035, z*0.035)` FBM inputs → better L1/L2 cache reuse in `avx_smoothnoise` /
  `avx_fbm`. AVX2 and scalar paths are unchanged (still per-scanline).
- **SDF grid gather prefetch** — before `sdf_grid_query_avx512/8` for grid `gi`, issues
  `_mm_prefetch(T1)` for the first 4 cache lines (256 bytes) of grid `gi+1`'s data array,
  plus a `_MM_HINT_T2` hint at the array midpoint. The `i32gather` instruction has ~100-cycle
  latency on current microarchitectures; the prefetch hides most of it behind the current
  grid's trilinear interpolation math. No overhead on single-grid scenes (the inner loop
  simply finds no `gi+1`).
- **Denoiser normal reuse** — `shade_batch_avx512` now accepts optional `nx_out/ny_out/nz_out`
  output arrays (matching `shade_batch_avx8`). The normals already computed for shading are
  forwarded directly to the denoiser `nrm` buffer. This eliminates the per-hit-pixel scalar
  `scene_normal()` call that was firing in `render_frame` after `shade_batch` returned (~6
  `eval_scene` calls per hit pixel; ~540k avoided evals at 400×225×12spp, 50% hit rate).

### cpurt13 — FBM 4→3 octaves for secondary rays + AVX2 2D tiling
- **FBM octaves 4→3 for all secondary rays** (shadow, GI, AO) — both AVX-512 and AVX2.
  Saves ~25% of residual secondary-ray terrain cost on top of the 7→4 reduction in cpurt10,
  bringing total secondary-ray FBM savings to ~57% vs primary rays. The `inv_tot_lut`
  already carried the oct=3 entry (1.142857). Visual impact is imperceptible: shadow
  penumbra and AO differ only in fine-grain noise that the bilateral denoiser smooths anyway.
- **2D tiling (8×4) for AVX2 render_frame** — mirrors the AVX-512 change from cpurt12.
  OMP loop now iterates over 4-row strips with `schedule(dynamic,1)`; within each strip the
  8-wide x-loop runs per row as before. Same FBM cache coherence benefit as AVX-512.

### cpurt14 — 2× tile unroll + shadow early-out + water/crystal normal reuse
- **2× unrolled x-loop (AVX-512)** — the inner x-loop steps by 32 and processes two
  16-wide tiles per iteration via a captured lambda. Halves loop-overhead iterations and
  lets the out-of-order core overlap setup of tile B with the march latency of tile A.
- **Soft-shadow early-out** — after the penumbra update each step, lanes whose accumulated
  `res` has dropped to ≤0.001 are retired immediately without waiting to formally hit.
  Fires frequently in deep-shadow regions (dense geometry, low sun angle). Both AVX-512 and
  AVX2 paths.
- **Water/crystal normal reuse** — `shade_known_hit` now accepts an optional `precomp_n`
  pointer. `shade_batch_avx512/8` passes the already-computed primary normal for each water
  or crystal lane, skipping the `scene_normal()` call (6 `eval_scene` each) that was
  redundantly firing inside `shade_known_hit`. The reflection ray march inside the water path
  remains scalar per-pixel (unavoidable — it's a data-dependent secondary ray).

### cpurt15 — AVX2 2× tile unroll + AO oct 2 + batched water reflections
- **AVX2 2× unrolled x-loop** — mirrors the AVX-512 change from cpurt14. x-loop steps by
  16 and processes two 8-wide tiles per iteration via a captured lambda. Same loop-overhead
  and OOO overlap benefit on AVX2 hardware.
- **AO oct 3→2** — AO sample points are only 0.03–0.43 units from the surface; the third
  FBM octave adds detail at ~0.05 world-unit scale, finer than the sample step itself.
  Saves 1 of 3 secondary eval calls per AO step (~33% of AO terrain cost). Both AVX-512
  and AVX2 paths.
- **Batched water reflection march** — new `shade_water_batch_avx512()` collects all water
  lanes in a tile, fires one `ray_march_avx512` for all reflection rays, then uses
  `scene_normal_avx512` and `soft_shadow_avx512` on the reflected hits. Replaces up to 16
  scalar `ray_march` calls per tile with one 16-wide vectorised march. The Fresnel blend,
  water colour, caustic shimmer, and specular remain scalar per-lane (arithmetic only, not
  marches). AVX-512 only — the AVX2 water path still uses the scalar route.

### cpurt16 — AVX2 water reflections + GI oct 2 + ray coherence sorting
- **AVX2 batched water reflections** — `shade_water_batch_avx8()` mirrors the AVX-512 path.
  Water lanes in a tile are collected into one 8-wide reflection march instead of up to eight
  scalar `ray_march()` calls. Reflected-hit normals and soft shadows are also vectorised.
- **GI oct 3→2** — GI bounce marches now evaluate terrain at oct=2. A single diffuse bounce
  only needs large-scale shape, so this saves another ~33% of GI terrain cost. Both ISAs.
- **Primary-ray coherence sorting** — each tile takes three short, low-quality march steps to
  estimate hit depth, then sorts its 16/8 rays by that estimate before the main march. Rays
  likely to hit soon are grouped together, reducing active-mask divergence. Applied from
  frame > 0 so the frame-0 GI cache stays pixel-exact.

### cpurt17 — cheaper/cached reflections + native Linux realtime viewer
- **Cheaper reflection marches** — reflection `t_max` is reduced to 80, reflected-hit normals
  and reflection terrain use oct=3, and reflected shadows use 32 steps instead of 64.
- **Half-resolution reflections** — only even-indexed water lanes are marched; odd lanes
  reuse their neighbour's result.
- **Grazing-angle early-out** — reflected rays nearly parallel to the water go straight to
  sky colour without a march.
- **Temporal reflection cache** — `GICache` now stores RGB reflections. New samples blend
  30% fresh / 70% cached, matching the existing temporal GI approach.
- **Realtime viewer port** — the active `game.cpp` uses SDL2 + GLEW OpenGL 3.3 core and
  builds natively on Linux (and under MSYS2 MINGW64). CPURT16/17 shader and scene fixes are
  preserved. `game_win32.cpp` remains only as a legacy Win32/WGL reference.

---

## `cpurt_render_buffer()` — game integration interface

```cpp
void cpurt_render_buffer(
    uint8_t* rgb_out,          // output: W×H×3 bytes, row-major, top-to-bottom
    int W, int H,              // render resolution
    float cx, float cy, float cz,   // camera position
    float fx, float fy, float fz,   // camera forward vector (normalized)
    float rx, float ry, float rz,   // camera right vector (normalized)
    float ux, float uy, float uz,   // camera up vector (normalized)
    float anim_t,              // animation time in seconds (drives animated objects)
    int threads                // OpenMP thread count
);
```

The game calls this when CPU render mode is selected. The function renders into `rgb_out`,
one byte per channel, sRGB gamma-corrected. It calls `render_frame()` with `SAMPLES=1`
(single sample per pixel — fast enough for real-time at reduced resolution).

Build the game with `-DNO_MAIN` to suppress `cpurt17.cpp`'s own `main()`.

---

## Optimization roadmap status

The three items that used to be listed as "what to optimize next" have now landed:

1. **Batched water reflections for AVX2** — implemented as `shade_water_batch_avx8()` in
   CPURT16 and carried into CPURT17.
2. **GI oct 3→2** — implemented in CPURT16 for both AVX-512 and AVX2 GI marches.
3. **Primary ray coherence sorting** — implemented in CPURT16 using a short pre-march and
   per-tile lane sorting from frame 1 onward.

Possible follow-ups:
- Make the CPURT17 reflection cache camera-aware instead of pixel-index-only.
- Expose GPU-side quality knobs so the realtime shader can mirror the CPU renderer's
  reflection shortcuts when needed.

---

## Scene animation (video mode)

In video mode the camera orbits the scene at `cam_angle = anim_t * 0.18` radians. The
torus (grid 1) rotates at `rot_y = anim_t * 0.9` and bobs vertically; the crystal sphere
(grid 2) bobs at a different frequency. The water ripple is driven by `anim_t` in both the
CPU SDF and the GLSL shader.

The warm-up frame (`"./source code/cpurt17" video 0 N`) initializes the GI cache so frame 0
doesn't have an empty reprojection buffer.

---

## Denoiser

`bilateral_denoise()` is a 5×5 edge-aware bilateral filter applied after all samples are
accumulated. It uses per-pixel depth (stored from the first sample) and normal data as
guidance to avoid blurring across geometric edges. Called once per frame on the full
accumulated float buffer before converting to 8-bit output. Normal data is now sourced
directly from `shade_batch_avx512/8` with no extra scene evaluations.
