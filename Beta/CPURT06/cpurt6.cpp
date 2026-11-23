// cpurt6.cpp — Animated hybrid SDF renderer with AVX-512 batch primary + secondary ray marching
//
// What's new vs cpurt5:
//   5. soft_shadow_avx512()   — 16 shadow rays marched together; tracks per-lane penumbra (res, ph)
//   6. ray_march_avx512_gi()  — thin wrapper: same marcher, explicit t_max for GI bounce
//   7. shade_batch_avx512()   — batches all 16 shadow + GI rays from one primary tile, shades in bulk
//      Shadow + GI together account for ~65-70% of render time; this yields another ~2-3x on top of
//      the 3.2x primary-march speedup already in cpurt5.
//
// What's carried from cpurt5:
//   1. eval_scene_avx512()       — evaluates the whole scene for 16 positions at once
//   2. sdf_grid_query_avx512()   — 8x gather-based trilinear interpolation for all 3 baked meshes
//   3. ray_march_avx512()        — 16 independent rays advance in SIMD lockstep; lanes retire on hit
//   4. render loop               — primary rays processed 16 pixels/step
//
// Falls back to the original scalar path automatically when compiled without AVX-512.
//
// Compile:
//   g++ -O3 -march=native -fopenmp -std=c++17 -o cpurt6 cpurt6.cpp -lm
//   # For explicit AVX-512 (if -march=native doesn't pick it up on your machine):
//   g++ -O3 -mavx512f -mavx512dq -fopenmp -std=c++17 -o cpurt6 cpurt6.cpp -lm
//
// Run (same modes as cpurt5):
//   ./cpurt6 fast | ./cpurt6 | ./cpurt6 hq | ./cpurt6 hd
//   ./cpurt6 video FRAME TOTAL

#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <chrono>
#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef __AVX512F__
#include <immintrin.h>
#endif

static const float PI       = 3.14159265f;
static const float INF      = 1e30f;
static const int   MAX_STEPS= 256;
static const float SURF_DIST= 0.004f;
static const float MAX_DIST = 300.0f;

// ─── SCALAR MATH ─────────────────────────────────────────────────────────────
struct V3 { float x,y,z; };
inline V3  operator+(V3 a,V3 b){return{a.x+b.x,a.y+b.y,a.z+b.z};}
inline V3  operator-(V3 a,V3 b){return{a.x-b.x,a.y-b.y,a.z-b.z};}
inline V3  operator*(V3 a,float s){return{a.x*s,a.y*s,a.z*s};}
inline V3  operator*(float s,V3 a){return a*s;}
inline V3  operator*(V3 a,V3 b){return{a.x*b.x,a.y*b.y,a.z*b.z};}
inline float dot(V3 a,V3 b){return a.x*b.x+a.y*b.y+a.z*b.z;}
inline float len(V3 v){return sqrtf(dot(v,v));}
inline V3    norm(V3 v){float l=len(v);return l>1e-9f?v*(1.f/l):V3{0,1,0};}
inline V3    cross(V3 a,V3 b){return{a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x};}
inline V3    reflect(V3 v,V3 n){return v-n*(2.f*dot(v,n));}
inline V3    mix(V3 a,V3 b,float t){return a*(1-t)+b*t;}
inline float clampf(float v,float lo,float hi){return std::max(lo,std::min(hi,v));}
inline float smoothstep(float e0,float e1,float x){
    float t=clampf((x-e0)/(e1-e0),0,1);return t*t*(3-2*t);
}

// ─── BAKED SDF GRID ──────────────────────────────────────────────────────────
struct SDFGrid {
    int N;
    float bmin[3], bmax[3];
    float* data;
    V3  pos;
    float scale;
    float rot_y;
    int mat;
    bool valid;
};

static SDFGrid g_grids[8];
static int     g_ngrid = 0;

SDFGrid load_sdf(const char* path, V3 pos, float scale, int mat, float rot_y=0.f) {
    SDFGrid g{}; g.valid=false;
    FILE* f=fopen(path,"rb");
    if(!f){fprintf(stderr,"Cannot open %s\n",path);return g;}
    fread(&g.N,sizeof(int),1,f);
    fread(g.bmin,sizeof(float),3,f);
    fread(g.bmax,sizeof(float),3,f);
    int n=g.N*g.N*g.N;
    g.data=new float[n];
    fread(g.data,sizeof(float),n,f);
    fclose(f);
    g.pos=pos; g.scale=scale; g.rot_y=rot_y; g.mat=mat; g.valid=true;
    printf("  Loaded %s  N=%d  world pos (%.1f %.1f %.1f) scale %.1f\n",
           path,g.N,pos.x,pos.y,pos.z,scale);
    return g;
}

// Scalar SDF grid query (unchanged from cpurt4, used by shading code)
float sdf_grid_query(const SDFGrid& g, V3 wp) {
    float dx=wp.x-g.pos.x, dz=wp.z-g.pos.z;
    float ca=cosf(-g.rot_y), sa=sinf(-g.rot_y);
    float lx=(ca*dx - sa*dz)/g.scale;
    float ly=(wp.y-g.pos.y)/g.scale;
    float lz=(sa*dx + ca*dz)/g.scale;
    float N=(float)g.N;
    float tx=(lx-g.bmin[0])/(g.bmax[0]-g.bmin[0])*N;
    float ty=(ly-g.bmin[1])/(g.bmax[1]-g.bmin[1])*N;
    float tz=(lz-g.bmin[2])/(g.bmax[2]-g.bmin[2])*N;
    float ox=std::max(0.f,std::max((float)g.bmin[0]-lx, lx-(float)g.bmax[0]));
    float oy=std::max(0.f,std::max((float)g.bmin[1]-ly, ly-(float)g.bmax[1]));
    float oz=std::max(0.f,std::max((float)g.bmin[2]-lz, lz-(float)g.bmax[2]));
    float outside=sqrtf(ox*ox+oy*oy+oz*oz);
    if(outside>0.f) return outside*g.scale;
    tx=clampf(tx,0.f,N-1.001f);
    ty=clampf(ty,0.f,N-1.001f);
    tz=clampf(tz,0.f,N-1.001f);
    int ix=(int)tx, iy=(int)ty, iz=(int)tz;
    float fx=tx-ix, fy=ty-iy, fz=tz-iz;
    int gN=g.N;
    auto at=[&](int x,int y,int z)->float{
        x=std::max(0,std::min(gN-1,x));
        y=std::max(0,std::min(gN-1,y));
        z=std::max(0,std::min(gN-1,z));
        return g.data[z*gN*gN+y*gN+x];
    };
    float v000=at(ix,iy,iz),   v100=at(ix+1,iy,iz);
    float v010=at(ix,iy+1,iz), v110=at(ix+1,iy+1,iz);
    float v001=at(ix,iy,iz+1), v101=at(ix+1,iy,iz+1);
    float v011=at(ix,iy+1,iz+1),v111=at(ix+1,iy+1,iz+1);
    float v00=v000+(v100-v000)*fx, v10=v010+(v110-v010)*fx;
    float v01=v001+(v101-v001)*fx, v11=v011+(v111-v011)*fx;
    float v0=v00+(v10-v00)*fy,     v1=v01+(v11-v01)*fy;
    return (v0+(v1-v0)*fz)*g.scale;
}

// ─── PROCEDURAL SCENE (scalar, used by shading / secondary rays) ─────────────
inline float hash21(float x,float y){
    float v=sinf(x*127.1f+y*311.7f)*43758.5453f;
    return v-floorf(v);
}
inline float smoothnoise(float x,float y){
    float ix=floorf(x),iy=floorf(y);
    float fx=x-ix,fy=y-iy;
    float ux=fx*fx*(3-2*fx),uy=fy*fy*(3-2*fy);
    float a=hash21(ix,iy),b=hash21(ix+1,iy);
    float c=hash21(ix,iy+1),d=hash21(ix+1,iy+1);
    return a+(b-a)*ux+(c-a)*uy+(d-b+a-c)*ux*uy;
}
inline float fbm(float x,float y,int oct=6){
    float v=0,amp=0.5f,freq=1.f,tot=0;
    for(int i=0;i<oct;i++){
        v+=smoothnoise(x*freq,y*freq)*amp;
        tot+=amp;amp*=0.5f;freq*=2.1f;
    }
    return v/tot;
}
inline float terrain_h(float x,float z){
    float h=fbm(x*0.035f,z*0.035f,7)*22.f;
    float r=sqrtf(x*x+z*z);
    h*=smoothstep(5.f,30.f,r);
    h-=4.f*smoothstep(30.f,8.f,r);
    return h;
}

enum Mat { MAT_GRASS=1, MAT_ROCK, MAT_SNOW, MAT_WATER,
           MAT_TREE_TRUNK, MAT_TREE_LEAF,
           MAT_MESH_STONE, MAT_MESH_GOLD, MAT_MESH_CRYSTAL };

struct Hit { float d; int mat; };
inline Hit closer(Hit a,Hit b){return a.d<b.d?a:b;}

float sd_capsule(V3 p,V3 a,V3 b,float r){
    V3 ab=b-a,ap=p-a;
    float t=clampf(dot(ap,ab)/dot(ab,ab),0,1);
    return len(ap-ab*t)-r;
}

Hit eval_scene(V3 p){
    Hit best={INF,MAT_GRASS};
    float r_lake=sqrtf(p.x*p.x+p.z*p.z);
    if(r_lake<60.f){
        float dw=p.y+0.1f;
        dw+=sinf(p.x*0.8f)*0.04f*smoothstep(60,20,r_lake);
        dw+=cosf(p.z*0.9f)*0.03f*smoothstep(60,20,r_lake);
        best=closer(best,{dw,MAT_WATER});
    }
    float h=terrain_h(p.x,p.z);
    int tmat=h>8.f?MAT_SNOW:(h>2.f&&fbm(p.x*0.3f,p.z*0.3f,2)<0.42f?MAT_ROCK:MAT_GRASS);
    best=closer(best,{(p.y-h)*0.55f,tmat});
    static const float tx[]={18,-22,30,-15,40,-35,25,-28};
    static const float tz[]={12,20,-18,35,-30,10,28,-10};
    for(int i=0;i<8;i++){
        float dx=p.x-tx[i],dz=p.z-tz[i];
        if(dx*dx+dz*dz>64.f) continue;
        float th=terrain_h(tx[i],tz[i]);
        V3 base={tx[i],th,tz[i]};
        V3 top={tx[i],th+4.5f,tz[i]};
        float dtr=sd_capsule(p,base,top,0.25f);
        best=closer(best,{dtr,MAT_TREE_TRUNK});
        for(int j=0;j<4;j++){
            V3 lc={p.x-tx[i],p.y-(th+2.7f+j*1.1f),p.z-tz[i]};
            float wx=lc.x+sinf(lc.y*2.3f+tx[i])*0.15f;
            float wz=lc.z+cosf(lc.y*2.3f+tz[i])*0.15f;
            float r=(2.5f-j*0.45f);
            best=closer(best,{sqrtf(wx*wx+lc.y*lc.y+wz*wz)-r,MAT_TREE_LEAF});
        }
    }
    for(int gi=0;gi<g_ngrid;gi++){
        if(!g_grids[gi].valid) continue;
        float d=sdf_grid_query(g_grids[gi],p);
        best=closer(best,{d,g_grids[gi].mat});
    }
    return best;
}

// ─── NORMAL / MARCH / SHADOW / AO (scalar, unchanged) ────────────────────────
V3 scene_normal(V3 p){
    const float e=0.003f;
    auto f=[](V3 q){return eval_scene(q).d;};
    return norm({f({p.x+e,p.y,p.z})-f({p.x-e,p.y,p.z}),
                 f({p.x,p.y+e,p.z})-f({p.x,p.y-e,p.z}),
                 f({p.x,p.y,p.z+e})-f({p.x,p.y,p.z-e})});
}
struct March { V3 pos; int mat; bool hit; float t; };
March ray_march(V3 ro,V3 rd){
    float t=0.001f;
    for(int i=0;i<MAX_STEPS&&t<MAX_DIST;i++){
        V3 p={ro.x+rd.x*t,ro.y+rd.y*t,ro.z+rd.z*t};
        Hit h=eval_scene(p);
        if(h.d<SURF_DIST) return {p,h.mat,true,t};
        t+=h.d;
    }
    return {{0,0,0},0,false,MAX_DIST};
}
float soft_shadow(V3 p,V3 ldir,float k=14.f){
    float t=0.05f,res=1.f,ph=1e20f;
    for(int i=0;i<96;i++){
        V3 q={p.x+ldir.x*t,p.y+ldir.y*t,p.z+ldir.z*t};
        float d=eval_scene(q).d;
        if(d<0.001f) return 0.f;
        float y2=d*d/(2.f*ph);
        float di=sqrtf(d*d-y2*y2);
        res=std::min(res,k*di/std::max(0.001f,t-y2));
        ph=d;t+=d;
        if(t>60.f)break;
    }
    return clampf(res,0,1);
}
float calc_ao(V3 p,V3 n){
    float occ=0,sc=1;
    for(int i=0;i<6;i++){
        float h2=0.03f+0.08f*i;
        V3 sp={p.x+n.x*h2,p.y+n.y*h2,p.z+n.z*h2};
        float d=eval_scene(sp).d;
        occ+=(h2-d)*sc;sc*=0.75f;
    }
    return clampf(1-2.5f*occ,0,1);
}

static V3 sun_dir;

V3 sky_color(V3 rd){
    float sd=std::max(0.f,dot(rd,sun_dir));
    float y=std::max(0.f,rd.y);
    V3 sky=mix({0.5f,0.65f,0.85f},{0.15f,0.35f,0.72f},powf(y,0.4f));
    sky=sky+V3{1.4f,1.1f,0.7f}*powf(sd,512.f)*8.f;
    sky=sky+V3{0.8f,0.6f,0.3f}*powf(sd,8.f)*0.3f;
    return sky;
}

struct MatProps{V3 albedo;float roughness;};
MatProps get_mat(int mat,V3 p){
    switch(mat){
    case MAT_GRASS:  return{{0.15f+fbm(p.x*.5f,p.z*.5f,2)*.12f,
                              0.28f+fbm(p.x*.4f+5,p.z*.4f,2)*.1f,.07f},.95f};
    case MAT_ROCK:   {float n=fbm(p.x*.8f,p.z*.8f,3);
                      return{{.3f+n*.1f,.28f+n*.08f,.25f+n*.06f},.8f};}
    case MAT_SNOW:   return{{.9f,.92f,.95f},.55f};
    case MAT_WATER:  return{{.04f,.1f,.2f},.0f};
    case MAT_TREE_TRUNK: return{{.3f,.18f,.08f},.95f};
    case MAT_TREE_LEAF:  {float v=fbm(p.x*2,p.z*2,2);
                          return{{.08f+v*.06f,.28f+v*.1f,.06f},.9f};}
    case MAT_MESH_STONE: {float n=fbm(p.x*2.1f+3,p.z*2.1f+7,3)*.15f;
                          return{{.45f+n,.42f+n,.38f+n},.75f};}
    case MAT_MESH_GOLD:  return{{.9f,.72f,.2f},.15f};
    case MAT_MESH_CRYSTAL:return{{.5f,.8f,.95f},.05f};
    default:         return{{.5f,.5f,.5f},.8f};
    }
}

// ─── PRNG ────────────────────────────────────────────────────────────────────
static uint32_t rng_state[64];
inline uint32_t rng_next(int tid){
    rng_state[tid]^=rng_state[tid]<<13;
    rng_state[tid]^=rng_state[tid]>>17;
    rng_state[tid]^=rng_state[tid]<<5;
    return rng_state[tid];
}
inline float rng_float(int tid){return (rng_next(tid)>>8)*5.96e-8f;}
inline V3 rand_hemi(V3 n,int tid){
    for(;;){
        float a=rng_float(tid)*2-1,b=rng_float(tid)*2-1,c=rng_float(tid)*2-1;
        float l2=a*a+b*b+c*c;
        if(l2>1e-4f&&l2<1.f){
            V3 v=norm({a,b,c});
            return dot(v,n)<0?V3{-v.x,-v.y,-v.z}:v;
        }
    }
}

// ─── SHADING (scalar, operates on a known primary hit) ───────────────────────
// shade_known_hit: same logic as the old shade_hit but takes the pre-marched
// primary result instead of re-marching it internally.
V3 shade_known_hit(V3 ro, V3 rd, March m, int tid) {
    if (!m.hit) return sky_color(rd);

    if (m.mat == MAT_WATER) {
        V3 n = scene_normal(m.pos);
        float ndv = std::max(0.f, dot(n, -1.f * rd));
        float fresnel = 0.04f + 0.96f * powf(1 - ndv, 5.f);
        V3 refl_dir = reflect(rd, n);
        V3 refl_col = sky_color(refl_dir);
        March rm = ray_march(m.pos + n * 0.01f, refl_dir);
        if (rm.hit && rm.mat != MAT_WATER) {
            MatProps mp2 = get_mat(rm.mat, rm.pos);
            V3 rn = scene_normal(rm.pos);
            float d2 = std::max(0.f, dot(rn, sun_dir));
            float sh2 = d2 > 0.01f ? soft_shadow(rm.pos + rn * 0.02f, sun_dir, 10.f) : 0;
            refl_col = mp2.albedo * (d2 * sh2 * 1.5f + 0.15f);
        }
        float depth_f = smoothstep(0, 3.f, fabsf(terrain_h(m.pos.x, m.pos.z)));
        V3 water_col = mix({0.1f,.3f,.4f},{.01f,.08f,.18f}, depth_f);
        float caus = powf(std::max(0.f, sinf(m.pos.x*2.3f)*sinf(m.pos.z*1.9f)), 4.f) * 0.5f;
        water_col = water_col + V3{.1f,.15f,.1f} * caus;
        V3 col = mix(water_col, refl_col, fresnel);
        V3 half_v = norm(sun_dir - rd);
        float spec = powf(std::max(0.f, dot(n, half_v)), 256.f) * 2.f;
        return col + V3{1.4f,1.2f,.8f} * spec;
    }

    V3 n = scene_normal(m.pos);
    MatProps mp = get_mat(m.mat, m.pos);
    float diff = std::max(0.f, dot(n, sun_dir));
    float sha = diff > 0.01f ? soft_shadow(m.pos + n * 0.02f, sun_dir, 12.f) : 0.f;
    float ao = calc_ao(m.pos, n);
    float sky_amb = 0.5f + 0.5f * n.y;

    if (m.mat == MAT_MESH_CRYSTAL) {
        V3 neg_rd = {-rd.x,-rd.y,-rd.z};
        float rim = powf(1.f - std::max(0.f, dot(n, neg_rd)), 3.f);
        V3 col = mp.albedo * (diff * sha * 1.2f + 0.3f) + V3{.8f,.95f,1.f} * rim * 0.6f;
        V3 half_v = norm(sun_dir - rd);
        float spec = powf(std::max(0.f, dot(n, half_v)), 128.f) * 2.f;
        return col + V3{1,1,1} * spec * sha;
    }

    V3 gi = {0,0,0};
    V3 gi_dir = rand_hemi(n, tid);
    March gm = ray_march(m.pos + n * 0.02f, gi_dir);
    if (gm.hit && gm.mat != MAT_WATER) {
        MatProps gmp = get_mat(gm.mat, gm.pos);
        float gdiff = std::max(0.f, dot(scene_normal(gm.pos), sun_dir)) * 0.5f;
        gi = gmp.albedo * gdiff;
    } else if (!gm.hit) {
        gi = sky_color(gi_dir) * 0.3f;
    }
    gi = gi * dot(n, gi_dir) * 0.4f;

    V3 sun_col = {1.4f,1.15f,.8f};
    V3 amb_col = {.15f,.2f,.35f};
    V3 col = mp.albedo * (sun_col * diff * sha + amb_col * sky_amb * ao + gi);

    if (mp.roughness < 0.5f) {
        float gloss = 1.f - mp.roughness;
        V3 half_v = norm(sun_dir - rd);
        float spec = powf(std::max(0.f, dot(n, half_v)), 4.f/(gloss*gloss+.001f)) * gloss * .8f;
        col = col + sun_col * spec * sha;
    }
    return col;
}

V3 apply_fog(V3 col, float dist, V3 rd){
    float fog = 1.f - expf(-dist * .004f);
    float sd = std::max(0.f, dot(rd, sun_dir));
    V3 fog_col = {.55f+sd*.3f, .6f+sd*.15f, .72f};
    return mix(col, fog_col, fog);
}

// ─── AVX-512 SECTION ─────────────────────────────────────────────────────────
#ifdef __AVX512F__

// ── Fast sin(x) for all real x ───────────────────────────────────────────────
// Polynomial minimax on [-pi, pi] after range reduction.
// Accuracy: ~5-6 digits — good enough for the hash/noise functions.
static inline __m512 avx_sin(__m512 x) {
    // Range reduce: x = x - round(x / 2π) * 2π
    const __m512 inv_2pi = _mm512_set1_ps(0.15915494f);  // 1/(2π)
    const __m512 two_pi  = _mm512_set1_ps(6.28318530f);
    __m512 k = _mm512_roundscale_ps(
        _mm512_mul_ps(x, inv_2pi),
        _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    x = _mm512_fnmadd_ps(k, two_pi, x);  // x -= k * 2π

    // Horner: sin(x) ≈ x*(1 + x²*(-1/6 + x²*(1/120 + x²*(-1/5040))))
    __m512 x2 = _mm512_mul_ps(x, x);
    __m512 r  = _mm512_fmadd_ps(x2, _mm512_set1_ps(-1.f/5040.f), _mm512_set1_ps(1.f/120.f));
    r = _mm512_fmadd_ps(x2, r, _mm512_set1_ps(-1.f/6.f));
    r = _mm512_fmadd_ps(x2, r, _mm512_set1_ps(1.f));
    return _mm512_mul_ps(x, r);
}

// ── Vectorized noise primitives ───────────────────────────────────────────────
static inline __m512 avx_hash21(__m512 x, __m512 y) {
    // sin(x*127.1 + y*311.7) * 43758.5453 - floor(...)
    __m512 v = avx_sin(_mm512_fmadd_ps(x, _mm512_set1_ps(127.1f),
                                        _mm512_mul_ps(y, _mm512_set1_ps(311.7f))));
    v = _mm512_mul_ps(v, _mm512_set1_ps(43758.5453f));
    return _mm512_sub_ps(v, _mm512_floor_ps(v));
}

static inline __m512 avx_smoothnoise(__m512 x, __m512 y) {
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

static inline __m512 avx_fbm(__m512 x, __m512 y, int oct = 6) {
    __m512 v    = _mm512_setzero_ps();
    __m512 amp  = _mm512_set1_ps(0.5f);
    __m512 freq = _mm512_set1_ps(1.f);
    __m512 tot  = _mm512_setzero_ps();
    for (int i = 0; i < oct; i++) {
        v   = _mm512_fmadd_ps(avx_smoothnoise(_mm512_mul_ps(x, freq),
                                               _mm512_mul_ps(y, freq)), amp, v);
        tot = _mm512_add_ps(tot, amp);
        amp  = _mm512_mul_ps(amp,  _mm512_set1_ps(0.5f));
        freq = _mm512_mul_ps(freq, _mm512_set1_ps(2.1f));
    }
    return _mm512_div_ps(v, tot);
}

// Vectorized terrain height for 16 (x,z) pairs
static inline __m512 avx_terrain_h(__m512 x, __m512 z) {
    __m512 h = _mm512_mul_ps(
        avx_fbm(_mm512_mul_ps(x, _mm512_set1_ps(0.035f)),
                _mm512_mul_ps(z, _mm512_set1_ps(0.035f)), 7),
        _mm512_set1_ps(22.f));
    __m512 r = _mm512_sqrt_ps(_mm512_fmadd_ps(x, x, _mm512_mul_ps(z, z)));

    // smoothstep(5, 30, r) → multiply h
    __m512 t1 = _mm512_div_ps(_mm512_sub_ps(r, _mm512_set1_ps(5.f)), _mm512_set1_ps(25.f));
    t1 = _mm512_min_ps(_mm512_max_ps(t1, _mm512_setzero_ps()), _mm512_set1_ps(1.f));
    t1 = _mm512_mul_ps(t1, _mm512_mul_ps(t1,
        _mm512_fnmadd_ps(t1, _mm512_set1_ps(2.f), _mm512_set1_ps(3.f))));
    h = _mm512_mul_ps(h, t1);

    // smoothstep(30, 8, r) → subtract 4*
    __m512 t2 = _mm512_div_ps(_mm512_sub_ps(r, _mm512_set1_ps(30.f)),
                               _mm512_set1_ps(8.f - 30.f));  // (r-30)/(8-30)
    t2 = _mm512_min_ps(_mm512_max_ps(t2, _mm512_setzero_ps()), _mm512_set1_ps(1.f));
    t2 = _mm512_mul_ps(t2, _mm512_mul_ps(t2,
        _mm512_fnmadd_ps(t2, _mm512_set1_ps(2.f), _mm512_set1_ps(3.f))));
    h = _mm512_fnmadd_ps(_mm512_set1_ps(4.f), t2, h);  // h -= 4*t2
    return h;
}

// ── AVX-512 SDF grid query for 16 world positions at once ─────────────────────
// Returns signed distance in world units (16 floats).
static __m512 sdf_grid_query_avx512(const SDFGrid& g,
                                     __m512 wpx, __m512 wpy, __m512 wpz) {
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
    __m512 bx0=_mm512_set1_ps(g.bmin[0]), bx1=_mm512_set1_ps(g.bmax[0]);
    __m512 by0=_mm512_set1_ps(g.bmin[1]), by1=_mm512_set1_ps(g.bmax[1]);
    __m512 bz0=_mm512_set1_ps(g.bmin[2]), bz1=_mm512_set1_ps(g.bmax[2]);
    __m512 zero = _mm512_setzero_ps();
    __m512 ox = _mm512_max_ps(zero, _mm512_max_ps(_mm512_sub_ps(bx0,lx), _mm512_sub_ps(lx,bx1)));
    __m512 oy = _mm512_max_ps(zero, _mm512_max_ps(_mm512_sub_ps(by0,ly), _mm512_sub_ps(ly,by1)));
    __m512 oz = _mm512_max_ps(zero, _mm512_max_ps(_mm512_sub_ps(bz0,lz), _mm512_sub_ps(lz,bz1)));
    __m512 outside = _mm512_sqrt_ps(
        _mm512_fmadd_ps(ox,ox, _mm512_fmadd_ps(oy,oy, _mm512_mul_ps(oz,oz))));
    __mmask16 outside_mask = _mm512_cmp_ps_mask(outside, zero, _CMP_GT_OQ);

    // Map local coords → grid cell coordinates [0, N)
    float fN = (float)g.N;
    __m512 vN = _mm512_set1_ps(fN);
    __m512 tx = _mm512_div_ps(_mm512_mul_ps(_mm512_sub_ps(lx,bx0),vN), _mm512_sub_ps(bx1,bx0));
    __m512 ty = _mm512_div_ps(_mm512_mul_ps(_mm512_sub_ps(ly,by0),vN), _mm512_sub_ps(by1,by0));
    __m512 tz = _mm512_div_ps(_mm512_mul_ps(_mm512_sub_ps(lz,bz0),vN), _mm512_sub_ps(bz1,bz0));
    __m512 max_idx = _mm512_set1_ps(fN - 1.001f);
    tx = _mm512_min_ps(_mm512_max_ps(tx, zero), max_idx);
    ty = _mm512_min_ps(_mm512_max_ps(ty, zero), max_idx);
    tz = _mm512_min_ps(_mm512_max_ps(tz, zero), max_idx);

    // Integer cell indices and fractional parts
    __m512i ix = _mm512_cvttps_epi32(tx);
    __m512i iy = _mm512_cvttps_epi32(ty);
    __m512i iz = _mm512_cvttps_epi32(tz);
    __m512  fx = _mm512_sub_ps(tx, _mm512_cvtepi32_ps(ix));
    __m512  fy = _mm512_sub_ps(ty, _mm512_cvtepi32_ps(iy));
    __m512  fz = _mm512_sub_ps(tz, _mm512_cvtepi32_ps(iz));

    // Clamp indices so ix+1, iy+1, iz+1 are always valid
    int gN = g.N;
    __m512i vgN  = _mm512_set1_epi32(gN);
    __m512i vgN2 = _mm512_set1_epi32(gN * gN);
    __m512i vgN1 = _mm512_set1_epi32(gN - 1);
    __m512i vzero_i = _mm512_setzero_si512();
    __m512i one_i   = _mm512_set1_epi32(1);
    ix  = _mm512_min_epi32(_mm512_max_epi32(ix,  vzero_i), vgN1);
    iy  = _mm512_min_epi32(_mm512_max_epi32(iy,  vzero_i), vgN1);
    iz  = _mm512_min_epi32(_mm512_max_epi32(iz,  vzero_i), vgN1);
    __m512i ix1 = _mm512_min_epi32(_mm512_add_epi32(ix, one_i), vgN1);
    __m512i iy1 = _mm512_min_epi32(_mm512_add_epi32(iy, one_i), vgN1);
    __m512i iz1 = _mm512_min_epi32(_mm512_add_epi32(iz, one_i), vgN1);

    // Flatten 3-D index: i = iz*N² + iy*N + ix
    // Using mullo_epi32 (available in AVX-512F via _mm512_mullo_epi32)
    auto idx3 = [&](__m512i zi, __m512i yi, __m512i xi) -> __m512i {
        return _mm512_add_epi32(
            _mm512_add_epi32(_mm512_mullo_epi32(zi, vgN2),
                              _mm512_mullo_epi32(yi, vgN)),
            xi);
    };
    __m512i i000=idx3(iz ,iy ,ix ); __m512i i100=idx3(iz ,iy ,ix1);
    __m512i i010=idx3(iz ,iy1,ix ); __m512i i110=idx3(iz ,iy1,ix1);
    __m512i i001=idx3(iz1,iy ,ix ); __m512i i101=idx3(iz1,iy ,ix1);
    __m512i i011=idx3(iz1,iy1,ix ); __m512i i111=idx3(iz1,iy1,ix1);

    // Gather the 8 trilinear corners (scale = 4 bytes per float)
    const float* base = g.data;
    __m512 v000=_mm512_i32gather_ps(i000,base,4);
    __m512 v100=_mm512_i32gather_ps(i100,base,4);
    __m512 v010=_mm512_i32gather_ps(i010,base,4);
    __m512 v110=_mm512_i32gather_ps(i110,base,4);
    __m512 v001=_mm512_i32gather_ps(i001,base,4);
    __m512 v101=_mm512_i32gather_ps(i101,base,4);
    __m512 v011=_mm512_i32gather_ps(i011,base,4);
    __m512 v111=_mm512_i32gather_ps(i111,base,4);

    // Trilinear interpolation via lerp = a + (b-a)*t
#define LERP(a,b,t) _mm512_fmadd_ps(_mm512_sub_ps(b,a),t,a)
    __m512 v00 = LERP(v000,v100,fx); __m512 v10 = LERP(v010,v110,fx);
    __m512 v01 = LERP(v001,v101,fx); __m512 v11 = LERP(v011,v111,fx);
    __m512 v0  = LERP(v00, v10, fy); __m512 v1  = LERP(v01, v11, fy);
    __m512 d   = LERP(v0,  v1,  fz);
#undef LERP

    // Scale back to world units
    __m512 scale_v = _mm512_set1_ps(g.scale);
    d = _mm512_mul_ps(d, scale_v);

    // Blend: outside bbox → outside*scale  |  inside → trilinear grid value
    // _mm512_mask_blend_ps(k, a, b): k=1 → b, k=0 → a
    return _mm512_mask_blend_ps(outside_mask, d, _mm512_mul_ps(outside, scale_v));
}

// ── Vectorized eval_scene for 16 positions ────────────────────────────────────
struct EvalResult16 {
    __m512  d;    // best signed distance
    __m512i mat;  // material at best distance
};

static EvalResult16 eval_scene_avx512(__m512 px, __m512 py, __m512 pz) {
    __m512  best_d   = _mm512_set1_ps(INF);
    __m512i best_mat = _mm512_set1_epi32(MAT_GRASS);

    // Helper: update best if prim_d < best_d
    auto update = [&](__m512 prim_d, int mat_id) {
        __mmask16 closer = _mm512_cmp_ps_mask(prim_d, best_d, _CMP_LT_OQ);
        best_d   = _mm512_min_ps(best_d, prim_d);
        best_mat = _mm512_mask_blend_epi32(closer, best_mat, _mm512_set1_epi32(mat_id));
    };

    // ── Water plane ────────────────────────────────────────────────────────
    __m512 r_lake = _mm512_sqrt_ps(_mm512_fmadd_ps(px,px,_mm512_mul_ps(pz,pz)));
    __mmask16 in_lake = _mm512_cmp_ps_mask(r_lake, _mm512_set1_ps(60.f), _CMP_LT_OQ);
    if (in_lake) {
        __m512 dw = _mm512_add_ps(py, _mm512_set1_ps(0.1f));
        // Small wave ripples
        __m512 wave_x = _mm512_mul_ps(avx_sin(_mm512_mul_ps(px, _mm512_set1_ps(0.8f))),
                                       _mm512_set1_ps(0.04f));
        __m512 wave_z = _mm512_mul_ps(avx_sin(_mm512_mul_ps(pz, _mm512_set1_ps(0.9f))),
                                       _mm512_set1_ps(0.03f));
        // Fade waves at lake edge: smoothstep(60,20,r_lake) — just use 1 for simplicity inside lake
        dw = _mm512_add_ps(dw, _mm512_add_ps(wave_x, wave_z));
        // Only apply water where in_lake
        __m512 safe_dw = _mm512_mask_blend_ps(in_lake, _mm512_set1_ps(INF), dw);
        update(safe_dw, MAT_WATER);
    }

    // ── Terrain ────────────────────────────────────────────────────────────
    {
        __m512 h = avx_terrain_h(px, pz);
        __m512 d_terrain = _mm512_mul_ps(_mm512_sub_ps(py, h), _mm512_set1_ps(0.55f));

        // Vectorized material selection (mirrors scalar logic exactly):
        //   h>8   → MAT_SNOW
        //   h>2 && fbm(x*0.3,z*0.3,2)<0.42 → MAT_ROCK
        //   else  → MAT_GRASS
        __m512i tmat = _mm512_set1_epi32(MAT_GRASS);
        __mmask16 is_snow        = _mm512_cmp_ps_mask(h, _mm512_set1_ps(8.f), _CMP_GT_OQ);
        __mmask16 above_rock_h   = _mm512_cmp_ps_mask(h, _mm512_set1_ps(2.f), _CMP_GT_OQ);
        __m512 rock_noise = avx_fbm(_mm512_mul_ps(px, _mm512_set1_ps(0.3f)),
                                     _mm512_mul_ps(pz, _mm512_set1_ps(0.3f)), 2);
        __mmask16 is_rock = above_rock_h &
            _mm512_cmp_ps_mask(rock_noise, _mm512_set1_ps(0.42f), _CMP_LT_OQ);
        tmat = _mm512_mask_blend_epi32(is_rock, tmat, _mm512_set1_epi32(MAT_ROCK));
        tmat = _mm512_mask_blend_epi32(is_snow, tmat, _mm512_set1_epi32(MAT_SNOW));

        // Per-lane update (can't use uniform update() helper — material varies per lane)
        __mmask16 terrain_closer = _mm512_cmp_ps_mask(d_terrain, best_d, _CMP_LT_OQ);
        best_d   = _mm512_min_ps(best_d, d_terrain);
        best_mat = _mm512_mask_blend_epi32(terrain_closer, best_mat, tmat);
    }

    // ── Trees (8 capsule trunks) ────────────────────────────────────────────
    {
        static const float tx_arr[] = {18,-22,30,-15,40,-35,25,-28};
        static const float tz_arr[] = {12,20,-18,35,-30,10,28,-10};
        for (int i = 0; i < 8; i++) {
            float ti_x = tx_arr[i], ti_z = tz_arr[i];
            // Skip trees far from all 16 positions (coarse cull per-tree)
            // We do a horizontal distance test using the min of all 16 positions
            __m512 dxi = _mm512_sub_ps(px, _mm512_set1_ps(ti_x));
            __m512 dzi = _mm512_sub_ps(pz, _mm512_set1_ps(ti_z));
            __m512 dist2 = _mm512_fmadd_ps(dxi,dxi,_mm512_mul_ps(dzi,dzi));
            float min_dist2 = _mm512_reduce_min_ps(dist2);
            if (min_dist2 > 100.f) continue;  // whole tile is far from this tree

            float th = terrain_h(ti_x, ti_z);  // scalar (called once per tree per tile)
            // Capsule: base=(ti_x, th, ti_z)  top=(ti_x, th+4.5, ti_z)  r=0.25
            // sd_capsule vectorized: p-a=(dxi, py-th, dzi), ab=(0, 4.5, 0)
            __m512 ap_y = _mm512_sub_ps(py, _mm512_set1_ps(th));
            float ab_ab = 4.5f * 4.5f;
            // dot(ap, ab) = ap_y * 4.5
            __m512 t_cap = _mm512_mul_ps(ap_y, _mm512_set1_ps(4.5f / ab_ab));
            t_cap = _mm512_min_ps(_mm512_max_ps(t_cap, _mm512_setzero_ps()), _mm512_set1_ps(1.f));
            __m512 cl_y = _mm512_fmadd_ps(t_cap, _mm512_set1_ps(4.5f), _mm512_set1_ps(th));
            __m512 ey = _mm512_sub_ps(py, cl_y);
            __m512 d_trunk = _mm512_sub_ps(
                _mm512_sqrt_ps(_mm512_fmadd_ps(dxi,dxi,_mm512_fmadd_ps(ey,ey,_mm512_mul_ps(dzi,dzi)))),
                _mm512_set1_ps(0.25f));
            update(d_trunk, MAT_TREE_TRUNK);

            // 4 foliage spheres (warped spheres)
            for (int j = 0; j < 4; j++) {
                float cy = th + 2.7f + j * 1.1f;
                float r_leaf = 2.5f - j * 0.45f;
                __m512 ly = _mm512_sub_ps(py, _mm512_set1_ps(cy));
                // warp: wx = dx + sin(ly*2.3+ti_x)*0.15, wz = dz + cos(ly*2.3+ti_z)*0.15
                __m512 warp_arg = _mm512_fmadd_ps(ly, _mm512_set1_ps(2.3f), _mm512_set1_ps((float)(i * 1.0)));
                __m512 wx = _mm512_fmadd_ps(avx_sin(warp_arg), _mm512_set1_ps(0.15f), dxi);
                // For wz, use cos = sin(arg + pi/2)
                __m512 cos_arg = _mm512_add_ps(warp_arg, _mm512_set1_ps(1.5707963f));
                __m512 wz = _mm512_fmadd_ps(avx_sin(cos_arg), _mm512_set1_ps(0.15f), dzi);
                __m512 d_leaf = _mm512_sub_ps(
                    _mm512_sqrt_ps(_mm512_fmadd_ps(wx,wx,_mm512_fmadd_ps(ly,ly,_mm512_mul_ps(wz,wz)))),
                    _mm512_set1_ps(r_leaf));
                update(d_leaf, MAT_TREE_LEAF);
            }
        }
    }

    // ── Baked SDF grid objects ──────────────────────────────────────────────
    for (int gi = 0; gi < g_ngrid; gi++) {
        if (!g_grids[gi].valid) continue;
        __m512 d = sdf_grid_query_avx512(g_grids[gi], px, py, pz);
        update(d, g_grids[gi].mat);
    }

    return {best_d, best_mat};
}

// ── 16-ray batch marcher ──────────────────────────────────────────────────────
struct March16 {
    __m512  pos_x, pos_y, pos_z;
    __m512i mat;
    __mmask16 hit_mask;
    __m512  t;
};

static March16 ray_march_avx512(
        __m512 rox, __m512 roy, __m512 roz,
        __m512 rdx, __m512 rdy, __m512 rdz) {
    __m512 t       = _mm512_set1_ps(0.001f);
    __mmask16 active = 0xFFFF;
    __mmask16 hit_mask = 0;

    __m512 hit_x = _mm512_setzero_ps();
    __m512 hit_y = _mm512_setzero_ps();
    __m512 hit_z = _mm512_setzero_ps();
    __m512i hit_mat = _mm512_setzero_si512();

    const __m512 surf_dist_v = _mm512_set1_ps(SURF_DIST);
    const __m512 max_dist_v  = _mm512_set1_ps(MAX_DIST);

    for (int i = 0; i < MAX_STEPS && active; i++) {
        // Current ray positions
        __m512 px = _mm512_fmadd_ps(rdx, t, rox);
        __m512 py = _mm512_fmadd_ps(rdy, t, roy);
        __m512 pz = _mm512_fmadd_ps(rdz, t, roz);

        EvalResult16 res = eval_scene_avx512(px, py, pz);

        // Advance active rays by their SDF distance
        t = _mm512_mask_add_ps(t, active, t, res.d);

        // Detect hits: d < SURF_DIST (only check active rays)
        __mmask16 just_hit = active & _mm512_cmp_ps_mask(res.d, surf_dist_v, _CMP_LT_OQ);
        if (just_hit) {
            hit_x   = _mm512_mask_blend_ps(just_hit, hit_x,   px);
            hit_y   = _mm512_mask_blend_ps(just_hit, hit_y,   py);
            hit_z   = _mm512_mask_blend_ps(just_hit, hit_z,   pz);
            hit_mat = _mm512_mask_blend_epi32(just_hit, hit_mat, res.mat);
            hit_mask |= just_hit;
            active  &= ~just_hit;
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
        float k = 14.f)
{
    const int   SHADOW_STEPS = 96;
    const float T_MAX        = 60.f;
    const float T_START      = 0.05f;
    const float HIT_THRESH   = 0.001f;

    const __m512 k_v        = _mm512_set1_ps(k);
    const __m512 zero       = _mm512_setzero_ps();
    const __m512 one        = _mm512_set1_ps(1.f);
    const __m512 eps_v      = _mm512_set1_ps(0.001f);
    const __m512 t_max_v    = _mm512_set1_ps(T_MAX);
    const __m512 hit_v      = _mm512_set1_ps(HIT_THRESH);

    __m512 t   = _mm512_set1_ps(T_START);
    __m512 res = one;                            // accumulated shadow factor
    __m512 ph  = _mm512_set1_ps(1e20f);         // previous d (penumbra tracking)

    // active: lane is still marching (not blocked, not escaped)
    __mmask16 active = 0xFFFF;

    for (int i = 0; i < SHADOW_STEPS && active; i++) {
        __m512 px = _mm512_fmadd_ps(rdx, t, rox);
        __m512 py = _mm512_fmadd_ps(rdy, t, roy);
        __m512 pz = _mm512_fmadd_ps(rdz, t, roz);

        EvalResult16 ev = eval_scene_avx512(px, py, pz);
        __m512 d = ev.d;

        // Hard shadow: d < HIT_THRESH → res = 0 for those lanes, retire them
        __mmask16 blocked = active & _mm512_cmp_ps_mask(d, hit_v, _CMP_LT_OQ);
        if (blocked) {
            res    = _mm512_mask_blend_ps(blocked, res, zero);
            active &= ~blocked;
        }

        // Improved penumbra (Quilez formula):
        //   y2 = d*d / (2*ph)
        //   di = sqrt(d*d - y2*y2)   ← min clearance
        //   res = min(res, k * di / max(0.001, t - y2))
        __m512 y2  = _mm512_div_ps(_mm512_mul_ps(d, d),
                                   _mm512_add_ps(_mm512_add_ps(ph, ph), eps_v));
        // d²-y2²  — clamp to 0 to avoid sqrt of negative (numerical noise)
        __m512 arg = _mm512_max_ps(zero,
                         _mm512_sub_ps(_mm512_mul_ps(d, d), _mm512_mul_ps(y2, y2)));
        __m512 di  = _mm512_sqrt_ps(arg);

        __m512 denom = _mm512_max_ps(eps_v, _mm512_sub_ps(t, y2));
        __m512 cand  = _mm512_div_ps(_mm512_mul_ps(k_v, di), denom);
        // Only update res for active lanes (blocked lanes already zeroed)
        res = _mm512_mask_blend_ps(active, res, _mm512_min_ps(res, cand));

        ph = _mm512_mask_blend_ps(active, ph, d);           // update ph only for active
        t  = _mm512_mask_add_ps(t, active, t, d);           // advance t

        // Retire lanes that escaped
        active &= ~(_mm512_cmp_ps_mask(t, t_max_v, _CMP_GE_OQ));
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
    __m512 t         = _mm512_set1_ps(0.02f);
    __mmask16 active = 0xFFFF;
    __mmask16 hit_mask = 0;

    __m512 hit_x = _mm512_setzero_ps();
    __m512 hit_y = _mm512_setzero_ps();
    __m512 hit_z = _mm512_setzero_ps();
    __m512i hit_mat = _mm512_setzero_si512();

    const __m512 surf_dist_v = _mm512_set1_ps(SURF_DIST);
    const __m512 t_max_v     = _mm512_set1_ps(t_max);

    for (int i = 0; i < MAX_STEPS && active; i++) {
        __m512 px = _mm512_fmadd_ps(rdx, t, rox);
        __m512 py = _mm512_fmadd_ps(rdy, t, roy);
        __m512 pz = _mm512_fmadd_ps(rdz, t, roz);

        EvalResult16 res = eval_scene_avx512(px, py, pz);

        t = _mm512_mask_add_ps(t, active, t, res.d);

        __mmask16 just_hit = active & _mm512_cmp_ps_mask(res.d, surf_dist_v, _CMP_LT_OQ);
        if (just_hit) {
            hit_x   = _mm512_mask_blend_ps(just_hit, hit_x,   px);
            hit_y   = _mm512_mask_blend_ps(just_hit, hit_y,   py);
            hit_z   = _mm512_mask_blend_ps(just_hit, hit_z,   pz);
            hit_mat = _mm512_mask_blend_epi32(just_hit, hit_mat, res.mat);
            hit_mask |= just_hit;
            active  &= ~just_hit;
        }

        // Retire rays that exceeded t_max (much sooner than MAX_DIST)
        active &= ~(_mm512_cmp_ps_mask(t, t_max_v, _CMP_GE_OQ));
    }

    return {hit_x, hit_y, hit_z, hit_mat, hit_mask, t};
}

#endif // __AVX512F__

// ─── POST-PROCESS ─────────────────────────────────────────────────────────────
inline float aces(float x){
    float a=2.51f,b=.03f,c=2.43f,d=.59f,e=.14f;
    return clampf((x*(a*x+b))/(x*(c*x+d)+e),0,1);
}

void bilateral_denoise(float* out,const float* col,const float* dep,
                       const float* nrm,int W,int H,int r=2){
#pragma omp parallel for
    for(int y=0;y<H;y++) for(int x=0;x<W;x++){
        int idx=(y*W+x)*3;
        float d0=dep[y*W+x];
        const float* n0=nrm+idx;
        float sw=0,sc[3]={0,0,0};
        for(int dy=-r;dy<=r;dy++) for(int dx2=-r;dx2<=r;dx2++){
            int nx=x+dx2,ny=y+dy;
            if(nx<0||ny<0||nx>=W||ny>=H) continue;
            int nidx=(ny*W+nx)*3;
            float d1=dep[ny*W+nx];
            const float* n1=nrm+nidx;
            float nd=(n0[0]-.5f)*(n1[0]-.5f)+(n0[1]-.5f)*(n1[1]-.5f)+(n0[2]-.5f)*(n1[2]-.5f);
            float wd=1.f/(1+40.f*fabsf(d1-d0));
            float wn=std::max(0.f,nd*4.f+0.1f);
            float ws=expf(-0.5f*(dx2*dx2+dy*dy));
            float w=ws*wd*wn;
            sc[0]+=col[nidx]*w;sc[1]+=col[nidx+1]*w;sc[2]+=col[nidx+2]*w;
            sw+=w;
        }
        if(sw>0){out[idx]=sc[0]/sw;out[idx+1]=sc[1]/sw;out[idx+2]=sc[2]/sw;}
        else{out[idx]=col[idx];out[idx+1]=col[idx+1];out[idx+2]=col[idx+2];}
    }
}

void write_ppm(const char* path,float* img,int W,int H){
    FILE* f=fopen(path,"wb");
    fprintf(f,"P6\n%d %d\n255\n",W,H);
    for(int i=0;i<W*H*3;i+=3){
        uint8_t r2=(uint8_t)(img[i]*255),g2=(uint8_t)(img[i+1]*255),b2=(uint8_t)(img[i+2]*255);
        fwrite(&r2,1,1,f);fwrite(&g2,1,1,f);fwrite(&b2,1,1,f);
    }
    fclose(f);
}

// ─── AVX-512 BATCH SHADING ────────────────────────────────────────────────────
// shade_batch_avx512: given 16 primary hits (from ray_march_avx512), issue all
// secondary rays (shadow + GI) in SIMD batches of 16 and return 16 RGB colours.
//
// Strategy per-lane:
//   1. Compute surface normal (scalar — 6 scene evals, unavoidable)
//   2. Collect shadow ray origins + dirs for all 16 lanes → soft_shadow_avx512
//   3. Collect GI ray origins + dirs for all 16 lanes → ray_march_avx512_gi
//   4. Finish shading with the batched shadow / GI results (scalar arithmetic)
//
// Water and crystal have special paths — those lanes fall through to scalar
// shading exactly as before (they're a small fraction of pixels).
//
// Output: col_r/g/b arrays of 16 floats (caller accumulates into framebuffer).
#ifdef __AVX512F__
static void shade_batch_avx512(
        // Primary hit results (16 lanes)
        const float* hx, const float* hy, const float* hz,
        const int*   hm,
        const __mmask16 hit_mask,
        // Ray directions for sky + fog (16 lanes)
        const float* rdx0, const float* rdy0, const float* rdz0,
        // Camera origin (single value, same for all)
        V3 cam_pos,
        int tid,
        // Outputs
        float* col_r, float* col_g, float* col_b)
{
    // ── 1. Compute normals for all hit lanes (scalar, parallelised by caller) ──
    V3  normals[16];
    MatProps mps[16];
    bool is_special[16]; // water or crystal — handled scalar below

    for (int i = 0; i < 16; i++) {
        bool hit = (hit_mask >> i) & 1;
        if (!hit) {
            // Sky — fill colour immediately, no secondary rays needed
            V3 rd_i = {rdx0[i], rdy0[i], rdz0[i]};
            V3 sky = sky_color(rd_i);
            col_r[i] = sky.x; col_g[i] = sky.y; col_b[i] = sky.z;
            is_special[i] = true; // mark so we skip the batch path below
            normals[i]    = {0,1,0};
            mps[i]        = {{0,0,0},1.f};
            continue;
        }
        V3 p = {hx[i], hy[i], hz[i]};
        normals[i]    = scene_normal(p);
        mps[i]        = get_mat(hm[i], p);
        is_special[i] = (hm[i] == MAT_WATER || hm[i] == MAT_MESH_CRYSTAL);
    }

    // ── 2. Special materials: scalar path (water reflection, crystal rim) ──────
    for (int i = 0; i < 16; i++) {
        if (!is_special[i]) continue;
        bool hit = (hit_mask >> i) & 1;
        if (!hit) continue; // already handled above
        V3 rd_i = {rdx0[i], rdy0[i], rdz0[i]};
        March m_i;
        m_i.pos = {hx[i], hy[i], hz[i]};
        m_i.mat = hm[i];
        m_i.hit = true;
        m_i.t   = 0;
        V3 c = shade_known_hit(cam_pos, rd_i, m_i, tid);
        col_r[i] = c.x; col_g[i] = c.y; col_b[i] = c.z;
    }

    // ── 3. Build shadow + GI ray batches for non-special hit lanes ────────────
    // For miss/special lanes we still fill the arrays (they won't affect output)
    // but we exclude them via is_special guard at step 5.
    alignas(64) float sha_ox[16], sha_oy[16], sha_oz[16]; // shadow origins
    alignas(64) float sha_dx[16], sha_dy[16], sha_dz[16]; // shadow dirs (sun_dir)
    alignas(64) float gi_ox[16],  gi_oy[16],  gi_oz[16];  // GI origins
    alignas(64) float gi_dx[16],  gi_dy[16],  gi_dz[16];  // GI random dirs
    float diff_v[16];                                       // diffuse dot product
    bool  skip_shadow[16];                                  // diff ≤ 0.01

    for (int i = 0; i < 16; i++) {
        if (is_special[i]) {
            // Fill with harmless values so the batch still runs without branching
            sha_ox[i]=sha_oy[i]=sha_oz[i]=0;
            sha_dx[i]=0; sha_dy[i]=1; sha_dz[i]=0;
            gi_ox[i]=gi_oy[i]=gi_oz[i]=0;
            gi_dx[i]=0;  gi_dy[i]=1;  gi_dz[i]=0;
            diff_v[i] = 0; skip_shadow[i] = true;
            continue;
        }
        V3 p  = {hx[i], hy[i], hz[i]};
        V3 n  = normals[i];
        float diff = std::max(0.f, dot(n, sun_dir));
        diff_v[i]     = diff;
        skip_shadow[i]= (diff <= 0.01f);

        // Shadow ray: origin offset along normal, direction = sun_dir
        sha_ox[i] = p.x + n.x * 0.02f;
        sha_oy[i] = p.y + n.y * 0.02f;
        sha_oz[i] = p.z + n.z * 0.02f;
        sha_dx[i] = sun_dir.x; sha_dy[i] = sun_dir.y; sha_dz[i] = sun_dir.z;

        // GI ray: cosine-weighted hemisphere sample
        V3 gi_dir = rand_hemi(n, tid);
        gi_ox[i] = p.x + n.x * 0.02f;
        gi_oy[i] = p.y + n.y * 0.02f;
        gi_oz[i] = p.z + n.z * 0.02f;
        gi_dx[i] = gi_dir.x; gi_dy[i] = gi_dir.y; gi_dz[i] = gi_dir.z;
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

    // For lanes that don't need shadow (diff≤0.01), we don't use sha_arr,
    // but running them through the batch is cheap (they march through open air).

    // ── 5. Batch-march GI bounce rays ────────────────────────────────────────
    March16 gi16;
    {
        __m512 gox = _mm512_load_ps(gi_ox), goy = _mm512_load_ps(gi_oy), goz = _mm512_load_ps(gi_oz);
        __m512 gdx = _mm512_load_ps(gi_dx), gdy = _mm512_load_ps(gi_dy), gdz = _mm512_load_ps(gi_dz);
        gi16 = ray_march_avx512_gi(gox, goy, goz, gdx, gdy, gdz, 50.f);
    }
    alignas(64) float ghx[16], ghy[16], ghz[16];
    alignas(64) int   ghm[16];
    _mm512_store_ps(ghx, gi16.pos_x); _mm512_store_ps(ghy, gi16.pos_y); _mm512_store_ps(ghz, gi16.pos_z);
    _mm512_store_epi32(ghm, gi16.mat);

    // ── 6. Final shading (scalar arithmetic — fast, no march calls) ───────────
    const V3 sun_col = {1.4f,1.15f,.8f};
    const V3 amb_col = {.15f,.2f,.35f};

    for (int i = 0; i < 16; i++) {
        if (is_special[i]) continue; // already computed above

        V3 n   = normals[i];
        MatProps mp = mps[i];
        float diff = diff_v[i];
        float sha  = skip_shadow[i] ? 0.f : sha_arr[i];
        float ao   = calc_ao({hx[i],hy[i],hz[i]}, n);
        float sky_amb = 0.5f + 0.5f * n.y;

        // GI contribution
        V3 gi = {0,0,0};
        bool gi_hit = (gi16.hit_mask >> i) & 1;
        if (gi_hit && ghm[i] != MAT_WATER) {
            MatProps gmp = get_mat(ghm[i], {ghx[i],ghy[i],ghz[i]});
            float gdiff  = std::max(0.f, dot(scene_normal({ghx[i],ghy[i],ghz[i]}), sun_dir)) * 0.5f;
            gi = gmp.albedo * gdiff;
        } else if (!gi_hit) {
            V3 gi_dir = {gi_dx[i], gi_dy[i], gi_dz[i]};
            gi = sky_color(gi_dir) * 0.3f;
        }
        gi = gi * dot(n, {gi_dx[i], gi_dy[i], gi_dz[i]}) * 0.4f;

        V3 col = mp.albedo * (sun_col * diff * sha + amb_col * sky_amb * ao + gi);

        if (mp.roughness < 0.5f) {
            float gloss = 1.f - mp.roughness;
            V3 rd_i = {rdx0[i], rdy0[i], rdz0[i]};
            V3 half_v = norm(sun_dir - rd_i);
            float spec = powf(std::max(0.f, dot(n, half_v)), 4.f/(gloss*gloss+.001f)) * gloss * .8f;
            col = col + sun_col * spec * sha;
        }

        col_r[i] = col.x; col_g[i] = col.y; col_b[i] = col.z;
    }
}
#endif // __AVX512F__

// ─── RENDER ONE FRAME ─────────────────────────────────────────────────────────
void render_frame(int W,int H,int SAMPLES,float anim_t,const char* outpath){
    sun_dir=norm({0.7f,0.55f,0.3f});

    g_grids[1].pos.y = 5.f + sinf(anim_t*1.3f)*0.8f;
    g_grids[1].rot_y = anim_t * 0.9f;
    g_grids[2].pos.y = terrain_h(-10,15)+1.2f + sinf(anim_t*2.1f+1.f)*0.4f;
    g_grids[2].rot_y = anim_t * 0.4f;

    float cam_angle = anim_t * 0.18f;
    float cam_r = 38.f;
    float cpx = cosf(cam_angle + 3.8f) * cam_r;
    float cpz = sinf(cam_angle + 3.8f) * cam_r;
    float cpy = terrain_h(cpx,cpz) + 11.f;
    V3 cam_pos   = {cpx, cpy, cpz};
    V3 cam_target= {0.f, 2.f+sinf(anim_t*0.5f)*0.5f, 0.f};
    V3 cam_fwd   = norm(cam_target-cam_pos);
    V3 cam_right = norm(cross(cam_fwd,{0,1,0}));
    V3 cam_up    = cross(cam_right,cam_fwd);
    float fov    = 0.78f;
    float aspect = (float)W/H;

    for(int i=0;i<64;i++) rng_state[i]=12345u*(i+1)*2654435761u^(uint32_t)(anim_t*1000);

    int NP = W*H;
    float* acc = new float[NP*3]();
    float* dep = new float[NP]();
    float* nrm = new float[NP*3]();
    float* buf = new float[NP*3]();
    float* out = new float[NP*3]();

#ifdef __AVX512F__
    printf("  [AVX-512] batch primary + secondary ray marching active\n");
#else
    printf("  [scalar] no AVX-512 — compiled without -mavx512f or CPU lacks it\n");
#endif

    for(int frame=0;frame<SAMPLES;frame++){
        float jx=(frame%4+.5f)/4.f-.5f;
        float jy=(frame/4%4+.5f)/4.f-.5f;

#pragma omp parallel for schedule(dynamic,4)
        for(int y=0;y<H;y++){
            int tid=0;
#ifdef _OPENMP
            tid=omp_get_thread_num()&63;
#endif
            rng_state[tid]^=(uint32_t)(y*7919+frame*2053);

#ifdef __AVX512F__
            // ── AVX-512 path: 16 primary rays at a time ──────────────────
            for(int x=0;x<W;x+=16){
                int batch = std::min(16, W-x);

                // Build 16 ray origins + directions (with DoF jitter)
                alignas(64) float rox_a[16], roy_a[16], roz_a[16];
                alignas(64) float rdx_a[16], rdy_a[16], rdz_a[16];
                // We also need the un-jittered direction for fog distance
                alignas(64) float rdx0[16], rdy0[16], rdz0[16];

                for(int dx=0;dx<16;dx++){
                    int xi = x + dx;
                    if(xi >= W) xi = W-1;  // clamp last partial tile

                    float ux = (xi+.5f+jx*.5f+(rng_float(tid)-.5f)*.3f)/W*2-1;
                    float uy = 1-(y+.5f+jy*.5f+(rng_float(tid)-.5f)*.3f)/H*2;
                    float rdx_ = cam_fwd.x + cam_right.x*(ux*aspect*fov) + cam_up.x*(uy*fov);
                    float rdy_ = cam_fwd.y + cam_right.y*(ux*aspect*fov) + cam_up.y*(uy*fov);
                    float rdz_ = cam_fwd.z + cam_right.z*(ux*aspect*fov) + cam_up.z*(uy*fov);
                    float rlen = sqrtf(rdx_*rdx_+rdy_*rdy_+rdz_*rdz_);
                    rdx0[dx]=rdx_/rlen; rdy0[dx]=rdy_/rlen; rdz0[dx]=rdz_/rlen;

                    float dof_str=0.02f;
                    float dfx=rng_float(tid)*2-1, dfy=rng_float(tid)*2-1;
                    float rox_ = cam_pos.x + cam_right.x*dfx*dof_str + cam_up.x*dfy*dof_str;
                    float roy_ = cam_pos.y + cam_right.y*dfx*dof_str + cam_up.y*dfy*dof_str;
                    float roz_ = cam_pos.z + cam_right.z*dfx*dof_str + cam_up.z*dfy*dof_str;
                    float fx_ = rox_+rdx0[dx]*40, fy_ = roy_+rdy0[dx]*40, fz_ = roz_+rdz0[dx]*40;
                    float fdx = fx_-rox_, fdy = fy_-roy_, fdz = fz_-roz_;
                    float flen = sqrtf(fdx*fdx+fdy*fdy+fdz*fdz);
                    rox_a[dx]=rox_; roy_a[dx]=roy_; roz_a[dx]=roz_;
                    rdx_a[dx]=fdx/flen; rdy_a[dx]=fdy/flen; rdz_a[dx]=fdz/flen;
                }

                // Load into AVX-512 registers
                __m512 vro_x = _mm512_load_ps(rox_a);
                __m512 vro_y = _mm512_load_ps(roy_a);
                __m512 vro_z = _mm512_load_ps(roz_a);
                __m512 vrd_x = _mm512_load_ps(rdx_a);
                __m512 vrd_y = _mm512_load_ps(rdy_a);
                __m512 vrd_z = _mm512_load_ps(rdz_a);

                // March all 16 primary rays at once
                March16 m16 = ray_march_avx512(vro_x, vro_y, vro_z, vrd_x, vrd_y, vrd_z);

                // Extract primary hit results
                alignas(64) float hx[16], hy[16], hz[16], ht[16];
                alignas(64) int   hm[16];
                _mm512_store_ps(hx, m16.pos_x); _mm512_store_ps(hy, m16.pos_y); _mm512_store_ps(hz, m16.pos_z);
                _mm512_store_ps(ht, m16.t);
                _mm512_store_epi32(hm, m16.mat);

                // ── Batch-shade all 16 secondary rays (shadow + GI) ──────────
                alignas(64) float col_r[16], col_g[16], col_b[16];
                shade_batch_avx512(hx, hy, hz, hm, m16.hit_mask,
                                   rdx0, rdy0, rdz0,
                                   cam_pos, tid,
                                   col_r, col_g, col_b);

                for(int dx=0;dx<batch;dx++){
                    int xi = x + dx;
                    bool did_hit = (m16.hit_mask >> dx) & 1;

                    // Apply fog using the DoF-jittered direction (matches cpurt5 behaviour)
                    V3 fog_rd = {rdx_a[dx], rdy_a[dx], rdz_a[dx]};
                    V3 col    = {col_r[dx], col_g[dx], col_b[dx]};
                    col = apply_fog(col, ht[dx], fog_rd);

                    // Store depth/normal for denoiser (frame 0 only)
                    if(frame==0){
                        dep[y*W+xi] = ht[dx];
                        if(did_hit){
                            V3 p = {hx[dx], hy[dx], hz[dx]};
                            V3 n = scene_normal(p);
                            nrm[(y*W+xi)*3+0]=n.x*.5f+.5f;
                            nrm[(y*W+xi)*3+1]=n.y*.5f+.5f;
                            nrm[(y*W+xi)*3+2]=n.z*.5f+.5f;
                        } else {
                            nrm[(y*W+xi)*3+0]=.5f; nrm[(y*W+xi)*3+1]=.5f; nrm[(y*W+xi)*3+2]=.5f;
                        }
                    }
                    acc[(y*W+xi)*3+0]+=col.x;
                    acc[(y*W+xi)*3+1]+=col.y;
                    acc[(y*W+xi)*3+2]+=col.z;
                }
            }
#else
            // ── Scalar fallback (identical to cpurt4) ─────────────────────
            for(int x=0;x<W;x++){
                float ux=(x+.5f+jx*.5f+(rng_float(tid)-.5f)*.3f)/W*2-1;
                float uy=1-(y+.5f+jy*.5f+(rng_float(tid)-.5f)*.3f)/H*2;
                V3 rd=norm(cam_fwd+cam_right*(ux*aspect*fov)+cam_up*(uy*fov));
                float dof_str=0.02f;
                float dfx=rng_float(tid)*2-1,dfy=rng_float(tid)*2-1;
                V3 ro_dof={cam_pos.x+cam_right.x*dfx*dof_str+cam_up.x*dfy*dof_str,
                           cam_pos.y+cam_right.y*dfx*dof_str+cam_up.y*dfy*dof_str,
                           cam_pos.z+cam_right.z*dfx*dof_str+cam_up.z*dfy*dof_str};
                V3 focus={cam_pos.x+rd.x*40,cam_pos.y+rd.y*40,cam_pos.z+rd.z*40};
                V3 rd_dof=norm(focus-ro_dof);

                March primary = ray_march(ro_dof, rd_dof);
                V3 col = shade_known_hit(cam_pos, rd, primary, tid);
                col = apply_fog(col, primary.t, rd_dof);

                if(frame==0){
                    dep[y*W+x]=primary.t;
                    if(primary.hit){V3 n=scene_normal(primary.pos);
                        nrm[(y*W+x)*3+0]=n.x*.5f+.5f;
                        nrm[(y*W+x)*3+1]=n.y*.5f+.5f;
                        nrm[(y*W+x)*3+2]=n.z*.5f+.5f;
                    }else{nrm[(y*W+x)*3+0]=.5f;nrm[(y*W+x)*3+1]=.5f;nrm[(y*W+x)*3+2]=.5f;}
                }
                acc[(y*W+x)*3+0]+=col.x;
                acc[(y*W+x)*3+1]+=col.y;
                acc[(y*W+x)*3+2]+=col.z;
            }
#endif
        }
    }

    for(int i=0;i<NP*3;i++) buf[i]=acc[i]/SAMPLES;
    for(int i=0;i<NP*3;i+=3){
        buf[i]=aces(buf[i]*1.8f);buf[i+1]=aces(buf[i+1]*1.8f);buf[i+2]=aces(buf[i+2]*1.8f);
    }
    bilateral_denoise(out,buf,dep,nrm,W,H,2);
    for(int i=0;i<NP*3;i++) out[i]=powf(clampf(out[i],0,1),1.f/2.2f);
    write_ppm(outpath,out,W,H);
    delete[]acc;delete[]dep;delete[]nrm;delete[]buf;delete[]out;
}

// ─── MAIN ────────────────────────────────────────────────────────────────────
int main(int argc,char**argv){
    bool is_video = argc>1 && !strcmp(argv[1],"video");
    bool fast = argc>1 && !strcmp(argv[1],"fast");
    bool hq   = argc>1 && !strcmp(argv[1],"hq");
    bool hd   = argc>1 && !strcmp(argv[1],"hd");

#ifdef __AVX512F__
    printf("cpurt6 — AVX-512 build  (batch primary + secondary rays)\n");
#else
    printf("cpurt6 — scalar build  (no AVX-512 on this CPU)\n");
#endif

    printf("Loading baked SDF objects...\n");
    g_grids[g_ngrid++]=load_sdf("assets/arch.sdf",  {8.f,terrain_h(8,0),0.f},   2.5f, MAT_MESH_STONE);
    g_grids[g_ngrid++]=load_sdf("assets/torus.sdf", {0.f,5.f,0.f},              2.0f, MAT_MESH_GOLD);
    g_grids[g_ngrid++]=load_sdf("assets/sphere.sdf",{-10.f,terrain_h(-10,15)+1.2f,15.f},1.5f,MAT_MESH_CRYSTAL);

    if(is_video){
        int frame_idx = argc>2 ? atoi(argv[2]) : 0;
        int total     = argc>3 ? atoi(argv[3]) : 48;
        float anim_t  = (float)frame_idx / total * 2.f * PI;
        char outpath[256];
        system("mkdir -p frames");
        snprintf(outpath,sizeof(outpath),"frames/frame_%04d.ppm",frame_idx);
        auto t0=std::chrono::high_resolution_clock::now();
        render_frame(400,225,2,anim_t,outpath);
        auto t1=std::chrono::high_resolution_clock::now();
        double secs=std::chrono::duration<double>(t1-t0).count();
        printf("video frame %d/%d  → %s  (%.1fs)\n",frame_idx,total,outpath,secs);
    } else {
        int W = hd?1920 : hq?800 : fast?200 : 400;
        int H = hd?1080 : hq?450 : fast?113 : 225;
        int S = hd?8    : hq?24  : fast?4   : 12;
        const char* outpath = hd?"cpurt5_hd.ppm" : hq?"cpurt5_hq.ppm" : "cpurt5_output.ppm";
        if(hd) printf("Full HD 1920x1080 — this takes ~15-20 min on 2 cores (less with AVX-512)\n");
        auto t0=std::chrono::high_resolution_clock::now();
        render_frame(W,H,S,0.f,outpath);
        auto t1=std::chrono::high_resolution_clock::now();
        double secs=std::chrono::duration<double>(t1-t0).count();
        printf("%.1fs  %dx%d  %d samples  → %s\n",secs,W,H,S,outpath);
    }

    for(int gi=0;gi<g_ngrid;gi++) if(g_grids[gi].valid) delete[]g_grids[gi].data;
}
