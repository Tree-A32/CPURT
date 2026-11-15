/*
 * game.cpp — Real-time SDF raymarching FPS (cpurt8 scene, GPU edition)
 *
 * This is the interactive counterpart to cpurt8.cpp. The SDF scene
 * (terrain FBM, water lake, arch/torus/sphere objects, sky) is ported
 * to a GLSL fragment shader and raymarched on the GPU at ~60fps.
 *
 * cpurt8 → game mapping:
 *   eval_scene_avx8()   →  sdScene() in GLSL
 *   sky_color()         →  skyColor() in GLSL
 *   soft_shadow_avx8()  →  softShadow() in GLSL
 *   calc_ao_avx8()      →  calcAO() in GLSL (4 steps, no GI)
 *   shade_batch_avx8()  →  main shading in mainImage()
 *
 * Build:
 *   g++ -O2 -std=c++17 -o game game.cpp $(sdl2-config --cflags --libs) -lGL -lGLEW -lm
 *
 * Controls:
 *   WASD move | Mouse look | Shift sprint | Space jump
 *   F11 fullscreen | Click capture mouse | Esc release mouse
 */

#define INTERNAL_SCALE 2   // render at 1/INTERNAL_SCALE resolution, upscale

#include <SDL2/SDL.h>
#include <GL/glew.h>
#include <GL/gl.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <algorithm>

// ─────────────────────────────────────────────────────────────────────────────
// GLSL fragment shader (inline)
// ─────────────────────────────────────────────────────────────────────────────
static const char* VERT_SRC = R"GLSL(
#version 330 core
layout(location=0) in vec2 aPos;
out vec2 vUV;
void main(){
    vUV = aPos*0.5+0.5;
    gl_Position = vec4(aPos,0,1);
}
)GLSL";

static const char* FRAG_SRC = R"GLSL(
#version 330 core
in  vec2 vUV;
out vec4 fragColor;

uniform vec3  uCamPos;
uniform vec3  uCamFwd;
uniform vec3  uCamRight;
uniform vec3  uCamUp;
uniform float uTime;
uniform vec2  uResolution;

// ── constants ────────────────────────────────────────────────────────────────
#define MAX_STEPS  200
#define MAX_DIST   300.0
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
            + fbm(xz*0.3 )*2.0
            + fbm(xz*1.2 )*0.4;
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
    // terrain
    float dT = sdTerrain(p);
    vec2 res = vec2(dT, MAT_TERRAIN);

    // water
    float dW = sdWater(p);
    if(dW < res.x) res = vec2(dW, MAT_WATER);

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
        float sha  = (matID==MAT_WATER)?1.0 : softShadow(p+n*0.08, sunDir, 0.05, 60.0, 14.0);
        float ao   = calcAO(p,n);
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
// C++ host
// ─────────────────────────────────────────────────────────────────────────────

static GLuint compileShader(GLenum type, const char* src) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);
    GLint ok; glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char buf[4096]; glGetShaderInfoLog(s, sizeof(buf), nullptr, buf);
        fprintf(stderr, "Shader error:\n%s\n", buf);
        exit(1);
    }
    return s;
}

static GLuint linkProgram(const char* vsrc, const char* fsrc) {
    GLuint vs = compileShader(GL_VERTEX_SHADER,   vsrc);
    GLuint fs = compileShader(GL_FRAGMENT_SHADER, fsrc);
    GLuint p  = glCreateProgram();
    glAttachShader(p, vs); glAttachShader(p, fs);
    glLinkProgram(p);
    GLint ok; glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) {
        char buf[1024]; glGetProgramInfoLog(p, sizeof(buf), nullptr, buf);
        fprintf(stderr, "Link error:\n%s\n", buf);
        exit(1);
    }
    glDeleteShader(vs); glDeleteShader(fs);
    return p;
}

struct Vec3 { float x,y,z; };
static Vec3 add(Vec3 a, Vec3 b){ return {a.x+b.x, a.y+b.y, a.z+b.z}; }
static Vec3 scale(Vec3 a, float s){ return {a.x*s, a.y*s, a.z*s}; }
static Vec3 norm(Vec3 a){ float l=sqrtf(a.x*a.x+a.y*a.y+a.z*a.z)+1e-9f; return {a.x/l,a.y/l,a.z/l}; }
static Vec3 cross(Vec3 a,Vec3 b){ return {a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x}; }

// Terrain height for collision (must match GLSL fbm)
static float hashf(float x, float y) {
    float px = x*127.1f + y*311.7f;
    float py = x*311.7f + y*127.1f;
    px = px - floorf(px); py = py - floorf(py);
    px += px*px + 19.3f*py; py += py*py + 19.3f*px;
    float v = px*py; v = v - floorf(v);
    return v*2.0f - 1.0f;
}
static float noiseC(float x, float z) {
    float ix=floorf(x), iz=floorf(z);
    float fx=x-ix, fz=z-iz;
    float ux=fx*fx*(3-2*fx), uz=fz*fz*(3-2*fz);
    return (hashf(ix,iz)*(1-ux)+hashf(ix+1,iz)*ux)*(1-uz)
          +(hashf(ix,iz+1)*(1-ux)+hashf(ix+1,iz+1)*ux)*uz;
}
static float fbmC(float x, float z) {
    float v=0,a=0.5f;
    float m[4]={1.6f,1.2f,-1.2f,1.6f};
    for(int i=0;i<6;i++){
        v+=a*noiseC(x,z);
        float nx=m[0]*x+m[1]*z, nz=m[2]*x+m[3]*z;
        x=nx; z=nz; a*=0.5f;
    }
    return v;
}
// Must match GLSL terrainH exactly for collision to align with visuals.
// GLSL uses: h -= smoothstep(8.,2.,r)*4.5  where smoothstep(edge0,edge1,x)
// maps x in [edge0,edge1] → [0,1] with cubic — note edge0 > edge1 so it
// goes from 0 at r=8 to 1 at r=2 (lake deepest near centre).
static float smoothstepC(float e0, float e1, float x) {
    float t = std::max(0.f, std::min(1.f, (x - e0) / (e1 - e0)));
    return t * t * (3.f - 2.f * t);
}
static float terrainH(float x, float z) {
    float h = fbmC(x*0.08f,z*0.08f)*8.0f
            + fbmC(x*0.3f, z*0.3f )*2.0f
            + fbmC(x*1.2f, z*1.2f )*0.4f;
    float r = sqrtf((x-2.f)*(x-2.f)+(z-4.f)*(z-4.f));
    h -= smoothstepC(8.f, 2.f, r) * 4.5f;   // matches GLSL smoothstep(8,2,r)
    return h;
}

int main(int /*argc*/, char** /*argv*/) {
    printf("WASD   move  |  Mouse   look  |  Shift  sprint  |  Space  jump\n");
    printf("F11    fullscreen  |  Click  capture mouse  |  Esc  release mouse\n\n");

    if (SDL_Init(SDL_INIT_VIDEO) < 0) { fprintf(stderr,"SDL_Init: %s\n",SDL_GetError()); return 1; }

    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION,3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION,3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK,SDL_GL_CONTEXT_PROFILE_CORE);
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER,1);

    int WIN_W=1280, WIN_H=720;
    SDL_Window* win = SDL_CreateWindow("cpurt8 realtime SDF",
        SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
        WIN_W, WIN_H,
        SDL_WINDOW_OPENGL|SDL_WINDOW_RESIZABLE);
    if(!win){ fprintf(stderr,"SDL_CreateWindow: %s\n",SDL_GetError()); return 1; }

    SDL_GLContext ctx = SDL_GL_CreateContext(win);
    if(!ctx){ fprintf(stderr,"SDL_GL_CreateContext: %s\n",SDL_GetError()); return 1; }
    SDL_GL_SetSwapInterval(1);

    glewExperimental = GL_TRUE;
    if(glewInit() != GLEW_OK){ fprintf(stderr,"glewInit failed\n"); return 1; }

    GLuint prog = linkProgram(VERT_SRC, FRAG_SRC);

    // fullscreen quad
    float quad[] = { -1,-1, 1,-1, -1,1, 1,1 };
    GLuint vao,vbo;
    glGenVertexArrays(1,&vao); glBindVertexArray(vao);
    glGenBuffers(1,&vbo); glBindBuffer(GL_ARRAY_BUFFER,vbo);
    glBufferData(GL_ARRAY_BUFFER,sizeof(quad),quad,GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0,2,GL_FLOAT,GL_FALSE,0,0);

    // offscreen FBO for internal resolution
    GLuint fbo=0, fboTex=0;
    int RW=WIN_W/INTERNAL_SCALE, RH=WIN_H/INTERNAL_SCALE;
    glGenFramebuffers(1,&fbo);
    glGenTextures(1,&fboTex);
    glBindTexture(GL_TEXTURE_2D,fboTex);
    glTexImage2D(GL_TEXTURE_2D,0,GL_RGB8,RW,RH,0,GL_RGB,GL_UNSIGNED_BYTE,nullptr);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
    glBindFramebuffer(GL_FRAMEBUFFER,fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,fboTex,0);
    glBindFramebuffer(GL_FRAMEBUFFER,0);

    // blit shader (simple textured quad)
    static const char* BLIT_FRAG = R"GLSL(
#version 330 core
in vec2 vUV;
out vec4 fragColor;
uniform sampler2D uTex;
void main(){ fragColor=texture(uTex,vUV); }
)GLSL";
    GLuint blitProg = linkProgram(VERT_SRC, BLIT_FRAG);

    // uniform locations
    GLint uCamPos_loc    = glGetUniformLocation(prog,"uCamPos");
    GLint uCamFwd_loc    = glGetUniformLocation(prog,"uCamFwd");
    GLint uCamRight_loc  = glGetUniformLocation(prog,"uCamRight");
    GLint uCamUp_loc     = glGetUniformLocation(prog,"uCamUp");
    GLint uTime_loc      = glGetUniformLocation(prog,"uTime");
    GLint uRes_loc       = glGetUniformLocation(prog,"uResolution");

    // camera state
    Vec3 camPos = {4.0f, 6.0f, -10.0f};
    float yaw   = 0.0f;   // radians
    float pitch = -0.3f;

    float velY  = 0.0f;
    bool  mouseCaptured = false;
    bool  fullscreen = false;

    Uint32 lastTick = SDL_GetTicks();
    Uint32 fpsTick  = lastTick;
    int    fpsFrames = 0;

    bool running = true;
    while (running) {
        Uint32 now = SDL_GetTicks();
        float dt = (now - lastTick) * 0.001f;
        if (dt > 0.05f) dt = 0.05f;
        lastTick = now;

        SDL_Event ev;
        while (SDL_PollEvent(&ev)) {
            if (ev.type == SDL_QUIT) { running=false; break; }
            if (ev.type == SDL_KEYDOWN) {
                if (ev.key.keysym.sym == SDLK_ESCAPE) {
                    if (mouseCaptured) {
                        SDL_SetRelativeMouseMode(SDL_FALSE);
                        mouseCaptured = false;
                    } else running = false;
                }
                if (ev.key.keysym.sym == SDLK_F11) {
                    fullscreen = !fullscreen;
                    SDL_SetWindowFullscreen(win, fullscreen ? SDL_WINDOW_FULLSCREEN_DESKTOP : 0);
                }
            }
            if (ev.type == SDL_MOUSEBUTTONDOWN) {
                if (!mouseCaptured) {
                    SDL_SetRelativeMouseMode(SDL_TRUE);
                    mouseCaptured = true;
                }
            }
            if (ev.type == SDL_MOUSEMOTION && mouseCaptured) {
                yaw   -= ev.motion.xrel * 0.002f;   // negate: right mouse = turn right
                pitch -= ev.motion.yrel * 0.002f;
                pitch = std::max(-1.4f, std::min(1.4f, pitch));
            }
            if (ev.type == SDL_WINDOWEVENT && ev.window.event == SDL_WINDOWEVENT_SIZE_CHANGED) {
                WIN_W = ev.window.data1;
                WIN_H = ev.window.data2;
                RW = WIN_W/INTERNAL_SCALE;
                RH = WIN_H/INTERNAL_SCALE;
                glBindTexture(GL_TEXTURE_2D,fboTex);
                glTexImage2D(GL_TEXTURE_2D,0,GL_RGB8,RW,RH,0,GL_RGB,GL_UNSIGNED_BYTE,nullptr);
            }
        }

        // build camera vectors
        Vec3 fwd  = {sinf(yaw)*cosf(pitch), sinf(pitch), cosf(yaw)*cosf(pitch)};
        Vec3 fwd_flat = {sinf(yaw), 0, cosf(yaw)};  // for movement
        Vec3 right_flat = {cosf(yaw), 0, -sinf(yaw)};
        Vec3 up   = {0,1,0};
        Vec3 right= norm(cross(fwd, up));
        up = norm(cross(right, fwd));

        // movement
        const Uint8* keys = SDL_GetKeyboardState(nullptr);
        float speed = (keys[SDL_SCANCODE_LSHIFT]||keys[SDL_SCANCODE_RSHIFT]) ? 12.f : 5.f;
        Vec3 mv = {0,0,0};
        if (keys[SDL_SCANCODE_W]) mv = add(mv, fwd_flat);
        if (keys[SDL_SCANCODE_S]) mv = add(mv, scale(fwd_flat,-1));
        if (keys[SDL_SCANCODE_A]) mv = add(mv, right_flat);
        if (keys[SDL_SCANCODE_D]) mv = add(mv, scale(right_flat,-1));
        float mlen = sqrtf(mv.x*mv.x+mv.z*mv.z);
        if (mlen > 0.001f) {
            mv.x /= mlen; mv.z /= mlen;
            camPos.x += mv.x * speed * dt;
            camPos.z += mv.z * speed * dt;
        }

        // jump + gravity
        float groundH = terrainH(camPos.x, camPos.z) + 1.75f;
        // Only enforce water surface when over the lake (r < 60 from lake centre)
        float lakeR = sqrtf((camPos.x-2.f)*(camPos.x-2.f)+(camPos.z-4.f)*(camPos.z-4.f));
        float waterH  = (lakeR < 60.f) ? (-1.2f + 1.75f) : -1e9f;
        float floorH  = std::max(groundH, waterH);
        if (keys[SDL_SCANCODE_SPACE] && camPos.y <= floorH + 0.05f) velY = 6.0f;
        velY -= 18.0f * dt;
        camPos.y += velY * dt;
        if (camPos.y < floorH) { camPos.y = floorH; velY = 0; }

        // render to FBO
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);
        glViewport(0,0,RW,RH);
        glUseProgram(prog);
        glUniform3f(uCamPos_loc,   camPos.x, camPos.y, camPos.z);
        glUniform3f(uCamFwd_loc,   fwd.x,    fwd.y,    fwd.z);
        glUniform3f(uCamRight_loc, right.x,  right.y,  right.z);
        glUniform3f(uCamUp_loc,    up.x,     up.y,     up.z);
        glUniform1f(uTime_loc,     now * 0.001f);
        glUniform2f(uRes_loc,      (float)RW, (float)RH);
        glBindVertexArray(vao);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

        // blit to window
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glViewport(0,0,WIN_W,WIN_H);
        glUseProgram(blitProg);
        glBindTexture(GL_TEXTURE_2D, fboTex);
        glUniform1i(glGetUniformLocation(blitProg,"uTex"),0);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

        SDL_GL_SwapWindow(win);

        // FPS title
        fpsFrames++;
        if (now - fpsTick >= 500) {
            float fps = fpsFrames * 1000.0f / (now - fpsTick);
            char title[128];
            snprintf(title,sizeof(title),"cpurt8 realtime SDF  |  %.0f fps  |  pos %.1f %.1f %.1f",
                     fps, camPos.x, camPos.y, camPos.z);
            SDL_SetWindowTitle(win, title);
            fpsTick = now;
            fpsFrames = 0;
        }
    }

    SDL_GL_DeleteContext(ctx);
    SDL_DestroyWindow(win);
    SDL_Quit();
    return 0;
}
