// baker.cpp — OBJ triangle mesh → signed distance field voxel grid
// Usage: ./baker input.obj output.sdf [resolution=64]
// Output .sdf binary format:
//   int32  N          (grid is N×N×N)
//   float32 bmin[3]   (world-space bounding box min, with padding)
//   float32 bmax[3]   (world-space bounding box max, with padding)
//   float32 data[N*N*N]  (signed distances, stored [z][y][x])
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>
#include <string>
#ifdef _OPENMP
#include <omp.h>
#endif

struct V3 { float x,y,z; };
inline V3 operator+(V3 a,V3 b){return{a.x+b.x,a.y+b.y,a.z+b.z};}
inline V3 operator-(V3 a,V3 b){return{a.x-b.x,a.y-b.y,a.z-b.z};}
inline V3 operator*(V3 a,float s){return{a.x*s,a.y*s,a.z*s};}
inline float dot(V3 a,V3 b){return a.x*b.x+a.y*b.y+a.z*b.z;}
inline float len2(V3 v){return dot(v,v);}
inline float len(V3 v){return sqrtf(len2(v));}
inline V3 cross(V3 a,V3 b){return{a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x};}
inline V3 norm(V3 v){float l=len(v);return l>1e-9f?v*(1.f/l):V3{0,1,0};}
inline float clampf(float v,float lo,float hi){return v<lo?lo:v>hi?hi:v;}

// ── Closest point on triangle ─────────────────────────────────────────────────
// Returns squared distance; sets 'closest' and face normal sign
float closest_pt_on_tri(V3 p, V3 a, V3 b, V3 c, V3& closest) {
    V3 ab=b-a, ac=c-a, ap=p-a;
    float d1=dot(ab,ap), d2=dot(ac,ap);
    if(d1<=0&&d2<=0){closest=a;return len2(p-a);}
    V3 bp=p-b;
    float d3=dot(ab,bp),d4=dot(ac,bp);
    if(d3>=0&&d4<=d3){closest=b;return len2(p-b);}
    V3 cp=p-c;
    float d5=dot(ab,cp),d6=dot(ac,cp);
    if(d6>=0&&d5<=d6){closest=c;return len2(p-c);}
    float vc=d1*d4-d3*d2;
    if(vc<=0&&d1>=0&&d3<=0){
        float v=d1/(d1-d3);
        closest=a+ab*v;return len2(p-closest);
    }
    float vb=d5*d2-d1*d6;
    if(vb<=0&&d2>=0&&d6<=0){
        float w=d2/(d2-d6);
        closest=a+ac*w;return len2(p-closest);
    }
    float va=d3*d6-d5*d4;
    if(va<=0&&(d4-d3)>=0&&(d5-d6)>=0){
        float w=(d4-d3)/((d4-d3)+(d5-d6));
        closest=b+(c-b)*w;return len2(p-closest);
    }
    float denom=1.f/(va+vb+vc);
    float v2=vb*denom,w=vc*denom;
    closest=a+ab*v2+ac*w;
    return len2(p-closest);
}

// ── Load OBJ ─────────────────────────────────────────────────────────────────
struct Tri { V3 a,b,c; V3 fn; }; // fn = face normal

std::vector<Tri> load_obj(const char* path) {
    FILE* f=fopen(path,"r");
    if(!f){fprintf(stderr,"Cannot open %s\n",path);exit(1);}
    std::vector<V3> verts;
    std::vector<Tri> tris;
    char line[256];
    while(fgets(line,sizeof(line),f)){
        if(line[0]=='v'&&line[1]==' '){
            float x,y,z;
            sscanf(line+2,"%f %f %f",&x,&y,&z);
            verts.push_back({x,y,z});
        } else if(line[0]=='f'&&line[1]==' '){
            // Parse face — supports v, v/vt, v/vt/vn, v//vn
            int i0,i1,i2;
            char s0[32],s1[32],s2[32];
            sscanf(line+2,"%s %s %s",s0,s1,s2);
            i0=atoi(s0); i1=atoi(s1); i2=atoi(s2);
            if(i0<0)i0+=(int)verts.size()+1;
            if(i1<0)i1+=(int)verts.size()+1;
            if(i2<0)i2+=(int)verts.size()+1;
            if(i0<1||i1<1||i2<1||i0>(int)verts.size()||i1>(int)verts.size()||i2>(int)verts.size()) continue;
            V3 a=verts[i0-1],b=verts[i1-1],c=verts[i2-1];
            V3 fn=norm(cross(b-a,c-a));
            tris.push_back({a,b,c,fn});
        }
    }
    fclose(f);
    printf("  Loaded: %zu verts, %zu triangles\n",verts.size(),tris.size());
    return tris;
}

int main(int argc,char**argv){
    if(argc<3){fprintf(stderr,"Usage: baker input.obj output.sdf [N=64]\n");return 1;}
    const char* inpath=argv[1];
    const char* outpath=argv[2];
    int N=argc>3?atoi(argv[3]):64;

    printf("Baking %s → %s  (grid %d³)\n",inpath,outpath,N);
    auto tris=load_obj(inpath);
    if(tris.empty()){fprintf(stderr,"No triangles found!\n");return 1;}

    // Bounding box
    float xmin=1e30,ymin=1e30,zmin=1e30;
    float xmax=-1e30,ymax=-1e30,zmax=-1e30;
    for(auto& t:tris){
        for(auto* v:{&t.a,&t.b,&t.c}){
            xmin=std::min(xmin,v->x);ymin=std::min(ymin,v->y);zmin=std::min(zmin,v->z);
            xmax=std::max(xmax,v->x);ymax=std::max(ymax,v->y);zmax=std::max(zmax,v->z);
        }
    }
    // Padding = 15% of longest axis
    float pad=std::max({xmax-xmin,ymax-ymin,zmax-zmin})*0.15f;
    xmin-=pad;ymin-=pad;zmin-=pad;
    xmax+=pad;ymax+=pad;zmax+=pad;
    printf("  BBox: (%.2f %.2f %.2f) → (%.2f %.2f %.2f)\n",xmin,ymin,zmin,xmax,ymax,zmax);

    std::vector<float> data(N*N*N);
    long total=(long)N*N*N;
    printf("  Computing %ldM distance samples...\n",total/1000000+1);

#pragma omp parallel for schedule(dynamic,4)
    for(int iz=0;iz<N;iz++){
        for(int iy=0;iy<N;iy++){
            for(int ix=0;ix<N;ix++){
                // World position of voxel center
                V3 p={
                    xmin+(ix+0.5f)/N*(xmax-xmin),
                    ymin+(iy+0.5f)/N*(ymax-ymin),
                    zmin+(iz+0.5f)/N*(zmax-zmin)
                };
                // Find closest triangle
                float best_d2=1e30f;
                V3 best_closest={0,0,0};
                V3 best_fn={0,1,0};
                for(auto& t:tris){
                    V3 cl;
                    float d2=closest_pt_on_tri(p,t.a,t.b,t.c,cl);
                    if(d2<best_d2){
                        best_d2=d2;
                        best_closest=cl;
                        best_fn=t.fn;
                    }
                }
                float d=sqrtf(best_d2);
                // Sign: dot of (p - closest) with face normal
                // Positive = outside (same side as normal), Negative = inside
                V3 diff=p-best_closest;
                float sgn=dot(diff,best_fn)>=0?1.f:-1.f;
                // Conservative scale: grid voxels have finite resolution
                // Multiply by 0.9 so the ray marcher never overshoots
                data[iz*N*N+iy*N+ix]=sgn*d*0.9f;
            }
        }
        if(iz%(N/4)==0) printf("  %d/%d slices\n",iz,N);
    }

    // Write .sdf file
    FILE* f=fopen(outpath,"wb");
    if(!f){fprintf(stderr,"Cannot write %s\n",outpath);return 1;}
    fwrite(&N,sizeof(int),1,f);
    float bbox[6]={xmin,ymin,zmin,xmax,ymax,zmax};
    fwrite(bbox,sizeof(float),6,f);
    fwrite(data.data(),sizeof(float),N*N*N,f);
    fclose(f);
    printf("  Saved %s  (%.1f KB)\n",outpath,(float)(sizeof(int)+6*sizeof(float)+N*N*N*sizeof(float))/1024.f);
    return 0;
}
