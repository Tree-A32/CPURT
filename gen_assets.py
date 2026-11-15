import math, os

def save_obj(path, verts, faces):
    with open(path,'w') as f:
        for v in verts: f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for t in faces: f.write(f"f {t[0]} {t[1]} {t[2]}\n")
    print(f"  {path}: {len(verts)} verts, {len(faces)} tris")

# ── UV Sphere (radius 1, centered at origin) ──────────────────────────────────
def make_sphere(lat=20, lon=24):
    verts, faces = [], []
    for i in range(lat+1):
        phi = math.pi * i / lat - math.pi/2
        for j in range(lon):
            theta = 2*math.pi * j / lon
            verts.append((math.cos(phi)*math.cos(theta),
                          math.sin(phi),
                          math.cos(phi)*math.sin(theta)))
    def idx(i,j): return i*(lon)+j+1
    for i in range(lat):
        for j in range(lon):
            nj=(j+1)%lon
            a,b,c,d = idx(i,j),idx(i,nj),idx(i+1,nj),idx(i+1,j)
            faces.append((a,b,c)); faces.append((a,c,d))
    return verts, faces

# ── Torus (major R=1, minor r=0.4) ───────────────────────────────────────────
def make_torus(R=1.0, r=0.38, seg_u=28, seg_v=16):
    verts, faces = [], []
    for i in range(seg_u):
        u = 2*math.pi*i/seg_u
        for j in range(seg_v):
            v = 2*math.pi*j/seg_v
            x = (R + r*math.cos(v))*math.cos(u)
            y = r*math.sin(v)
            z = (R + r*math.cos(v))*math.sin(u)
            verts.append((x, y, z))
    def idx(i,j): return i*seg_v + j + 1
    for i in range(seg_u):
        ni = (i+1)%seg_u
        for j in range(seg_v):
            nj = (j+1)%seg_v
            a,b,c,d = idx(i,j),idx(ni,j),idx(ni,nj),idx(i,nj)
            faces.append((a,b,c)); faces.append((a,c,d))
    return verts, faces

# ── Arch (two pillars + curved lintel) ────────────────────────────────────────
def make_arch():
    verts, faces = [], []
    vi = [0]
    def add_v(x,y,z): verts.append((x,y,z)); vi[0]+=1; return vi[0]
    def quad(a,b,c,d): faces.append((a,b,c)); faces.append((a,c,d))

    # Box: add 8 verts, 12 tris
    def box(x0,y0,z0,x1,y1,z1):
        a=add_v(x0,y0,z0); b=add_v(x1,y0,z0)
        c=add_v(x1,y1,z0); d=add_v(x0,y1,z0)
        e=add_v(x0,y0,z1); f=add_v(x1,y0,z1)
        g=add_v(x1,y1,z1); h=add_v(x0,y1,z1)
        quad(a,b,c,d); quad(e,h,g,f)  # bottom, top
        quad(a,e,f,b); quad(d,c,g,h)  # front, back
        quad(a,d,h,e); quad(b,f,g,c)  # left, right

    # Pillars
    box(-2.2, 0, -0.4, -1.0, 4.5, 0.4)
    box( 1.0, 0, -0.4,  2.2, 4.5, 0.4)

    # Curved lintel (semicircle arc, 14 segments)
    segs = 14
    cx, cy = 0.0, 4.5
    R_out, R_in = 1.6, 0.8
    arc_verts = []
    for i in range(segs+1):
        ang = math.pi * i / segs  # 0..pi
        x = cx - math.cos(ang)
        y = cy + math.sin(ang)
        arc_verts.append((x, y))
    # Build arch ring (extruded in z)
    zlo, zhi = -0.4, 0.4
    start = vi[0]
    for (x,y) in arc_verts:
        for (xr,r) in [(-x, R_out), (-x, R_in)]:  # outer/inner
            xv = xr * (R_out if r==R_out else R_in) / R_out
            # Actually just use x with radius scale
            pass
    # Simpler: just extrude the arc path with thickness
    base = vi[0]
    for i, (x,y) in enumerate(arc_verts):
        nx_v = -math.sin(math.pi*i/segs) if i<segs else -1
        ny_v =  math.cos(math.pi*i/segs)
        thick = 0.5
        # outer point
        add_v(x + nx_v*thick, y + ny_v*thick, zlo)
        add_v(x + nx_v*thick, y + ny_v*thick, zhi)
        # inner point
        add_v(x - nx_v*thick*0.5, y - ny_v*thick*0.5, zlo)
        add_v(x - nx_v*thick*0.5, y - ny_v*thick*0.5, zhi)
    # Connect segments (4 verts per station: outer_lo, outer_hi, inner_lo, inner_hi)
    for i in range(segs):
        s = base + i*4
        n = base + (i+1)*4
        ol,oh,il,ih = s,s+1,s+2,s+3
        nl,nh,nil_,nih = n,n+1,n+2,n+3
        quad(ol,nl,nh,oh)   # outer face
        quad(il,ih,nih,nil_) # inner face
        quad(oh,nh,nih,ih)   # top face
        quad(ol,il,nil_,nl)  # bottom face
    # Caps
    s0 = base; s1 = base + segs*4
    quad(s0, s0+1, s0+3, s0+2)
    quad(s1, s1+2, s1+3, s1+1)
    return verts, faces

os.makedirs("assets", exist_ok=True)
print("Generating assets...")
save_obj("assets/sphere.obj", *make_sphere())
save_obj("assets/torus.obj",  *make_torus())
save_obj("assets/arch.obj",   *make_arch())
print("Done.")
