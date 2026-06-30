import taichi as ti

ti.init(arch=ti.gpu)

# 常量定义
SCREEN_SIZE = 800
EPSILON = 1e-4

# 材质参数（可调节）
Ka = ti.field(dtype=ti.f32, shape=())
Kd = ti.field(dtype=ti.f32, shape=())
Ks = ti.field(dtype=ti.f32, shape=())
Shininess = ti.field(dtype=ti.f32, shape=())

Ka[None] = 0.2
Kd[None] = 0.7
Ks[None] = 0.5
Shininess[None] = 32.0

# 球体参数
SPHERE_CENTER = ti.Vector([-1.2, -0.2, 0.0])
SPHERE_RADIUS = 1.2
SPHERE_COLOR = ti.Vector([0.8, 0.1, 0.1])

# 圆锥参数
CONE_VERTEX = ti.Vector([1.2, 1.2, 0.0])
CONE_BASE_Y = -1.4
CONE_BASE_RADIUS = 1.2
CONE_COLOR = ti.Vector([0.6, 0.2, 0.8])

# 相机和光照参数
CAMERA_POS = ti.Vector([0.0, 0.0, 5.0])
LIGHT_POS = ti.Vector([2.0, 3.0, 4.0])
LIGHT_COLOR = ti.Vector([1.0, 1.0, 1.0])
BACKGROUND_COLOR = ti.Vector([0.0, 0.3, 0.3])

# 像素缓冲区
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(SCREEN_SIZE, SCREEN_SIZE))

@ti.func
def ray_sphere_intersect(ro, rd, center, radius):
    oc = ro - center
    a = rd.dot(rd)
    b = 2.0 * oc.dot(rd)
    c = oc.dot(oc) - radius * radius
    discriminant = b * b - 4.0 * a * c
    
    hit = ti.cast(False, ti.i32)
    t = -1.0
    hit_pos = ti.Vector([0.0, 0.0, 0.0])
    normal = ti.Vector([0.0, 0.0, 0.0])
    
    if discriminant >= 0.0:
        sqrt_disc = ti.sqrt(discriminant)
        t1 = (-b - sqrt_disc) / (2.0 * a)
        t2 = (-b + sqrt_disc) / (2.0 * a)
        
        t_candidate = -1.0
        if t1 > 0.001:
            t_candidate = t1
        elif t2 > 0.001:
            t_candidate = t2
        
        if t_candidate > 0.001:
            hit = ti.cast(True, ti.i32)
            t = t_candidate
            hit_pos = ro + rd * t
            normal = (hit_pos - center).normalized()
    
    return hit, t, hit_pos, normal

@ti.func
def ray_cone_intersect(ro, rd, vertex, base_y, base_radius):
    hit = ti.cast(False, ti.i32)
    t = -1.0
    hit_pos = ti.Vector([0.0, 0.0, 0.0])
    normal = ti.Vector([0.0, 0.0, 0.0])
    
    cone_height = vertex.y - base_y
    
    valid_cone = cone_height > 0.0
    
    if valid_cone:
        axis = ti.Vector([0.0, 1.0, 0.0])
        tan_theta = base_radius / cone_height
        
        oc = ro - vertex
        rd_parallel = rd.dot(axis)
        rd_perp = rd - rd_parallel * axis
        oc_parallel = oc.dot(axis)
        oc_perp = oc - oc_parallel * axis
        
        a = rd_perp.dot(rd_perp) - tan_theta * tan_theta * rd_parallel * rd_parallel
        b = 2.0 * (oc_perp.dot(rd_perp) - tan_theta * tan_theta * oc_parallel * rd_parallel)
        c = oc_perp.dot(oc_perp) - tan_theta * tan_theta * oc_parallel * oc_parallel
        
        discriminant = b * b - 4.0 * a * c
        
        if discriminant >= 0.0:
            sqrt_disc = ti.sqrt(discriminant)
            t1 = (-b - sqrt_disc) / (2.0 * a)
            t2 = (-b + sqrt_disc) / (2.0 * a)
            
            min_t_candidate = 1e10
            
            if t1 > 0.001:
                hit_candidate = ro + rd * t1
                if base_y <= hit_candidate.y <= vertex.y:
                    if t1 < min_t_candidate:
                        min_t_candidate = t1
            
            if t2 > 0.001:
                hit_candidate = ro + rd * t2
                if base_y <= hit_candidate.y <= vertex.y:
                    if t2 < min_t_candidate:
                        min_t_candidate = t2
            
            if min_t_candidate < 1e10:
                t = min_t_candidate
    
    if t > 0.001:
        hit = ti.cast(True, ti.i32)
        hit_pos = ro + rd * t
        
        axis = ti.Vector([0.0, 1.0, 0.0])
        tan_theta = base_radius / cone_height
        
        to_hit = hit_pos - vertex
        axis_component = to_hit.dot(axis)
        perp_component = to_hit - axis_component * axis
        normal = perp_component - (axis_component * tan_theta * tan_theta) * axis
        normal = normal.normalized()
    
    return hit, t, hit_pos, normal

@ti.func
def phong_shading(hit_pos, normal, obj_color):
    L = (LIGHT_POS - hit_pos).normalized()
    V = (CAMERA_POS - hit_pos).normalized()
    R = 2.0 * normal.dot(L) * normal - L
    R = R.normalized()
    
    ambient = Ka[None] * LIGHT_COLOR * obj_color
    NdotL = ti.max(0.0, normal.dot(L))
    diffuse = Kd[None] * NdotL * LIGHT_COLOR * obj_color
    RdotV = ti.max(0.0, R.dot(V))
    specular = Ks[None] * ti.pow(RdotV, Shininess[None]) * LIGHT_COLOR
    
    return ambient + diffuse + specular

@ti.kernel
def render():
    for i, j in ti.ndrange(SCREEN_SIZE, SCREEN_SIZE):
        x = (2.0 * i - SCREEN_SIZE) / SCREEN_SIZE
        y = (SCREEN_SIZE - 2.0 * j) / SCREEN_SIZE
        
        rd = ti.Vector([x, y, -1.0]).normalized()
        ro = CAMERA_POS
        
        min_t = 1e10
        hit_color = BACKGROUND_COLOR
        hit_normal = ti.Vector([0.0, 0.0, 0.0])
        
        sphere_hit, sphere_t, sphere_pos, sphere_normal = ray_sphere_intersect(ro, rd, SPHERE_CENTER, SPHERE_RADIUS)
        if sphere_hit and sphere_t < min_t:
            min_t = sphere_t
            hit_color = SPHERE_COLOR
            hit_normal = sphere_normal
        
        cone_hit, cone_t, cone_pos, cone_normal = ray_cone_intersect(ro, rd, CONE_VERTEX, CONE_BASE_Y, CONE_BASE_RADIUS)
        if cone_hit and cone_t < min_t:
            min_t = cone_t
            hit_color = CONE_COLOR
            hit_normal = cone_normal
        
        if min_t < 1e10:
            hit_pos = ro + rd * min_t
            final_color = phong_shading(hit_pos, hit_normal, hit_color)
            pixels[i, j] = final_color
        else:
            pixels[i, j] = BACKGROUND_COLOR

def main():
    window = ti.ui.Window("Phong Lighting", res=(SCREEN_SIZE, SCREEN_SIZE))
    canvas = window.get_canvas()
    gui = window.get_gui()
    
    print("=" * 60)
    print("Phong Lighting Model")
    print("=" * 60)
    print("Instructions:")
    print("  - Use sliders to adjust material parameters")
    print("  - Ka: Ambient coefficient (0.0-1.0)")
    print("  - Kd: Diffuse coefficient (0.0-1.0)")
    print("  - Ks: Specular coefficient (0.0-1.0)")
    print("  - Shininess: Specular exponent (1.0-128.0)")
    print("  - Press ESC to exit")
    print("=" * 60)
    
    while window.running:
        render()
        canvas.set_image(pixels)
        
        with gui.sub_window("Material Params", 0.05, 0.05, 0.25, 0.35) as w:
            w.text("Phong Parameters")
            Ka[None] = w.slider_float("Ka (Ambient)", Ka[None], 0.0, 1.0)
            Kd[None] = w.slider_float("Kd (Diffuse)", Kd[None], 0.0, 1.0)
            Ks[None] = w.slider_float("Ks (Specular)", Ks[None], 0.0, 1.0)
            Shininess[None] = w.slider_float("Shininess", Shininess[None], 1.0, 128.0)
        
        window.show()

if __name__ == '__main__':
    main()
