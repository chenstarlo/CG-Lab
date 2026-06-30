import taichi as ti

ti.init(arch=ti.gpu)

# 常量定义
SCREEN_SIZE = 800
EPSILON = 1e-4

# 材质类型枚举
DIFFUSE = 0
MIRROR = 1
PLANE = 2

# 材质参数（固定值）
Ka = 0.2
Kd = 0.7
Ks = 0.5
Shininess = 32.0
MirrorReflectivity = 0.8

# 光源参数（可调节）
LightPos = ti.Vector.field(3, dtype=ti.f32, shape=())
LightPos[None] = ti.Vector([2.0, 5.0, 3.0])
LightColor = ti.Vector([1.0, 1.0, 1.0])

# 渲染参数（可调节）
MaxBounces = ti.field(dtype=ti.i32, shape=())
MaxBounces[None] = 3

# 球体 1 参数（漫反射材质）
SPHERE_CENTER_1 = ti.Vector([-1.5, 0.0, 0.0])
SPHERE_RADIUS_1 = 1.0
SPHERE_COLOR_1 = ti.Vector([0.8, 0.1, 0.1])
SPHERE_MATERIAL_1 = DIFFUSE

# 球体 2 参数（镜面反射材质）
SPHERE_CENTER_2 = ti.Vector([1.5, 0.0, 0.0])
SPHERE_RADIUS_2 = 1.0
SPHERE_COLOR_2 = ti.Vector([0.9, 0.9, 0.9])
SPHERE_MATERIAL_2 = MIRROR

# 平面参数
PLANE_Y = -1.0
PLANE_NORMAL = ti.Vector([0.0, 1.0, 0.0])

# 摄像机参数
CAMERA_POS = ti.Vector([0.0, 0.0, 6.0])
BACKGROUND_COLOR = ti.Vector([0.1, 0.1, 0.15])

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
    material = DIFFUSE
    color = ti.Vector([0.0, 0.0, 0.0])

    if discriminant >= 0.0:
        sqrt_disc = ti.sqrt(discriminant)
        t1 = (-b - sqrt_disc) / (2.0 * a)
        t2 = (-b + sqrt_disc) / (2.0 * a)

        t_candidate = -1.0
        if t1 > EPSILON:
            t_candidate = t1
        elif t2 > EPSILON:
            t_candidate = t2

        if t_candidate > EPSILON:
            hit = ti.cast(True, ti.i32)
            t = t_candidate
            hit_pos = ro + rd * t
            normal = (hit_pos - center).normalized()

    return hit, t, hit_pos, normal, material, color

@ti.func
def ray_plane_intersect(ro, rd, plane_y, plane_normal):
    hit = ti.cast(False, ti.i32)
    t = -1.0
    hit_pos = ti.Vector([0.0, 0.0, 0.0])
    normal = plane_normal
    material = PLANE
    color = ti.Vector([0.5, 0.5, 0.5])

    denom = rd.dot(plane_normal)
    if ti.abs(denom) > EPSILON:
        t = (plane_y - ro.y) / denom
        if t > EPSILON:
            hit = ti.cast(True, ti.i32)
            hit_pos = ro + rd * t

            x = int(ti.floor(hit_pos.x))
            z = int(ti.floor(hit_pos.z))
            if (x + z) % 2 == 0:
                color = ti.Vector([0.9, 0.9, 0.9])
            else:
                color = ti.Vector([0.1, 0.1, 0.1])

    return hit, t, hit_pos, normal, material, color

@ti.func
def check_shadow(hit_pos, normal, light_pos):
    light_dir = light_pos - hit_pos
    light_dist = light_dir.norm()
    light_dir_normalized = light_dir / light_dist

    shadow_ray_origin = hit_pos + normal * EPSILON

    shadow_hit = ti.cast(False, ti.i32)

    _, t1, _, _, _, _ = ray_sphere_intersect(shadow_ray_origin, light_dir_normalized, SPHERE_CENTER_1, SPHERE_RADIUS_1)
    if t1 > EPSILON and t1 < light_dist:
        shadow_hit = ti.cast(True, ti.i32)

    if not shadow_hit:
        _, t2, _, _, _, _ = ray_sphere_intersect(shadow_ray_origin, light_dir_normalized, SPHERE_CENTER_2, SPHERE_RADIUS_2)
        if t2 > EPSILON and t2 < light_dist:
            shadow_hit = ti.cast(True, ti.i32)

    if not shadow_hit:
        _, t3, _, _, _, _ = ray_plane_intersect(shadow_ray_origin, light_dir_normalized, PLANE_Y, PLANE_NORMAL)
        if t3 > EPSILON and t3 < light_dist:
            shadow_hit = ti.cast(True, ti.i32)

    return shadow_hit

@ti.func
def phong_shading(hit_pos, normal, obj_color, light_pos):
    L = (light_pos - hit_pos).normalized()
    V = (CAMERA_POS - hit_pos).normalized()
    R = 2.0 * normal.dot(L) * normal - L
    R = R.normalized()

    ambient = Ka * LightColor * obj_color
    NdotL = ti.max(0.0, normal.dot(L))
    diffuse = Kd * NdotL * LightColor * obj_color
    RdotV = ti.max(0.0, R.dot(V))
    specular = Ks * ti.pow(RdotV, Shininess) * LightColor

    shadow = check_shadow(hit_pos, normal, light_pos)

    result = ambient
    if shadow:
        pass
    else:
        result = ambient + diffuse + specular

    return result

@ti.func
def reflect_vector(d, n):
    return d - 2.0 * d.dot(n) * n

@ti.kernel
def render():
    for i, j in ti.ndrange(SCREEN_SIZE, SCREEN_SIZE):
        x = (2.0 * i - SCREEN_SIZE) / SCREEN_SIZE
        y = (2.0 * j - SCREEN_SIZE) / SCREEN_SIZE

        rd = ti.Vector([x, y, -1.0]).normalized()
        ro = CAMERA_POS

        final_color = ti.Vector([0.0, 0.0, 0.0])
        throughput = ti.Vector([1.0, 1.0, 1.0])

        for bounce in range(MaxBounces[None]):
            min_t = 1e10
            hit = ti.cast(False, ti.i32)
            hit_pos = ti.Vector([0.0, 0.0, 0.0])
            hit_normal = ti.Vector([0.0, 0.0, 0.0])
            hit_material = DIFFUSE
            hit_color = ti.Vector([0.0, 0.0, 0.0])

            sphere1_hit, t1, pos1, norm1, mat1, col1 = ray_sphere_intersect(ro, rd, SPHERE_CENTER_1, SPHERE_RADIUS_1)
            if sphere1_hit and t1 < min_t:
                min_t = t1
                hit = ti.cast(True, ti.i32)
                hit_pos = pos1
                hit_normal = norm1
                hit_material = SPHERE_MATERIAL_1
                hit_color = SPHERE_COLOR_1

            sphere2_hit, t2, pos2, norm2, mat2, col2 = ray_sphere_intersect(ro, rd, SPHERE_CENTER_2, SPHERE_RADIUS_2)
            if sphere2_hit and t2 < min_t:
                min_t = t2
                hit = ti.cast(True, ti.i32)
                hit_pos = pos2
                hit_normal = norm2
                hit_material = SPHERE_MATERIAL_2
                hit_color = SPHERE_COLOR_2

            plane_hit, t3, pos3, norm3, mat3, col3 = ray_plane_intersect(ro, rd, PLANE_Y, PLANE_NORMAL)
            if plane_hit and t3 < min_t:
                min_t = t3
                hit = ti.cast(True, ti.i32)
                hit_pos = pos3
                hit_normal = norm3
                hit_material = PLANE
                hit_color = col3

            if not hit:
                final_color = final_color + throughput * BACKGROUND_COLOR
                break

            if hit_material == DIFFUSE or hit_material == PLANE:
                shading = phong_shading(hit_pos, hit_normal, hit_color, LightPos[None])
                final_color = final_color + throughput * shading
                break

            elif hit_material == MIRROR:
                ro = hit_pos + hit_normal * EPSILON
                rd = reflect_vector(rd, hit_normal).normalized()
                throughput = throughput * MirrorReflectivity

        pixels[i, j] = final_color

def main():
    window = ti.ui.Window("Whitted Ray Tracing", res=(SCREEN_SIZE, SCREEN_SIZE))
    canvas = window.get_canvas()
    gui = window.get_gui()

    while window.running:
        render()
        canvas.set_image(pixels)

        with gui.sub_window("Controls", 0.05, 0.05, 0.25, 0.30) as w:
            w.text("Light Position")
            lx = LightPos[None][0]
            ly = LightPos[None][1]
            lz = LightPos[None][2]
            lx = w.slider_float("Light X", lx, -5.0, 5.0)
            ly = w.slider_float("Light Y", ly, 1.0, 10.0)
            lz = w.slider_float("Light Z", lz, -5.0, 5.0)
            LightPos[None] = ti.Vector([lx, ly, lz])

            w.text("Rendering")
            MaxBounces[None] = int(w.slider_float("Max Bounces", float(MaxBounces[None]), 1.0, 5.0))

        window.show()

if __name__ == '__main__':
    main()
