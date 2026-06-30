import taichi as ti

ti.init(arch=ti.cpu)

IMAGE_SIZE = 256
LEAKY_ALPHA = 0.1
LEARNING_RATE = 0.01
BETA1 = 0.9
BETA2 = 0.999
EPSILON = 1e-8

SPHERE_CENTER = ti.Vector([0.5, 0.5, 0.5])
SPHERE_RADIUS = 0.3
TARGET_LIGHT = ti.Vector([0.8, 0.8, 0.2])
INIT_LIGHT = ti.Vector([0.2, 0.2, 0.8])

light_pos = ti.Vector.field(3, dtype=ti.f32, shape=(), needs_grad=True)
target_image = ti.field(dtype=ti.f32, shape=(IMAGE_SIZE, IMAGE_SIZE))
render_image = ti.field(dtype=ti.f32, shape=(IMAGE_SIZE, IMAGE_SIZE), needs_grad=True)
loss = ti.field(dtype=ti.f32, shape=(), needs_grad=True)

m = ti.Vector.field(3, dtype=ti.f32, shape=())
v = ti.Vector.field(3, dtype=ti.f32, shape=())
t = ti.field(dtype=ti.i32, shape=())

display_pixels = ti.Vector.field(3, dtype=ti.f32, shape=(IMAGE_SIZE * 2, IMAGE_SIZE))


@ti.func
def ray_sphere_intersect(ro, rd, center, radius):
    oc = ro - center
    a = rd.dot(rd)
    b = 2.0 * oc.dot(rd)
    c = oc.dot(oc) - radius * radius
    discriminant = b * b - 4.0 * a * c

    hit = False
    hit_pos = ti.Vector([0.0, 0.0, 0.0])
    normal = ti.Vector([0.0, 0.0, 0.0])

    if discriminant >= 0.0:
        sqrt_disc = ti.sqrt(discriminant)
        t1 = (-b - sqrt_disc) / (2.0 * a)
        t2 = (-b + sqrt_disc) / (2.0 * a)

        t_candidate = -1.0
        if t1 > 1e-4:
            t_candidate = t1
        elif t2 > 1e-4:
            t_candidate = t2

        if t_candidate > 1e-4:
            hit = True
            hit_pos = ro + rd * t_candidate
            normal = (hit_pos - center).normalized()

    return hit, hit_pos, normal


@ti.func
def leaky_lambertian(normal, light_dir, alpha):
    n_dot_l = normal.dot(light_dir)
    return ti.max(alpha * n_dot_l, n_dot_l)


@ti.kernel
def render_target():
    for i, j in ti.ndrange(IMAGE_SIZE, IMAGE_SIZE):
        u = float(i) / IMAGE_SIZE
        v_coord = float(j) / IMAGE_SIZE

        ro = ti.Vector([u, v_coord, 0.0])
        rd = ti.Vector([0.0, 0.0, 1.0])

        intensity = 0.0
        hit, hit_pos, normal = ray_sphere_intersect(ro, rd, SPHERE_CENTER, SPHERE_RADIUS)

        if hit:
            light_dir = (TARGET_LIGHT - hit_pos).normalized()
            intensity = ti.max(0.0, normal.dot(light_dir))

        target_image[i, j] = intensity


@ti.kernel
def render_current():
    for i, j in ti.ndrange(IMAGE_SIZE, IMAGE_SIZE):
        u = float(i) / IMAGE_SIZE
        v_coord = float(j) / IMAGE_SIZE

        ro = ti.Vector([u, v_coord, 0.0])
        rd = ti.Vector([0.0, 0.0, 1.0])

        intensity = 0.0
        hit, hit_pos, normal = ray_sphere_intersect(ro, rd, SPHERE_CENTER, SPHERE_RADIUS)

        if hit:
            light_dir = (light_pos[None] - hit_pos).normalized()
            intensity = leaky_lambertian(normal, light_dir, LEAKY_ALPHA)

        render_image[i, j] = intensity


@ti.kernel
def reset_loss():
    loss[None] = 0.0


@ti.kernel
def compute_loss():
    for i, j in ti.ndrange(IMAGE_SIZE, IMAGE_SIZE):
        diff = render_image[i, j] - target_image[i, j]
        ti.atomic_add(loss[None], diff * diff / (IMAGE_SIZE * IMAGE_SIZE))


@ti.kernel
def adam_step():
    grad = light_pos.grad[None]
    m[None] = BETA1 * m[None] + (1.0 - BETA1) * grad
    v[None] = BETA2 * v[None] + (1.0 - BETA2) * grad * grad
    m_hat = m[None] / (1.0 - BETA1 ** t[None])
    v_hat = v[None] / (1.0 - BETA2 ** t[None])
    light_pos[None] = light_pos[None] - LEARNING_RATE * m_hat / (ti.sqrt(v_hat) + EPSILON)


@ti.kernel
def update_display():
    for i, j in ti.ndrange(IMAGE_SIZE, IMAGE_SIZE):
        val = target_image[i, j]
        display_pixels[i, j] = ti.Vector([val, val, val])
        val2 = ti.max(0.0, render_image[i, j])
        display_pixels[i + IMAGE_SIZE, j] = ti.Vector([val2, val2, val2])


def main():
    render_target()
    light_pos[None] = INIT_LIGHT
    m[None] = ti.Vector([0.0, 0.0, 0.0])
    v[None] = ti.Vector([0.0, 0.0, 0.0])
    t[None] = 0

    gui = ti.GUI("Differentiable Rendering - Light Optimization",
                 res=(IMAGE_SIZE * 2, IMAGE_SIZE))

    iteration = 0
    while gui.running:
        t[None] += 1

        reset_loss()
        with ti.ad.Tape(loss):
            render_current()
            compute_loss()

        adam_step()
        update_display()

        if iteration % 10 == 0:
            lp = light_pos[None]
            print(f"Iter {iteration:4d} | Loss: {loss[None]:.6f} | "
                  f"Light: ({lp[0]:.3f}, {lp[1]:.3f}, {lp[2]:.3f}) | "
                  f"Target: ({TARGET_LIGHT[0]:.3f}, {TARGET_LIGHT[1]:.3f}, {TARGET_LIGHT[2]:.3f})")

        gui.set_image(display_pixels)

        gui.text(f"Iter: {iteration}", pos=(0.02, 0.98), color=0xFFFFFF, font_size=20)
        gui.text(f"Loss: {loss[None]:.6f}", pos=(0.02, 0.93), color=0xFFFFFF, font_size=16)
        lp = light_pos[None]
        gui.text(f"Light: ({lp[0]:.3f}, {lp[1]:.3f}, {lp[2]:.3f})",
                 pos=(0.52, 0.98), color=0xFFFFFF, font_size=16)
        gui.text(f"Target: ({TARGET_LIGHT[0]:.3f}, {TARGET_LIGHT[1]:.3f}, {TARGET_LIGHT[2]:.3f})",
                 pos=(0.52, 0.93), color=0x88FF88, font_size=16)

        gui.text("Target", pos=(0.02, 0.05), color=0xFFFF00, font_size=20)
        gui.text("Current", pos=(0.52, 0.05), color=0x00FFFF, font_size=20)

        iteration += 1
        gui.show()


if __name__ == '__main__':
    main()
