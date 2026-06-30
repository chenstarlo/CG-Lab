import taichi as ti
import numpy as np

ti.init(arch=ti.cuda, device_memory_GB=2)

GRID_SIZE = 20
NUM_PARTICLES = GRID_SIZE * GRID_SIZE
NUM_LINES = 2 * GRID_SIZE * (GRID_SIZE - 1)

REST_LENGTH = 0.08
SPRING_K = 200.0
DAMPING_K = 0.1
MASS = 0.1
GRAVITY = 9.8
DELTA_T = 0.01
MAX_VELOCITY = 5.0

positions = ti.Vector.field(3, dtype=ti.f32, shape=NUM_PARTICLES)
velocities = ti.Vector.field(3, dtype=ti.f32, shape=NUM_PARTICLES)
forces = ti.Vector.field(3, dtype=ti.f32, shape=NUM_PARTICLES)
neighbors = ti.field(dtype=ti.i32, shape=(NUM_PARTICLES, 4))
num_neighbors = ti.field(dtype=ti.i32, shape=NUM_PARTICLES)
lines = ti.Vector.field(2, dtype=ti.i32, shape=NUM_LINES)
fixed = ti.field(dtype=ti.i32, shape=NUM_PARTICLES)

integrator_mode = ti.field(dtype=ti.i32, shape=())
is_paused = ti.field(dtype=ti.i32, shape=())

old_pos = ti.Vector.field(3, dtype=ti.f32, shape=NUM_PARTICLES)
old_vel = ti.Vector.field(3, dtype=ti.f32, shape=NUM_PARTICLES)

@ti.kernel
def init_positions():
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            idx = i * GRID_SIZE + j
            positions[idx] = ti.Vector([
                -0.8 + i * REST_LENGTH,
                0.5,
                -0.8 + j * REST_LENGTH
            ])
            velocities[idx] = ti.Vector([0.0, 0.0, 0.0])
            forces[idx] = ti.Vector([0.0, 0.0, 0.0])
            fixed[idx] = 1 if j == 0 else 0
            num_neighbors[idx] = 0

@ti.kernel
def init_neighbors():
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            idx = i * GRID_SIZE + j
            count = 0
            
            if i < GRID_SIZE - 1:
                neighbors[idx, count] = (i + 1) * GRID_SIZE + j
                count += 1
            if i > 0:
                neighbors[idx, count] = (i - 1) * GRID_SIZE + j
                count += 1
            if j < GRID_SIZE - 1:
                neighbors[idx, count] = i * GRID_SIZE + (j + 1)
                count += 1
            if j > 0:
                neighbors[idx, count] = i * GRID_SIZE + (j - 1)
                count += 1
            
            num_neighbors[idx] = count

@ti.kernel
def init_lines():
    line_idx = 0
    
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE - 1):
            idx1 = i * GRID_SIZE + j
            idx2 = i * GRID_SIZE + (j + 1)
            lines[line_idx] = ti.Vector([idx1, idx2])
            line_idx += 1
    
    for i in range(GRID_SIZE - 1):
        for j in range(GRID_SIZE):
            idx1 = i * GRID_SIZE + j
            idx2 = (i + 1) * GRID_SIZE + j
            lines[line_idx] = ti.Vector([idx1, idx2])
            line_idx += 1

@ti.func
def compute_forces_on(idx: ti.i32):
    force = ti.Vector([0.0, -MASS * GRAVITY, 0.0])
    force += -DAMPING_K * velocities[idx]
    
    for n in range(num_neighbors[idx]):
        other = neighbors[idx, n]
        delta = positions[idx] - positions[other]
        dist = delta.norm()
        if dist > 1e-6:
            spring_force = -SPRING_K * (dist - REST_LENGTH) * delta / dist
            force += spring_force
    
    return force

@ti.func
def clamp_velocity(idx: ti.i32):
    vel_norm = velocities[idx].norm()
    if vel_norm > MAX_VELOCITY:
        velocities[idx] = velocities[idx] * (MAX_VELOCITY / vel_norm)

@ti.kernel
def step_explicit():
    for i in range(NUM_PARTICLES):
        if fixed[i] == 1:
            continue
        
        f = compute_forces_on(i)
        a = f / MASS
        
        positions[i] += velocities[i] * DELTA_T
        velocities[i] += a * DELTA_T
        
        clamp_velocity(i)

@ti.kernel
def step_semi_implicit():
    for i in range(NUM_PARTICLES):
        if fixed[i] == 1:
            continue
        
        f = compute_forces_on(i)
        a = f / MASS
        
        velocities[i] += a * DELTA_T
        clamp_velocity(i)
        positions[i] += velocities[i] * DELTA_T

@ti.kernel
def step_implicit_iter():
    for i in range(NUM_PARTICLES):
        old_pos[i] = positions[i]
        old_vel[i] = velocities[i]
    
    for _ in range(3):
        for i in range(NUM_PARTICLES):
            if fixed[i] == 1:
                continue
            
            f = compute_forces_on(i)
            a = f / MASS
            
            velocities[i] = old_vel[i] + a * DELTA_T
            clamp_velocity(i)
            positions[i] = old_pos[i] + velocities[i] * DELTA_T

def main():
    init_positions()
    init_neighbors()
    init_lines()
    
    integrator_mode[None] = 1
    is_paused[None] = 0
    
    window = ti.ui.Window("Work6: Mass-Spring Cloth Simulation", (1280, 720))
    canvas = window.get_canvas()
    
    camera = ti.ui.Camera()
    camera.position(2.0, 2.0, 2.0)
    camera.lookat(0.0, 0.0, 0.0)
    
    gui = window.get_gui()
    
    while window.running:
        camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
        
        scene = window.get_scene()
        scene.set_camera(camera)
        
        scene.ambient_light((0.5, 0.5, 0.5))
        scene.point_light(pos=(5.0, 5.0, 5.0), color=(1.0, 1.0, 1.0))
        
        with gui.sub_window("Control Panel", 0.05, 0.05, 0.2, 0.3) as w:
            w.text("Integrator")
            
            if w.button("Explicit Euler"):
                integrator_mode[None] = 0
            
            if w.button("Semi-Implicit"):
                integrator_mode[None] = 1
            
            if w.button("Implicit Euler"):
                integrator_mode[None] = 2
            
            w.text("")
            w.text("Actions")
            
            if w.button("Pause/Resume"):
                is_paused[None] = 1 - is_paused[None]
            
            if w.button("Reset"):
                init_positions()
                init_neighbors()
                is_paused[None] = 0
            
            w.text("")
            mode_names = ["Explicit", "Semi-Implicit", "Implicit"]
            w.text(f"Mode: {mode_names[integrator_mode[None]]}")
            w.text(f"Status: {'Paused' if is_paused[None] else 'Running'}")
        
        if is_paused[None] == 0:
            if integrator_mode[None] == 0:
                step_explicit()
            elif integrator_mode[None] == 1:
                step_semi_implicit()
            elif integrator_mode[None] == 2:
                step_implicit_iter()
        
        scene.lines(positions, indices=lines, color=(0.8, 0.6, 0.4), width=2.0)
        scene.particles(positions, radius=0.015, color=(1.0, 0.8, 0.6))
        
        canvas.scene(scene)
        window.show()

if __name__ == "__main__":
    main()