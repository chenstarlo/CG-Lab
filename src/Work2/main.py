import taichi as ti
import numpy as np

ti.init(arch=ti.cpu)

# 常量定义
SCREEN_SIZE = 800
NUM_SEGMENTS = 500
MAX_CONTROL_POINTS = 100
CURVE_COLOR = (0.0, 1.0, 0.0)    # 绿色
POINT_COLOR = (1.0, 0.0, 0.0)    # 红色
BG_COLOR = (0.0, 0.0, 0.0)       # 黑色

# Taichi Field 定义
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(SCREEN_SIZE, SCREEN_SIZE))
curve_points_field = ti.Vector.field(2, dtype=ti.f32, shape=(NUM_SEGMENTS + 1))
gui_points = ti.Vector.field(2, dtype=ti.f32, shape=(MAX_CONTROL_POINTS))

def de_casteljau(points, t):
    """迭代版本的 De Casteljau 算法，避免递归开销"""
    n = len(points)
    # 将列表转换为数组以便修改
    pts = np.array(points, dtype=np.float32)
    
    for k in range(1, n):
        for i in range(n - k):
            pts[i] = (1 - t) * pts[i] + t * pts[i + 1]
    
    return pts[0].tolist()

def uniform_cubic_bspline(points, t):
    """修复矩阵乘法顺序"""
    M = np.array([
        [-1,  3, -3,  1],
        [ 3, -6,  3,  0],
        [-3,  0,  3,  0],
        [ 1,  4,  1,  0]
    ]) / 6.0
    
    # T: (1, 4), M: (4, 4), P: (4, 2)
    # 正确的计算顺序: (T @ M) @ P
    T = np.array([[t**3, t**2, t, 1]])  # 形状: (1, 4)
    P = np.array(points)  # 形状: (4, 2)
    
    result = T @ M @ P  # 结果形状: (1, 2)
    return result[0].tolist()

def compute_bezier_curve(points):
    curve_points = []
    for i in range(NUM_SEGMENTS + 1):
        t = i / NUM_SEGMENTS
        point = de_casteljau(points, t)
        curve_points.append(point)
    return curve_points

def compute_bspline_curve(points):
    curve_points = []
    if len(points) < 4:
        return curve_points
    num_segments = len(points) - 3
    for segment_idx in range(num_segments):
        segment_points = points[segment_idx:segment_idx+4]
        for i in range(NUM_SEGMENTS // num_segments + 1):
            t = i / (NUM_SEGMENTS // num_segments)
            if t <= 1.0:
                point = uniform_cubic_bspline(segment_points, t)
                curve_points.append(point)
    return curve_points

@ti.kernel
def draw_curve(n: ti.i32):
    for i in range(n):
        point = curve_points_field[i]
        x = int(point[0] * SCREEN_SIZE)
        y = int(point[1] * SCREEN_SIZE)
        if 0 <= x < SCREEN_SIZE and 0 <= y < SCREEN_SIZE:
            pixels[x, y] = CURVE_COLOR

@ti.kernel
def draw_points(n: ti.i32):
    for i in range(n):
        point = gui_points[i]
        x = int(point[0] * SCREEN_SIZE)
        y = int(point[1] * SCREEN_SIZE)
        if 0 <= x < SCREEN_SIZE and 0 <= y < SCREEN_SIZE:
            pixels[x, y] = POINT_COLOR

def main():
    control_points = []
    current_mode = 'bezier'
    
    window = ti.ui.Window("Bézier & B-Spline", res=(SCREEN_SIZE, SCREEN_SIZE))
    canvas = window.get_canvas()
    
    print("操作说明:")
    print("  - 鼠标左键点击: 添加控制点")
    print("  - 按 C 键: 清空")
    print("  - 按 B 键: 切换模式")
    print("  - 按 ESC 键: 退出")
    
    last_pressed = False
    
    while window.running:
        if window.get_event(ti.ui.PRESS):
            if window.event.key == ti.ui.ESCAPE:
                break
            elif window.event.key == 'c':
                control_points.clear()
            elif window.event.key == 'b':
                current_mode = 'bspline' if current_mode == 'bezier' else 'bezier'
                print(f"模式: {'贝塞尔' if current_mode == 'bezier' else 'B样条'}")
        
        current_pressed = window.is_pressed(ti.ui.LMB)
        if current_pressed and not last_pressed:
            if len(control_points) < MAX_CONTROL_POINTS:
                pos = window.get_cursor_pos()
                control_points.append([pos[0], pos[1]])
                print(f"控制点 #{len(control_points)}: ({pos[0]:.3f}, {pos[1]:.3f})")
        last_pressed = current_pressed
        
        pixels.fill(BG_COLOR)
        
        if current_mode == 'bezier' and len(control_points) >= 2:
            curve = compute_bezier_curve(control_points)
            curve_np = np.array(curve, dtype=np.float32)
            curve_points_field.from_numpy(curve_np)
            draw_curve(len(curve))
            
        elif current_mode == 'bspline' and len(control_points) >= 4:
            curve = compute_bspline_curve(control_points)
            if len(curve) > 0:
                curve_np = np.array(curve[:NUM_SEGMENTS + 1], dtype=np.float32)
                curve_points_field.from_numpy(curve_np)
                draw_curve(len(curve_np))
        
        if len(control_points) >= 1:
            gui_np = np.array(control_points + [[-10, -10]] * (MAX_CONTROL_POINTS - len(control_points)), dtype=np.float32)
            gui_points.from_numpy(gui_np)
            draw_points(len(control_points))
        
        canvas.set_image(pixels)
        window.show()

if __name__ == '__main__':
    main()
