import taichi as ti
import numpy as np

# 初始化 Taichi，指定使用 GPU 后端以获得最佳性能
ti.init(arch=ti.gpu)

# 常量定义
SCREEN_SIZE = 800
NUM_SEGMENTS = 1000
MAX_CONTROL_POINTS = 100

# GPU 缓冲区
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(SCREEN_SIZE, SCREEN_SIZE))
curve_points_field = ti.Vector.field(2, dtype=ti.f32, shape=(NUM_SEGMENTS + 1))
gui_points = ti.Vector.field(2, dtype=ti.f32, shape=(MAX_CONTROL_POINTS))

# 控制点列表（存储在 CPU 端）
control_points = []

def de_casteljau(points, t):
    """
    De Casteljau 算法：递归线性插值求贝塞尔曲线上的点
    points: 控制点列表
    t: 参数 [0, 1]
    返回: 贝塞尔曲线在参数 t 处的 [x, y] 坐标
    """
    if len(points) == 1:
        return points[0]

    new_points = []
    for i in range(len(points) - 1):
        # 线性插值: P' = (1-t) * P0 + t * P1
        new_point = [
            (1 - t) * points[i][0] + t * points[i + 1][0],
            (1 - t) * points[i][1] + t * points[i + 1][1]
        ]
        new_points.append(new_point)

    return de_casteljau(new_points, t)

@ti.kernel
def draw_curve_kernel(n: ti.i32):
    """
    GPU 绘制内核：并行点亮贝塞尔曲线上的像素
    n: 曲线点的数量
    """
    for i in range(n):
        point = curve_points_field[i]
        x = int(point[0] * SCREEN_SIZE)
        y = int(point[1] * SCREEN_SIZE)

        # 越界检查
        if 0 <= x < SCREEN_SIZE and 0 <= y < SCREEN_SIZE:
            pixels[x, y] = [0.0, 1.0, 0.0]  # 绿色

@ti.kernel
def draw_control_points_kernel(n: ti.i32):
    """
    GPU 绘制内核：绘制控制点
    n: 控制点数量
    """
    for i in range(n):
        point = gui_points[i]
        x = int(point[0] * SCREEN_SIZE)
        y = int(point[1] * SCREEN_SIZE)

        # 越界检查
        if 0 <= x < SCREEN_SIZE and 0 <= y < SCREEN_SIZE:
            pixels[x, y] = [1.0, 0.0, 0.0]  # 红色

@ti.kernel
def draw_control_lines_kernel(n: ti.i32):
    """
    GPU 绘制内核：绘制控制多边形（连接控制点的线）
    n: 控制点数量
    """
    for i in range(n - 1):
        p1 = gui_points[i]
        p2 = gui_points[i + 1]

        x1 = int(p1[0] * SCREEN_SIZE)
        y1 = int(p1[1] * SCREEN_SIZE)
        x2 = int(p2[0] * SCREEN_SIZE)
        y2 = int(p2[1] * SCREEN_SIZE)

        # Bresenham 算法绘制线段
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy

        while True:
            if 0 <= x1 < SCREEN_SIZE and 0 <= y1 < SCREEN_SIZE:
                pixels[x1, y1] = [0.5, 0.5, 0.5]  # 灰色

            if x1 == x2 and y1 == y2:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x1 += sx
            if e2 < dx:
                err += dx
                y1 += sy

def main():
    # 创建窗口
    window = ti.ui.Window("Bézier Curve - De Casteljau Algorithm", res=(SCREEN_SIZE, SCREEN_SIZE))
    canvas = window.get_canvas()

    print("=" * 60)
    print("贝塞尔曲线绘制程序 - De Casteljau 算法")
    print("=" * 60)
    print("操作说明:")
    print("  - 鼠标左键点击: 添加控制点")
    print("  - 按 C 键: 清空所有控制点")
    print("  - 按 ESC 键: 退出程序")
    print("=" * 60)

    while window.running:
        # 处理键盘事件
        if window.get_event(ti.ui.PRESS):
            if window.event.key == ti.ui.ESCAPE:
                break
            elif window.event.key == 'c' or window.event.key == 'C':
                # 清空控制点
                control_points.clear()
                print("已清空所有控制点")
            elif window.event.key == ti.ui.LMB:
                # 添加控制点
                if len(control_points) < MAX_CONTROL_POINTS:
                    mouse_pos = window.get_cursor_pos()
                    control_points.append([mouse_pos[0], mouse_pos[1]])
                    print(f"添加控制点 #{len(control_points)}: [{mouse_pos[0]:.3f}, {mouse_pos[1]:.3f}]")

        # 清空像素缓冲区
        pixels.fill([0.0, 0.0, 0.0])  # 黑色背景

        # 如果有控制点
        if len(control_points) >= 2:
            # 计算贝塞尔曲线上的所有点
            curve_points = []
            for i in range(NUM_SEGMENTS + 1):
                t = i / NUM_SEGMENTS
                point = de_casteljau(control_points, t)
                curve_points.append(point)

            # 将曲线点拷贝到 GPU
            curve_points_np = np.array(curve_points, dtype=np.float32)
            curve_points_field.from_numpy(curve_points_np)

            # 调用 GPU 内核绘制曲线
            draw_curve_kernel(NUM_SEGMENTS + 1)

            # 绘制控制多边形（灰线）
            gui_points_np = np.array(control_points + [[-10, -10]] * (MAX_CONTROL_POINTS - len(control_points)), dtype=np.float32)
            gui_points.from_numpy(gui_points_np)
            draw_control_lines_kernel(len(control_points))

        # 如果有控制点，绘制控制点
        if len(control_points) >= 1:
            gui_points_np = np.array(control_points + [[-10, -10]] * (MAX_CONTROL_POINTS - len(control_points)), dtype=np.float32)
            gui_points.from_numpy(gui_points_np)
            draw_control_points_kernel(len(control_points))

        # 显示画面
        canvas.set_image(pixels)

        # 显示窗口
        window.show()

if __name__ == '__main__':
    main()
