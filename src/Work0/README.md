# Work0: 粒子物理模拟实验

基于 Taichi 框架的粒子物理模拟实验项目

## 实验简介

这是一个使用 Taichi GPU 并行计算框架实现的交互式粒子物理模拟系统。项目展示了如何利用现代 GPU 计算技术实现高性能的粒子系统，支持实时物理交互和渲染。

## 技术栈

- **Taichi** >=1.7.4 - 高性能并行计算框架
- **Python** >=3.12

## 项目结构

```
src/Work0/
├── config.py      # 系统配置参数
├── physics.py     # 物理引擎核心
├── main.py        # 主程序入口
└── test.py        # 测试文件
```

## 核心功能

### 1. 粒子物理模拟
- 支持 10,000 个粒子的实时物理计算
- GPU 并行计算，性能优异
- 鼠标引力交互效果

### 2. 物理系统特性
- **引力系统**: 粒子跟随鼠标移动
- **空气阻力**: 模拟真实的运动衰减
- **边界碰撞**: 粒子在窗口边缘反弹

### 3. 渲染系统
- 实时粒子渲染
- 天蓝色粒子视觉效果
- 800x600 窗口分辨率

## 配置参数

### 物理系统参数
- `NUM_PARTICLES = 10000` - 粒子总数（性能不佳时可调至 2000）
- `GRAVITY_STRENGTH = 0.001` - 鼠标引力强度
- `DRAG_COEF = 0.98` - 空气阻力系数
- `BOUNCE_COEF = -0.8` - 边界反弹能量损耗

### 渲染系统参数
- `WINDOW_RES = (800, 600)` - 窗口分辨率
- `PARTICLE_RADIUS = 1.5` - 粒子绘制半径
- `PARTICLE_COLOR = 0x00BFFF` - 粒子颜色（天蓝色）

## 快速开始

### 安装依赖

```bash
uv sync
```

### 运行程序

```bash
uv run python -m src.Work0.main
```

### 使用说明

1. 运行程序后会弹出一个窗口
2. 移动鼠标即可看到粒子被引力吸引的效果
3. 粒子会在窗口边缘反弹
4. 按 ESC 键或关闭窗口退出程序

## 技术亮点

1. **GPU 并行计算**: 使用 Taichi 的 `@ti.kernel` 装饰器将物理计算编译为 GPU 内核
2. **显存管理**: 粒子数据直接存储在显存中，避免 CPU-GPU 数据传输
3. **高性能**: 可同时模拟 10,000 个粒子的实时物理交互
4. **交互式**: 粒子会跟随鼠标移动，形成引力效果

## 模块说明

### config.py
定义系统的核心参数，包括物理系统参数和渲染系统参数。

### physics.py
实现物理引擎核心逻辑：
- `pos` - 粒子位置向量场（存储在显存中）
- `vel` - 粒子速度向量场（存储在显存中）
- `init_particles()` - 初始化粒子随机位置
- `update_particles(mouse_x, mouse_y)` - 并行计算每个粒子的物理状态

### main.py
程序入口和渲染循环：
- 初始化 Taichi GPU 环境
- 创建 GUI 窗口
- 主渲染循环：获取鼠标位置 → 驱动 GPU 计算 → 渲染粒子

## 性能优化建议

如果遇到卡顿，可以调整 `config.py` 中的参数：
- 将 `NUM_PARTICLES` 从 10000 调整为 2000
- 调整 `GRAVITY_STRENGTH` 改变引力强度
- 调整 `DRAG_COEF` 改变阻力效果

## 效果展示

![粒子系统效果](https://private-user-images.githubusercontent.com/182183290/569691423-657673af-97dd-49b0-a976-b81a4f808acd.gif?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NzQ1MjQ0OTUsIm5iZiI6MTc3NDUyNDE5NSwicGF0aCI6Ii8xODIxODMyOTAvNTY5NjkxNDIzLTY1NzY3M2FmLTk3ZGQtNDliMC1hOTc2LWI4MWE0ZjgwOGFjZC5naWY_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjYwMzI2JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI2MDMyNlQxMTIzMTVaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT02YzU1OGFkOTMyMTljODg1ZDA1ODk0ZTZjNjcxN2E4ODk1MGYyODJhZDNkOTYzY2RiYzBmMjYwYmEyYTAxMjI0JlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.5OyqaIZZq9bNMxJsYZ-qfEW_y6vHnPZY6BGvBlxuduA)

### 交互演示

1. 移动鼠标吸引粒子
2. 粒子在边界反弹
3. 实时物理计算