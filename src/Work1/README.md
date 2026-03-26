# Work1: 3D 坐标变换实验

基于 Taichi 框架的 3D 坐标变换与 MVP 矩阵实现

## 实验简介

本实验实现了 3D 空间中的坐标变换流程，包括模型变换（Model）、视图变换（View）和投影变换（Projection）矩阵的推导与实现。通过本实验，你将深入理解计算机图形学中的 MVP 变换过程。

## 技术栈

- **Taichi** >=1.7.4 - 高性能并行计算框架
- **Python** >=3.12

## 项目结构

```
src/Work1/
├── __init__.py
└── main.py        # 主程序，包含所有变换矩阵实现
```

## 核心功能

### 1. 模型变换（Model）
- 绕 Z 轴旋转
- 支持任意角度的旋转

### 2. 视图变换（View）
- 相机平移到原点
- 简化后续变换计算

### 3. 投影变换（Projection）
- 透视投影到正交投影的转换
- 视锥体挤压为正交长方体
- 标准化设备坐标（NDC）映射

### 4. 实时交互
- 按 A 键：顺时针旋转三角形
- 按 D 键：逆时针旋转三角形
- 按 ESC 键：退出程序

## 技术实现

### 变换矩阵函数

#### `get_model_matrix(angle)`
- **功能**：生成绕 Z 轴旋转的模型变换矩阵
- **参数**：旋转角度（角度制）
- **返回**：4x4 模型变换矩阵

#### `get_view_matrix(eye_pos)`
- **功能**：生成视图变换矩阵
- **参数**：相机位置（三维向量）
- **返回**：4x4 视图变换矩阵

#### `get_projection_matrix(eye_fov, aspect_ratio, zNear, zFar)`
- **功能**：生成透视投影矩阵
- **参数**：
  - `eye_fov`：视场角（角度制）
  - `aspect_ratio`：屏幕长宽比
  - `zNear`：近截面距离
  - `zFar`：远截面距离
- **返回**：4x4 投影变换矩阵

## 快速开始

### 安装依赖

```bash
uv sync
```

### 运行程序

```bash
uv run python -m src.Work1.main
```

### 使用说明

1. 运行程序后会弹出一个 700x700 的窗口
2. 按 **A 键**：顺时针旋转三角形
3. 按 **D 键**：逆时针旋转三角形
4. 按 **ESC 键**：退出程序

## 技术亮点

1. **Taichi 并行计算**：使用 `@ti.kernel` 装饰器实现并行计算
2. **GPU 加速**：利用 Taichi 的 GPU 后端加速矩阵计算
3. **数学正确性**：严格按照计算机图形学标准实现 MVP 变换
4. **实时渲染**：流畅的 3D 旋转效果

## 坐标变换流程

1. **模型变换**：将模型绕 Z 轴旋转
2. **视图变换**：将相机平移到原点
3. **投影变换**：将 3D 坐标投影到 2D 屏幕
4. **透视除法**：将齐次坐标转换为 NDC 坐标
5. **视口变换**：将 NDC 坐标映射到屏幕坐标

## 性能优化

- 使用 Taichi 的 `@ti.func` 装饰器优化计算函数
- 利用 Taichi 的并行计算能力加速矩阵运算
- 使用 CPU 后端确保兼容性，也可切换到 GPU 后端获得更高性能

## 预期效果

程序运行后，会在窗口中显示一个彩色的线框三角形。通过按 A/D 键，可以观察三角形绕 Z 轴旋转的 3D 效果，直观理解 MVP 变换的作用。
https://private-user-images.githubusercontent.com/182183290/569703189-f9a6e30e-a17a-4b22-9739-0f55c88cac0e.gif?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NzQ1MjUzNTIsIm5iZiI6MTc3NDUyNTA1MiwicGF0aCI6Ii8xODIxODMyOTAvNTY5NzAzMTg5LWY5YTZlMzBlLWExN2EtNGIyMi05NzM5LTBmNTVjODhjYWMwZS5naWY_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjYwMzI2JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI2MDMyNlQxMTM3MzJaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT0zMzMyNGIwNWM5ODdlOGZjZTVmOGI3NTJjNWQ4Mjc3ZjI1YThjYTFjNmNmMTA2ZWNiMjI4YmM1MzM0ZjM3ZDU1JlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.dyGNVWc-BWa88VKJ4Z8yCcfWzqa_Gp5frf2qbkqn_uI