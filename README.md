# CG-Lab

计算机图形学实验项目集合

## 项目简介

这是一个计算机图形学实验项目集合，包含多个基于 Taichi 框架的图形学实验，涵盖粒子系统、3D 变换、渲染等核心概念。

## 技术栈

- **Taichi** >=1.7.4 - 高性能并行计算框架
- **Python** >=3.12
- **uv** - 现代 Python 包管理器

## 项目结构

```
CG-Lab/
├── README.md          # 项目总说明
├── pyproject.toml     # 依赖配置
├── uv.lock            # 依赖锁文件
└── src/
    ├── Work0/         # 粒子物理模拟实验
    │   ├── __init__.py
    │   ├── config.py
    │   ├── main.py
    │   ├── physics.py
    │   ├── test.py
    │   └── README.md  # Work0 实验说明
    └── Work1/         # 3D 坐标变换实验
        ├── __init__.py
        ├── main.py
        └── README.md  # Work1 实验说明
```

## 实验列表

### Work0: 粒子物理模拟
- **描述**：基于 Taichi GPU 并行计算的交互式粒子系统
- **功能**：鼠标引力、空气阻力、边界碰撞
- **运行**：`uv run python -m src.Work0.main`
- **详细说明**：[Work0/README.md](src/Work0/README.md)

### Work1: 3D 坐标变换
- **描述**：实现 MVP (Model-View-Projection) 变换矩阵
- **功能**：3D 旋转、透视投影、实时渲染
- **运行**：`uv run python -m src.Work1.main`
- **详细说明**：[Work1/README.md](src/Work1/README.md)

## 快速开始

### 安装依赖

```bash
# 安装项目依赖
uv sync
```

### 运行实验

```bash
# 运行 Work0 粒子系统
uv run python -m src.Work0.main

# 运行 Work1 3D 变换
uv run python -m src.Work1.main
```

## 开发指南

### 添加新实验

1. 在 `src/` 目录下创建新的实验目录（如 `Work2/`）
2. 创建必要的代码文件和 `README.md` 说明文档
3. 更新根目录的 `README.md`，添加新实验的链接和说明

### 依赖管理

- **添加依赖**：`uv add package-name`
- **更新依赖**：`uv sync`
- **清理依赖**：`uv clean`

## 许可证

MIT License