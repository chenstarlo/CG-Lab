# 实验四：Phong 光照模型交互式渲染

**姓名：赵春哲 | 学号：202411998378 | 专业：人工智能**

## 实验目标

1. **理论理解**：理解并掌握局部光照的基本原理，区分环境光（Ambient）、漫反射（Diffuse）和镜面高光（Specular）。

2. **数学基础**：熟练掌握三维空间中的向量运算（法向量计算、光线方向、视线方向与反射向量）。

3. **工程实践**：掌握如何利用 Taichi 实现交互式渲染，通过 UI 控件实时调节材质参数。

## 实验原理

Phong 光照模型将物体表面反射的光分为三个独立分量：

$$I = I_{环境光} + I_{漫反射} + I_{镜面高光}$$

### 环境光
模拟场景中经过多次反射后均匀分布的背景光：
$$I_{环境光} = K_a \times C_{光源} \times C_{物体}$$

### 漫反射
模拟粗糙表面向各个方向均匀散射的光：
$$I_{漫反射} = K_d \times \max(0, \mathbf{N} \cdot \mathbf{L}) \times C_{光源} \times C_{物体}$$

### 镜面高光
模拟光滑表面反射的强光：
$$I_{镜面高光} = K_s \times \max(0, \mathbf{R} \cdot \mathbf{V})^n \times C_{光源}$$

**符号说明**：
- $\mathbf{N}$：表面法向量
- $\mathbf{L}$：指向光源的方向向量
- $\mathbf{V}$：指向摄像机的方向向量
- $\mathbf{R}$：光线的理想反射向量
- $n$：高光指数

## 场景设置

### 几何体
- **红色球体**：圆心 (-1.2, -0.2, 0)，半径 1.2，颜色 (0.8, 0.1, 0.1)
- **紫色圆锥**：顶点 (1.2, 1.2, 0)，底面 y=-1.4，底面半径 1.2，颜色 (0.6, 0.2, 0.8)

### 摄像机与光源
- **摄像机位置**：(0, 0, 5)
- **点光源位置**：(2, 3, 4)
- **光源颜色**：(1.0, 1.0, 1.0)（纯白）
- **背景颜色**：深青色 (0.0, 0.3, 0.3)

## 交互说明

程序提供 4 个滑动条控件实时调节材质参数：

| 参数 | 范围 | 默认值 | 说明 |
|------|------|--------|------|
| Ka | 0.0 ~ 1.0 | 0.2 | 环境光系数 |
| Kd | 0.0 ~ 1.0 | 0.7 | 漫反射系数 |
| Ks | 0.0 ~ 1.0 | 0.5 | 镜面高光系数 |
| Shininess | 1.0 ~ 128.0 | 32.0 | 高光指数 |

## 运行方式

```bash
uv run python -m src.Work3.main
```

## 功能实现

1. **光线投射**：为屏幕每个像素发射射线
2. **求交检测**：实现射线与球体、圆锥的求交算法
3. **深度测试**：Z-buffer 深度竞争逻辑，选择最近交点
4. **Phong 着色**：实现完整的环境光、漫反射、镜面高光计算
5. **UI 交互**：实时调节材质参数并更新渲染结果

## 技术亮点

- 使用 Taichi GPU 并行计算加速渲染
- 实现精确的圆锥求交算法
- 交互式参数调节，即时反馈渲染效果
- 正确处理物体遮挡关系

## 预期效果

程序运行后，会在窗口中显示一个深青色背景的 3D 场景，包含一个红色球体和一个紫色圆锥。场景上方有四个滑动条控件，可以实时调节材质参数。

### 视觉效果

- **深青色背景**：模拟环境光效果
- **红色球体**：显示环境光 + 漫反射 + 镜面高光
- **紫色圆锥**：同样显示完整的 Phong 光照效果
- **高光区域**：球体和圆锥上的白色高光斑点

### 参数调节效果

- **Ka 增大**：整体颜色变亮（环境光增强）
- **Kd 增大**：漫反射增强，颜色更饱和
- **Ks 增大**：镜面高光增强，高光斑点更亮
- **Shininess 增大**：高光斑点变小变锐利

## 交互演示

### 参数调节

通过拖动窗口上方的四个滑动条，可以实时调节材质参数：

1. **Ka（环境光系数）**：控制物体整体亮度，即使在阴影中也能看到
2. **Kd（漫反射系数）**：控制物体受光面的亮度
3. **Ks（镜面高光系数）**：控制高光强度
4. **Shininess（高光指数）**：控制高光大小和锐利度

### 观察要点

- 调节 Ka 时，物体在阴影部分也会变亮或变暗
- 调节 Ks 和 Shininess 时，高光斑点会明显变化
- 移动光源位置时（通过 GUI 控件），阴影和高光位置会实时变化

## 效果展示

![Phong 光照模型演示](https://private-user-images.githubusercontent.com/182183290/615250580-48893b69-c16e-4310-bd5b-6fc94bf4486b.gif?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3ODI4MzQxODcsIm5iZiI6MTc4MjgzMzg4NywicGF0aCI6Ii8xODIxODMyOTAvNjE1MjUwNTgwLTQ4ODkzYjY5LWMxNmUtNDMxMC1iZDViLTZmYzk0YmY0NDg2Yi5naWY_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjYwNjMwJTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI2MDYzMFQxNTM4MDdaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT1lZTVhMjYzMjM0YmNmNWI3NTRmOTczN2RjNDM3ZmJkODQ0M2Y3Nzk5MTI5MWViNWM5NmU2ODhkYzNkODQ0N2ZlJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCZyZXNwb25zZS1jb250ZW50LXR5cGU9aW1hZ2UlMkZnaWYifQ.TkvPZjuWvYFu8jGA3vrco1Cnv232KHOs9TYqcDA6Ve8)

