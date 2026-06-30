# Work4: Whitted-Style 光线追踪

**姓名：赵春哲 | 学号：202411998378 | 专业：人工智能**

## 实验目标

1. **理论理解**：理解光线投射（Ray Casting）与光线追踪（Ray Tracing）的本质区别
2. **全局光照**：掌握如何通过发射"次级射线"来实现硬阴影和理想镜面反射
3. **GPU 编程思维**：学习如何将递归光线追踪改写为适合 GPU 的迭代模式

## 实验原理

Whitted-Style 光线追踪模型的工作流程：

1. **阴影测试**：从交点向光源发射"暗影射线"，检测是否处于阴影中
2. **材质分支**：
   - 漫反射材质：按 Phong 模型计算颜色，终止光线传播
   - 镜面反射材质：计算反射方向，生成新射线继续传播

反射向量公式：
$$\mathbf{R} = \mathbf{L}_{in} - 2(\mathbf{L}_{in} \cdot \mathbf{N})\mathbf{N}$$

## 场景设置

### 几何体
- **无限大平面** (y = -1.0)：黑白棋盘格纹理，漫反射材质
- **红色漫反射球** (-1.5, 0, 0)，半径 1.0
- **银色镜面反射球** (1.5, 0, 0)，半径 1.0

### 摄像机
- 位置：(0, 0, 6)

### 材质系统
| 物体 | 材质类型 | 基础颜色 |
|------|----------|----------|
| 平面 | 漫反射 (Diffuse) | 黑白棋盘格 |
| 红球 | 漫反射 (Diffuse) | (0.8, 0.1, 0.1) |
| 银球 | 镜面反射 (Mirror) | (0.9, 0.9, 0.9) |

## 核心算法

### 迭代式光线追踪
```python
throughput = 1.0  # 光线吞吐量
final_color = 0.0  # 最终颜色

for bounce in range(max_bounces):
    # 发射射线，找到最近交点
    if 击中漫反射物体:
        final_color += throughput * phong_shading()
        break
    elif 击中镜面物体:
        # 反射光线，继续追踪
        throughput *= reflectivity
        更新射线起点和方向
```

### 阴影 Acne 避坑
射线起点必须沿法线方向偏移微小量：
$$\mathbf{P}_{new} = \mathbf{P} + \mathbf{N} \times \epsilon$$
防止射线与自身表面立刻相交。

## 交互说明

| 控件 | 范围 | 说明 |
|------|------|------|
| Light X/Y/Z | 可调 | 点光源三维坐标 |
| Ka | 0.0-1.0 | 环境光系数 |
| Kd | 0.0-1.0 | 漫反射系数 |
| Ks | 0.0-1.0 | 镜面高光系数 |
| Shininess | 1.0-128.0 | 高光指数 |
| Mirror Reflectivity | 0.0-1.0 | 镜面反射率 |
| Max Bounces | 1-5 | 最大弹射次数 |

## 运行方式

```bash
uv run python -m src.Work4.main
```

## 功能亮点

1. **迭代式光线追踪**：GPU 友好的循环结构代替递归
2. **硬阴影**：暗影射线检测实现真实阴影
3. **镜面反射**：银色球体展示多次反射效果
4. **棋盘格纹理**：经典程序化纹理
5. **实时交互**：所有参数可实时调节

## 现象观察

- **Max Bounces = 1**：无反射，仅直接光照
- **Max Bounces > 1**：可见镜中球体及棋盘格倒影
- **移动光源**：阴影位置实时变化

## 效果展示

![Whitted-Style 光线追踪演示](https://private-user-images.githubusercontent.com/182183290/615250700-7cbe18a3-6d7c-4de2-8545-0643f0213c5e.gif?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3ODI4MzQxOTksIm5iZiI6MTc4MjgzMzg5OSwicGF0aCI6Ii8xODIxODMyOTAvNjE1MjUwNzAwLTdjYmUxOGEzLTZkN2MtNGRlMi04NTQ1LTA2NDNmMDIxM2M1ZS5naWY_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjYwNjMwJTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI2MDYzMFQxNTM4MTlaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT05ZDYyNzUwOTdiYWIwYTM3Mjg3MmVlNDY0YWM1YjE2N2VkMmNiMjE4YjdkNjE1YzZiNzQ2NmExNjdhMjdiMDc2JlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCZyZXNwb25zZS1jb250ZW50LXR5cGU9aW1hZ2UlMkZnaWYifQ.aAJUm9yA5KunmVWKCZuEUfAc7Sh63QaHUMCv-HPNb2E)

