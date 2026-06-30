# Work5: 可微光栅化与网格优化

**姓名：赵春哲 | 学号：202411998378 | 专业：人工智能**

## 实验目标

- 理解并掌握可微光栅化的原理，特别是在处理离散几何体（Mesh）边界时的数学近似方法
- 掌握如何通过多视角的二维图像（剪影/RGB）反推并优化三维空间中的网格顶点坐标
- 深刻理解在网格优化过程中，正则化对于防止拓扑崩坏和陷入局部最优的决定性作用

## 实验原理

将一个初始的"球体"通过梯度下降，逐渐"捏"成目标形状。这个过程需要解决两个问题：

### 防梯度消失：软光栅化 (Soft Rasterization)

在传统渲染（硬光栅化）中，像素要么在三角形内，要么在三角形外。这种阶跃变化导致边界处的梯度为 0（即发生梯度消失），网络无法知道顶点该往哪个方向移动。

软光栅化通过计算像素到三角形边缘的距离，并利用 Sigmoid 函数在边界处产生平滑的概率过渡：

$$A(d) = \text{sigmoid}\left(\frac{d}{\sigma}\right)$$

其中 $\sigma$ 控制边缘的模糊程度。

### 防局部最优：网格正则化 (Mesh Regularization)

仅依靠图像差异（Loss）去移动顶点会导致顶点交叉、重叠，变成一团"刺猬"。必须引入三种正则化损失：

- **拉普拉斯平滑 (Laplacian Smoothing)**：约束相邻顶点，防止表面出现尖锐突起
- **边长一致性 (Edge Length Penalty)**：惩罚过长或过短的边，防止三角形严重拉伸
- **法线一致性 (Normal Consistency)**：约束相邻三角形面的法线方向接近，保持表面平滑

总 Loss 公式：

$$L_{total} = L_{silhouette} + w_{lap}L_{lap} + w_{edge}L_{edge} + w_{normal}L_{normal}$$

## 环境配置

```bash
# 安装PyTorch（建议使用CUDA版本以加速）
pip install torch torchvision

# 安装PyTorch3D（Windows用户建议通过Conda安装）
# conda install pytorch pytorch torchvision cudatoolkit=11.3 -c pytorch
# pip install pytorch3d

# 其他依赖
pip install matplotlib numpy
```

## 算法实现

### 软光栅化

核心思想：对于每个像素，计算其到三角形边缘的符号距离，使用 Sigmoid 函数产生平滑的概率过渡。

```python
def soft_rasterize(verts, faces, camera_pos, sigma=1.0/IMAGE_SIZE):
    # 1. 将3D顶点投影到2D屏幕坐标
    # 2. 对每个像素，计算到三角形边缘的最近距离 d
    # 3. 使用 sigmoid(d / sigma) 计算该像素被覆盖的概率
    # 4. 概率平滑地从0过渡到1，避免硬边界
```

### 拉普拉斯平滑损失

通过图的拉普拉斯矩阵约束相邻顶点的位置差异：

$$L_{lap} = \frac{1}{N} \sum_{i=1}^{N} || \sum_{j \in N(i)} (v_i - v_j) ||^2$$

### 边长一致性损失

惩罚边长偏离目标值（通常为网格的平均边长）：

$$L_{edge} = \frac{1}{M} \sum_{e \in Edges} (||e|| - l_{target})^2$$

### 法线一致性损失

约束共享同一条边的两个三角形的法线方向接近：

$$L_{normal} = \frac{1}{E} \sum_{(i,j) \in Edges} (1 - n_i \cdot n_j)$$

### 优化循环

```python
deform_verts = source_sphere_verts.clone()
deform_verts.requires_grad_(True)
optimizer = torch.optim.Adam([deform_verts], lr=0.01)

for iteration in range(NUM_ITERATIONS):
    optimizer.zero_grad()

    # 多视角剪影损失
    total_silhouette_loss = 0
    for cam_pos in camera_views:
        rendered = soft_rasterize(deform_verts, faces, cam_pos)
        total_silhouette_loss += MSE(rendered, target_silhouette[cam_pos])

    # 正则化损失
    lap_loss = compute_laplacian_loss(deform_verts, faces)
    edge_loss = compute_edge_length_loss(deform_verts, faces)
    normal_loss = compute_normal_consistency_loss(deform_verts, faces)

    # 总损失
    total_loss = (
        total_silhouette_loss / NUM_VIEWS +
        LAPLACIAN_WEIGHT * lap_loss +
        EDGE_WEIGHT * edge_loss +
        NORMAL_WEIGHT * normal_loss
    )

    total_loss.backward()
    optimizer.step()
```

## 交互说明

运行程序：

```bash
uv run python -m src.Work5.main
```

程序将自动：

1. 创建目标网格（椭球体作为简化奶牛形状）
2. 创建源网格（高细分球体）
3. 从 8 个不同视角渲染目标剪影
4. 执行 500 次迭代优化
5. 实时显示优化过程图像

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| IMAGE_SIZE | 256 | 渲染图像分辨率 |
| NUM_VIEWS | 8 | 相机视角数量 |
| SIGMA | 1/256 | 软光栅化模糊程度 |
| LEARNING_RATE | 0.01 | 优化器学习率 |
| NUM_ITERATIONS | 500 | 优化迭代次数 |
| VISUALIZE_EVERY | 50 | 可视化间隔 |
| LAPLACIAN_WEIGHT | 1.0 | 拉普拉斯正则化权重 |
| EDGE_WEIGHT | 0.5 | 边长正则化权重 |
| NORMAL_WEIGHT | 0.3 | 法线正则化权重 |

## 实验观察要点

1. **观察梯度消失问题**：如果不使用软光栅化，边界梯度为0，顶点无法正确移动
2. **观察正则化作用**：增加正则化权重使网格更平滑，但可能降低目标匹配精度
3. **观察多视角约束**：单视角优化会产生歧义，多视角确保3D一致性
4. **观察迭代过程**：球体逐渐"捏"成目标椭球形状

## 实现特性

- 使用 PyTorch 自动微分实现梯度计算
- 多视角剪影融合确保 3D 重建完整性
- 三种正则化协同防止网格退化
- 实时可视化优化过程
- 支持提前中断（Ctrl+C）查看当前结果
