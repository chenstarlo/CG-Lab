# 实验八：SMPL LBS (Linear Blend Skinning) 蒙皮过程可视化

**姓名：赵春哲 | 学号：202411998378 | 专业：人工智能**

## 实验目标

本实验基于 SMPL 模型完成一次完整的 LBS (Linear Blend Skinning) 蒙皮过程可视化。具体目标为：

1. 理解参数化人体模型中模板网格、形状参数、姿态参数、关节回归器和蒙皮权重之间的关系。
2. 理解 LBS 四个阶段：
   - (a) 模板网格 $$\bar{T}$$ 与蒙皮权重 $$\mathcal{W}$$
   - (b) 形状校正后网格 $$\bar{T} + B_S(\beta)$$ 以及关节 $$J(\beta)$$
   - (c) 姿态校正后网格 $$T_P(\beta,\theta)=\bar{T}+B_S(\beta)+B_P(\theta)$$
   - (d) 经过 LBS 之后的最终姿态结果
3. 学会调用 SMPL 模型，并把官方 lbs() 实现中的关键中间量单独提取出来做可视化。

## 实验原理

### LBS 四个阶段

**(a) 模板网格与蒙皮权重**
- 初始状态是模板人体网格 $$\bar{T}$$，通常处于 T-pose
- 每个顶点带有一组对各关节的影响权重 $$\mathcal{W}$$

**(b) 加入形状参数**
$$T_{shape} = \bar{T} + B_S(\beta)$$
$$J(\beta) = \mathcal{J}(T_{shape})$$

**(c) 加入姿态相关校正**
$$T_P(\beta,\theta) = \bar{T} + B_S(\beta) + B_P(\theta)$$

**(d) 线性混合蒙皮**
$$v_i' = \sum_{k=1}^{K} w_{ik} \, G_k(\theta, J(\beta)) \begin{bmatrix} v_i^{posed} \\ 1 \end{bmatrix}$$

## 实现细节

### 五个核心对象

1. **v_template**：模板顶点（T-pose）
2. **v_shaped**：加了形状形变后的顶点
3. **J**：由 v_shaped 回归出的关节位置
4. **v_posed**：加了姿态校正后的顶点
5. **verts**：完成 LBS 之后的最终顶点

### 关键函数

- **batch_rodrigues()**：轴角转旋转矩阵
- **batch_rigid_transform()**：计算关节全局变换
- **lbs_forward()**：手写 LBS 前向传播

## 使用说明

### 1. 下载 SMPL 模型

从师大云盘或 SMPL 官网下载 `SMPL_NEUTRAL.pkl`，放置在项目根目录。

### 2. 安装依赖

```bash
uv add smplx trimesh pyrender
```

### 3. 运行程序

```bash
cd d:\COD\CG-Lab
uv run python -m src.Work7.main
```

## 输出文件

| 文件 | 说明 |
|------|------|
| `outputs/stage_a_template_weights.png` | 模板网格与单关节权重热力图 |
| `outputs/all_joint_weights.png` | 全关节主导权重分布图 |
| `outputs/stage_b_shaped_joints.png` | 形状校正与关节回归 |
| `outputs/stage_c_pose_offsets.png` | 姿态校正可视化 |
| `outputs/stage_d_lbs_result.png` | 完整 LBS 结果 |
| `outputs/comparison_grid.png` | 四阶段对比图 |
| `outputs/summary.txt` | LBS 一致性验证结果 |

## 验证结果

手写 LBS 与官方实现对比：
- Mean Absolute Error (MAE)
- Max Absolute Error

## 预期效果

程序运行后，会在终端显示进度信息，并自动生成 7 个输出文件到 `outputs/` 目录。

### 阶段 (a)：模板网格与蒙皮权重

- **stage_a_template_weights.png**：左侧显示 T-pose 模板网格，右侧显示左肩关节的权重热力图（红色表示受影响强，蓝色表示受影响弱）
- **all_joint_weights.png**：显示每个顶点由哪个关节主导控制，不同颜色代表不同关节

### 阶段 (b)：形状校正与关节回归

- **stage_b_shaped_joints.png**：显示形状校正后的人体网格（体型已变化），叠加红色关节点显示回归出的关节位置

### 阶段 (c)：姿态校正

- **stage_c_pose_offsets.png**：显示姿态校正后的网格，颜色表示 pose_offsets 的大小（弯曲部位颜色更明显）

### 阶段 (d)：完整 LBS 结果

- **stage_d_lbs_result.png**：显示最终姿态下的人体网格，叠加红色关节点

### 对比图

- **comparison_grid.png**：四阶段对比图（2×2 布局），清晰展示从模板到最终蒙皮结果的完整过程

## 交互演示

本实验为非交互式程序，运行后自动完成所有计算和可视化：

1. **模型加载**：自动加载 SMPL_NEUTRAL.pkl
2. **信息输出**：打印模型基本信息（顶点数、面片数、关节数等）
3. **阶段可视化**：依次生成四个阶段的可视化图像
4. **对比图生成**：生成四阶段对比图
5. **验证总结**：生成 summary.txt 记录验证结果

### 观察要点

- **阶段 (a)**：观察肩部权重如何影响周围区域
- **阶段 (b)**：观察体型变化后关节位置如何随之变化
- **阶段 (c)**：观察弯曲部位（如肘部）的 pose_offsets 较大
- **阶段 (d)**：观察最终姿态下人体的自然弯曲效果

## 效果展示

### 模板网格与关节权重

![模板网格与关节权重](https://github.com/chenstarlo/CG-Lab/raw/main/src/Work7/outputs/stage_a_template_weights.png)

展示 T-pose 模板网格以及左肩关节的权重热力图，颜色越亮表示该关节对顶点的影响越强。

### 全关节主导权重分布

![全关节主导权重分布](https://github.com/chenstarlo/CG-Lab/raw/main/src/Work7/outputs/all_joint_weights.png)

每个面片根据"主导影响关节"分配颜色，直观展示 SMPL 模板网格各区域主要受哪些关节控制。

### 形状校正与关节回归

![形状校正与关节回归](https://github.com/chenstarlo/CG-Lab/raw/main/src/Work7/outputs/stage_b_shaped_joints.png)

设置非零 shape 参数 β 后，人体体型发生变化，关节点（红色）由形状后的网格回归得到，位于身体内部合理位置。

### 姿态校正偏移量

![姿态校正偏移量](https://github.com/chenstarlo/CG-Lab/raw/main/src/Work7/outputs/stage_c_pose_offsets.png)

姿态相关校正主要集中在发生弯曲的部位附近（如肘部、肩部），颜色越亮表示 pose_offsets 越大。

### 完整 LBS 蒙皮结果

![完整LBS蒙皮结果](https://github.com/chenstarlo/CG-Lab/raw/main/src/Work7/outputs/stage_d_lbs_result.png)

经过 LBS 之后，人体进入最终姿态，关节点跟随骨骼一起运动，呈现自然的弯曲效果。

### 四阶段对比图

![四阶段对比图](https://github.com/chenstarlo/CG-Lab/raw/main/src/Work7/outputs/comparison_grid.png)

将 (a) 模板 + 权重、(b) 形状 + 关节、(c) 姿态偏移、(d) 最终蒙皮四个阶段排成 2×2 对比图，差异一目了然。