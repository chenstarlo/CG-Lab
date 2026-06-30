# Work7: SMPL LBS (Linear Blend Skinning) 蒙皮过程可视化

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

## 效果展示

![LBS 蒙皮过程演示](待补充)

### 演示说明

请上传你的演示视频或截图，替换上面的链接地址。