import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import math
import os

try:
    from pytorch3d.ops import cot_laplacian
    from pytorch3d.structures import Meshes
    from pytorch3d.renderer import (
        PerspectiveCameras,
        SoftSilhouetteShader,
        RasterizationSettings,
        MeshRenderer,
        MeshRasterizer
    )
    PYTORCH3D_AVAILABLE = True
except ImportError:
    PYTORCH3D_AVAILABLE = False
    print("PyTorch3D not available, using fallback renderer")

torch.set_num_threads(8)

# 自动检测 CUDA
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("Using CUDA device:", torch.cuda.get_device_name(0))
    print("   CUDA version:", torch.version.cuda)
else:
    DEVICE = torch.device("cpu")
    print("Using CPU (CUDA not available)")
print("Using device:", DEVICE)

IMAGE_SIZE = 64
NUM_VIEWS = 2
SIGMA = 1.0 / IMAGE_SIZE

LAPLACIAN_WEIGHT = 1.0
EDGE_WEIGHT = 0.5
NORMAL_WEIGHT = 0.3

LEARNING_RATE = 0.1
NUM_ITERATIONS = 30
VISUALIZE_EVERY = 10

def create_sphere_mesh(subdivision=8):
    verts = []
    faces = []
    
    phi = math.pi * (3.0 - math.sqrt(5.0))
    num_verts = subdivision * subdivision
    
    for i in range(num_verts):
        y = 1.0 - (i / float(num_verts - 1)) * 2.0
        radius = math.sqrt(1.0 - y * y)
        theta = phi * i
        x = math.cos(theta) * radius
        z = math.sin(theta) * radius
        verts.append([x, y, z])
    
    verts = torch.tensor(verts, dtype=torch.float32, device=DEVICE)
    
    for i in range(subdivision - 1):
        for j in range(subdivision - 1):
            idx = i * subdivision + j
            faces.append([idx, idx + 1, idx + subdivision])
            faces.append([idx + 1, idx + subdivision + 1, idx + subdivision])
    
    for i in range(subdivision - 1):
        idx = i * subdivision + subdivision - 1
        faces.append([idx, idx + subdivision, idx + 1])
    
    faces = torch.tensor(faces, dtype=torch.int64, device=DEVICE)
    return verts, faces

def create_target_cow_mesh():
    verts = []
    faces = []
    
    rows, cols = 20, 20
    for i in range(rows):
        for j in range(cols):
            u = i / (rows - 1)
            v = j / (cols - 1)
            
            theta = u * np.pi
            phi = v * 2 * np.pi
            
            x = 1.5 * np.sin(theta) * np.cos(phi)
            y = 0.8 * np.cos(theta)
            z = 1.0 * np.sin(theta) * np.sin(phi)
            
            verts.append([x, y, z])
    
    for i in range(rows - 1):
        for j in range(cols - 1):
            idx = i * cols + j
            faces.append([idx, idx + cols, idx + 1])
            faces.append([idx + 1, idx + cols, idx + cols + 1])
    
    verts_tensor = torch.tensor(verts, dtype=torch.float32, device=DEVICE)
    faces_tensor = torch.tensor(faces, dtype=torch.int64, device=DEVICE)
    
    return verts_tensor, faces_tensor

def compute_laplacian_loss(verts, faces):
    """向量化的 Laplacian 平滑损失计算"""
    V = verts.shape[0]
    
    # 创建稀疏 Laplacian 矩阵
    faces_np = faces.detach().cpu().numpy()
    
    # 收集所有边
    edges = []
    for i, j, k in faces_np:
        edges.append((i, j))
        edges.append((j, k))
        edges.append((k, i))
    
    # 构建邻接关系
    adj = [set() for _ in range(V)]
    for i, j in edges:
        adj[i].add(j)
        adj[j].add(i)
    
    # 计算 Laplacian 损失（向量化版本）
    loss = torch.tensor(0.0, device=DEVICE)
    for i in range(V):
        neighbors = list(adj[i])
        if len(neighbors) > 0:
            neighbor_verts = verts[torch.tensor(neighbors, device=DEVICE)]
            mean_neighbor = neighbor_verts.mean(dim=0)
            loss += ((verts[i] - mean_neighbor) ** 2).sum()
    
    return loss / V

def compute_edge_length_loss(verts, faces):
    """向量化的边长度损失计算"""
    # 获取三角形的三个顶点
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    
    # 计算三条边的长度
    edge_lengths = torch.cat([
        torch.norm(v0 - v1, dim=1),
        torch.norm(v1 - v2, dim=1),
        torch.norm(v2 - v0, dim=1)
    ])
    
    target_length = 0.1
    loss = torch.mean((edge_lengths - target_length) ** 2)
    
    return loss

def compute_normal_consistency_loss(verts, faces):
    """向量化的法向量一致性损失计算"""
    V = verts.shape[0]
    F = faces.shape[0]
    
    # 获取三角形的三个顶点
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    
    # 计算每个面的法向量
    edge1 = v1 - v0
    edge2 = v2 - v0
    face_normals = torch.cross(edge1, edge2)
    
    # 归一化面法向量
    face_normals_norm = torch.norm(face_normals, dim=1, keepdim=True) + 1e-6
    face_normals = face_normals / face_normals_norm
    
    # 将面法向量累加到顶点上
    normals = torch.zeros(V, 3, device=DEVICE)
    normals.index_add_(0, faces[:, 0], face_normals)
    normals.index_add_(0, faces[:, 1], face_normals)
    normals.index_add_(0, faces[:, 2], face_normals)
    
    # 归一化顶点法向量
    normals_norm = torch.norm(normals, dim=1, keepdim=True) + 1e-6
    normals = normals / normals_norm
    
    # 计算相邻顶点法向量的一致性损失
    n0 = normals[faces[:, 0]]
    n1 = normals[faces[:, 1]]
    n2 = normals[faces[:, 2]]
    
    dot_01 = torch.sum(n0 * n1, dim=1)
    dot_12 = torch.sum(n1 * n2, dim=1)
    dot_20 = torch.sum(n2 * n0, dim=1)
    
    loss = torch.mean(1.0 - dot_01 + 1.0 - dot_12 + 1.0 - dot_20)
    
    return loss

class SoftRasterizer:
    def __init__(self, image_size=256):
        self.image_size = image_size
    
    def rasterize(self, verts, faces, camera_pos, focal_length=1.0):
        batch_size = verts.shape[0] if verts.dim() > 2 else 1
        
        if verts.dim() == 2:
            verts = verts.unsqueeze(0)
        if faces.dim() == 2:
            faces = faces.unsqueeze(0)
        
        verts_np = verts.detach().cpu().numpy()
        faces_np = faces.detach().cpu().numpy()
        
        images = []
        
        for b in range(batch_size):
            v = verts_np[b]
            f = faces_np[b]
            
            image = np.zeros((self.image_size, self.image_size), dtype=np.float32)
            
            cam_x, cam_y, cam_z = camera_pos
            cam_pos = np.array([cam_x, cam_y, cam_z])
            
            for face_idx in range(len(f)):
                i, j, k = f[face_idx]
                
                v0, v1, v2 = v[i], v[j], v[k]
                
                v0_proj = v0[:2] / (v0[2] + 4.0) * focal_length * self.image_size / 2 + self.image_size / 2
                v1_proj = v1[:2] / (v1[2] + 4.0) * focal_length * self.image_size / 2 + self.image_size / 2
                v2_proj = v2[:2] / (v2[2] + 4.0) * focal_length * self.image_size / 2 + self.image_size / 2
                
                x_min = max(0, int(min(v0_proj[0], v1_proj[0], v2_proj[0])))
                x_max = min(self.image_size - 1, int(max(v0_proj[0], v1_proj[0], v2_proj[0])) + 1)
                y_min = max(0, int(min(v0_proj[1], v1_proj[1], v2_proj[1])))
                y_max = min(self.image_size - 1, int(max(v0_proj[1], v1_proj[1], v2_proj[1])) + 1)
                
                for px in range(x_min, x_max + 1):
                    for py in range(y_min, y_max + 1):
                        p = np.array([px, py])
                        
                        if self.point_in_triangle(p, v0_proj, v1_proj, v2_proj):
                            dist = self.distance_to_triangle_edge(p, v0_proj, v1_proj, v2_proj)
                            prob = 1.0 / (1.0 + np.exp(-dist / SIGMA))
                            image[py, px] = max(image[py, px], prob)
            
            images.append(image)
        
        return torch.tensor(np.array(images), device=DEVICE)
    
    def point_in_triangle(self, p, v0, v1, v2):
        def sign(p1, p2, p3):
            return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
        
        d1 = sign(p, v0, v1)
        d2 = sign(p, v1, v2)
        d3 = sign(p, v2, v0)
        
        has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
        has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
        
        return not (has_neg and has_pos)
    
    def distance_to_triangle_edge(self, p, v0, v1, v2):
        def point_to_segment_distance(p, a, b):
            ab = b - a
            ap = p - a
            t = np.dot(ap, ab) / (np.dot(ab, ab) + 1e-6)
            t = np.clip(t, 0, 1)
            closest = a + t * ab
            return np.linalg.norm(p - closest)
        
        d1 = point_to_segment_distance(p, v0, v1)
        d2 = point_to_segment_distance(p, v1, v2)
        d3 = point_to_segment_distance(p, v2, v0)
        
        return -min(d1, d2, d3)

def get_camera_views(num_views=8, radius=4.0):
    views = []
    for i in range(num_views):
        angle = 2 * np.pi * i / num_views
        x = radius * np.sin(angle)
        z = radius * np.cos(angle)
        y = 1.0 + 0.5 * np.sin(angle * 2)
        views.append([x, y, z])
    return views

def main():
    print("=" * 50)
    print("Work5: Differentiable Rasterization")
    print("=" * 50)
    
    print("\n[Step 1] Creating target mesh (ellipsoid)...")
    target_verts, target_faces = create_target_cow_mesh()
    print(f"  Target mesh: {target_verts.shape[0]} vertices, {target_faces.shape[0]} faces")
    
    print("\n[Step 2] Creating source mesh (sphere)...")
    source_verts, source_faces = create_sphere_mesh(subdivision=8)
    print(f"  Source mesh: {source_verts.shape[0]} vertices, {source_faces.shape[0]} faces")
    
    renderer = SoftRasterizer(image_size=IMAGE_SIZE)
    camera_views = get_camera_views(NUM_VIEWS)
    
    print(f"\n[Step 3] Rendering target silhouettes from {NUM_VIEWS} views...")
    target_images = []
    for i, cam_pos in enumerate(camera_views):
        img = renderer.rasterize(target_verts.unsqueeze(0), target_faces.unsqueeze(0), cam_pos)
        target_images.append(img)
        print(f"  View {i+1}/{NUM_VIEWS}: camera at ({cam_pos[0]:.2f}, {cam_pos[1]:.2f}, {cam_pos[2]:.2f})")
    
    target_images = torch.cat(target_images, dim=0)
    
    deform_verts = source_verts.clone()
    deform_verts.requires_grad_(True)
    
    optimizer = torch.optim.Adam([deform_verts], lr=LEARNING_RATE)
    
    print(f"\n[Step 4] Starting differentiable optimization...")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Regularization weights: lap={LAPLACIAN_WEIGHT}, edge={EDGE_WEIGHT}, normal={NORMAL_WEIGHT}")
    print(f"  Iterations: {NUM_ITERATIONS}")
    print("-" * 50)
    
    has_display = os.environ.get('DISPLAY', None) is not None or os.name == 'nt'
    
    if has_display:
        plt.ion()
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    try:
        for iteration in range(NUM_ITERATIONS):
            optimizer.zero_grad()
            
            total_silhouette_loss = torch.tensor(0.0, device=DEVICE)
            
            for view_idx, cam_pos in enumerate(camera_views):
                rendered = renderer.rasterize(
                    deform_verts.unsqueeze(0),
                    source_faces.unsqueeze(0),
                    cam_pos
                )
                
                silhouette_loss = torch.mean((rendered - target_images[view_idx]) ** 2)
                total_silhouette_loss += silhouette_loss
            
            total_silhouette_loss = total_silhouette_loss / NUM_VIEWS
            
            lap_loss = compute_laplacian_loss(deform_verts, source_faces)
            edge_loss = compute_edge_length_loss(deform_verts, source_faces)
            normal_loss = compute_normal_consistency_loss(deform_verts, source_faces)
            
            total_loss = (
                total_silhouette_loss +
                LAPLACIAN_WEIGHT * lap_loss +
                EDGE_WEIGHT * edge_loss +
                NORMAL_WEIGHT * normal_loss
            )
            
            total_loss.backward()
            optimizer.step()
            
            if iteration % 10 == 0:
                print(f"  Iter {iteration:4d}/{NUM_ITERATIONS} | "
                      f"Silhouette: {total_silhouette_loss.item():.6f} | "
                      f"Lap: {lap_loss.item():.6f} | "
                      f"Edge: {edge_loss.item():.6f} | "
                      f"Normal: {normal_loss.item():.6f} | "
                      f"Total: {total_loss.item():.6f}")
            
            if has_display and iteration % VISUALIZE_EVERY == 0:
                ax = axes.flat
                for view_idx in range(min(3, NUM_VIEWS)):
                    rendered = renderer.rasterize(
                        deform_verts.unsqueeze(0),
                        source_faces.unsqueeze(0),
                        camera_views[view_idx]
                    )
                    
                    ax[view_idx].clear()
                    ax[view_idx].imshow(rendered.squeeze().detach().cpu().numpy(), cmap='gray')
                    ax[view_idx].set_title(f'Iter {iteration}, View {view_idx}')
                    ax[view_idx].axis('off')
                
                ax[3].clear()
                verts_np = deform_verts.detach().cpu().numpy()
                ax[3].scatter(verts_np[:, 0], verts_np[:, 2], s=1, c=verts_np[:, 1], cmap='viridis')
                ax[3].set_title(f'Mesh (Iter {iteration})')
                ax[3].set_xlabel('X')
                ax[3].set_ylabel('Z')
                
                verts_np = target_verts.detach().cpu().numpy()
                ax[4].clear()
                ax[4].scatter(verts_np[:, 0], verts_np[:, 2], s=1, c=verts_np[:, 1], cmap='viridis')
                ax[4].set_title('Target Mesh')
                ax[4].set_xlabel('X')
                ax[4].set_ylabel('Z')
                
                ax[5].clear()
                loss_components = [
                    total_silhouette_loss.item(),
                    lap_loss.item() * LAPLACIAN_WEIGHT,
                    edge_loss.item() * EDGE_WEIGHT,
                    normal_loss.item() * NORMAL_WEIGHT
                ]
                ax[5].bar(['Silhouette', 'Laplacian', 'Edge', 'Normal'], loss_components)
                ax[5].set_title(f'Loss Components (Iter {iteration})')
                ax[5].set_ylabel('Loss')
                
                plt.tight_layout()
                plt.pause(0.01)
    
    except KeyboardInterrupt:
        print("\nOptimization interrupted by user")
    
    if has_display:
        plt.ioff()
        plt.show()
    
    print("\n[Step 5] Saving final results...")
    
    fig = plt.figure(figsize=(15, 5))
    
    ax1 = fig.add_subplot(131)
    final_rendered = renderer.rasterize(
        deform_verts.unsqueeze(0),
        source_faces.unsqueeze(0),
        camera_views[0]
    )
    ax1.imshow(final_rendered.squeeze().detach().cpu().numpy(), cmap='gray')
    ax1.set_title('Final Optimized Silhouette')
    ax1.axis('off')
    
    ax2 = fig.add_subplot(132, projection='3d')
    verts_final = deform_verts.detach().cpu().numpy()
    ax2.scatter(verts_final[:, 0], verts_final[:, 1], verts_final[:, 2], s=1, c='blue')
    ax2.set_title('Final Mesh Shape')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    
    ax3 = fig.add_subplot(133, projection='3d')
    verts_target = target_verts.detach().cpu().numpy()
    ax3.scatter(verts_target[:, 0], verts_target[:, 1], verts_target[:, 2], s=1, c='red')
    ax3.set_title('Target Mesh Shape')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    
    plt.tight_layout()
    plt.savefig('work5_result.png', dpi=150)
    print("  Result saved to work5_result.png")
    
    fig2 = plt.figure(figsize=(12, 4))
    for view_idx in range(min(4, NUM_VIEWS)):
        ax = fig2.add_subplot(1, 4, view_idx + 1)
        rendered = renderer.rasterize(
            deform_verts.unsqueeze(0),
            source_faces.unsqueeze(0),
            camera_views[view_idx]
        )
        ax.imshow(rendered.squeeze().detach().cpu().numpy(), cmap='gray')
        ax.set_title(f'View {view_idx + 1}')
        ax.axis('off')
    plt.tight_layout()
    plt.savefig('work5_views.png', dpi=150)
    print("  Multi-view result saved to work5_views.png")
    
    print("\n" + "=" * 50)
    print("Optimization complete!")
    print("=" * 50)

if __name__ == '__main__':
    main()
