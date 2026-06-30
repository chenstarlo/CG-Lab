import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import os
import sys

try:
    import smplx
    SMPLX_AVAILABLE = True
except ImportError:
    SMPLX_AVAILABLE = False
    print("smplx not installed, please install with: pip install smplx")
    sys.exit(1)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_PATH = "SMPL_NEUTRAL.pkl"

def load_smpl_model(model_path):
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        print("Please download SMPL_NEUTRAL.pkl and place it in the project root")
        sys.exit(1)
    
    model = smplx.create(
        model_path,
        model_type='smpl',
        gender='neutral',
        device=DEVICE
    )
    return model

def print_model_info(model):
    print("\n" + "="*60)
    print("SMPL Model Information")
    print("="*60)
    print(f"Number of vertices: {model.v_template.shape[1]}")
    print(f"Number of faces: {model.faces.shape[0]}")
    print(f"Number of joints: {model.J_regressor.shape[0]}")
    print(f"Shape parameters (betas) dimension: {model.num_betas}")
    print(f"Pose parameters dimension: {model.num_body_joints * 3}")
    print(f"Number of blend shapes: {model.shapedirs.shape[0]}")
    print("="*60)

def visualize_mesh(ax, vertices, faces, title, colors=None, cmap='viridis', vmin=None, vmax=None):
    x = vertices[:, 0].cpu().numpy()
    y = vertices[:, 1].cpu().numpy()
    z = vertices[:, 2].cpu().numpy()
    
    if colors is None:
        colors = np.ones(len(x))
    else:
        colors = colors.cpu().numpy()
    
    ax.plot_trisurf(x, y, z, triangles=faces.cpu().numpy(), 
                    cmap=cmap, vmin=vmin, vmax=vmax,
                    edgecolor='gray', linewidth=0.1, alpha=0.8)
    
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    x_lim = max(abs(x)) * 1.2
    y_lim = max(abs(y)) * 1.2
    z_lim = max(abs(z)) * 1.2
    ax.set_xlim(-x_lim, x_lim)
    ax.set_ylim(-y_lim, y_lim)
    ax.set_zlim(-z_lim, z_lim)
    
    ax.view_init(elev=20, azim=-45)

def visualize_joints(ax, joints, color='red', marker='o', s=50):
    x = joints[:, 0].cpu().numpy()
    y = joints[:, 1].cpu().numpy()
    z = joints[:, 2].cpu().numpy()
    ax.scatter(x, y, z, c=color, marker=marker, s=s, alpha=1.0)

def visualize_stage_a_template_weights(model):
    print("\n[Task 2] Visualizing template mesh and skinning weights...")
    
    v_template = model.v_template[0]
    faces = model.faces
    
    fig = plt.figure(figsize=(16, 8))
    
    ax1 = fig.add_subplot(121, projection='3d')
    visualize_mesh(ax1, v_template, faces, '(a) Template Mesh (T-pose)')
    
    joint_idx = 11
    joint_name = "Left Shoulder"
    weights = model.lbs_weights[0, :, joint_idx]
    
    ax2 = fig.add_subplot(122, projection='3d')
    visualize_mesh(ax2, v_template, faces, f'(a) Weight Heatmap - {joint_name}', 
                   colors=weights, cmap='coolwarm', vmin=0, vmax=1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'stage_a_template_weights.png'), dpi=150)
    print("  Saved: stage_a_template_weights.png")
    
    fig2 = plt.figure(figsize=(10, 8))
    ax3 = fig2.add_subplot(111, projection='3d')
    dominant_joint = torch.argmax(model.lbs_weights[0], dim=1)
    visualize_mesh(ax3, v_template, faces, 'All Joint Dominant Weights', 
                   colors=dominant_joint, cmap='tab20')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'all_joint_weights.png'), dpi=150)
    print("  Saved: all_joint_weights.png")
    
    plt.close('all')

def visualize_stage_b_shaped_joints(model):
    print("\n[Task 3] Visualizing shape correction and joint regression...")
    
    betas = torch.zeros(1, model.num_betas, device=DEVICE)
    betas[0, 0] = 3.0
    betas[0, 1] = -2.0
    betas[0, 2] = 1.5
    
    v_shaped = model.v_template + torch.matmul(betas, model.shapedirs.transpose(1, 2)).view(1, -1, 3)
    J = torch.matmul(model.J_regressor, v_shaped[0])
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    visualize_mesh(ax, v_shaped[0], model.faces, '(b) Shape Corrected Mesh + Joints')
    visualize_joints(ax, J, color='red', marker='o', s=80)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'stage_b_shaped_joints.png'), dpi=150)
    print("  Saved: stage_b_shaped_joints.png")
    
    plt.close('all')
    
    return betas, v_shaped, J

def batch_rodrigues(rot_vecs):
    theta = torch.norm(rot_vecs, dim=-1, keepdim=True)
    theta = torch.clamp(theta, min=1e-8)
    axis = rot_vecs / theta
    
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    
    one = torch.ones_like(cos_theta)
    zero = torch.zeros_like(cos_theta)
    
    R = torch.stack([
        cos_theta + axis[:, 0:1] ** 2 * (1 - cos_theta),
        axis[:, 0:1] * axis[:, 1:2] * (1 - cos_theta) - axis[:, 2:3] * sin_theta,
        axis[:, 0:1] * axis[:, 2:3] * (1 - cos_theta) + axis[:, 1:2] * sin_theta,
        axis[:, 1:2] * axis[:, 0:1] * (1 - cos_theta) + axis[:, 2:3] * sin_theta,
        cos_theta + axis[:, 1:2] ** 2 * (1 - cos_theta),
        axis[:, 1:2] * axis[:, 2:3] * (1 - cos_theta) - axis[:, 0:1] * sin_theta,
        axis[:, 2:3] * axis[:, 0:1] * (1 - cos_theta) - axis[:, 1:2] * sin_theta,
        axis[:, 2:3] * axis[:, 1:2] * (1 - cos_theta) + axis[:, 0:1] * sin_theta,
        cos_theta + axis[:, 2:3] ** 2 * (1 - cos_theta)
    ], dim=1).view(-1, 3, 3)
    
    return R

def visualize_stage_c_pose_offsets(model, betas, v_shaped):
    print("\n[Task 4] Visualizing pose correction offsets...")
    
    body_pose = torch.zeros(1, model.num_body_joints * 3, device=DEVICE)
    body_pose[0, 3] = np.pi / 4
    body_pose[0, 6] = np.pi / 3
    body_pose[0, 36] = -np.pi / 4
    
    global_orient = torch.zeros(1, 3, device=DEVICE)
    
    rot_mats = batch_rodrigues(torch.cat([global_orient, body_pose], dim=1).view(-1, 3))
    rot_mats = rot_mats.view(1, -1, 3, 3)
    
    pose_feature = (rot_mats[:, 1:, :, :] - torch.eye(3, device=DEVICE).unsqueeze(0)).view(1, -1)
    pose_offsets = torch.matmul(pose_feature, model.posedirs).view(1, -1, 3)
    
    v_posed = v_shaped + pose_offsets
    
    offset_magnitude = torch.norm(pose_offsets[0], dim=1)
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    visualize_mesh(ax, v_posed[0], model.faces, '(c) Pose Offsets Visualization', 
                   colors=offset_magnitude, cmap='coolwarm', vmin=0, vmax=offset_magnitude.max())
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'stage_c_pose_offsets.png'), dpi=150)
    print("  Saved: stage_c_pose_offsets.png")
    
    plt.close('all')
    
    return global_orient, body_pose, v_posed, pose_offsets

def batch_rigid_transform(rot_mats, joints, parents):
    batch_size = rot_mats.shape[0]
    num_joints = joints.shape[1]
    
    J_transformed = torch.zeros(batch_size, num_joints, 4, 4, device=DEVICE)
    J_transformed[:, :, 3, 3] = 1.0
    
    for i in range(num_joints):
        if parents[i] == -1:
            transform = torch.eye(4, device=DEVICE).unsqueeze(0)
            transform[:, :3, :3] = rot_mats[:, i]
            transform[:, :3, 3] = joints[:, i]
            J_transformed[:, i] = transform
        else:
            transform = torch.eye(4, device=DEVICE).unsqueeze(0)
            transform[:, :3, :3] = rot_mats[:, i]
            transform[:, :3, 3] = joints[:, i] - joints[:, parents[i]]
            
            J_transformed[:, i] = torch.matmul(J_transformed[:, parents[i]], transform)
    
    J_transformed[:, :, :3, 3] -= torch.matmul(
        J_transformed[:, :, :3, :3], 
        joints.unsqueeze(-1)
    )[:, :, :3, 0]
    
    return J_transformed

def lbs_forward(model, betas, global_orient, body_pose):
    v_shaped = model.v_template + torch.matmul(betas, model.shapedirs.transpose(1, 2)).view(1, -1, 3)
    J = torch.matmul(model.J_regressor, v_shaped[0]).unsqueeze(0)
    
    rot_mats = batch_rodrigues(torch.cat([global_orient, body_pose], dim=1).view(-1, 3))
    rot_mats = rot_mats.view(1, -1, 3, 3)
    
    pose_feature = (rot_mats[:, 1:, :, :] - torch.eye(3, device=DEVICE).unsqueeze(0)).view(1, -1)
    pose_offsets = torch.matmul(pose_feature, model.posedirs).view(1, -1, 3)
    v_posed = v_shaped + pose_offsets
    
    A = batch_rigid_transform(rot_mats, J, model.parents)
    
    W = model.lbs_weights.unsqueeze(0)
    T = torch.matmul(W, A.view(1, -1, 16)).view(1, -1, 4, 4)
    
    v_homo = torch.cat([v_posed, torch.ones(1, v_posed.shape[1], 1, device=DEVICE)], dim=-1)
    v_homo = v_homo.unsqueeze(-1)
    
    verts_homo = torch.matmul(T, v_homo)
    verts = verts_homo[:, :, :3, 0]
    
    return verts, v_shaped, v_posed, J, A

def visualize_stage_d_lbs_result(model, betas, global_orient, body_pose):
    print("\n[Task 5] Visualizing complete LBS result...")
    
    verts, v_shaped, v_posed, J, A = lbs_forward(model, betas, global_orient, body_pose)
    
    J_transformed = A[:, :, :3, 3]
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    visualize_mesh(ax, verts[0], model.faces, '(d) Final Skinned Mesh')
    visualize_joints(ax, J_transformed[0], color='red', marker='o', s=60)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'stage_d_lbs_result.png'), dpi=150)
    print("  Saved: stage_d_lbs_result.png")
    
    plt.close('all')
    
    return verts

def create_comparison_grid(model):
    print("\n[Task 6] Creating comparison grid...")
    
    betas = torch.zeros(1, model.num_betas, device=DEVICE)
    betas[0, 0] = 3.0
    betas[0, 1] = -2.0
    betas[0, 2] = 1.5
    
    body_pose = torch.zeros(1, model.num_body_joints * 3, device=DEVICE)
    body_pose[0, 3] = np.pi / 4
    body_pose[0, 6] = np.pi / 3
    body_pose[0, 36] = -np.pi / 4
    global_orient = torch.zeros(1, 3, device=DEVICE)
    
    v_template = model.v_template[0]
    
    v_shaped = model.v_template + torch.matmul(betas, model.shapedirs.transpose(1, 2)).view(1, -1, 3)
    J = torch.matmul(model.J_regressor, v_shaped[0])
    
    rot_mats = batch_rodrigues(torch.cat([global_orient, body_pose], dim=1).view(-1, 3))
    rot_mats = rot_mats.view(1, -1, 3, 3)
    pose_feature = (rot_mats[:, 1:, :, :] - torch.eye(3, device=DEVICE).unsqueeze(0)).view(1, -1)
    pose_offsets = torch.matmul(pose_feature, model.posedirs).view(1, -1, 3)
    v_posed = v_shaped + pose_offsets
    
    verts, _, _, _, _ = lbs_forward(model, betas, global_orient, body_pose)
    
    fig = plt.figure(figsize=(20, 15))
    
    ax1 = fig.add_subplot(221, projection='3d')
    visualize_mesh(ax1, v_template, model.faces, '(a) Template + Weights')
    
    ax2 = fig.add_subplot(222, projection='3d')
    visualize_mesh(ax2, v_shaped[0], model.faces, '(b) Shape + Joints')
    visualize_joints(ax2, J, color='red', marker='o', s=40)
    
    ax3 = fig.add_subplot(223, projection='3d')
    offset_magnitude = torch.norm(pose_offsets[0], dim=1)
    visualize_mesh(ax3, v_posed[0], model.faces, '(c) Pose Offsets', 
                   colors=offset_magnitude, cmap='coolwarm')
    
    ax4 = fig.add_subplot(224, projection='3d')
    visualize_mesh(ax4, verts[0], model.faces, '(d) Final Skinned Mesh')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'comparison_grid.png'), dpi=150)
    print("  Saved: comparison_grid.png")
    
    plt.close('all')

def verify_lbs_consistency(model):
    print("\n[Task 7] Verifying LBS consistency with official implementation...")
    
    betas = torch.zeros(1, model.num_betas, device=DEVICE)
    betas[0, 0] = 2.5
    betas[0, 1] = 1.0
    betas[0, 2] = -1.5
    betas[0, 3] = 0.8
    
    body_pose = torch.zeros(1, model.num_body_joints * 3, device=DEVICE)
    body_pose[0, 3] = np.pi / 3
    body_pose[0, 6] = np.pi / 4
    body_pose[0, 9] = -np.pi / 6
    body_pose[0, 36] = np.pi / 4
    
    global_orient = torch.zeros(1, 3, device=DEVICE)
    global_orient[0, 1] = np.pi / 6
    
    my_verts, _, _, _, _ = lbs_forward(model, betas, global_orient, body_pose)
    
    output = model(
        betas=betas,
        body_pose=body_pose,
        global_orient=global_orient
    )
    official_verts = output.vertices
    
    diff = my_verts - official_verts
    mae = torch.mean(torch.abs(diff)).item()
    max_ae = torch.max(torch.abs(diff)).item()
    
    with open(os.path.join(OUTPUT_DIR, 'summary.txt'), 'w') as f:
        f.write("="*60 + "\n")
        f.write("LBS Consistency Verification Summary\n")
        f.write("="*60 + "\n\n")
        f.write(f"Model: SMPL Neutral\n")
        f.write(f"Device: {DEVICE}\n\n")
        f.write(f"Shape parameters (betas):\n")
        f.write(f"  beta[0] = {betas[0, 0].item():.4f} (height)\n")
        f.write(f"  beta[1] = {betas[0, 1].item():.4f} (weight)\n")
        f.write(f"  beta[2] = {betas[0, 2].item():.4f}\n")
        f.write(f"  beta[3] = {betas[0, 3].item():.4f}\n\n")
        f.write(f"Pose parameters:\n")
        f.write(f"  Global rotation: {global_orient[0].cpu().numpy()}\n")
        f.write(f"  Body pose: First 3 joints modified\n\n")
        f.write("="*60 + "\n")
        f.write(f"Mean Absolute Error (MAE): {mae:.10f}\n")
        f.write(f"Max Absolute Error: {max_ae:.10f}\n")
        f.write("="*60 + "\n\n")
        f.write("Verification Result: ")
        if mae < 1e-5:
            f.write("PASS - Hand-written LBS matches official implementation!\n")
        else:
            f.write("FAIL - Differences detected!\n")
    
    print(f"  Mean Absolute Error: {mae:.10f}")
    print(f"  Max Absolute Error: {max_ae:.10f}")
    print(f"  Verification: {'PASS' if mae < 1e-5 else 'FAIL'}")
    print("  Saved: summary.txt")

def main():
    print("="*60)
    print("Work7: SMPL LBS (Linear Blend Skinning) Visualization")
    print("="*60)
    
    if not SMPLX_AVAILABLE:
        sys.exit(1)
    
    model = load_smpl_model(MODEL_PATH)
    print_model_info(model)
    
    visualize_stage_a_template_weights(model)
    
    betas, v_shaped, J = visualize_stage_b_shaped_joints(model)
    
    global_orient, body_pose, v_posed, pose_offsets = visualize_stage_c_pose_offsets(model, betas, v_shaped)
    
    verts = visualize_stage_d_lbs_result(model, betas, global_orient, body_pose)
    
    create_comparison_grid(model)
    
    verify_lbs_consistency(model)
    
    print("\n" + "="*60)
    print("All tasks completed! Outputs saved to ./outputs/")
    print("="*60)

if __name__ == "__main__":
    main()