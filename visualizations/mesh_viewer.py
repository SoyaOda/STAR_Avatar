"""
Mesh visualization utilities using Open3D and Matplotlib
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Try to import Open3D (optional)
try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False


def visualize_mesh_open3d(vertices, faces, window_name="STAR Mesh", point_size=1.0):
    """
    Visualize 3D mesh using Open3D

    Args:
        vertices: np.array [N, 3] or torch.Tensor
        faces: np.array [F, 3] or torch.Tensor
        window_name: Window title
        point_size: Point cloud size if no faces
    """
    if not HAS_OPEN3D:
        print("⚠️  Open3D not installed. Falling back to Matplotlib...")
        visualize_mesh_matplotlib(vertices, faces, title=window_name)
        return

    # Convert to numpy if torch tensor
    if isinstance(vertices, torch.Tensor):
        vertices = vertices.detach().cpu().numpy()
    if isinstance(faces, torch.Tensor):
        faces = faces.detach().cpu().numpy()

    # Handle batch dimension
    if len(vertices.shape) == 3:
        vertices = vertices[0]  # Take first in batch

    print(f"Visualizing mesh:")
    print(f"  - Vertices: {vertices.shape}")
    print(f"  - Faces: {faces.shape}")
    print(f"  - Vertex range: [{vertices.min():.3f}, {vertices.max():.3f}]")

    # Create Open3D mesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)

    # Compute normals for better visualization
    mesh.compute_vertex_normals()

    # Set color
    mesh.paint_uniform_color([0.7, 0.7, 0.9])  # Light blue

    # Create coordinate frame for reference
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.5, origin=[0, 0, 0]
    )

    # Visualize
    print(f"\nOpening 3D viewer: '{window_name}'")
    print("Controls:")
    print("  - Left mouse: Rotate")
    print("  - Right mouse: Pan")
    print("  - Scroll: Zoom")
    print("  - Press 'H' for help")
    print("  - Press 'Q' to quit")

    o3d.visualization.draw_geometries(
        [mesh, coord_frame],
        window_name=window_name,
        width=1024,
        height=768,
        mesh_show_back_face=True
    )


def visualize_mesh_matplotlib(vertices, faces, title="STAR Mesh", save_path=None):
    """
    Visualize 3D mesh using Matplotlib (lighter, no interaction)

    Args:
        vertices: np.array [N, 3] or torch.Tensor
        faces: np.array [F, 3] or torch.Tensor
        title: Plot title
        save_path: If provided, save figure to this path
    """
    # Convert to numpy if torch tensor
    if isinstance(vertices, torch.Tensor):
        vertices = vertices.detach().cpu().numpy()
    if isinstance(faces, torch.Tensor):
        faces = faces.detach().cpu().numpy()

    # Handle batch dimension
    if len(vertices.shape) == 3:
        vertices = vertices[0]

    fig = plt.figure(figsize=(12, 8))

    # Create 4 subplots for different views
    views = [
        (221, (30, 45), "Front-Right"),
        (222, (30, 135), "Front-Left"),
        (223, (30, -45), "Back-Right"),
        (224, (-30, 45), "Top"),
    ]

    for subplot_idx, (elev, azim), view_name in views:
        ax = fig.add_subplot(subplot_idx, projection='3d')

        # Plot mesh
        ax.plot_trisurf(
            vertices[:, 0], vertices[:, 1], vertices[:, 2],
            triangles=faces,
            color='lightblue',
            alpha=0.8,
            edgecolor='gray',
            linewidth=0.1
        )

        # Set labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f"{view_name} View")

        # Set view angle
        ax.view_init(elev=elev, azim=azim)

        # Set aspect ratio
        max_range = np.array([
            vertices[:, 0].max() - vertices[:, 0].min(),
            vertices[:, 1].max() - vertices[:, 1].min(),
            vertices[:, 2].max() - vertices[:, 2].min()
        ]).max() / 2.0

        mid_x = (vertices[:, 0].max() + vertices[:, 0].min()) * 0.5
        mid_y = (vertices[:, 1].max() + vertices[:, 1].min()) * 0.5
        mid_z = (vertices[:, 2].max() + vertices[:, 2].min()) * 0.5

        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved figure to: {save_path}")

    plt.show()


def save_mesh_obj(vertices, faces, filepath):
    """
    Save mesh to OBJ file

    Args:
        vertices: np.array [N, 3]
        faces: np.array [F, 3]
        filepath: Output .obj file path
    """
    # Convert to numpy if torch tensor
    if isinstance(vertices, torch.Tensor):
        vertices = vertices.detach().cpu().numpy()
    if isinstance(faces, torch.Tensor):
        faces = faces.detach().cpu().numpy()

    # Handle batch dimension
    if len(vertices.shape) == 3:
        vertices = vertices[0]

    with open(filepath, 'w') as f:
        # Write vertices
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

        # Write faces (OBJ indices start at 1)
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

    print(f"✓ Saved mesh to: {filepath}")


def visualize_joints(joints, title="STAR Joints"):
    """
    Visualize joint positions

    Args:
        joints: np.array [num_joints, 3] or torch.Tensor
        title: Plot title
    """
    if isinstance(joints, torch.Tensor):
        joints = joints.detach().cpu().numpy()

    if len(joints.shape) == 3:
        joints = joints[0]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot joints as scatter points
    ax.scatter(
        joints[:, 0], joints[:, 1], joints[:, 2],
        c='red', s=100, alpha=0.8, marker='o'
    )

    # Label each joint
    for i, joint in enumerate(joints):
        ax.text(joint[0], joint[1], joint[2], f'{i}', fontsize=8)

    # Connect joints with lines (simplified skeleton)
    # SMPL/STAR joint connections (simplified)
    connections = [
        (0, 1), (0, 2), (0, 3),  # Root to hips and spine
        (1, 4), (2, 5),  # Hips to knees
        (4, 7), (5, 8),  # Knees to ankles
        (3, 6), (6, 9),  # Spine to chest to head
        (9, 12), (9, 13), (9, 14),  # Chest to shoulders and neck
        (12, 15), (13, 16),  # Shoulders to elbows
        (15, 18), (16, 19),  # Elbows to wrists
    ]

    for start, end in connections:
        if start < len(joints) and end < len(joints):
            ax.plot(
                [joints[start, 0], joints[end, 0]],
                [joints[start, 1], joints[end, 1]],
                [joints[start, 2], joints[end, 2]],
                'b-', linewidth=2, alpha=0.6
            )

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Test with dummy data
    print("Testing visualization functions...")

    # Create a simple cube mesh for testing
    vertices = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],  # Bottom
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],  # Top
    ], dtype=np.float32)

    faces = np.array([
        [0, 1, 2], [0, 2, 3],  # Bottom
        [4, 5, 6], [4, 6, 7],  # Top
        [0, 1, 5], [0, 5, 4],  # Front
        [2, 3, 7], [2, 7, 6],  # Back
        [0, 3, 7], [0, 7, 4],  # Left
        [1, 2, 6], [1, 6, 5],  # Right
    ], dtype=np.int32)

    print("\nTest 1: Matplotlib visualization")
    visualize_mesh_matplotlib(vertices, faces, title="Test Cube")

    # Uncomment to test Open3D (requires display)
    # print("\nTest 2: Open3D visualization")
    # visualize_mesh_open3d(vertices, faces, window_name="Test Cube")
