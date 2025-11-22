"""
PyTorch-based Mesh Renderer for Synthetic Data Generation
Implements spec1.md requirements for normal/depth/segmentation/joint heatmap rendering
"""
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from typing import Tuple, Optional, Dict


class STARRenderer:
    """
    PyTorch-based renderer for generating synthetic training data

    Outputs:
    - Normal maps: Camera-space surface normals (H×W×3)
    - Depth maps: Distance from camera (H×W×1)
    - Segmentation maps: Body part labels (H×W)
    - Joint heatmaps: 2D Gaussian heatmaps for each joint (H×W×K)

    Based on spec1.md合成データ生成 section
    """

    def __init__(
        self,
        image_size: int = 512,
        focal_length: float = 50.0,  # mm, 50mm equivalent
        sensor_width: float = 36.0,   # mm, full-frame sensor
        device: str = 'cpu'
    ):
        self.image_size = image_size
        self.focal_length = focal_length
        self.sensor_width = sensor_width
        self.device = device

        # Camera intrinsics (focal length in pixels)
        self.fx = (focal_length / sensor_width) * image_size
        self.fy = self.fx
        self.cx = image_size / 2.0
        self.cy = image_size / 2.0

    def setup_camera(
        self,
        camera_distance: float = 3.0,
        view: str = 'front',
        delta_x: float = 0.0,
        delta_y: float = 0.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Setup camera position and rotation matrix

        Args:
            camera_distance: Distance from origin in meters (D in spec)
            view: 'front', 'back', 'side', 'left', 'right'
            delta_x, delta_y: Small positional offsets (±2% of distance)

        Returns:
            camera_pos: [3] camera position in world coords
            R: [3, 3] rotation matrix (world to camera)
        """
        if view == 'front':
            # Camera at (Δx, Δy, -D) looking toward +Z
            camera_pos = torch.tensor(
                [delta_x, delta_y, -camera_distance],
                dtype=torch.float32,
                device=self.device
            )
            R = torch.eye(3, dtype=torch.float32, device=self.device)

        elif view == 'back':
            # Camera at (-Δx, Δy, +D) looking toward -Z
            camera_pos = torch.tensor(
                [-delta_x, delta_y, camera_distance],
                dtype=torch.float32,
                device=self.device
            )
            # 180° rotation around Y-axis
            R = torch.tensor([
                [-1, 0, 0],
                [0, 1, 0],
                [0, 0, -1]
            ], dtype=torch.float32, device=self.device)

        elif view in ['side', 'right']:
            # Camera at (D, Δy, Δx) looking at origin
            camera_pos = torch.tensor(
                [camera_distance, delta_y, delta_x],
                dtype=torch.float32,
                device=self.device
            )
            # 90° rotation around Y-axis
            R = torch.tensor([
                [0, 0, 1],
                [0, 1, 0],
                [-1, 0, 0]
            ], dtype=torch.float32, device=self.device)

        elif view == 'left':
            camera_pos = torch.tensor(
                [-camera_distance, delta_y, delta_x],
                dtype=torch.float32,
                device=self.device
            )
            # -90° rotation around Y-axis
            R = torch.tensor([
                [0, 0, -1],
                [0, 1, 0],
                [1, 0, 0]
            ], dtype=torch.float32, device=self.device)
        else:
            raise ValueError(f"Invalid view: {view}")

        return camera_pos, R

    def project_vertices(
        self,
        vertices: torch.Tensor,
        camera_pos: torch.Tensor,
        R: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Transform vertices to camera coords and project to 2D

        Args:
            vertices: [N, 3] or [B, N, 3] vertex positions in world coords
            camera_pos: [3] camera position
            R: [3, 3] rotation matrix

        Returns:
            vertices_2d: [N, 2] or [B, N, 2] projected pixel coordinates
            depth: [N] or [B, N] depth values (Z in camera coords)
        """
        # Handle batch dimension
        if vertices.dim() == 2:
            vertices = vertices.unsqueeze(0)  # [1, N, 3]

        batch_size = vertices.shape[0]

        # Transform to camera coordinates
        # 1. Translate to camera origin
        vertices_cam = vertices - camera_pos.view(1, 1, 3)

        # 2. Rotate to camera orientation
        vertices_cam = torch.matmul(vertices_cam, R.T)  # [B, N, 3]

        # Extract depth (Z coordinate in camera space)
        depth = vertices_cam[:, :, 2]  # [B, N]

        # Perspective projection
        # Avoid division by zero
        depth_safe = torch.clamp(depth, min=0.01)

        x_2d = self.fx * (vertices_cam[:, :, 0] / depth_safe) + self.cx
        y_2d = self.fy * (vertices_cam[:, :, 1] / depth_safe) + self.cy

        vertices_2d = torch.stack([x_2d, y_2d], dim=-1)  # [B, N, 2]

        return vertices_2d.squeeze(0), depth.squeeze(0)

    def compute_vertex_normals(
        self,
        vertices: torch.Tensor,
        faces: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute per-vertex normals from mesh geometry

        Args:
            vertices: [N, 3] vertex positions
            faces: [F, 3] face indices

        Returns:
            normals: [N, 3] unit normal vectors
        """
        # Get face vertices
        v0 = vertices[faces[:, 0]]  # [F, 3]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]

        # Compute face normals via cross product
        edge1 = v1 - v0
        edge2 = v2 - v0
        face_normals = torch.cross(edge1, edge2, dim=1)  # [F, 3]

        # Normalize face normals
        face_normals = F.normalize(face_normals, dim=1)

        # Average face normals to get vertex normals
        vertex_normals = torch.zeros_like(vertices)

        for i in range(3):
            # Accumulate normals from all adjacent faces
            vertex_normals.index_add_(0, faces[:, i], face_normals)

        # Normalize vertex normals
        vertex_normals = F.normalize(vertex_normals, dim=1)

        return vertex_normals

    def render_normal_map(
        self,
        vertices: torch.Tensor,
        faces: torch.Tensor,
        camera_pos: torch.Tensor,
        R: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> np.ndarray:
        """
        Render camera-space normal map

        Returns:
            normal_map: [H, W, 3] RGB image with normals encoded as (n+1)/2
        """
        # Compute vertex normals in world space
        vertex_normals = self.compute_vertex_normals(vertices, faces)

        # Transform normals to camera space (rotate only, no translation)
        normals_cam = torch.matmul(vertex_normals, R.T)  # [N, 3]

        # Encode normals to RGB: (nx, ny, nz) -> ((nx+1)/2, (ny+1)/2, (nz+1)/2)
        normals_encoded = (normals_cam + 1.0) / 2.0  # [N, 3] in [0, 1]

        # Project vertices to 2D
        vertices_2d, depth = self.project_vertices(vertices, camera_pos, R)

        # Simple rasterization: for each face, fill triangle with interpolated normals
        normal_map = np.zeros((self.image_size, self.image_size, 3), dtype=np.float32)

        # For each face, rasterize
        for face_idx, face in enumerate(faces):
            # Get triangle vertices in 2D
            pts_2d = vertices_2d[face].cpu().numpy()  # [3, 2]
            pts_2d = pts_2d.astype(np.int32)

            # Get triangle normals (encoded)
            face_normals = normals_encoded[face].cpu().numpy()  # [3, 3]

            # Simple triangle fill with average normal
            avg_normal = face_normals.mean(axis=0)
            # Create a temporary single-channel mask and then fill RGB
            mask_temp = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
            cv2.fillPoly(mask_temp, [pts_2d], color=255)
            for c in range(3):
                normal_map[:, :, c][mask_temp > 0] = avg_normal[c]

        # Convert to uint8
        normal_map = (normal_map * 255).astype(np.uint8)

        return normal_map

    def render_depth_map(
        self,
        vertices: torch.Tensor,
        faces: torch.Tensor,
        camera_pos: torch.Tensor,
        R: torch.Tensor,
        normalize_by_height: Optional[float] = None
    ) -> np.ndarray:
        """
        Render depth map (Z-buffer)

        Args:
            normalize_by_height: If provided, scale depth so person height = this value

        Returns:
            depth_map: [H, W] depth values in meters
        """
        # Project vertices
        vertices_2d, depth = self.project_vertices(vertices, camera_pos, R)

        # Normalize depth by height if requested (spec requirement)
        if normalize_by_height is not None:
            # Compute current height from vertices
            y_min = vertices[:, 1].min()
            y_max = vertices[:, 1].max()
            current_height = y_max - y_min
            scale = normalize_by_height / current_height
            depth = depth * scale

        # Initialize depth map (background = 0)
        depth_map = np.zeros((self.image_size, self.image_size), dtype=np.float32)

        # Rasterize triangles
        vertices_2d_np = vertices_2d.cpu().numpy()
        depth_np = depth.cpu().numpy()

        for face in faces:
            pts_2d = vertices_2d_np[face.cpu().numpy()].astype(np.int32)
            avg_depth = depth_np[face.cpu().numpy()].mean()
            cv2.fillPoly(depth_map, [pts_2d], color=float(avg_depth))

        return depth_map

    def render_segmentation_map(
        self,
        vertices: torch.Tensor,
        faces: torch.Tensor,
        vertex_labels: torch.Tensor,
        camera_pos: torch.Tensor,
        R: torch.Tensor
    ) -> np.ndarray:
        """
        Render body part segmentation map

        Args:
            vertex_labels: [N] integer labels for each vertex (0=background, 1=head, etc.)

        Returns:
            seg_map: [H, W] integer label map
        """
        # Project vertices
        vertices_2d, _ = self.project_vertices(vertices, camera_pos, R)

        # Initialize segmentation map
        seg_map = np.zeros((self.image_size, self.image_size), dtype=np.uint8)

        vertices_2d_np = vertices_2d.cpu().numpy()
        labels_np = vertex_labels.cpu().numpy()

        # Rasterize with face labels
        for face in faces:
            pts_2d = vertices_2d_np[face.cpu().numpy()].astype(np.int32)
            # Use most common label among triangle vertices
            face_labels = labels_np[face.cpu().numpy()]
            label = int(np.bincount(face_labels).argmax())
            cv2.fillPoly(seg_map, [pts_2d], color=label)

        return seg_map

    def generate_joint_heatmaps(
        self,
        joints_2d: torch.Tensor,
        sigma: float = 5.0,
        confidence: Optional[torch.Tensor] = None
    ) -> np.ndarray:
        """
        Generate Gaussian heatmaps for 2D joint positions

        Args:
            joints_2d: [K, 2] joint positions in pixel coordinates
            sigma: Standard deviation of Gaussian (in pixels)
            confidence: [K] optional confidence scores

        Returns:
            heatmaps: [H, W, K] heatmap for each joint
        """
        num_joints = joints_2d.shape[0]
        heatmaps = np.zeros(
            (self.image_size, self.image_size, num_joints),
            dtype=np.float32
        )

        joints_2d_np = joints_2d.cpu().numpy()

        # Create coordinate grid
        y_grid, x_grid = np.mgrid[0:self.image_size, 0:self.image_size]

        for k in range(num_joints):
            u, v = joints_2d_np[k]

            # Skip if confidence is too low
            if confidence is not None and confidence[k] < 0.3:
                continue

            # Compute Gaussian
            gaussian = np.exp(
                -((x_grid - u)**2 + (y_grid - v)**2) / (2 * sigma**2)
            )

            heatmaps[:, :, k] = gaussian

        return heatmaps

    def render_all(
        self,
        vertices: torch.Tensor,
        faces: torch.Tensor,
        joints_3d: torch.Tensor,
        vertex_labels: Optional[torch.Tensor] = None,
        camera_distance: float = 3.0,
        view: str = 'front',
        normalize_height: float = 1.7
    ) -> Dict[str, np.ndarray]:
        """
        Generate all outputs for synthetic data (spec1.md requirements)

        Args:
            vertices: [N, 3] mesh vertices in world coords
            faces: [F, 3] face indices
            joints_3d: [K, 3] 3D joint positions
            vertex_labels: [N] body part labels (optional)
            camera_distance: Distance in meters
            view: Camera view direction
            normalize_height: Target height for depth normalization (meters)

        Returns:
            Dictionary with keys:
            - 'normal': [H, W, 3] normal map
            - 'depth': [H, W] depth map
            - 'segmentation': [H, W] segmentation map (if vertex_labels provided)
            - 'joint_heatmaps': [H, W, K] joint heatmaps
            - 'mask': [H, W] binary human mask
        """
        # Setup camera
        camera_pos, R = self.setup_camera(camera_distance, view)

        # Render normal map
        normal_map = self.render_normal_map(vertices, faces, camera_pos, R)

        # Render depth map
        depth_map = self.render_depth_map(
            vertices, faces, camera_pos, R,
            normalize_by_height=normalize_height
        )

        # Create binary mask (person vs background)
        mask = (depth_map > 0).astype(np.uint8)

        # Project joints to 2D
        joints_2d, _ = self.project_vertices(joints_3d, camera_pos, R)

        # Generate joint heatmaps
        joint_heatmaps = self.generate_joint_heatmaps(joints_2d)

        outputs = {
            'normal': normal_map,
            'depth': depth_map,
            'joint_heatmaps': joint_heatmaps,
            'mask': mask
        }

        # Optionally render segmentation
        if vertex_labels is not None:
            seg_map = self.render_segmentation_map(
                vertices, faces, vertex_labels, camera_pos, R
            )
            outputs['segmentation'] = seg_map

        return outputs


def test_renderer():
    """Test the PyTorch renderer"""
    print("Testing STARRenderer...")

    # Create simple test mesh (cube)
    vertices = torch.tensor([
        [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5],
        [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5],
        [-0.5, -0.5, 0.5], [0.5, -0.5, 0.5],
        [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5],
    ], dtype=torch.float32)

    faces = torch.tensor([
        [0,1,2], [0,2,3], [4,5,6], [4,6,7],
        [0,1,5], [0,5,4], [2,3,7], [2,7,6],
        [0,3,7], [0,7,4], [1,2,6], [1,6,5],
    ], dtype=torch.long)

    # Dummy joints
    joints = torch.tensor([
        [0, -0.3, 0],
        [0, 0.3, 0]
    ], dtype=torch.float32)

    # Create renderer
    renderer = STARRenderer(image_size=256)

    # Render all outputs
    outputs = renderer.render_all(
        vertices, faces, joints,
        camera_distance=2.0,
        view='front'
    )

    print(f"✓ Normal map: {outputs['normal'].shape}")
    print(f"✓ Depth map: {outputs['depth'].shape}")
    print(f"✓ Joint heatmaps: {outputs['joint_heatmaps'].shape}")
    print(f"✓ Mask: {outputs['mask'].shape}")

    # Save test outputs
    import os
    os.makedirs('outputs/test_renders', exist_ok=True)
    cv2.imwrite('outputs/test_renders/test_normal.png', outputs['normal'])
    cv2.imwrite('outputs/test_renders/test_depth.png',
                (outputs['depth'] * 50).astype(np.uint8))  # Scale for visibility
    cv2.imwrite('outputs/test_renders/test_mask.png', outputs['mask'] * 255)

    print("\n✓ Test outputs saved to outputs/test_renders/")


if __name__ == "__main__":
    test_renderer()
