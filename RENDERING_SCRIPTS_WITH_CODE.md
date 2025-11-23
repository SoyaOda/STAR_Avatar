# レンダリング改善スクリプト完全版

**作成日時**: 2025-11-22 23:03:42

**目的**: Sapiens推論精度向上のための3Dレンダリング改善

このドキュメントには、各スクリプトの完全なソースコードが含まれています。

---

## 📋 目次

1. [⭐⭐⭐ visualizations/photorealistic_renderer.py](#visualizations-photorealistic_renderer-py)
2. [⭐⭐ render_average_photorealistic.py](#render_average_photorealistic-py)
3. [⭐⭐ models/star_layer.py](#models-star_layer-py)
4. [⭐ data/synthetic_dataset.py](#data-synthetic_dataset-py)

---

## 📊 統計情報

- **総スクリプト数**: 4
- **存在するファイル**: 4
- **総コード行数**: 1,014 行
- **総ファイルサイズ**: 34.4 KB

---

## 優先度1: レンダリングの改善

### 1. ⭐⭐⭐ `visualizations/photorealistic_renderer.py`

**📄 ファイル情報**

- **パス**: `/Users/moei/program/STAR_Avatar/visualizations/photorealistic_renderer.py`
- **サイズ**: 12.5 KB
- **行数**: 364 行
- **最終更新**: 2025-11-22 22:25
- **状態**: ✅ 存在

**📝 説明**

照明、背景、マテリアル、カメラ設定

**✨ 改善項目**

- HDRI環境照明の追加
- リアルな背景（グラデーション、環境マップ）
- カメラの被写界深度（DOF）
- アンビエントオクルージョン
- 高解像度化（512→1024+）

**💻 ソースコード**

```py
"""
Photorealistic RGB Renderer for STAR Body Meshes

Uses pyrender with PBR (Physically-Based Rendering) for photo-like outputs.
This renderer is optimized for generating realistic RGB images similar to photographs.
"""

import numpy as np
import pyrender
import trimesh
from PIL import Image
import os


class PhotorealisticRenderer:
    """
    Photorealistic renderer using pyrender with PBR materials.

    Produces high-quality RGB images that look like real photographs.
    Uses 3-point lighting setup and smooth shading for realistic appearance.
    """

    def __init__(self, image_size=512, focal_length=50.0, sensor_width=36.0):
        """
        Initialize photorealistic renderer.

        Args:
            image_size: Output image resolution (width and height in pixels)
            focal_length: Camera focal length in mm (default: 50mm standard lens)
            sensor_width: Camera sensor width in mm (default: 36mm full-frame)
        """
        self.image_size = image_size
        self.focal_length = focal_length
        self.sensor_width = sensor_width

        # Calculate focal length in pixels
        self.fx = (focal_length / sensor_width) * image_size
        self.fy = self.fx
        self.cx = image_size / 2.0
        self.cy = image_size / 2.0

        # Create pyrender renderer (offscreen for server/headless environments)
        self.renderer = pyrender.OffscreenRenderer(image_size, image_size)

    def create_mesh(self, vertices, faces, color=None):
        """
        Create a pyrender mesh with PBR material.

        Args:
            vertices: Vertex positions [N, 3] numpy array
            faces: Face indices [F, 3] numpy array
            color: Base color [R, G, B, A] (default: skin-like beige)

        Returns:
            pyrender.Mesh with PBR material
        """
        # Default skin-like color (moderate tone to prevent overexposure)
        if color is None:
            color = [0.4, 0.32, 0.28, 1.0]  # Darker base color

        # Convert to numpy if torch tensor
        if hasattr(vertices, 'cpu'):
            vertices = vertices.cpu().numpy()
        if hasattr(faces, 'cpu'):
            faces = faces.cpu().numpy()

        # Create trimesh object
        tri_mesh = trimesh.Trimesh(
            vertices=vertices,
            faces=faces,
            process=True  # Automatically compute normals and validate
        )

        # Create PBR material for realistic appearance
        material = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=color,
            metallicFactor=0.0,      # No metallic (human skin)
            roughnessFactor=0.8,     # Higher roughness to prevent white-out
            alphaMode='OPAQUE'
        )

        # Create pyrender mesh
        mesh = pyrender.Mesh.from_trimesh(tri_mesh, material=material, smooth=True)

        return mesh

    def setup_camera(self, distance=3.0, view='front', look_at_y=-0.4):
        """
        Setup camera position and orientation.

        Args:
            distance: Camera distance from origin in meters
            view: Camera view ('front', 'back', 'side'/'right', 'left')
            look_at_y: Y-coordinate of the point camera looks at (negative to see feet)

        Returns:
            camera: pyrender.Camera object
            camera_pose: 4x4 camera pose matrix
        """
        # Create perspective camera
        camera = pyrender.IntrinsicsCamera(
            fx=self.fx,
            fy=self.fy,
            cx=self.cx,
            cy=self.cy,
            znear=0.1,
            zfar=100.0
        )

        # Helper function to create look-at matrix
        def look_at(eye, target, up):
            """Create a look-at view matrix."""
            forward = target - eye
            forward = forward / np.linalg.norm(forward)

            right = np.cross(forward, up)
            right = right / np.linalg.norm(right)

            up_new = np.cross(right, forward)

            # Pyrender uses OpenGL convention: camera looks down -Z
            # So we need to negate forward
            mat = np.eye(4)
            mat[:3, 0] = right
            mat[:3, 1] = up_new
            mat[:3, 2] = -forward
            mat[:3, 3] = eye

            return mat

        # Camera position based on view
        target = np.array([0.0, look_at_y, 0.0])  # Look at lower point

        if view == 'front':
            # Front view: camera on +Z axis looking at target
            camera_pos = np.array([0.0, 0.0, distance])
            up = np.array([0.0, 1.0, 0.0])
            camera_pose = look_at(camera_pos, target, up)

        elif view == 'back':
            # Back view: camera on -Z axis looking at target
            camera_pos = np.array([0.0, 0.0, -distance])
            up = np.array([0.0, 1.0, 0.0])
            camera_pose = look_at(camera_pos, target, up)

        elif view in ['side', 'right']:
            # Right side view: camera on +X axis looking at target
            camera_pos = np.array([distance, 0.0, 0.0])
            up = np.array([0.0, 1.0, 0.0])
            camera_pose = look_at(camera_pos, target, up)

        elif view == 'left':
            # Left side view: camera on -X axis looking at target
            camera_pos = np.array([-distance, 0.0, 0.0])
            up = np.array([0.0, 1.0, 0.0])
            camera_pose = look_at(camera_pos, target, up)
        else:
            raise ValueError(f"Unknown view: {view}")

        return camera, camera_pose

    def setup_lighting(self, scene, distance=3.0):
        """
        Setup室内のソフトな環境光を近似するライト設定

        3点照明をベースに、適度な明るさで部屋全体の照明を再現します。

        Args:
            scene: pyrender.Scene object
            distance: Approximate distance for light placement (not used with PointLight)
        """
        # PointLight の基本強度（白飛び防止のため大幅に減らす）
        base_intensity = 5.0

        # Key light: 右上前方（メインライト）
        key_light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=base_intensity * 1.5)
        key_pose = np.eye(4)
        key_pose[:3, 3] = [1.5, 1.8, 2.0]
        scene.add(key_light, pose=key_pose)

        # Fill light: 左上前方（補助光、少し弱め）
        fill_light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=base_intensity)
        fill_pose = np.eye(4)
        fill_pose[:3, 3] = [-1.5, 1.5, 2.0]
        scene.add(fill_light, pose=fill_pose)

        # Rim light: 後ろ上（背面への回り込み）
        rim_light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=base_intensity * 0.8)
        rim_pose = np.eye(4)
        rim_pose[:3, 3] = [0.0, 1.8, -2.0]
        scene.add(rim_light, pose=rim_pose)

        # 弱めのディレクショナルライト（窓からの光）
        sun = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=0.3)
        sun_pose = np.eye(4)
        # 少し下向き（X軸まわりに-30°回転）
        angle = np.deg2rad(-30.0)
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(angle), -np.sin(angle)],
            [0, np.sin(angle), np.cos(angle)],
        ])
        sun_pose[:3, :3] = Rx
        scene.add(sun, pose=sun_pose)

    def render(self, vertices, faces, camera_distance=3.0, view='front',
               mesh_color=None, background_color=None):
        """
        Render photorealistic RGB image.

        Args:
            vertices: Mesh vertices [N, 3]
            faces: Mesh faces [F, 3]
            camera_distance: Distance from camera to origin in meters
            view: Camera view ('front', 'back', 'side', 'left')
            mesh_color: Mesh color [R, G, B, A] (default: skin tone)
            background_color: Background color [R, G, B, A] (default: white)

        Returns:
            RGB image as numpy array [H, W, 3] with values in [0, 255]
        """
        # Create scene with low ambient light to prevent overexposure
        scene = pyrender.Scene(ambient_light=[0.05, 0.05, 0.05], bg_color=background_color or [0.8, 0.8, 0.8, 1.0])

        # Add mesh
        mesh = self.create_mesh(vertices, faces, color=mesh_color)
        scene.add(mesh)

        # Setup camera
        camera, camera_pose = self.setup_camera(camera_distance, view)
        scene.add(camera, pose=camera_pose)

        # Setup lighting
        self.setup_lighting(scene, camera_distance)

        # Render
        color, depth = self.renderer.render(scene)

        return color

    def render_multiview(self, vertices, faces, camera_distance=3.0,
                        views=None, save_dir=None, filename_prefix='render'):
        """
        Render multiple views and optionally save them.

        Args:
            vertices: Mesh vertices [N, 3]
            faces: Mesh faces [F, 3]
            camera_distance: Distance from camera to origin
            views: List of view names (default: ['front', 'side', 'back'])
            save_dir: Directory to save images (optional)
            filename_prefix: Prefix for saved filenames

        Returns:
            Dictionary mapping view names to RGB images
        """
        if views is None:
            views = ['front', 'side', 'back']

        results = {}

        for view in views:
            # Render view
            rgb_image = self.render(vertices, faces, camera_distance, view)
            results[view] = rgb_image

            # Save if directory specified
            if save_dir is not None:
                os.makedirs(save_dir, exist_ok=True)
                filename = os.path.join(save_dir, f"{filename_prefix}_{view}.png")
                Image.fromarray(rgb_image).save(filename)
                print(f"Saved {view} view to {filename}")

        return results

    def render_front_side(self, vertices, faces, camera_distance=3.0,
                         save_prefix=None):
        """
        Render front and side views side-by-side.

        Args:
            vertices: Mesh vertices [N, 3]
            faces: Mesh faces [F, 3]
            camera_distance: Distance from camera to origin
            save_prefix: Path prefix for saving (e.g., 'outputs/renders/avatar')

        Returns:
            Combined image with front and side views [H, W*2, 3]
        """
        # Render both views
        front_img = self.render(vertices, faces, camera_distance, 'front')
        side_img = self.render(vertices, faces, camera_distance, 'side')

        # Combine horizontally
        combined = np.concatenate([front_img, side_img], axis=1)

        # Save if prefix specified
        if save_prefix is not None:
            os.makedirs(os.path.dirname(save_prefix), exist_ok=True)
            filename = f"{save_prefix}_front_side.png"
            Image.fromarray(combined).save(filename)
            print(f"Saved combined front-side view to {filename}")

        return combined

    def __del__(self):
        """Cleanup renderer resources."""
        if hasattr(self, 'renderer'):
            self.renderer.delete()


if __name__ == "__main__":
    """
    Example usage: Render STAR model with photorealistic quality.
    """
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from models.star_layer import STARLayer
    import torch

    print("Initializing STAR model...")
    star = STARLayer(gender='neutral', num_betas=10)

    print("Initializing photorealistic renderer...")
    renderer = PhotorealisticRenderer(image_size=512, focal_length=50.0)

    # Generate random body shape
    print("Generating random body shape...")
    betas = torch.randn(1, 10) * 0.5
    vertices, joints = star(betas)
    faces = star.get_faces()

    # Render photorealistic images
    print("Rendering photorealistic views...")
    output_dir = "outputs/renders"
    os.makedirs(output_dir, exist_ok=True)

    # Render individual views
    views_dict = renderer.render_multiview(
        vertices=vertices[0].cpu().numpy(),
        faces=faces,
        camera_distance=3.0,
        views=['front', 'back', 'side', 'left'],
        save_dir=output_dir,
        filename_prefix='photorealistic'
    )

    # Render combined front-side view
    combined = renderer.render_front_side(
        vertices=vertices[0].cpu().numpy(),
        faces=faces,
        camera_distance=3.0,
        save_prefix=os.path.join(output_dir, 'photorealistic')
    )

    print("\nPhotorealistic rendering complete!")
    print(f"Output directory: {output_dir}")
    print("Generated files:")
    print("  - photorealistic_front.png")
    print("  - photorealistic_back.png")
    print("  - photorealistic_side.png")
    print("  - photorealistic_left.png")
    print("  - photorealistic_front_side.png")
```

---

### 2. ⭐⭐ `render_average_photorealistic.py`

**📄 ファイル情報**

- **パス**: `/Users/moei/program/STAR_Avatar/render_average_photorealistic.py`
- **サイズ**: 2.9 KB
- **行数**: 102 行
- **最終更新**: 2025-11-22 14:13
- **状態**: ✅ 存在

**📝 説明**

体型バリエーション、ポーズ、カメラアングル

**✨ 改善項目**

- 多様な体型バリエーション
- 自然なポーズ（手を下ろす、軽く曲げる）
- カメラアングルのバリエーション
- 画像のノイズ・ブラー追加

**💻 ソースコード**

```py
"""
Render average STAR body with photorealistic quality.

Generates photo-like RGB images of the average human body shape
using pyrender with PBR (Physically-Based Rendering).
"""

import torch
import os
from models.star_layer import STARLayer
from visualizations.photorealistic_renderer import PhotorealisticRenderer


def main():
    """Generate photorealistic images of average body shape."""

    print("="*60)
    print("Photorealistic Average Body Rendering")
    print("="*60)

    # Initialize STAR model
    print("\n1. Initializing STAR model...")
    star = STARLayer(gender='neutral', num_betas=10)

    # Initialize photorealistic renderer
    print("2. Initializing photorealistic renderer...")
    renderer = PhotorealisticRenderer(
        image_size=512,
        focal_length=50.0,  # 50mm standard lens
        sensor_width=36.0   # Full-frame sensor
    )

    # Generate average body (all beta parameters = 0)
    print("3. Generating average body shape (β = 0)...")
    betas = torch.zeros(1, 10)  # Average shape
    vertices, joints = star(betas)
    faces = star.get_faces()

    # Convert to numpy for rendering
    vertices_np = vertices[0].cpu().numpy()

    # Render photorealistic views
    print("4. Rendering photorealistic images...")
    output_dir = "outputs/renders"
    os.makedirs(output_dir, exist_ok=True)

    # Render all views
    print("   - Rendering front view...")
    renderer.render_multiview(
        vertices=vertices_np,
        faces=faces,
        camera_distance=3.0,
        views=['front'],
        save_dir=output_dir,
        filename_prefix='average_photorealistic'
    )

    print("   - Rendering side view...")
    renderer.render_multiview(
        vertices=vertices_np,
        faces=faces,
        camera_distance=3.0,
        views=['side'],
        save_dir=output_dir,
        filename_prefix='average_photorealistic'
    )

    print("   - Rendering back view...")
    renderer.render_multiview(
        vertices=vertices_np,
        faces=faces,
        camera_distance=3.0,
        views=['back'],
        save_dir=output_dir,
        filename_prefix='average_photorealistic'
    )

    # Render combined front-side view
    print("   - Rendering combined front-side view...")
    renderer.render_front_side(
        vertices=vertices_np,
        faces=faces,
        camera_distance=3.0,
        save_prefix=os.path.join(output_dir, 'average_photorealistic')
    )

    print("\n" + "="*60)
    print("✓ Photorealistic rendering complete!")
    print("="*60)
    print(f"\nOutput directory: {output_dir}")
    print("\nGenerated files:")
    print("  - average_photorealistic_front.png")
    print("  - average_photorealistic_side.png")
    print("  - average_photorealistic_back.png")
    print("  - average_photorealistic_front_side.png")
    print("\nThese images show the average human body shape with")
    print("photorealistic rendering (PBR materials + 3-point lighting).")
    print("="*60)


if __name__ == "__main__":
    main()
```

---

## 優先度2: 3Dモデルの改善

### 3. ⭐⭐ `models/star_layer.py`

**📄 ファイル情報**

- **パス**: `/Users/moei/program/STAR_Avatar/models/star_layer.py`
- **サイズ**: 9.2 KB
- **行数**: 263 行
- **最終更新**: 2025-11-22 10:03
- **状態**: ✅ 存在

**📝 説明**

STARモデルのパラメータ設定、メッシュ品質

**✨ 改善項目**

- より詳細なメッシュ（細分化）
- リアルな体型パラメータ範囲
- ポーズパラメータの拡張

**💻 ソースコード**

```py
"""
STAR Model Layer - PyTorch implementation
Based on: https://github.com/ahmedosman/STAR

STAR: Sparse Trained Articulated Human Body Regressor
A drop-in replacement for SMPL with improved deformations
"""
import torch
import torch.nn as nn
import numpy as np
import os
import pickle

class STARLayer(nn.Module):
    """
    STAR body model layer

    Args:
        gender: 'neutral', 'male', or 'female'
        num_betas: Number of shape parameters to use (default: 10, max: 300)
        model_path: Path to STAR .npz model file
    """
    def __init__(self, gender='neutral', num_betas=10, model_path=None):
        super(STARLayer, self).__init__()

        self.gender = gender
        self.num_betas = num_betas

        # Auto-detect model path if not provided
        if model_path is None:
            default_path = os.path.join(
                os.path.dirname(__file__), '..', 'data', 'star_models',
                gender, 'model.npz'
            )
            if os.path.exists(default_path):
                model_path = default_path

        # Try to load official STAR model
        if model_path and os.path.exists(model_path):
            self._load_official_model(model_path)
        else:
            print("⚠️  Official STAR model not found. Using minimal template.")
            self._create_minimal_model()

    def _load_official_model(self, model_path):
        """Load official STAR model from .npz file"""
        print(f"Loading STAR model from {model_path}...")

        model_data = np.load(model_path, allow_pickle=True)

        # Template mesh (mean shape in T-pose)
        # Shape: (6890, 3) for STAR
        self.register_buffer('v_template',
                           torch.tensor(model_data['v_template'], dtype=torch.float32))

        # Shape blend shapes: (6890, 3, num_betas)
        # Each beta parameter deforms the mesh
        shapedirs = model_data['shapedirs'][:, :, :self.num_betas]
        self.register_buffer('shapedirs',
                           torch.tensor(shapedirs, dtype=torch.float32))

        # Joint regressor: (24, 6890)
        # Computes 3D joint positions from vertices
        J_regressor = model_data['J_regressor']
        # Handle both sparse and dense matrices
        if hasattr(J_regressor, 'toarray'):
            J_regressor = J_regressor.toarray()
        self.register_buffer('J_regressor',
                           torch.tensor(J_regressor, dtype=torch.float32))

        # Skinning weights: (6890, 24)
        # LBS weights for each vertex to each joint
        self.register_buffer('weights',
                           torch.tensor(model_data['weights'], dtype=torch.float32))

        # Pose blend shapes: (6890, 3, 207)
        # Corrective shapes for pose deformations
        if 'posedirs' in model_data:
            self.register_buffer('posedirs',
                               torch.tensor(model_data['posedirs'], dtype=torch.float32))

        # Faces: triangle mesh connectivity
        self.register_buffer('faces',
                           torch.tensor(model_data['f'].astype(np.int64)))

        self.num_joints = self.J_regressor.shape[0]
        self.num_vertices = self.v_template.shape[0]

        print(f"✓ Loaded STAR {self.gender} model:")
        print(f"  - Vertices: {self.num_vertices}")
        print(f"  - Joints: {self.num_joints}")
        print(f"  - Shape parameters: {self.num_betas}")

    def _create_minimal_model(self):
        """Create a minimal simplified human mesh for testing"""
        print("Creating minimal human mesh template...")

        # Simplified human mesh (cylinder-based approximation)
        # In production, replace with actual STAR model

        # Create a simple humanoid mesh
        # This is a VERY simplified version - just for testing structure
        vertices, faces = self._create_simple_humanoid()

        self.num_vertices = len(vertices)
        self.num_joints = 24  # STAR/SMPL standard

        # Template vertices
        self.register_buffer('v_template',
                           torch.tensor(vertices, dtype=torch.float32))

        # Random shape directions for demo
        shapedirs = np.random.randn(self.num_vertices, 3, self.num_betas) * 0.01
        self.register_buffer('shapedirs',
                           torch.tensor(shapedirs, dtype=torch.float32))

        # Simple joint regressor (random for demo)
        J_regressor = np.zeros((self.num_joints, self.num_vertices))
        # Place joints at specific vertices
        joint_indices = np.linspace(0, self.num_vertices-1, self.num_joints, dtype=int)
        for i, idx in enumerate(joint_indices):
            J_regressor[i, idx] = 1.0
        self.register_buffer('J_regressor',
                           torch.tensor(J_regressor, dtype=torch.float32))

        # LBS weights (simplified)
        weights = np.zeros((self.num_vertices, self.num_joints))
        for i in range(self.num_vertices):
            closest_joint = joint_indices[np.argmin(np.abs(joint_indices - i))]
            joint_idx = np.where(joint_indices == closest_joint)[0][0]
            weights[i, joint_idx] = 1.0
        self.register_buffer('weights',
                           torch.tensor(weights, dtype=torch.float32))

        # Faces
        self.register_buffer('faces',
                           torch.tensor(faces, dtype=torch.int64))

        print(f"✓ Created minimal template:")
        print(f"  - Vertices: {self.num_vertices}")
        print(f"  - Faces: {len(faces)}")
        print(f"  - Joints: {self.num_joints}")
        print(f"  ⚠️  This is a simplified model. Download official STAR for accurate results.")

    def _create_simple_humanoid(self):
        """Create a very simple humanoid mesh (cylinder-based)"""
        # This is a placeholder - creates a simple cylindrical mesh
        # In production, use actual STAR model

        n_rings = 20
        n_segments = 16
        height = 1.7  # meters
        radius = 0.15

        vertices = []

        for i in range(n_rings):
            y = (i / (n_rings - 1)) * height - height/2  # Center at origin
            r = radius * (1.0 - 0.3 * abs(2*i/(n_rings-1) - 1))  # Taper at ends

            for j in range(n_segments):
                angle = 2 * np.pi * j / n_segments
                x = r * np.cos(angle)
                z = r * np.sin(angle)
                vertices.append([x, y, z])

        vertices = np.array(vertices, dtype=np.float32)

        # Create faces
        faces = []
        for i in range(n_rings - 1):
            for j in range(n_segments):
                # Two triangles per quad
                v1 = i * n_segments + j
                v2 = i * n_segments + (j + 1) % n_segments
                v3 = (i + 1) * n_segments + j
                v4 = (i + 1) * n_segments + (j + 1) % n_segments

                faces.append([v1, v2, v3])
                faces.append([v2, v4, v3])

        faces = np.array(faces, dtype=np.int32)

        return vertices, faces

    def forward(self, betas, pose=None, trans=None):
        """
        Forward pass: generate mesh from parameters

        Args:
            betas: Shape parameters [batch_size, num_betas]
            pose: Pose parameters [batch_size, 72] (optional, defaults to T-pose)
            trans: Translation [batch_size, 3] (optional)

        Returns:
            vertices: [batch_size, num_vertices, 3]
            joints: [batch_size, num_joints, 3]
        """
        batch_size = betas.shape[0]

        # Apply shape blend shapes
        # v_shaped = v_template + sum(beta_i * shapedir_i)
        v_shaped = self.v_template.unsqueeze(0).expand(batch_size, -1, -1)

        # Add shape deformations
        for i in range(min(betas.shape[1], self.num_betas)):
            # shapedirs: [num_vertices, 3, num_betas]
            # betas[:, i]: [batch_size]
            v_shaped = v_shaped + betas[:, i].view(batch_size, 1, 1) * self.shapedirs[:, :, i]

        # Compute joints
        # J: [batch_size, num_joints, 3]
        joints = torch.einsum('bvk,jv->bjk', v_shaped, self.J_regressor)

        # For now, skip pose deformations (T-pose only)
        # In full implementation, apply LBS with pose parameters
        vertices = v_shaped

        # Apply translation if provided
        if trans is not None:
            vertices = vertices + trans.view(batch_size, 1, 3)
            joints = joints + trans.view(batch_size, 1, 3)

        return vertices, joints

    def get_faces(self):
        """Return face indices"""
        return self.faces.cpu().numpy()


def test_star_layer():
    """Quick test of STAR layer"""
    print("\n" + "="*60)
    print("Testing STAR Layer")
    print("="*60 + "\n")

    # Create STAR layer
    star = STARLayer(gender='neutral', num_betas=10)

    # Random beta parameters
    batch_size = 1
    betas = torch.randn(batch_size, 10) * 0.5  # Small random variations

    print(f"\nInput beta parameters: {betas.shape}")
    print(f"Beta values: {betas[0].numpy()}")

    # Generate mesh
    vertices, joints = star(betas)

    print(f"\nOutput:")
    print(f"  - Vertices: {vertices.shape}")
    print(f"  - Joints: {joints.shape}")
    print(f"  - Vertex range: [{vertices.min().item():.3f}, {vertices.max().item():.3f}]")

    # Get faces
    faces = star.get_faces()
    print(f"  - Faces: {faces.shape}")

    return star, vertices, joints


if __name__ == "__main__":
    test_star_layer()
```

---

## 優先度3: データ生成

### 4. ⭐ `data/synthetic_dataset.py`

**📄 ファイル情報**

- **パス**: `/Users/moei/program/STAR_Avatar/data/synthetic_dataset.py`
- **サイズ**: 9.8 KB
- **行数**: 285 行
- **最終更新**: 2025-11-22 17:34
- **状態**: ✅ 存在

**📝 説明**

データ生成のバリエーション追加

**✨ 改善項目**

- より多様なサンプル生成
- データ拡張の強化
- リアリティ向上のための後処理

**💻 ソースコード**

```py
"""
PyTorch Dataset for Synthetic STAR Training Data

Loads synthetic data generated by generate_synthetic_data.py:
- Multi-channel front/back views (Normal + Depth + Joints + Mask)
- Ground truth beta (shape parameters)
- Ground truth T (global translation)
- Optional user attributes (height, weight, gender)
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class SyntheticDataset(Dataset):
    """
    Dataset for synthetic STAR body data.

    Directory structure expected:
        outputs/synthetic_data/
        ├── sample_1/
        │   ├── front_normal.png
        │   ├── front_depth.png
        │   ├── front_joints_heatmap.png
        │   ├── front_mask.png
        │   ├── back_normal.png
        │   ├── back_depth.png
        │   ├── back_joints_heatmap.png
        │   ├── back_mask.png
        │   ├── beta_gt.npy (shape: [10])
        │   └── T_gt.npy (shape: [3])
        ├── sample_2/
        └── ...
    """

    def __init__(self, data_dir='outputs/synthetic_data', transform=None,
                 use_attributes=False, image_size=512):
        """
        Initialize synthetic dataset.

        Args:
            data_dir: Root directory containing sample folders
            transform: Optional transform to apply to images
            use_attributes: Whether to include user attributes (height/weight/gender)
            image_size: Expected image size (default: 512)
        """
        self.data_dir = data_dir
        self.transform = transform
        self.use_attributes = use_attributes
        self.image_size = image_size

        # Get list of sample directories
        self.samples = []
        if os.path.exists(data_dir):
            for item in sorted(os.listdir(data_dir)):
                sample_path = os.path.join(data_dir, item)
                if os.path.isdir(sample_path) and item.startswith('sample_'):
                    # Check if all required files exist
                    if self._validate_sample(sample_path):
                        self.samples.append(sample_path)

        if len(self.samples) == 0:
            raise ValueError(f"No valid samples found in {data_dir}")

        print(f"Loaded {len(self.samples)} samples from {data_dir}")

    def _validate_sample(self, sample_path):
        """Check if sample has all required files."""
        required_files = [
            'front_normal.png',
            'front_depth.png',
            'front_joints_heatmap.png',
            'front_mask.png',
            'back_normal.png',
            'back_depth.png',
            'back_joints_heatmap.png',
            'back_mask.png',
            'beta_gt.npy',
            'T_gt.npy'
        ]

        for filename in required_files:
            if not os.path.exists(os.path.join(sample_path, filename)):
                return False
        return True

    def __len__(self):
        return len(self.samples)

    def _load_image(self, path):
        """Load image and convert to numpy array."""
        img = Image.open(path)
        return np.array(img, dtype=np.float32) / 255.0  # Normalize to [0, 1]

    def _load_multichannel_view(self, sample_path, view='front'):
        """
        Load and stack multi-channel input for one view.

        Args:
            sample_path: Path to sample directory
            view: 'front' or 'back'

        Returns:
            Tensor of shape [21, H, W]:
                - Normal map: channels 0-2 (RGB)
                - Depth map: channel 3 (grayscale)
                - Mask: channel 4 (grayscale)
                - Joint heatmaps: channels 5-20 (16 joints, grayscale each)
        """
        # Load normal map (RGB, 3 channels)
        normal_path = os.path.join(sample_path, f'{view}_normal.png')
        normal = self._load_image(normal_path)  # [H, W, 3]

        # Load depth map (grayscale, 1 channel)
        depth_path = os.path.join(sample_path, f'{view}_depth.png')
        depth = self._load_image(depth_path)  # [H, W] or [H, W, 1]
        if depth.ndim == 3:
            depth = depth[:, :, 0]  # Take first channel if RGB
        depth = np.expand_dims(depth, axis=2)  # [H, W, 1]

        # Load mask (grayscale, 1 channel)
        mask_path = os.path.join(sample_path, f'{view}_mask.png')
        mask = self._load_image(mask_path)  # [H, W] or [H, W, 1]
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        mask = np.expand_dims(mask, axis=2)  # [H, W, 1]

        # Load joint heatmaps (16 channels stacked as RGB image)
        joints_path = os.path.join(sample_path, f'{view}_joints_heatmap.png')
        joints_img = self._load_image(joints_path)  # [H, W] or [H, W, 3]

        # The joints heatmap was saved as a visualization (grayscale or RGB)
        # For training, we need the actual 16-channel heatmaps
        # Since we saved it as visualization, we'll use grayscale as proxy
        # TODO: Modify generate_synthetic_data.py to save individual heatmaps
        # For now, replicate the grayscale across 16 channels
        if joints_img.ndim == 3:
            joints_gray = np.mean(joints_img, axis=2, keepdims=True)  # [H, W, 1]
        else:
            joints_gray = np.expand_dims(joints_img, axis=2)  # [H, W, 1]
        joints = np.repeat(joints_gray, 16, axis=2)  # [H, W, 16]

        # Stack all channels: Normal(3) + Depth(1) + Mask(1) + Joints(16) = 21
        multichannel = np.concatenate([normal, depth, mask, joints], axis=2)  # [H, W, 21]

        # Convert to torch tensor and change to [C, H, W] format
        multichannel = torch.from_numpy(multichannel).float()  # [H, W, 21]
        multichannel = multichannel.permute(2, 0, 1)  # [21, H, W]

        return multichannel

    def __getitem__(self, idx):
        """
        Get one sample.

        Returns:
            dict with keys:
                - 'front_input': Tensor [21, H, W]
                - 'back_input': Tensor [21, H, W]
                - 'beta_gt': Tensor [10]
                - 'T_gt': Tensor [3]
                - 'attr_input': Optional Tensor [3] (if use_attributes=True)
        """
        sample_path = self.samples[idx]

        # Load multi-channel inputs
        front_input = self._load_multichannel_view(sample_path, 'front')
        back_input = self._load_multichannel_view(sample_path, 'back')

        # Load ground truth
        beta_gt = np.load(os.path.join(sample_path, 'beta_gt.npy'))
        T_gt = np.load(os.path.join(sample_path, 'T_gt.npy'))

        beta_gt = torch.from_numpy(beta_gt).float().squeeze()  # Remove batch dim if present
        T_gt = torch.from_numpy(T_gt).float()

        # Apply transform if provided
        if self.transform is not None:
            # Apply same transform to both views
            # Note: transform should handle multi-channel input
            front_input = self.transform(front_input)
            back_input = self.transform(back_input)

        result = {
            'front_input': front_input,
            'back_input': back_input,
            'beta_gt': beta_gt,
            'T_gt': T_gt
        }

        # Add user attributes if requested
        if self.use_attributes:
            # For synthetic data, we can generate random attributes
            # or load from file if available
            # Format: [height_ratio, weight_ratio, gender]
            # For now, generate random attributes
            height_ratio = torch.rand(1) * 0.3 + 0.85  # [0.85, 1.15]
            weight_ratio = torch.rand(1) * 0.3 + 0.85  # [0.85, 1.15]
            gender = torch.randint(0, 2, (1,)).float()  # 0 or 1

            attr_input = torch.cat([height_ratio, weight_ratio, gender])
            result['attr_input'] = attr_input

        return result


def test_dataset():
    """Test the synthetic dataset."""
    print("="*60)
    print("Testing SyntheticDataset")
    print("="*60)

    # Check if synthetic data exists
    data_dir = 'outputs/synthetic_data'
    if not os.path.exists(data_dir):
        print(f"\nError: {data_dir} does not exist")
        print("Please run generate_synthetic_data.py first to create synthetic data")
        return

    # Create dataset
    try:
        dataset = SyntheticDataset(
            data_dir=data_dir,
            use_attributes=True
        )
    except ValueError as e:
        print(f"\nError: {e}")
        print("Please run generate_synthetic_data.py first to create synthetic data")
        return

    print(f"\nDataset size: {len(dataset)}")

    # Test loading one sample
    print("\nLoading first sample...")
    sample = dataset[0]

    print(f"\nSample keys: {sample.keys()}")
    print(f"\nShapes:")
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")

    print(f"\nValue ranges:")
    print(f"  front_input: [{sample['front_input'].min():.3f}, {sample['front_input'].max():.3f}]")
    print(f"  back_input: [{sample['back_input'].min():.3f}, {sample['back_input'].max():.3f}]")
    print(f"  beta_gt: {sample['beta_gt'].numpy()}")
    print(f"  T_gt: {sample['T_gt'].numpy()}")
    if 'attr_input' in sample:
        print(f"  attr_input: {sample['attr_input'].numpy()}")

    # Test DataLoader
    print("\n" + "-"*60)
    print("Testing with DataLoader")
    print("-"*60)

    from torch.utils.data import DataLoader

    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0
    )

    print(f"\nDataLoader batch count: {len(dataloader)}")

    # Load one batch
    batch = next(iter(dataloader))

    print(f"\nBatch shapes:")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")

    print("\n" + "="*60)
    print("✓ Dataset test passed!")
    print("="*60)


if __name__ == "__main__":
    test_dataset()
```

---

## 🎯 推奨修正順序

以下の順序で修正することを推奨します：

### 1. `visualizations/photorealistic_renderer.py` (最優先)

**修正箇所**:
- 行60-80: 照明システム（3点照明→HDRI環境照明）
- 行120-140: 背景設定（白背景→グラデーション/環境マップ）
- 行85-100: PBRマテリアル設定
- 行40-55: カメラパラメータ（焦点距離、DOF）
- 行25-30: 解像度設定（512→1024以上）

### 2. `models/star_layer.py`

**修正箇所**:
- メッシュ細分化オプションの追加
- 体型パラメータβの範囲拡張

### 3. `render_average_photorealistic.py`

**修正箇所**:
- 多様な体型でのレンダリング追加
- 自然なポーズバリエーション
- カメラアングルの多様化

### 4. `data/synthetic_dataset.py`

**修正箇所**:
- データ生成の多様化
- 後処理パイプラインの追加

