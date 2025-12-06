"""
Photorealistic RGB Renderer for STAR/MHR Body Meshes

Uses pyrender with PBR (Physically-Based Rendering) for photo-like outputs.
This renderer is optimized for generating realistic RGB images similar to photographs.

Camera parameters are compatible with SAM 3D Body / HMR format:
- Weak perspective: scale, tx, ty
- Full perspective: intrinsic matrix K, extrinsic matrix [R|t]
"""

import numpy as np
import pyrender
import trimesh
from PIL import Image
import os

from src.rendering.camera import CameraParams, create_sam3db_camera


class PhotorealisticRenderer:
    """
    Photorealistic renderer using pyrender with PBR materials.

    Produces high-quality RGB images that look like real photographs.
    Uses 3-point lighting setup and smooth shading for realistic appearance.

    Camera parameters follow SAM 3D Body / HMR conventions:
    - Default focal length: 5000 pixels (HMR standard)
    - Weak perspective output: scale = f / z
    """

    # HMR standard focal length (pixels)
    HMR_FOCAL_LENGTH = 5000.0

    def __init__(self, image_width=512, image_height=None, focal_length_pixels=None,
                 use_hmr_camera=True, camera_preset=None):
        """
        Initialize photorealistic renderer with SAM 3D Body compatible camera.

        Args:
            image_width: Output image width in pixels
            image_height: Output image height in pixels (defaults to image_width for square)
            focal_length_pixels: Focal length in pixels (default: 5000 for HMR standard)
            use_hmr_camera: If True, use HMR standard focal length (5000 pixels)
            camera_preset: Camera preset name ('iphone_1080p', 'iphone_4k', etc.)
        """
        # Handle iPhone presets
        if camera_preset is not None:
            from src.rendering.camera import IPHONE_PRESETS
            if camera_preset in IPHONE_PRESETS:
                p = IPHONE_PRESETS[camera_preset]
                image_width = p['width']
                image_height = p['height']
                focal_length_pixels = p['fx']
                use_hmr_camera = False

        self.image_width = image_width
        self.image_height = image_height if image_height is not None else image_width
        self.image_size = self.image_width  # For backward compatibility

        # Use HMR standard focal length by default
        if focal_length_pixels is None:
            if use_hmr_camera:
                focal_length_pixels = self.HMR_FOCAL_LENGTH
            else:
                # Legacy: 50mm on 36mm sensor
                focal_length_pixels = (50.0 / 36.0) * image_width

        self.focal_length_pixels = focal_length_pixels
        self.fx = focal_length_pixels
        self.fy = focal_length_pixels
        self.cx = self.image_width / 2.0
        self.cy = self.image_height / 2.0

        # Create pyrender renderer (offscreen for server/headless environments)
        self.renderer = pyrender.OffscreenRenderer(self.image_width, self.image_height)

        # Camera params helper (will be set per-render with distance)
        self._camera_params = None
        self.camera_preset = camera_preset

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
        # Default skin-like color (natural peachy skin tone)
        if color is None:
            color = [0.85, 0.70, 0.60, 1.0]  # Natural peachy skin tone

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

    def setup_camera(self, distance=3.0, view='front', look_at_y=0.1):
        """
        Setup camera position and orientation.

        Args:
            distance: Camera distance from origin in meters
            view: Camera view ('front', 'back', 'side'/'right', 'left')
            look_at_y: Y-coordinate of the point camera looks at (negative moves person down in image)

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

    def get_camera_params(self, camera_distance: float, azimuth: float = 0.0) -> dict:
        """
        Get SAM 3D Body / HMR compatible camera parameters

        Args:
            camera_distance: Distance from camera to subject in meters
            azimuth: View angle in degrees (0 = front)

        Returns:
            dict with camera parameters in SAM 3D Body format
        """
        camera_params = CameraParams(
            focal_length_pixels=self.focal_length_pixels,
            image_width=self.image_width,
            image_height=self.image_height,
            camera_distance=camera_distance,
            principal_point=(self.cx, self.cy)
        )

        return camera_params.get_full_camera_dict(azimuth=azimuth, elevation=0.0)

    def render(self, vertices, faces, camera_distance=3.0, view='front',
               mesh_color=None, background_color=None, return_camera_params=False):
        """
        Render photorealistic RGB image.

        Args:
            vertices: Mesh vertices [N, 3]
            faces: Mesh faces [F, 3]
            camera_distance: Distance from camera to origin in meters
            view: Camera view ('front', 'back', 'side', 'left')
            mesh_color: Mesh color [R, G, B, A] (default: skin tone)
            background_color: Background color [R, G, B, A] (default: white)
            return_camera_params: If True, also return camera parameters

        Returns:
            RGB image as numpy array [H, W, 3] with values in [0, 255]
            If return_camera_params=True: (image, camera_params_dict)
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

        if return_camera_params:
            # Convert view to azimuth for camera params
            view_to_azimuth = {
                'front': 0.0,
                'right': 90.0,
                'side': 90.0,
                'back': 180.0,
                'left': 270.0
            }
            azimuth = view_to_azimuth.get(view, 0.0)
            camera_params = self.get_camera_params(camera_distance, azimuth)
            return color, camera_params

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


class EnhancedPhotorealisticRenderer:
    """
    Enhanced photorealistic renderer with HDRI-style lighting and realistic backgrounds.

    Improvements over PhotorealisticRenderer:
    - HDRI-style multi-light setup (ambient + sun + multiple fill lights)
    - Realistic background options (gradient, solid color, varied, HDRI images)
    - Higher default resolution (1024px)
    - Better material settings for realism
    - Randomized lighting for diversity
    - HDRI background compositing for Sapiens-ready outputs
    """

    def __init__(self, image_size=1024, focal_length=50.0, sensor_width=36.0,
                 use_hdri_lighting=True, background_mode='gradient',
                 hdri_background_manager=None, clothing_generator=None):
        """
        Initialize enhanced photorealistic renderer.

        Args:
            image_size: Output image resolution (default: 1024)
            focal_length: Camera focal length in mm
            sensor_width: Camera sensor width in mm
            use_hdri_lighting: Use HDRI-style lighting (default: True)
            background_mode: 'gradient', 'solid', 'varied', or 'hdri' (default: 'gradient')
            hdri_background_manager: HDRIBackgroundManager instance (optional)
            clothing_generator: SimpleClothingGenerator instance (optional)
        """
        self.image_size = image_size
        self.focal_length = focal_length
        self.sensor_width = sensor_width
        self.use_hdri_lighting = use_hdri_lighting
        self.background_mode = background_mode
        self.hdri_background_manager = hdri_background_manager
        self.clothing_generator = clothing_generator

        # Calculate focal length in pixels
        self.fx = (focal_length / sensor_width) * image_size
        self.fy = self.fx
        self.cx = image_size / 2.0
        self.cy = image_size / 2.0

        # Create pyrender renderer
        self.renderer = pyrender.OffscreenRenderer(image_size, image_size)

    def create_mesh(self, vertices, faces, color=None):
        """Create a pyrender mesh with realistic PBR skin material."""
        # Realistic human skin tones (linear sRGB values from physicallybased.info)
        if color is None:
            # Six skin tone categories representing diverse ethnicities
            SKIN_TONES = [
                [0.847, 0.638, 0.552],  # Skin I (lightest)
                [0.799, 0.485, 0.347],  # Skin II
                [0.623, 0.433, 0.343],  # Skin III
                [0.436, 0.227, 0.131],  # Skin IV
                [0.283, 0.148, 0.079],  # Skin V
                [0.090, 0.050, 0.020],  # Skin VI (darkest)
            ]

            # Randomly select a skin tone category
            base_skin = SKIN_TONES[np.random.randint(0, len(SKIN_TONES))]

            # Add slight variation within the category (±3%)
            r = np.clip(base_skin[0] + np.random.uniform(-0.03, 0.03), 0.0, 1.0)
            g = np.clip(base_skin[1] + np.random.uniform(-0.03, 0.03), 0.0, 1.0)
            b = np.clip(base_skin[2] + np.random.uniform(-0.03, 0.03), 0.0, 1.0)

            color = [r, g, b, 1.0]

        # Convert to numpy if torch tensor
        if hasattr(vertices, 'cpu'):
            vertices = vertices.cpu().numpy()
        if hasattr(faces, 'cpu'):
            faces = faces.cpu().numpy()

        # Create trimesh object
        tri_mesh = trimesh.Trimesh(
            vertices=vertices,
            faces=faces,
            process=True
        )

        # Realistic PBR material for skin
        # Roughness: 0.5-0.7 for skin (not too glossy, not too matte)
        # Metallic: 0.0 (skin is dielectric, not metallic)
        roughness = 0.55 + np.random.uniform(-0.05, 0.15)  # 0.5-0.7 range
        material = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=color,
            metallicFactor=0.0,
            roughnessFactor=roughness,
            alphaMode='OPAQUE'
        )

        mesh = pyrender.Mesh.from_trimesh(tri_mesh, material=material, smooth=True)
        return mesh

    def setup_camera(self, distance=3.0, view='front', look_at_y=-0.4):
        """Setup camera with optional randomization."""
        # Create perspective camera
        camera = pyrender.IntrinsicsCamera(
            fx=self.fx,
            fy=self.fy,
            cx=self.cx,
            cy=self.cy,
            znear=0.1,
            zfar=100.0
        )

        def look_at(eye, target, up):
            """Create a look-at view matrix."""
            forward = target - eye
            forward = forward / np.linalg.norm(forward)

            right = np.cross(forward, up)
            right = right / np.linalg.norm(right)

            up_new = np.cross(right, forward)

            mat = np.eye(4)
            mat[:3, 0] = right
            mat[:3, 1] = up_new
            mat[:3, 2] = -forward
            mat[:3, 3] = eye

            return mat

        # Camera position based on view
        target = np.array([0.0, look_at_y, 0.0])

        if view == 'front':
            camera_pos = np.array([0.0, 0.0, distance])
            up = np.array([0.0, 1.0, 0.0])
            camera_pose = look_at(camera_pos, target, up)
        elif view == 'back':
            camera_pos = np.array([0.0, 0.0, -distance])
            up = np.array([0.0, 1.0, 0.0])
            camera_pose = look_at(camera_pos, target, up)
        elif view in ['side', 'right']:
            camera_pos = np.array([distance, 0.0, 0.0])
            up = np.array([0.0, 1.0, 0.0])
            camera_pose = look_at(camera_pos, target, up)
        elif view == 'left':
            camera_pos = np.array([-distance, 0.0, 0.0])
            up = np.array([0.0, 1.0, 0.0])
            camera_pose = look_at(camera_pos, target, up)
        else:
            raise ValueError(f"Unknown view: {view}")

        return camera, camera_pose

    def setup_hdri_lighting(self, scene):
        """
        Setup HDRI-style lighting with ambient + sun + multiple fill lights.

        This simulates realistic outdoor/studio lighting conditions.
        """
        # Base intensity for realistic lighting
        base_intensity = 6.0

        # Main sun light (directional) - stronger, from above-front
        sun = pyrender.DirectionalLight(color=[1.0, 0.98, 0.95], intensity=0.5)
        sun_pose = np.eye(4)
        angle_x = np.deg2rad(-35.0)  # Down from horizon
        angle_y = np.deg2rad(25.0)   # Slight rotation
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(angle_x), -np.sin(angle_x)],
            [0, np.sin(angle_x), np.cos(angle_x)],
        ])
        Ry = np.array([
            [np.cos(angle_y), 0, np.sin(angle_y)],
            [0, 1, 0],
            [-np.sin(angle_y), 0, np.cos(angle_y)],
        ])
        sun_pose[:3, :3] = Rx @ Ry
        scene.add(sun, pose=sun_pose)

        # Key light (main fill from front-right)
        key_light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=base_intensity * 1.8)
        key_pose = np.eye(4)
        key_pose[:3, 3] = [2.0, 2.0, 2.5]
        scene.add(key_light, pose=key_pose)

        # Fill light (softer, from front-left)
        fill_light = pyrender.PointLight(color=[1.0, 0.98, 0.96], intensity=base_intensity * 1.2)
        fill_pose = np.eye(4)
        fill_pose[:3, 3] = [-1.8, 1.5, 2.0]
        scene.add(fill_light, pose=fill_pose)

        # Back light (rim light from behind)
        back_light = pyrender.PointLight(color=[0.95, 0.97, 1.0], intensity=base_intensity * 1.0)
        back_pose = np.eye(4)
        back_pose[:3, 3] = [0.0, 2.0, -2.5]
        scene.add(back_light, pose=back_pose)

        # Side fill lights for better coverage
        side_right = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=base_intensity * 0.8)
        side_right_pose = np.eye(4)
        side_right_pose[:3, 3] = [2.5, 1.0, 0.0]
        scene.add(side_right, pose=side_right_pose)

        side_left = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=base_intensity * 0.8)
        side_left_pose = np.eye(4)
        side_left_pose[:3, 3] = [-2.5, 1.0, 0.0]
        scene.add(side_left, pose=side_left_pose)

    def get_background_color(self):
        """Get background color based on mode."""
        if self.background_mode == 'gradient':
            # Gradient-like effect by using a light gray
            # (True gradients would need post-processing)
            gray_value = np.random.uniform(0.75, 0.90)
            return [gray_value, gray_value, gray_value, 1.0]

        elif self.background_mode == 'solid':
            # Light neutral background
            return [0.85, 0.85, 0.85, 1.0]

        elif self.background_mode == 'varied':
            # Random backgrounds (light colors to avoid harsh contrast)
            mode = np.random.choice(['light_gray', 'warm', 'cool'])

            if mode == 'light_gray':
                gray = np.random.uniform(0.75, 0.92)
                return [gray, gray, gray, 1.0]
            elif mode == 'warm':
                return [0.88, 0.85, 0.82, 1.0]
            else:  # cool
                return [0.82, 0.85, 0.88, 1.0]

        else:
            return [0.85, 0.85, 0.85, 1.0]

    def render(self, vertices, faces, camera_distance=3.0, view='front',
               mesh_color=None, background_color=None, use_hdri_background=False,
               add_clothing=False, clothing_types=None):
        """
        Render enhanced photorealistic RGB image.

        Args:
            vertices: Mesh vertices [N, 3]
            faces: Mesh faces [F, 3]
            camera_distance: Distance from camera to origin
            view: Camera view ('front', 'back', 'side', 'left')
            mesh_color: Mesh color [R, G, B, A]
            background_color: Background color (default: auto from mode)
            use_hdri_background: If True, composite with HDRI background image
            add_clothing: If True, add simple clothing to body (default: False)
            clothing_types: List of clothing types (default: ['shorts', 'sports_bra'])

        Returns:
            RGB image as numpy array [H, W, 3] with values in [0, 255]
        """
        # Add clothing if requested (do this BEFORE any rendering)
        clothing_masks = None
        if add_clothing and self.clothing_generator is not None:
            if clothing_types is None:
                clothing_types = ['shorts', 'sports_bra']
            vertices, clothing_masks = self.clothing_generator.add_clothing(
                vertices.copy(),
                clothing_types=clothing_types,
                return_masks=True
            )

        # If HDRI background requested, render with transparency
        # Note: vertices already have clothing if add_clothing=True
        if use_hdri_background and self.hdri_background_manager is not None:
            return self.render_with_hdri_background(
                vertices, faces, camera_distance, view, mesh_color,
                clothing_masks
            )

        # Get background color
        if background_color is None:
            background_color = self.get_background_color()

        # Create scene with appropriate ambient light
        ambient_intensity = 0.08 if self.use_hdri_lighting else 0.05
        scene = pyrender.Scene(
            ambient_light=[ambient_intensity] * 3,
            bg_color=background_color
        )

        # Add mesh (with clothing colors if applicable)
        if clothing_masks is not None:
            # Render clothing and skin as separate meshes with different colors
            self._add_clothed_mesh(scene, vertices, faces, clothing_masks, mesh_color)
        else:
            # Render as single mesh
            mesh = self.create_mesh(vertices, faces, color=mesh_color)
            scene.add(mesh)

        # Setup camera
        camera, camera_pose = self.setup_camera(camera_distance, view)
        scene.add(camera, pose=camera_pose)

        # Setup lighting
        if self.use_hdri_lighting:
            self.setup_hdri_lighting(scene)
        else:
            # Fall back to standard 3-point lighting
            self._setup_standard_lighting(scene)

        # Render
        color, depth = self.renderer.render(scene)

        return color

    def _add_clothed_mesh(self, scene, vertices, faces, clothing_masks, base_color=None):
        """
        Add clothed mesh with different colors for clothing regions.

        Args:
            scene: pyrender Scene
            vertices: Mesh vertices [N, 3]
            faces: Mesh faces [F, 3]
            clothing_masks: Dict mapping clothing types to vertex masks
            base_color: Base skin color (optional)
        """
        # Define clothing colors (athletic wear colors)
        clothing_colors = {
            'shorts': [0.15, 0.15, 0.20, 1.0],       # Dark gray/black shorts
            'sports_bra': [0.25, 0.25, 0.30, 1.0],   # Charcoal gray sports bra
            'tank_top': [0.30, 0.35, 0.40, 1.0],     # Medium gray tank top
            'leggings': [0.10, 0.10, 0.15, 1.0],     # Black leggings
        }

        # Create combined clothing mask
        clothing_vertex_mask = np.zeros(len(vertices), dtype=bool)
        for mask in clothing_masks.values():
            clothing_vertex_mask |= mask

        # Determine which faces are clothing vs skin
        # A face is "clothing" if at least 2 of its 3 vertices are clothing
        faces_clothing_count = np.sum(clothing_vertex_mask[faces], axis=1)
        clothing_faces_mask = faces_clothing_count >= 2

        # Split faces into skin and clothing
        skin_faces = faces[~clothing_faces_mask]
        clothing_faces = faces[clothing_faces_mask]

        # Add skin mesh
        if len(skin_faces) > 0:
            skin_mesh = self.create_mesh(vertices, skin_faces, color=base_color)
            scene.add(skin_mesh)

        # Add clothing mesh with darker color
        if len(clothing_faces) > 0:
            # Use first clothing type's color
            first_clothing_type = list(clothing_masks.keys())[0]
            clothing_color = clothing_colors.get(first_clothing_type, [0.2, 0.2, 0.25, 1.0])

            clothing_mesh = self.create_mesh(vertices, clothing_faces, color=clothing_color)
            scene.add(clothing_mesh)

    def render_with_hdri_background(self, vertices, faces, camera_distance=3.0,
                                    view='front', mesh_color=None,
                                    clothing_masks=None):
        """
        Render with HDRI background compositing.

        This method renders the mesh with transparent background (RGBA),
        then composites it with a random HDRI background image.

        Note: Clothing should already be added to vertices before calling this method.

        Args:
            vertices: Mesh vertices [N, 3] (may already include clothing)
            faces: Mesh faces [F, 3]
            camera_distance: Distance from camera to origin
            view: Camera view ('front', 'back', 'side', 'left')
            mesh_color: Mesh color [R, G, B, A]

        Returns:
            RGB image composited with HDRI background [H, W, 3]
        """
        if self.hdri_background_manager is None:
            raise RuntimeError(
                "HDRIBackgroundManager not provided. "
                "Pass hdri_background_manager to constructor."
            )

        # Create scene with transparent background
        ambient_intensity = 0.08 if self.use_hdri_lighting else 0.05
        scene = pyrender.Scene(
            ambient_light=[ambient_intensity] * 3,
            bg_color=[0.0, 0.0, 0.0, 0.0]  # Transparent background
        )

        # Add mesh (with clothing colors if applicable)
        if clothing_masks is not None:
            # Render clothing and skin as separate meshes with different colors
            self._add_clothed_mesh(scene, vertices, faces, clothing_masks, mesh_color)
        else:
            # Render as single mesh
            mesh = self.create_mesh(vertices, faces, color=mesh_color)
            scene.add(mesh)

        # Setup camera
        camera, camera_pose = self.setup_camera(camera_distance, view)
        scene.add(camera, pose=camera_pose)

        # Setup lighting
        if self.use_hdri_lighting:
            self.setup_hdri_lighting(scene)
        else:
            self._setup_standard_lighting(scene)

        # Render with alpha channel
        # Note: pyrender.OffscreenRenderer.render() returns RGB only
        # We need to use flags to get RGBA
        try:
            color, depth = self.renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        except:
            # Fallback: render RGB and create alpha from depth
            color, depth = self.renderer.render(scene)

            # Create alpha channel from depth (non-zero depth = foreground)
            alpha = (depth > 0).astype(np.uint8) * 255
            alpha = alpha[:, :, np.newaxis]  # Add channel dimension

            # Combine to RGBA
            rgba = np.concatenate([color, alpha], axis=2)
        else:
            rgba = color  # Already RGBA from render flags

        # Get random HDRI background
        background = self.hdri_background_manager.get_random_background(
            target_size=(self.image_size, self.image_size)
        )

        # Composite
        composited = self.hdri_background_manager.composite_rgba_with_background(
            rgba, background
        )

        return composited

    def _setup_standard_lighting(self, scene):
        """Standard 3-point lighting (fallback)."""
        base_intensity = 5.0

        key_light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=base_intensity * 1.5)
        key_pose = np.eye(4)
        key_pose[:3, 3] = [1.5, 1.8, 2.0]
        scene.add(key_light, pose=key_pose)

        fill_light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=base_intensity)
        fill_pose = np.eye(4)
        fill_pose[:3, 3] = [-1.5, 1.5, 2.0]
        scene.add(fill_light, pose=fill_pose)

        rim_light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=base_intensity * 0.8)
        rim_pose = np.eye(4)
        rim_pose[:3, 3] = [0.0, 1.8, -2.0]
        scene.add(rim_light, pose=rim_pose)

    def render_multiview(self, vertices, faces, camera_distance=3.0,
                        views=None, save_dir=None, filename_prefix='render',
                        use_hdri_background=False, add_clothing=False,
                        clothing_types=None):
        """Render multiple views."""
        if views is None:
            views = ['front', 'side', 'back']

        results = {}

        for view in views:
            rgb_image = self.render(
                vertices, faces, camera_distance, view,
                use_hdri_background=use_hdri_background,
                add_clothing=add_clothing,
                clothing_types=clothing_types
            )
            results[view] = rgb_image

            if save_dir is not None:
                os.makedirs(save_dir, exist_ok=True)
                filename = os.path.join(save_dir, f"{filename_prefix}_{view}.png")
                Image.fromarray(rgb_image).save(filename)
                print(f"Saved {view} view to {filename}")

        return results

    def render_front_side(self, vertices, faces, camera_distance=3.0,
                         save_prefix=None, use_hdri_background=False,
                         add_clothing=False, clothing_types=None):
        """Render front and side views side-by-side."""
        front_img = self.render(
            vertices, faces, camera_distance, 'front',
            use_hdri_background=use_hdri_background,
            add_clothing=add_clothing,
            clothing_types=clothing_types
        )
        side_img = self.render(
            vertices, faces, camera_distance, 'side',
            use_hdri_background=use_hdri_background,
            add_clothing=add_clothing,
            clothing_types=clothing_types
        )

        combined = np.concatenate([front_img, side_img], axis=1)

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
