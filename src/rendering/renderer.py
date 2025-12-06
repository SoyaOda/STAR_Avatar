"""
Renderer Component
Handles photorealistic 2D rendering of 3D bodies

Camera parameters are SAM 3D Body / HMR compatible:
- Weak perspective: scale, tx, ty
- Full perspective: intrinsic matrix K, extrinsic matrix [R|t]
- Default focal length: 5000 pixels (HMR standard)

Supports camera presets:
- HMR standard (5000px focal length, square images)
- iPhone presets (1080p, 4K, photo modes)
"""
import numpy as np
from src.rendering.photorealistic_renderer import PhotorealisticRenderer
from src.rendering.camera import CameraParams, create_sam3db_camera, IPHONE_PRESETS


class Renderer:
    """
    Renderer component for generating 2D images from 3D bodies

    Features:
    - Photorealistic PBR rendering
    - Natural skin tones
    - SAM 3D Body / HMR compatible camera parameters
    - iPhone camera simulation
    - Transparent background support
    """

    # HMR standard focal length
    HMR_FOCAL_LENGTH = 5000.0

    def __init__(self, image_width=1024, image_height=None, focal_length_pixels=None,
                 use_hmr_camera=True, camera_preset=None):
        """
        Initialize renderer with configurable camera

        Args:
            image_width: Output width in pixels
            image_height: Output height in pixels (defaults to image_width)
            focal_length_pixels: Focal length in pixels (default: 5000 for HMR)
            use_hmr_camera: If True, use HMR standard focal length
            camera_preset: Camera preset ('iphone_1080p', 'iphone_4k', etc.)
        """
        self.camera_preset = camera_preset

        # Handle iPhone presets
        if camera_preset is not None and camera_preset in IPHONE_PRESETS:
            p = IPHONE_PRESETS[camera_preset]
            image_width = p['width']
            image_height = p['height']
            focal_length_pixels = p['fx']
            use_hmr_camera = False

        self.image_width = image_width
        self.image_height = image_height if image_height is not None else image_width
        self.image_size = self.image_width  # For backward compatibility

        # Use HMR standard by default
        if focal_length_pixels is None:
            focal_length_pixels = self.HMR_FOCAL_LENGTH if use_hmr_camera else None

        self.focal_length_pixels = focal_length_pixels or self.HMR_FOCAL_LENGTH

        print("="*70)
        print("Initializing Photorealistic Renderer")
        print("="*70)
        if camera_preset:
            print(f"  Camera preset: {camera_preset}")
        print(f"  Focal length: {self.focal_length_pixels} pixels")
        print(f"  Image size: {self.image_width}x{self.image_height}")

        self.renderer = PhotorealisticRenderer(
            image_width=self.image_width,
            image_height=self.image_height,
            focal_length_pixels=self.focal_length_pixels,
            use_hmr_camera=use_hmr_camera,
            camera_preset=None  # Already handled above
        )

        print("✓ Renderer initialized")

    def get_camera_params(self, camera_distance: float, azimuth: float = 0.0) -> dict:
        """
        Get SAM 3D Body / HMR compatible camera parameters

        Args:
            camera_distance: Distance from camera to subject in meters
            azimuth: View angle in degrees (0 = front)

        Returns:
            dict with camera parameters including:
            - weak_perspective: {scale, tx, ty}
            - intrinsics: {fx, fy, cx, cy}
            - intrinsic_matrix: 3x3 matrix
            - extrinsic_matrix: 4x4 matrix
            - focal_length, image_size, camera_distance, azimuth
        """
        return self.renderer.get_camera_params(camera_distance, azimuth)

    def render(
        self,
        vertices,
        faces,
        camera_distance=3.0,
        view='front',
        mesh_color=None,
        background_color=None,
        return_camera_params=False
    ):
        """
        Render 3D mesh to 2D image

        Args:
            vertices: Vertex positions [N, 3]
            faces: Face indices [F, 3]
            camera_distance: Distance from camera (meters)
            view: Camera view ('front', 'back', 'side', 'left')
            mesh_color: Mesh color [R, G, B, A] (default: skin tone)
            background_color: Background color [R, G, B, A] (default: black)
            return_camera_params: If True, also return camera parameters

        Returns:
            RGB image [H, W, 3] as numpy uint8 array
            If return_camera_params=True: (image, camera_params_dict)
        """
        # Default skin color
        if mesh_color is None:
            mesh_color = [0.85, 0.70, 0.60, 1.0]  # Natural peachy skin tone

        # Default background (black for alpha masking)
        if background_color is None:
            background_color = [0, 0, 0, 0]

        # Render using photorealistic renderer
        result = self.renderer.render(
            vertices=vertices,
            faces=faces,
            camera_distance=camera_distance,
            view=view,
            mesh_color=mesh_color,
            background_color=background_color,
            return_camera_params=return_camera_params
        )

        return result

    def render_with_alpha(
        self,
        vertices,
        faces,
        camera_distance=3.0,
        view='front',
        mesh_color=None,
        return_camera_params=False
    ):
        """
        Render with alpha channel for compositing

        Args:
            vertices: Vertex positions [N, 3]
            faces: Face indices [F, 3]
            camera_distance: Distance from camera (meters)
            view: Camera view
            mesh_color: Mesh color [R, G, B, A]
            return_camera_params: If True, also return camera parameters

        Returns:
            RGBA image [H, W, 4] as numpy uint8 array
            If return_camera_params=True: (image, camera_params_dict)
        """
        # Render with black background
        result = self.render(
            vertices=vertices,
            faces=faces,
            camera_distance=camera_distance,
            view=view,
            mesh_color=mesh_color,
            background_color=[0, 0, 0, 0],
            return_camera_params=return_camera_params
        )

        if return_camera_params:
            img_rgb, camera_params = result
        else:
            img_rgb = result
            camera_params = None

        # Create alpha channel from non-black pixels
        # Any pixel with intensity > 10 is considered foreground
        alpha = np.any(img_rgb > 10, axis=2).astype(np.uint8) * 255

        # Combine RGB + Alpha
        img_rgba = np.dstack([img_rgb, alpha])

        if return_camera_params:
            return img_rgba, camera_params

        return img_rgba

    def render_with_azimuth(
        self,
        vertices,
        faces,
        camera_distance=3.0,
        azimuth=0.0,
        mesh_color=None,
        return_camera_params=True
    ):
        """
        Render with specific azimuth angle (for multi-view generation)

        This method rotates the mesh and renders from front view,
        which is equivalent to rotating the camera around the subject.

        Args:
            vertices: Vertex positions [N, 3]
            faces: Face indices [F, 3]
            camera_distance: Distance from camera (meters)
            azimuth: Rotation angle in degrees (0 = front, 90 = right, etc.)
            mesh_color: Mesh color [R, G, B, A]
            return_camera_params: If True, also return camera parameters

        Returns:
            RGBA image [H, W, 4] as numpy uint8 array
            If return_camera_params=True: (image, camera_params_dict)
        """
        # Rotate vertices for this view angle
        rotated_vertices = rotate_vertices_y_axis(vertices, azimuth)

        # Render from front (mesh is rotated)
        result = self.render_with_alpha(
            vertices=rotated_vertices,
            faces=faces,
            camera_distance=camera_distance,
            view='front',
            mesh_color=mesh_color,
            return_camera_params=return_camera_params
        )

        if return_camera_params:
            img_rgba, camera_params = result
            # Update azimuth in camera params
            camera_params['azimuth'] = float(azimuth)
            return img_rgba, camera_params

        return result

    def close(self):
        """Clean up renderer resources"""
        if hasattr(self.renderer, 'renderer'):
            self.renderer.renderer.delete()


def rotate_vertices_y_axis(vertices, angle_degrees):
    """
    Rotate vertices around Y-axis

    Args:
        vertices: Vertex positions [N, 3]
        angle_degrees: Rotation angle in degrees

    Returns:
        Rotated vertices [N, 3]
    """
    angle_rad = np.radians(angle_degrees)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    rotation_matrix = np.array([
        [cos_a, 0, sin_a],
        [0, 1, 0],
        [-sin_a, 0, cos_a]
    ])

    rotated = vertices @ rotation_matrix.T
    return rotated


if __name__ == "__main__":
    # Test
    print("\nTesting Renderer with SAM 3D Body compatible camera...")

    # Create simple test mesh (cube)
    vertices = np.array([
        [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5],
        [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5],
        [-0.5, -0.5, 0.5], [0.5, -0.5, 0.5],
        [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5]
    ])
    faces = np.array([
        [0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6],
        [0, 3, 7], [0, 7, 4], [1, 2, 6], [1, 6, 5]
    ])

    renderer = Renderer(image_size=512)

    print("\nRendering test mesh with camera params...")
    img, cam_params = renderer.render(
        vertices, faces,
        camera_distance=2.0,
        return_camera_params=True
    )

    print(f"Rendered image: {img.shape}, dtype: {img.dtype}")
    print(f"\nCamera Parameters (SAM 3D Body format):")
    print(f"  Weak perspective:")
    print(f"    scale: {cam_params['weak_perspective']['scale']:.2f}")
    print(f"    tx: {cam_params['weak_perspective']['tx']}")
    print(f"    ty: {cam_params['weak_perspective']['ty']}")
    print(f"  Intrinsics:")
    print(f"    fx: {cam_params['intrinsics']['fx']:.2f}")
    print(f"    fy: {cam_params['intrinsics']['fy']:.2f}")
    print(f"    cx: {cam_params['intrinsics']['cx']:.2f}")
    print(f"    cy: {cam_params['intrinsics']['cy']:.2f}")
    print(f"  Distance: {cam_params['camera_distance']}m")
    print(f"  Azimuth: {cam_params['azimuth']}°")

    renderer.close()

    print("\n✓ Renderer test complete")
