"""
Renderer Component
Handles photorealistic 2D rendering of 3D bodies
"""
import numpy as np
from src.rendering.photorealistic_renderer import PhotorealisticRenderer


class Renderer:
    """
    Renderer component for generating 2D images from 3D bodies

    Features:
    - Photorealistic PBR rendering
    - Natural skin tones
    - Configurable camera positions
    - Transparent background support
    """

    def __init__(self, image_size=1024, focal_length=50.0):
        """
        Initialize renderer

        Args:
            image_size: Output resolution (pixels)
            focal_length: Camera focal length (mm)
        """
        self.image_size = image_size

        print("="*70)
        print("Initializing Photorealistic Renderer")
        print("="*70)

        self.renderer = PhotorealisticRenderer(
            image_size=image_size,
            focal_length=focal_length
        )

        print("✓ Renderer initialized")

    def render(
        self,
        vertices,
        faces,
        camera_distance=3.0,
        view='front',
        mesh_color=None,
        background_color=None
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

        Returns:
            RGB image [H, W, 3] as numpy uint8 array
        """
        # Default skin color
        if mesh_color is None:
            mesh_color = [0.85, 0.70, 0.60, 1.0]  # Natural peachy skin tone

        # Default background (black for alpha masking)
        if background_color is None:
            background_color = [0, 0, 0, 0]

        # Render using photorealistic renderer
        img_rgb = self.renderer.render(
            vertices=vertices,
            faces=faces,
            camera_distance=camera_distance,
            view=view,
            mesh_color=mesh_color,
            background_color=background_color
        )

        return img_rgb

    def render_with_alpha(
        self,
        vertices,
        faces,
        camera_distance=3.0,
        view='front',
        mesh_color=None
    ):
        """
        Render with alpha channel for compositing

        Args:
            vertices: Vertex positions [N, 3]
            faces: Face indices [F, 3]
            camera_distance: Distance from camera (meters)
            view: Camera view
            mesh_color: Mesh color [R, G, B, A]

        Returns:
            RGBA image [H, W, 4] as numpy uint8 array
        """
        # Render with black background
        img_rgb = self.render(
            vertices=vertices,
            faces=faces,
            camera_distance=camera_distance,
            view=view,
            mesh_color=mesh_color,
            background_color=[0, 0, 0, 0]
        )

        # Create alpha channel from non-black pixels
        # Any pixel with intensity > 10 is considered foreground
        alpha = np.any(img_rgb > 10, axis=2).astype(np.uint8) * 255

        # Combine RGB + Alpha
        img_rgba = np.dstack([img_rgb, alpha])

        return img_rgba

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
    print("\nTesting Renderer...")

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

    print("\nRendering test mesh...")
    img = renderer.render(vertices, faces, camera_distance=2.0)

    print(f"Rendered image: {img.shape}, dtype: {img.dtype}")
    print(f"Value range: [{img.min()}, {img.max()}]")

    print("\nRendering with alpha...")
    img_rgba = renderer.render_with_alpha(vertices, faces, camera_distance=2.0)

    print(f"Rendered RGBA: {img_rgba.shape}")
    print(f"Alpha range: [{img_rgba[:,:,3].min()}, {img_rgba[:,:,3].max()}]")

    renderer.close()

    print("\n✓ Renderer test complete")
