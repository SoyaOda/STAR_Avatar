"""
2D Image Renderer from 3D Mesh
Renders front and back views of 3D body mesh

Based on spec1.md - synthetic data generation approach
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PolyCollection
import cv2
import os


class MeshRenderer:
    """
    Simple mesh renderer for generating 2D views from 3D meshes

    Args:
        image_size: Output image resolution (default: 512x512)
        camera_distance: Distance from camera to subject in meters (default: 3.0)
        focal_length: Camera focal length in mm (default: 50mm)
    """

    def __init__(self, image_size=512, camera_distance=3.0, focal_length=50.0):
        self.image_size = image_size
        self.camera_distance = camera_distance
        self.focal_length = focal_length

        # Camera intrinsics (simplified perspective projection)
        # Approximate focal length in pixels
        sensor_width = 36.0  # Full-frame sensor width in mm
        self.fx = (focal_length / sensor_width) * image_size
        self.fy = self.fx
        self.cx = image_size / 2.0
        self.cy = image_size / 2.0

    def _to_numpy(self, tensor):
        """Convert torch tensor to numpy array"""
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().numpy()
        return tensor

    def _setup_camera(self, view='front'):
        """
        Setup camera position and rotation

        Args:
            view: 'front', 'back', 'side', 'left', or 'right'

        Returns:
            camera_pos: [3] camera position
            R: [3, 3] rotation matrix
        """
        if view == 'front':
            # Camera in front of subject (looking at +Z direction)
            camera_pos = np.array([0.0, 0.0, -self.camera_distance])
            # No rotation needed (identity)
            R = np.eye(3)
        elif view == 'back':
            # Camera behind subject (looking at -Z direction)
            camera_pos = np.array([0.0, 0.0, self.camera_distance])
            # Rotate 180 degrees around Y axis
            R = np.array([
                [-1,  0,  0],
                [ 0,  1,  0],
                [ 0,  0, -1]
            ], dtype=np.float32)
        elif view in ['side', 'right']:
            # Camera on right side (looking from +X direction toward origin)
            camera_pos = np.array([self.camera_distance, 0.0, 0.0])
            # Rotate 90 degrees around Y axis (clockwise from top)
            R = np.array([
                [ 0,  0,  1],
                [ 0,  1,  0],
                [-1,  0,  0]
            ], dtype=np.float32)
        elif view == 'left':
            # Camera on left side (looking from -X direction toward origin)
            camera_pos = np.array([-self.camera_distance, 0.0, 0.0])
            # Rotate -90 degrees around Y axis (counter-clockwise from top)
            R = np.array([
                [ 0,  0, -1],
                [ 0,  1,  0],
                [ 1,  0,  0]
            ], dtype=np.float32)
        else:
            raise ValueError(f"Invalid view: {view}. Must be 'front', 'back', 'side', 'left', or 'right'")

        return camera_pos, R

    def _transform_vertices(self, vertices, camera_pos, R):
        """
        Transform vertices to camera coordinate system

        Args:
            vertices: [N, 3] vertex positions in world coords
            camera_pos: [3] camera position
            R: [3, 3] rotation matrix

        Returns:
            vertices_cam: [N, 3] vertices in camera coords
        """
        # Translate to camera origin
        vertices_centered = vertices - camera_pos

        # Rotate to camera orientation
        vertices_cam = vertices_centered @ R.T

        return vertices_cam

    def _project_to_2d(self, vertices_cam):
        """
        Project 3D vertices to 2D image plane

        Args:
            vertices_cam: [N, 3] vertices in camera coords

        Returns:
            vertices_2d: [N, 2] projected 2D coordinates (in pixels)
            depth: [N] depth values (Z coordinate)
        """
        # Perspective projection
        # x_2d = fx * (X / Z) + cx
        # y_2d = fy * (Y / Z) + cy

        X = vertices_cam[:, 0]
        Y = vertices_cam[:, 1]
        Z = vertices_cam[:, 2]

        # Avoid division by zero
        Z = np.where(Z > 0.01, Z, 0.01)

        x_2d = self.fx * (X / Z) + self.cx
        y_2d = self.fy * (Y / Z) + self.cy

        vertices_2d = np.stack([x_2d, y_2d], axis=1)

        return vertices_2d, Z

    def _compute_face_visibility(self, vertices_cam, faces):
        """
        Compute which faces are visible (back-face culling)

        Args:
            vertices_cam: [N, 3] vertices in camera coords
            faces: [F, 3] face indices

        Returns:
            visible: [F] boolean array indicating visible faces
        """
        # Get face vertices
        v0 = vertices_cam[faces[:, 0]]
        v1 = vertices_cam[faces[:, 1]]
        v2 = vertices_cam[faces[:, 2]]

        # Compute face normals using cross product
        edge1 = v1 - v0
        edge2 = v2 - v0
        normals = np.cross(edge1, edge2)

        # View direction (from face to camera)
        # Camera is at origin in camera coords
        face_centers = (v0 + v1 + v2) / 3.0
        view_dirs = -face_centers  # Vector from face to camera (at origin)

        # Dot product: positive means facing camera
        dot_products = np.sum(normals * view_dirs, axis=1)

        visible = dot_products > 0

        return visible

    def render_view(self, vertices, faces, view='front', return_depth=False):
        """
        Render mesh from specified view

        Args:
            vertices: [N, 3] or [1, N, 3] vertex positions
            faces: [F, 3] face indices
            view: 'front' or 'back'
            return_depth: If True, also return depth map

        Returns:
            image: [H, W, 3] rendered RGB image
            depth_map: [H, W] depth map (if return_depth=True)
        """
        # Convert to numpy
        vertices = self._to_numpy(vertices)
        faces = self._to_numpy(faces)

        # Handle batch dimension
        if len(vertices.shape) == 3:
            vertices = vertices[0]

        # Setup camera
        camera_pos, R = self._setup_camera(view)

        # Transform vertices to camera coords
        vertices_cam = self._transform_vertices(vertices, camera_pos, R)

        # Project to 2D
        vertices_2d, depth = self._project_to_2d(vertices_cam)

        # Compute visible faces
        visible = self._compute_face_visibility(vertices_cam, faces)

        # Create figure with exact size
        dpi = 100
        figsize = self.image_size / dpi
        fig = plt.figure(figsize=(figsize, figsize), dpi=dpi)
        ax = fig.add_subplot(111)

        # Plot visible faces
        face_colors = []
        polygons = []
        face_depths = []

        # First pass: collect depths
        for i, face in enumerate(faces):
            if not visible[i]:
                continue
            avg_depth = depth[face].mean()
            face_depths.append(avg_depth)

        # Calculate depth range
        if face_depths:
            max_depth = max(face_depths)
            min_depth = min(face_depths)
        else:
            max_depth = 1.0
            min_depth = 0.0

        # Second pass: create polygons with colors
        face_depths = []  # Reset for sorting
        for i, face in enumerate(faces):
            if not visible[i]:
                continue

            # Get 2D coordinates of face vertices
            face_2d = vertices_2d[face]

            # Compute average depth for sorting
            avg_depth = depth[face].mean()
            face_depths.append(avg_depth)

            polygons.append(face_2d)

            # Simple shading based on depth
            # Farther = darker
            if max_depth > min_depth:
                brightness = 1.0 - 0.5 * (avg_depth - min_depth) / (max_depth - min_depth)
            else:
                brightness = 0.8

            face_colors.append([brightness * 0.7, brightness * 0.7, brightness * 0.9])

        # Sort faces by depth (painter's algorithm)
        if polygons:
            sorted_indices = np.argsort(face_depths)[::-1]  # Far to near
            polygons = [polygons[i] for i in sorted_indices]
            face_colors = [face_colors[i] for i in sorted_indices]

            # Create polygon collection
            poly_collection = PolyCollection(
                polygons,
                facecolors=face_colors,
                edgecolors='none',
                linewidths=0
            )
            ax.add_collection(poly_collection)

        # Set axis limits
        ax.set_xlim(0, self.image_size)
        ax.set_ylim(0, self.image_size)
        ax.set_aspect('equal')
        ax.axis('off')

        # Render to array
        fig.canvas.draw()

        # Get the actual rendered buffer
        buf = fig.canvas.buffer_rgba()
        buf_array = np.frombuffer(buf, dtype=np.uint8)

        # Calculate actual dimensions from buffer size
        # Buffer is RGBA, so divide by 4 to get pixel count
        total_pixels = len(buf_array) // 4
        actual_size = int(np.sqrt(total_pixels))

        # Reshape with actual size
        image = buf_array.reshape((actual_size, actual_size, 4))

        # Convert RGBA to RGB
        image = image[:, :, :3]

        # Resize to exact target size if needed
        if actual_size != self.image_size:
            image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)

        plt.close(fig)

        if return_depth:
            # Create depth map
            depth_map = self._create_depth_map(vertices_2d, faces, depth, visible)
            return image, depth_map

        return image

    def _create_depth_map(self, vertices_2d, faces, depth, visible):
        """Create depth map image"""
        depth_map = np.zeros((self.image_size, self.image_size), dtype=np.float32)

        for i, face in enumerate(faces):
            if not visible[i]:
                continue

            # Get 2D coordinates and depths
            pts = vertices_2d[face].astype(np.int32)
            face_depth = depth[face].mean()

            # Rasterize triangle
            cv2.fillPoly(depth_map, [pts], face_depth)

        return depth_map

    def render_front_back(self, vertices, faces, save_prefix=None):
        """
        Render both front and back views

        Args:
            vertices: [N, 3] or [1, N, 3] vertex positions
            faces: [F, 3] face indices
            save_prefix: If provided, save images as {prefix}_front.png and {prefix}_back.png

        Returns:
            front_image: [H, W, 3] front view
            back_image: [H, W, 3] back view
        """
        print(f"\nRendering front and back views...")
        print(f"  - Resolution: {self.image_size}x{self.image_size}")
        print(f"  - Camera distance: {self.camera_distance:.2f}m")

        # Render front view
        front_image = self.render_view(vertices, faces, view='front')
        print(f"  ✓ Front view rendered")

        # Render back view
        back_image = self.render_view(vertices, faces, view='back')
        print(f"  ✓ Back view rendered")

        # Save if requested
        if save_prefix:
            front_path = f"{save_prefix}_front.png"
            back_path = f"{save_prefix}_back.png"

            cv2.imwrite(front_path, cv2.cvtColor(front_image, cv2.COLOR_RGB2BGR))
            cv2.imwrite(back_path, cv2.cvtColor(back_image, cv2.COLOR_RGB2BGR))

            print(f"  ✓ Saved: {front_path}")
            print(f"  ✓ Saved: {back_path}")

        return front_image, back_image

    def render_front_side(self, vertices, faces, save_prefix=None):
        """
        Render both front and side views

        Args:
            vertices: [N, 3] or [1, N, 3] vertex positions
            faces: [F, 3] face indices
            save_prefix: If provided, save images as {prefix}_front.png and {prefix}_side.png

        Returns:
            front_image: [H, W, 3] front view
            side_image: [H, W, 3] side view
        """
        print(f"\nRendering front and side views...")
        print(f"  - Resolution: {self.image_size}x{self.image_size}")
        print(f"  - Camera distance: {self.camera_distance:.2f}m")

        # Render front view
        front_image = self.render_view(vertices, faces, view='front')
        print(f"  ✓ Front view rendered")

        # Render side view
        side_image = self.render_view(vertices, faces, view='side')
        print(f"  ✓ Side view rendered")

        # Save if requested
        if save_prefix:
            front_path = f"{save_prefix}_front.png"
            side_path = f"{save_prefix}_side.png"

            cv2.imwrite(front_path, cv2.cvtColor(front_image, cv2.COLOR_RGB2BGR))
            cv2.imwrite(side_path, cv2.cvtColor(side_image, cv2.COLOR_RGB2BGR))

            print(f"  ✓ Saved: {front_path}")
            print(f"  ✓ Saved: {side_path}")

        return front_image, side_image

    def render_multi_view_figure(self, vertices, faces, title="STAR Body Model", save_path=None, views=['front', 'back']):
        """
        Create a figure showing multiple views side-by-side

        Args:
            vertices: [N, 3] or [1, N, 3] vertex positions
            faces: [F, 3] face indices
            title: Figure title
            save_path: If provided, save figure to this path
            views: List of views to render (default: ['front', 'back'])
        """
        # Render all requested views
        images = []
        view_titles = []
        for view in views:
            image = self.render_view(vertices, faces, view=view)
            images.append(image)
            view_titles.append(f"{view.capitalize()} View")

        # Create side-by-side figure
        fig, axes = plt.subplots(1, len(views), figsize=(6*len(views), 6))

        # Handle single view case
        if len(views) == 1:
            axes = [axes]

        for i, (image, view_title) in enumerate(zip(images, view_titles)):
            axes[i].imshow(image)
            axes[i].set_title(view_title, fontsize=14, fontweight='bold')
            axes[i].axis('off')

        fig.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved multi-view figure: {save_path}")
            plt.close(fig)
        else:
            plt.show()


def test_renderer():
    """Test renderer with simple mesh"""
    print("\n" + "="*70)
    print("Testing Mesh Renderer")
    print("="*70)

    # Create simple test mesh (cube)
    vertices = np.array([
        [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5],
        [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5],
        [-0.5, -0.5, 0.5], [0.5, -0.5, 0.5],
        [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5],
    ], dtype=np.float32)

    faces = np.array([
        [0,1,2], [0,2,3], [4,5,6], [4,6,7],
        [0,1,5], [0,5,4], [2,3,7], [2,7,6],
        [0,3,7], [0,7,4], [1,2,6], [1,6,5],
    ], dtype=np.int32)

    # Create renderer
    renderer = MeshRenderer(image_size=256, camera_distance=2.0)

    # Render views
    renderer.render_multi_view_figure(vertices, faces, title="Test Cube")

    print("\n✓ Test completed")


if __name__ == "__main__":
    test_renderer()
