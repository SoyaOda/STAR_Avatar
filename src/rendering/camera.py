"""
Camera utilities for SAM 3D Body / HMR compatible rendering

Provides camera parameter handling compatible with:
- SAM 3D Body (Meta)
- HMR (Human Mesh Recovery)
- iPhone camera simulation
- Standard computer vision conventions

Camera Models:
- Weak Perspective: x = s * (RX)[:2] + [tx, ty]
- Full Perspective: x = K @ (R @ X + t)
"""
import numpy as np


# iPhone Camera Presets (based on calibration data)
# Reference: Stack Overflow, Apple Developer Documentation
# Note: Portrait orientation (vertical) - width < height
IPHONE_PRESETS = {
    # iPhone main camera (wide) - 26mm equivalent - Portrait 1080p
    'iphone_1080p': {
        'width': 1080,   # Portrait: narrower width
        'height': 1920,  # Portrait: taller height
        'fx': 1500.0,    # Approximate focal length for 1080p
        'fy': 1500.0,
        'cx': 540.0,     # width / 2
        'cy': 960.0,     # height / 2
        'fov_degrees': 69.0,  # Typical iPhone wide camera FOV
        'focal_length_mm': 6.86,  # Physical focal length
        'sensor_width_mm': 7.6,   # Approximate sensor width
    },
    # iPhone 4K video mode - Portrait
    'iphone_4k': {
        'width': 2160,   # Portrait
        'height': 3840,  # Portrait
        'fx': 3000.0,    # Scaled from 1080p
        'fy': 3000.0,
        'cx': 1080.0,
        'cy': 1920.0,
        'fov_degrees': 69.0,
        'focal_length_mm': 6.86,
        'sensor_width_mm': 7.6,
    },
    # iPhone photo mode (12MP - 3:4 aspect) - Portrait
    'iphone_photo_12mp': {
        'width': 3024,   # Portrait
        'height': 4032,  # Portrait
        'fx': 3150.0,
        'fy': 3150.0,
        'cx': 1512.0,
        'cy': 2016.0,
        'fov_degrees': 69.0,
        'focal_length_mm': 6.86,
        'sensor_width_mm': 7.6,
    },
}


class CameraParams:
    """
    Camera parameters in SAM 3D Body / HMR format

    Supports both weak perspective and full perspective models.

    Weak Perspective Parameters:
        scale: focal_length / depth (scalar)
        tx, ty: 2D translation in normalized image coordinates

    Full Perspective Parameters:
        focal_length: in pixels
        principal_point: (cx, cy) in pixels
        rotation: 3x3 rotation matrix
        translation: 3D translation vector
    """

    # HMR standard focal length (commonly used assumption)
    HMR_FOCAL_LENGTH = 5000.0

    def __init__(
        self,
        focal_length_pixels: float = 5000.0,
        image_width: int = 1024,
        image_height: int = None,
        camera_distance: float = 4.0,
        principal_point: tuple = None
    ):
        """
        Initialize camera parameters

        Args:
            focal_length_pixels: Focal length in pixels
            image_width: Image width in pixels
            image_height: Image height in pixels (defaults to image_width for square)
            camera_distance: Distance from camera to subject in meters
            principal_point: (cx, cy) principal point, defaults to image center
        """
        self.focal_length = focal_length_pixels
        self.image_width = image_width
        self.image_height = image_height if image_height is not None else image_width
        self.image_size = self.image_width  # For backward compatibility
        self.camera_distance = camera_distance

        if principal_point is None:
            self.cx = self.image_width / 2.0
            self.cy = self.image_height / 2.0
        else:
            self.cx, self.cy = principal_point

    @classmethod
    def from_physical_camera(
        cls,
        focal_length_mm: float = 50.0,
        sensor_width_mm: float = 36.0,
        image_size: int = 1024,
        camera_distance: float = 4.0
    ):
        """
        Create camera params from physical camera specifications

        Args:
            focal_length_mm: Focal length in millimeters
            sensor_width_mm: Sensor width in millimeters (36mm = full frame)
            image_size: Output image size in pixels
            camera_distance: Distance to subject in meters

        Returns:
            CameraParams instance
        """
        # Convert mm focal length to pixels
        focal_length_pixels = (focal_length_mm / sensor_width_mm) * image_size

        return cls(
            focal_length_pixels=focal_length_pixels,
            image_width=image_size,
            camera_distance=camera_distance
        )

    @classmethod
    def from_hmr_standard(cls, image_size: int = 1024, camera_distance: float = 4.0):
        """
        Create camera params using HMR standard (5000 pixel focal length)

        This is the most common assumption in HMR methods to minimize
        perspective distortion on human bodies.

        Args:
            image_size: Output image size
            camera_distance: Distance to subject

        Returns:
            CameraParams instance
        """
        return cls(
            focal_length_pixels=cls.HMR_FOCAL_LENGTH,
            image_width=image_size,
            camera_distance=camera_distance
        )

    @classmethod
    def from_iphone(cls, preset: str = 'iphone_1080p', camera_distance: float = 3.0):
        """
        Create camera params simulating iPhone camera

        iPhone main camera specs (approximate):
        - FOV: ~69° (wide camera)
        - Focal length: ~26mm equivalent (35mm)
        - Physical focal length: ~6.86mm
        - Sensor: ~7.6mm width

        Args:
            preset: iPhone preset name ('iphone_1080p', 'iphone_4k', 'iphone_photo_12mp')
            camera_distance: Distance to subject in meters

        Returns:
            CameraParams instance
        """
        if preset not in IPHONE_PRESETS:
            raise ValueError(f"Unknown iPhone preset: {preset}. "
                           f"Available: {list(IPHONE_PRESETS.keys())}")

        p = IPHONE_PRESETS[preset]

        return cls(
            focal_length_pixels=p['fx'],
            image_width=p['width'],
            image_height=p['height'],
            camera_distance=camera_distance,
            principal_point=(p['cx'], p['cy'])
        )

    def get_weak_perspective_params(self, tx: float = 0.0, ty: float = 0.0) -> dict:
        """
        Get weak perspective camera parameters (HMR format)

        Weak perspective: x = s * Π(RX) + [tx, ty]
        where s = focal_length / depth

        Args:
            tx: 2D translation X (normalized, typically 0 for centered)
            ty: 2D translation Y (normalized, typically 0 for centered)

        Returns:
            dict with scale, tx, ty
        """
        scale = self.focal_length / self.camera_distance

        return {
            'scale': float(scale),
            'tx': float(tx),
            'ty': float(ty)
        }

    def get_intrinsic_matrix(self) -> np.ndarray:
        """
        Get 3x3 camera intrinsic matrix K

        K = [[fx,  0, cx],
             [ 0, fy, cy],
             [ 0,  0,  1]]

        Returns:
            3x3 numpy array
        """
        K = np.array([
            [self.focal_length, 0, self.cx],
            [0, self.focal_length, self.cy],
            [0, 0, 1]
        ], dtype=np.float64)

        return K

    def get_intrinsics_dict(self) -> dict:
        """
        Get intrinsic parameters as dictionary

        Returns:
            dict with fx, fy, cx, cy
        """
        return {
            'fx': float(self.focal_length),
            'fy': float(self.focal_length),
            'cx': float(self.cx),
            'cy': float(self.cy)
        }

    def get_extrinsic_matrix(
        self,
        azimuth: float = 0.0,
        elevation: float = 0.0
    ) -> np.ndarray:
        """
        Get 4x4 camera extrinsic matrix [R|t]

        Camera placed at distance looking at origin.

        Args:
            azimuth: Horizontal rotation in degrees (0 = front)
            elevation: Vertical rotation in degrees (0 = eye level)

        Returns:
            4x4 numpy array
        """
        # Convert to radians
        az_rad = np.radians(azimuth)
        el_rad = np.radians(elevation)

        # Camera position in world coordinates
        x = self.camera_distance * np.sin(az_rad) * np.cos(el_rad)
        y = self.camera_distance * np.sin(el_rad)
        z = self.camera_distance * np.cos(az_rad) * np.cos(el_rad)

        camera_pos = np.array([x, y, z])

        # Look-at matrix (camera looks at origin)
        target = np.array([0.0, 0.0, 0.0])
        up = np.array([0.0, 1.0, 0.0])

        forward = target - camera_pos
        forward = forward / np.linalg.norm(forward)

        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)

        up_new = np.cross(right, forward)

        # Rotation matrix (world to camera)
        R = np.array([
            right,
            -up_new,  # Flip Y for image coordinates
            forward
        ])

        # Translation
        t = -R @ camera_pos

        # 4x4 matrix
        extrinsic = np.eye(4)
        extrinsic[:3, :3] = R
        extrinsic[:3, 3] = t

        return extrinsic

    def get_full_camera_dict(self, azimuth: float = 0.0, elevation: float = 0.0) -> dict:
        """
        Get complete camera parameters in SAM 3D Body compatible format

        Args:
            azimuth: View angle in degrees
            elevation: Elevation angle in degrees

        Returns:
            dict with all camera parameters
        """
        weak_persp = self.get_weak_perspective_params()
        intrinsics = self.get_intrinsics_dict()
        extrinsic = self.get_extrinsic_matrix(azimuth, elevation)

        return {
            # Weak perspective (HMR format)
            'weak_perspective': weak_persp,

            # Full perspective intrinsics
            'intrinsics': intrinsics,

            # Intrinsic matrix as list
            'intrinsic_matrix': self.get_intrinsic_matrix().tolist(),

            # Extrinsic matrix as list
            'extrinsic_matrix': extrinsic.tolist(),

            # Raw parameters
            'focal_length': float(self.focal_length),
            'image_width': self.image_width,
            'image_height': self.image_height,
            'image_size': self.image_size,  # For backward compatibility
            'camera_distance': float(self.camera_distance),
            'azimuth': float(azimuth),
            'elevation': float(elevation)
        }

    def project_points_weak_perspective(
        self,
        points_3d: np.ndarray,
        rotation: np.ndarray = None
    ) -> np.ndarray:
        """
        Project 3D points using weak perspective projection

        Args:
            points_3d: 3D points [N, 3]
            rotation: Optional 3x3 rotation matrix

        Returns:
            2D points [N, 2] in pixel coordinates
        """
        if rotation is not None:
            points_3d = points_3d @ rotation.T

        # Weak perspective projection
        scale = self.focal_length / self.camera_distance

        # Project (drop Z, scale, translate to image center)
        points_2d = points_3d[:, :2] * scale
        points_2d[:, 0] += self.cx
        points_2d[:, 1] += self.cy

        return points_2d

    def project_points_perspective(
        self,
        points_3d: np.ndarray,
        azimuth: float = 0.0,
        elevation: float = 0.0
    ) -> np.ndarray:
        """
        Project 3D points using full perspective projection

        Args:
            points_3d: 3D points [N, 3] in world coordinates
            azimuth: Camera azimuth angle
            elevation: Camera elevation angle

        Returns:
            2D points [N, 2] in pixel coordinates
        """
        # Get camera matrices
        K = self.get_intrinsic_matrix()
        extrinsic = self.get_extrinsic_matrix(azimuth, elevation)

        R = extrinsic[:3, :3]
        t = extrinsic[:3, 3]

        # Transform to camera coordinates
        points_cam = (R @ points_3d.T).T + t

        # Perspective division
        points_2d_homo = (K @ points_cam.T).T
        points_2d = points_2d_homo[:, :2] / points_2d_homo[:, 2:3]

        return points_2d


def create_sam3db_camera(image_size: int = 1024, camera_distance: float = 4.0) -> CameraParams:
    """
    Create camera parameters compatible with SAM 3D Body

    Uses HMR standard focal length (5000 pixels) which is commonly
    assumed in human mesh recovery methods.

    Args:
        image_size: Output image size
        camera_distance: Distance to subject in meters

    Returns:
        CameraParams instance
    """
    return CameraParams.from_hmr_standard(image_size, camera_distance)


if __name__ == "__main__":
    # Test camera parameters
    print("Testing Camera Parameters\n")

    # Create SAM 3D Body compatible camera
    camera = create_sam3db_camera(image_size=1024, camera_distance=4.0)

    print("SAM 3D Body Compatible Camera:")
    print(f"  Focal length: {camera.focal_length} pixels")
    print(f"  Image size: {camera.image_size}")
    print(f"  Camera distance: {camera.camera_distance}m")

    # Get weak perspective params
    weak_persp = camera.get_weak_perspective_params()
    print(f"\nWeak Perspective (HMR format):")
    print(f"  scale: {weak_persp['scale']:.2f}")
    print(f"  tx: {weak_persp['tx']}")
    print(f"  ty: {weak_persp['ty']}")

    # Get intrinsics
    intrinsics = camera.get_intrinsics_dict()
    print(f"\nIntrinsics:")
    print(f"  fx: {intrinsics['fx']:.2f}")
    print(f"  fy: {intrinsics['fy']:.2f}")
    print(f"  cx: {intrinsics['cx']:.2f}")
    print(f"  cy: {intrinsics['cy']:.2f}")

    # Full camera dict
    full_params = camera.get_full_camera_dict(azimuth=45.0)
    print(f"\nFull camera parameters for 45° view:")
    print(f"  Keys: {list(full_params.keys())}")

    # Test projection
    test_point = np.array([[0, 0, 0], [0, 1, 0], [0.5, 0, 0]])
    projected = camera.project_points_weak_perspective(test_point)
    print(f"\nTest projection (weak perspective):")
    print(f"  Origin -> {projected[0]}")
    print(f"  (0,1,0) -> {projected[1]}")
    print(f"  (0.5,0,0) -> {projected[2]}")

    print("\n✓ Camera test complete")
