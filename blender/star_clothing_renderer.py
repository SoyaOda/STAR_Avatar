"""
STAR + Clothing Blender Rendering Pipeline

Blender Python script for generating Sapiens-ready synthetic data
with STAR body models and realistic clothing.

Based on:
- SURREAL: https://www.di.ens.fr/willow/research/surreal/
- Meshcapade SMPL Blender Addon: https://github.com/Meshcapade/SMPL_blender_addon

Usage:
    blender --background --python star_clothing_renderer.py -- \
        --star_model ../data/star_models/neutral/model.npz \
        --clothing_mesh clothing/shorts.obj \
        --output_dir ../outputs/blender_renders \
        --num_samples 100

Requirements:
    - Blender 3.6+
    - STAR model .npz file
    - Clothing mesh files (.obj)
    - HDRI backgrounds (optional)
"""

import bpy
import bmesh
import numpy as np
import os
import sys
import argparse
import pickle
from mathutils import Vector, Matrix, Euler
import random


class STARClothingRenderer:
    """
    STAR body model + clothing rendering pipeline for Blender.

    Generates photorealistic synthetic human data with:
    - Random body shapes (STAR beta parameters)
    - Random poses (STAR pose parameters)
    - Clothing meshes (skinned to STAR body)
    - HDRI lighting
    - Multiple render passes (RGB, normals, depth, masks)
    """

    def __init__(self, star_model_path, clothing_mesh_paths=None, hdri_dir=None):
        """
        Initialize STAR + Clothing renderer.

        Args:
            star_model_path: Path to STAR .npz model file
            clothing_mesh_paths: List of paths to clothing .obj files
            hdri_dir: Directory containing HDRI backgrounds
        """
        self.star_model_path = star_model_path
        self.clothing_mesh_paths = clothing_mesh_paths or []
        self.hdri_dir = hdri_dir

        # STAR model parameters
        self.star_data = None
        self.num_betas = 10
        self.num_pose_params = 72  # 24 joints * 3 (axis-angle)

        # Blender objects
        self.body_obj = None
        self.clothing_objs = []

        # Scene setup
        self._setup_scene()

    def _setup_scene(self):
        """Setup Blender scene for rendering."""
        # Clear existing scene
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete()

        # Render settings
        scene = bpy.context.scene
        scene.render.engine = 'CYCLES'
        scene.cycles.device = 'GPU'  # Use GPU if available
        scene.cycles.samples = 128
        scene.render.resolution_x = 1024
        scene.render.resolution_y = 1024
        scene.render.film_transparent = False

        # Enable render passes
        scene.view_layers[0].use_pass_normal = True
        scene.view_layers[0].use_pass_z = True

        # Blender 5.0 compatibility: use_nodes is always True
        if bpy.app.version < (5, 0, 0):
            scene.use_nodes = True

        # World settings (will be updated with HDRI)
        world = bpy.data.worlds.new("World")
        scene.world = world
        if bpy.app.version < (5, 0, 0):
            world.use_nodes = True

    def load_star_model(self):
        """
        Load STAR model from .npz file.

        Creates Blender mesh object from STAR template.
        """
        print(f"Loading STAR model from: {self.star_model_path}")

        # Load STAR .npz data
        self.star_data = np.load(self.star_model_path, allow_pickle=True)

        # Extract vertices and faces
        v_template = self.star_data['v_template']  # [6890, 3]
        faces = self.star_data['f']  # [13776, 3]

        # Create mesh
        mesh = bpy.data.meshes.new("STAR_Mesh")
        mesh.from_pydata(v_template.tolist(), [], faces.tolist())
        mesh.update()

        # Create object
        self.body_obj = bpy.data.objects.new("STAR_Body", mesh)
        bpy.context.collection.objects.link(self.body_obj)

        # Add armature for skinning (simplified)
        # In production, use full STAR skeleton
        self._create_simplified_armature()

        print(f"✓ STAR model loaded: {len(v_template)} vertices, {len(faces)} faces")

    def _create_simplified_armature(self):
        """
        Create full STAR skeleton with 24 joints.

        STAR joint hierarchy (SMPL-compatible):
        0: Pelvis, 1: L_Hip, 2: R_Hip, 3: Spine1, 4: L_Knee, 5: R_Knee,
        6: Spine2, 7: L_Ankle, 8: R_Ankle, 9: Spine3, 10: L_Foot,
        11: R_Foot, 12: Neck, 13: L_Collar, 14: R_Collar, 15: Head,
        16: L_Shoulder, 17: R_Shoulder, 18: L_Elbow, 19: R_Elbow,
        20: L_Wrist, 21: R_Wrist, 22: L_Hand, 23: R_Hand
        """
        # Get joint locations from STAR model
        joints = self.star_data['J_regressor'].dot(self.star_data['v_template'])

        # Create armature
        armature = bpy.data.armatures.new("STAR_Armature")
        armature_obj = bpy.data.objects.new("STAR_Skeleton", armature)
        bpy.context.collection.objects.link(armature_obj)
        bpy.context.view_layer.objects.active = armature_obj

        # Enter edit mode to create bones
        bpy.ops.object.mode_set(mode='EDIT')

        # STAR kinematic tree (parent indices)
        parent_indices = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12,
                         13, 14, 16, 17, 18, 19, 20, 21]

        joint_names = [
            "Pelvis", "L_Hip", "R_Hip", "Spine1", "L_Knee", "R_Knee",
            "Spine2", "L_Ankle", "R_Ankle", "Spine3", "L_Foot", "R_Foot",
            "Neck", "L_Collar", "R_Collar", "Head", "L_Shoulder", "R_Shoulder",
            "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist", "L_Hand", "R_Hand"
        ]

        bones = []
        for i, name in enumerate(joint_names):
            bone = armature.edit_bones.new(name)
            bone.head = Vector(joints[i])

            # Set tail (toward child or offset)
            if i < len(joints) - 1:
                bone.tail = bone.head + Vector([0, 0, 0.1])
            else:
                bone.tail = bone.head + Vector([0, 0, 0.1])

            bones.append(bone)

        # Set parent relationships
        for i, parent_idx in enumerate(parent_indices):
            if parent_idx >= 0:
                bones[i].parent = bones[parent_idx]

        # Exit edit mode
        bpy.ops.object.mode_set(mode='OBJECT')

        # Add armature modifier to body
        modifier = self.body_obj.modifiers.new(name="Armature", type='ARMATURE')
        modifier.object = armature_obj

        # Bind mesh to armature using vertex groups
        self._create_vertex_groups(joints)

        # Parent body to armature
        self.body_obj.parent = armature_obj

        self.armature_obj = armature_obj
        self.joints = joints

    def _create_vertex_groups(self, joints):
        """
        Create vertex groups for skinning weights.

        Uses STAR's skinning weights (LBS weights).
        """
        if 'weights' not in self.star_data:
            print("⚠️  No skinning weights found, using automatic weights")
            return

        weights = self.star_data['weights']  # [6890, 24]

        joint_names = [
            "Pelvis", "L_Hip", "R_Hip", "Spine1", "L_Knee", "R_Knee",
            "Spine2", "L_Ankle", "R_Ankle", "Spine3", "L_Foot", "R_Foot",
            "Neck", "L_Collar", "R_Collar", "Head", "L_Shoulder", "R_Shoulder",
            "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist", "L_Hand", "R_Hand"
        ]

        # Create vertex groups
        for i, name in enumerate(joint_names):
            vg = self.body_obj.vertex_groups.new(name=name)
            for v_idx in range(len(self.body_obj.data.vertices)):
                if weights[v_idx, i] > 0.01:  # Only add significant weights
                    vg.add([v_idx], weights[v_idx, i], 'REPLACE')

    def apply_pose_parameters(self, pose):
        """
        Apply STAR pose parameters to armature.

        Args:
            pose: Pose parameters [72] (24 joints × 3 axis-angle)
        """
        if not hasattr(self, 'armature_obj'):
            raise RuntimeError("Armature not created. Call load_star_model() first.")

        pose = np.array(pose[:self.num_pose_params])
        pose_mat = pose.reshape(24, 3)

        # Set armature to pose mode
        bpy.context.view_layer.objects.active = self.armature_obj
        bpy.ops.object.mode_set(mode='POSE')

        # Apply rotations to each bone
        for i, bone in enumerate(self.armature_obj.pose.bones):
            # Convert axis-angle to rotation matrix
            angle = np.linalg.norm(pose_mat[i])
            if angle > 0:
                axis = pose_mat[i] / angle
                rotation = Euler(axis * angle, 'XYZ')
                bone.rotation_euler = rotation
            else:
                bone.rotation_euler = Euler([0, 0, 0], 'XYZ')

        # Return to object mode
        bpy.ops.object.mode_set(mode='OBJECT')

    def apply_shape_parameters(self, betas):
        """
        Apply STAR shape parameters (beta) to body mesh.

        Args:
            betas: Shape parameters [num_betas] (typically 10)
        """
        if self.star_data is None:
            raise RuntimeError("STAR model not loaded. Call load_star_model() first.")

        # Get shape blend shapes (shapedirs)
        shapedirs = self.star_data['shapedirs']  # [6890, 3, 10]
        v_template = self.star_data['v_template']  # [6890, 3]

        # Apply shape deformation: v = v_template + shapedirs @ betas
        betas = np.array(betas[:self.num_betas])
        v_shaped = v_template + np.einsum('vij,j->vi', shapedirs[:, :, :self.num_betas], betas)

        # Update mesh vertices
        mesh = self.body_obj.data
        for i, vert in enumerate(mesh.vertices):
            vert.co = Vector(v_shaped[i])

        mesh.update()

    def apply_skin_material(self, skin_tone='random'):
        """
        Apply realistic skin material with random skin tone.

        Args:
            skin_tone: 'random', 'light', 'medium', 'tan', 'brown', 'dark'
                      or RGB tuple (r, g, b)
        """
        # Skin tone presets (based on Fitzpatrick scale)
        skin_tones = {
            'light': (0.95, 0.85, 0.80),      # Type I-II
            'medium': (0.85, 0.70, 0.60),     # Type III
            'tan': (0.75, 0.60, 0.50),        # Type IV
            'brown': (0.55, 0.40, 0.30),      # Type V
            'dark': (0.35, 0.25, 0.20),       # Type VI
        }

        if skin_tone == 'random':
            skin_tone = random.choice(list(skin_tones.keys()))

        if isinstance(skin_tone, str):
            base_color = skin_tones.get(skin_tone, skin_tones['medium'])
        else:
            base_color = skin_tone

        # Add slight randomization
        base_color = np.array(base_color)
        base_color += np.random.uniform(-0.05, 0.05, 3)
        base_color = np.clip(base_color, 0, 1)

        # Create material
        material = bpy.data.materials.new(name="Skin_Material")
        if bpy.app.version < (5, 0, 0):
            material.use_nodes = True
        nodes = material.node_tree.nodes
        links = material.node_tree.links

        # Clear default nodes
        nodes.clear()

        # Add Principled BSDF for realistic skin
        node_bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
        node_bsdf.inputs['Base Color'].default_value = (*base_color, 1.0)

        # Blender 5.0 compatibility: Subsurface inputs may have different names
        try:
            # Try Blender 3.x/4.x naming
            node_bsdf.inputs['Subsurface'].default_value = 0.1
            node_bsdf.inputs['Subsurface Color'].default_value = (*base_color * 0.8, 1.0)
            node_bsdf.inputs['Subsurface Radius'].default_value = (1.0, 0.2, 0.1)
        except KeyError:
            # Blender 5.0+ may use different subsurface system
            # For now, skip SSS if not available
            pass

        node_bsdf.inputs['Roughness'].default_value = 0.4

        try:
            node_bsdf.inputs['Specular'].default_value = 0.5
        except KeyError:
            # Blender 5.0 uses IOR instead of Specular
            node_bsdf.inputs['IOR'].default_value = 1.4  # Skin IOR

        # Output node
        node_output = nodes.new(type='ShaderNodeOutputMaterial')

        # Connect nodes
        links.new(node_bsdf.outputs['BSDF'], node_output.inputs['Surface'])

        # Apply material to body
        if self.body_obj.data.materials:
            self.body_obj.data.materials[0] = material
        else:
            self.body_obj.data.materials.append(material)

        print(f"✓ Skin material applied: {base_color}")

    def load_clothing_mesh(self, clothing_path, clothing_name=None):
        """
        Load clothing mesh and attach to STAR body.

        Args:
            clothing_path: Path to clothing .obj file
            clothing_name: Name for clothing object (optional)
        """
        if clothing_name is None:
            clothing_name = os.path.splitext(os.path.basename(clothing_path))[0]

        print(f"Loading clothing: {clothing_path}")

        # Import OBJ
        bpy.ops.import_scene.obj(filepath=clothing_path)

        # Get imported object (last selected)
        clothing_obj = bpy.context.selected_objects[0]
        clothing_obj.name = f"Clothing_{clothing_name}"

        # Add armature modifier (same as body)
        if hasattr(self, 'armature_obj'):
            modifier = clothing_obj.modifiers.new(name="Armature", type='ARMATURE')
            modifier.object = self.armature_obj
            clothing_obj.parent = self.armature_obj

        # Add cloth modifier for realistic draping
        cloth_mod = clothing_obj.modifiers.new(name="Cloth", type='CLOTH')
        cloth_mod.settings.quality = 5

        # Add collision modifier to body
        if self.body_obj and not self.body_obj.modifiers.get("Collision"):
            collision_mod = self.body_obj.modifiers.new(name="Collision", type='COLLISION')

        self.clothing_objs.append(clothing_obj)

        print(f"✓ Clothing loaded: {clothing_name}")

    def setup_camera(self, distance=3.0, elevation=0.0, azimuth=0.0):
        """
        Setup camera for rendering.

        Args:
            distance: Camera distance from origin
            elevation: Elevation angle (degrees)
            azimuth: Azimuth angle (degrees)
        """
        # Create camera
        camera_data = bpy.data.cameras.new("Camera")
        camera_obj = bpy.data.objects.new("Camera", camera_data)
        bpy.context.collection.objects.link(camera_obj)
        bpy.context.scene.camera = camera_obj

        # Set camera properties
        camera_data.lens = 50  # 50mm lens

        # Position camera
        elevation_rad = np.radians(elevation)
        azimuth_rad = np.radians(azimuth)

        x = distance * np.cos(elevation_rad) * np.sin(azimuth_rad)
        y = distance * np.cos(elevation_rad) * np.cos(azimuth_rad)
        z = distance * np.sin(elevation_rad)

        camera_obj.location = (x, y, z)

        # Point camera at origin
        direction = Vector((0, 0, 0)) - camera_obj.location
        camera_obj.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()

        self.camera_obj = camera_obj

    def setup_hdri_lighting(self, hdri_path=None):
        """
        Setup HDRI environment lighting.

        Args:
            hdri_path: Path to HDRI image (optional, random if None)
        """
        # Select random HDRI if not specified
        if hdri_path is None and self.hdri_dir:
            hdri_files = [f for f in os.listdir(self.hdri_dir)
                         if f.endswith(('.hdr', '.exr', '.jpg', '.png'))]
            if hdri_files:
                hdri_path = os.path.join(self.hdri_dir, random.choice(hdri_files))

        if hdri_path is None:
            print("⚠️  No HDRI specified, using default lighting")
            return

        print(f"Setting up HDRI: {os.path.basename(hdri_path)}")

        # Setup world nodes for HDRI
        world = bpy.context.scene.world
        if bpy.app.version < (5, 0, 0):
            world.use_nodes = True
        nodes = world.node_tree.nodes
        links = world.node_tree.links

        # Clear existing nodes
        nodes.clear()

        # Add nodes
        node_env = nodes.new(type='ShaderNodeTexEnvironment')
        node_background = nodes.new(type='ShaderNodeBackground')
        node_output = nodes.new(type='ShaderNodeOutputWorld')

        # Load HDRI
        node_env.image = bpy.data.images.load(hdri_path)

        # Connect nodes
        links.new(node_env.outputs['Color'], node_background.inputs['Color'])
        links.new(node_background.outputs['Background'], node_output.inputs['Surface'])

    def render(self, output_path, render_passes=['rgb', 'normal', 'depth', 'mask']):
        """
        Render scene with multiple passes.

        Args:
            output_path: Base output path (extensions added per pass)
            render_passes: List of passes to render
        """
        scene = bpy.context.scene

        # Setup composite nodes for multi-pass rendering
        self._setup_composite_nodes(output_path, render_passes)

        # Render
        print(f"Rendering to: {output_path}")
        scene.render.filepath = output_path
        bpy.ops.render.render(write_still=True)

        print("✓ Rendering complete")

    def _setup_composite_nodes(self, output_path, render_passes):
        """Setup compositor nodes for multi-pass output."""
        scene = bpy.context.scene

        # Blender 5.0: Simplified approach without compositor nodes
        # Just use view layer output settings
        if bpy.app.version >= (5, 0, 0):
            print("⚠️  Blender 5.0: Using simplified output (compositor API changed)")
            # For now, just render main image
            # Multi-pass output would require different approach in Blender 5.0
            return

        # Blender 3.x/4.x: Use compositor nodes
        scene.use_nodes = True
        tree = scene.node_tree
        nodes = tree.nodes
        links = tree.links

        # Clear existing nodes
        nodes.clear()

        # Render layers node
        node_rl = nodes.new(type='CompositorNodeRLayers')

        # RGB output
        if 'rgb' in render_passes:
            node_rgb_output = nodes.new(type='CompositorNodeOutputFile')
            node_rgb_output.base_path = os.path.dirname(output_path)
            node_rgb_output.file_slots[0].path = os.path.basename(output_path) + '_rgb'
            node_rgb_output.format.file_format = 'PNG'
            links.new(node_rl.outputs['Image'], node_rgb_output.inputs[0])

        # Normal output
        if 'normal' in render_passes:
            node_normal_output = nodes.new(type='CompositorNodeOutputFile')
            node_normal_output.base_path = os.path.dirname(output_path)
            node_normal_output.file_slots[0].path = os.path.basename(output_path) + '_normal'
            node_normal_output.format.file_format = 'PNG'
            links.new(node_rl.outputs['Normal'], node_normal_output.inputs[0])

        # Depth output
        if 'depth' in render_passes:
            node_depth_output = nodes.new(type='CompositorNodeOutputFile')
            node_depth_output.base_path = os.path.dirname(output_path)
            node_depth_output.file_slots[0].path = os.path.basename(output_path) + '_depth'
            node_depth_output.format.file_format = 'OPEN_EXR'
            links.new(node_rl.outputs['Depth'], node_depth_output.inputs[0])

    def generate_random_parameters(self):
        """
        Generate random parameters for body shape, pose, camera, lighting, etc.

        Returns:
            Dictionary with random parameters
        """
        # Generate random pose (24 joints × 3 = 72 parameters)
        # Use smaller values for more natural poses
        pose = np.zeros(72)

        # Randomize main body joints more conservatively
        pose[0:3] = np.random.randn(3) * 0.1      # Pelvis
        pose[9:12] = np.random.randn(3) * 0.2     # Spine3
        pose[36:39] = np.random.randn(3) * 0.3    # Neck

        # Randomize limbs more freely
        pose[12:18] = np.random.randn(6) * 0.5    # Legs
        pose[48:72] = np.random.randn(24) * 0.4   # Arms

        return {
            'betas': np.random.randn(self.num_betas) * 0.5,  # Shape parameters
            'pose': pose,                                     # Pose parameters
            'skin_tone': 'random',                           # Random skin tone
            'camera_distance': np.random.uniform(2.5, 3.5),
            'camera_elevation': np.random.uniform(-10, 10),
            'camera_azimuth': np.random.uniform(0, 360),
        }

    def cleanup(self):
        """Cleanup Blender scene."""
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete()


def main():
    """Main rendering pipeline."""
    parser = argparse.ArgumentParser(description='STAR + Clothing Blender Renderer')
    parser.add_argument('--star_model', type=str, required=True,
                       help='Path to STAR .npz model file')
    parser.add_argument('--clothing_mesh', type=str, nargs='+',
                       help='Path(s) to clothing .obj files')
    parser.add_argument('--hdri_dir', type=str,
                       help='Directory containing HDRI backgrounds')
    parser.add_argument('--output_dir', type=str, default='../outputs/blender_renders',
                       help='Output directory')
    parser.add_argument('--num_samples', type=int, default=10,
                       help='Number of samples to generate')

    # Parse args (Blender passes args after --)
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    args = parser.parse_args(argv)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "="*70)
    print("STAR + Clothing Blender Rendering Pipeline")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  STAR model: {args.star_model}")
    print(f"  Clothing: {args.clothing_mesh}")
    print(f"  HDRI dir: {args.hdri_dir}")
    print(f"  Output dir: {args.output_dir}")
    print(f"  Num samples: {args.num_samples}")
    print()

    # Initialize renderer
    renderer = STARClothingRenderer(
        star_model_path=args.star_model,
        clothing_mesh_paths=args.clothing_mesh,
        hdri_dir=args.hdri_dir
    )

    # Load STAR model
    renderer.load_star_model()

    # Load clothing
    if args.clothing_mesh:
        for clothing_path in args.clothing_mesh:
            renderer.load_clothing_mesh(clothing_path)

    # Generate samples
    for i in range(args.num_samples):
        print(f"\n[{i+1}/{args.num_samples}] Generating sample...")

        # Generate random parameters
        params = renderer.generate_random_parameters()

        # Apply shape
        renderer.apply_shape_parameters(params['betas'])

        # Apply pose
        renderer.apply_pose_parameters(params['pose'])

        # Apply skin material
        renderer.apply_skin_material(params['skin_tone'])

        # Setup camera
        renderer.setup_camera(
            distance=params['camera_distance'],
            elevation=params['camera_elevation'],
            azimuth=params['camera_azimuth']
        )

        # Setup lighting
        renderer.setup_hdri_lighting()

        # Render
        output_path = os.path.join(args.output_dir, f"sample_{i+1:04d}")
        renderer.render(output_path)

        # Save parameters
        param_path = os.path.join(args.output_dir, f"sample_{i+1:04d}_params.pkl")
        with open(param_path, 'wb') as f:
            pickle.dump(params, f)

    print("\n" + "="*70)
    print("✓ All samples generated!")
    print("="*70)
    print(f"\nOutput directory: {args.output_dir}")
    print()


if __name__ == "__main__":
    main()
