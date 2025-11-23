"""
Blender Script: Generate Properly Fitted Clothed 3D Models

Improvements over previous version:
1. Automatic skinning weight transfer from STAR body to clothing
2. Clothing properly follows body pose and shape
3. Output to outputs/ directory
4. Data Transfer Modifier for accurate weight transfer

Usage:
    blender --background --python generate_fitted_clothed_models.py -- \
        --star_model ../data/star_models/female/model.npz \
        --clothing_mesh ../clothing/sports_outfit.obj \
        --output_dir ../outputs/fitted_clothed_models \
        --num_samples 5
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


class FittedClothedModelGenerator:
    """
    Generate clothed 3D STAR models with proper skinning weight transfer.

    Uses Blender's Data Transfer Modifier to automatically transfer
    vertex group weights from STAR body to clothing mesh.
    """

    def __init__(self, star_model_path, clothing_mesh_paths=None):
        """Initialize generator."""
        self.star_model_path = star_model_path
        self.clothing_mesh_paths = clothing_mesh_paths or []

        # STAR parameters
        self.star_data = None
        self.num_betas = 10
        self.num_pose_params = 72

        # Blender objects
        self.body_obj = None
        self.clothing_objs = []

        # Setup scene
        self._setup_scene()

    def _setup_scene(self):
        """Setup clean Blender scene."""
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete()
        print("✓ Scene cleared")

    def load_star_model(self):
        """Load STAR model from .npz file."""
        print(f"Loading STAR model: {self.star_model_path}")

        # Load STAR data
        self.star_data = np.load(self.star_model_path, allow_pickle=True)

        # Extract template
        v_template = self.star_data['v_template']  # [6890, 3]
        faces = self.star_data['f']  # [13776, 3]

        # Create mesh
        mesh = bpy.data.meshes.new("STAR_Mesh")
        mesh.from_pydata(v_template.tolist(), [], faces.tolist())
        mesh.update()

        # Create object
        self.body_obj = bpy.data.objects.new("STAR_Body", mesh)
        bpy.context.collection.objects.link(self.body_obj)

        # Create skeleton with weights
        self._create_skeleton_with_weights()

        print(f"✓ STAR loaded: {len(v_template)} verts, {len(faces)} faces")

    def _create_skeleton_with_weights(self):
        """Create STAR skeleton and apply LBS weights."""
        # Get joint locations
        joints = self.star_data['J_regressor'].dot(self.star_data['v_template'])

        # Create armature
        armature = bpy.data.armatures.new("STAR_Armature")
        armature_obj = bpy.data.objects.new("STAR_Skeleton", armature)
        bpy.context.collection.objects.link(armature_obj)
        bpy.context.view_layer.objects.active = armature_obj

        # Enter edit mode
        bpy.ops.object.mode_set(mode='EDIT')

        # STAR kinematic tree
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

        # Apply LBS weights from STAR
        if 'weights' in self.star_data:
            weights = self.star_data['weights']
            for i, name in enumerate(joint_names):
                vg = self.body_obj.vertex_groups.new(name=name)
                for v_idx in range(len(self.body_obj.data.vertices)):
                    if weights[v_idx, i] > 0.01:
                        vg.add([v_idx], weights[v_idx, i], 'REPLACE')
            print("✓ LBS weights applied to body")

        # Parent body to armature
        self.body_obj.parent = armature_obj

        self.armature_obj = armature_obj
        self.joints = joints

    def apply_shape_parameters(self, betas):
        """Apply STAR shape parameters."""
        if self.star_data is None:
            raise RuntimeError("STAR model not loaded")

        shapedirs = self.star_data['shapedirs']
        v_template = self.star_data['v_template']

        betas = np.array(betas[:self.num_betas])
        v_shaped = v_template + np.einsum('vij,j->vi', shapedirs[:, :, :self.num_betas], betas)

        # Update mesh
        mesh = self.body_obj.data
        for i, vert in enumerate(mesh.vertices):
            vert.co = Vector(v_shaped[i])
        mesh.update()

    def apply_pose_parameters(self, pose):
        """Apply STAR pose parameters to skeleton."""
        if not hasattr(self, 'armature_obj'):
            raise RuntimeError("Armature not created")

        pose = np.array(pose[:self.num_pose_params])
        pose_mat = pose.reshape(24, 3)

        # Set to pose mode
        bpy.context.view_layer.objects.active = self.armature_obj
        bpy.ops.object.mode_set(mode='POSE')

        # Apply rotations
        for i, bone in enumerate(self.armature_obj.pose.bones):
            angle = np.linalg.norm(pose_mat[i])
            if angle > 0:
                axis = pose_mat[i] / angle
                rotation = Euler(axis * angle, 'XYZ')
                bone.rotation_euler = rotation
            else:
                bone.rotation_euler = Euler([0, 0, 0], 'XYZ')

        # Return to object mode
        bpy.ops.object.mode_set(mode='OBJECT')

    def load_clothing_with_weight_transfer(self, clothing_path):
        """
        Load clothing and automatically transfer skinning weights from body.

        This is the KEY improvement: uses Data Transfer Modifier to copy
        vertex group weights from STAR body to clothing mesh.
        """
        print(f"Loading clothing: {os.path.basename(clothing_path)}")

        # Import OBJ
        bpy.ops.wm.obj_import(filepath=clothing_path)

        # Get imported object
        clothing_obj = bpy.context.selected_objects[0]
        clothing_obj.name = f"Clothing_{os.path.splitext(os.path.basename(clothing_path))[0]}"

        # CRITICAL: Transfer vertex groups (skinning weights) from body to clothing
        print("  Transferring skinning weights from body...")

        # Add Data Transfer Modifier
        data_transfer = clothing_obj.modifiers.new(name="DataTransfer", type='DATA_TRANSFER')
        data_transfer.object = self.body_obj  # Source: STAR body
        data_transfer.use_vert_data = True
        data_transfer.data_types_verts = {'VGROUP_WEIGHTS'}  # Transfer weights
        data_transfer.vert_mapping = 'NEAREST'  # Use nearest vertex
        data_transfer.mix_mode = 'REPLACE'  # Replace existing weights

        # Apply the data transfer
        bpy.context.view_layer.objects.active = clothing_obj
        bpy.ops.object.modifier_apply(modifier="DataTransfer")

        print("  ✓ Skinning weights transferred")

        # Add armature modifier (now clothing has proper weights)
        if hasattr(self, 'armature_obj'):
            modifier = clothing_obj.modifiers.new(name="Armature", type='ARMATURE')
            modifier.object = self.armature_obj
            clothing_obj.parent = self.armature_obj
            print("  ✓ Clothing rigged to armature")

        # Add cloth physics (optional, for settling)
        cloth_mod = clothing_obj.modifiers.new(name="Cloth", type='CLOTH')
        cloth_mod.settings.quality = 5
        cloth_mod.settings.time_scale = 1.0

        # Add collision to body
        if self.body_obj and not self.body_obj.modifiers.get("Collision"):
            collision_mod = self.body_obj.modifiers.new(name="Collision", type='COLLISION')
            collision_mod.settings.thickness_outer = 0.02

        self.clothing_objs.append(clothing_obj)
        print("✓ Clothing loaded and fitted")

    def apply_cloth_simulation(self, frames=50):
        """Run cloth physics simulation for natural draping."""
        print(f"Running cloth simulation ({frames} frames)...")

        # Set frame range
        bpy.context.scene.frame_start = 1
        bpy.context.scene.frame_end = frames

        # Run simulation by advancing frames
        for frame in range(1, frames + 1):
            bpy.context.scene.frame_set(frame)

        # Go to final frame
        bpy.context.scene.frame_set(frames)

        print("✓ Cloth simulation complete")

    def export_clothed_model(self, output_path):
        """Export clothed model as OBJ and numpy arrays."""
        print(f"Exporting: {output_path}")

        # Select body and clothing
        bpy.ops.object.select_all(action='DESELECT')
        self.body_obj.select_set(True)
        for clothing_obj in self.clothing_objs:
            clothing_obj.select_set(True)

        # Export as OBJ
        obj_path = output_path + ".obj"
        bpy.ops.wm.obj_export(
            filepath=obj_path,
            export_selected_objects=True,
            apply_modifiers=True
        )

        # Export vertices/faces as numpy
        depsgraph = bpy.context.evaluated_depsgraph_get()

        # Get body with all modifiers applied
        body_eval = self.body_obj.evaluated_get(depsgraph)
        body_mesh = body_eval.to_mesh()

        vertices = np.array([v.co for v in body_mesh.vertices])
        faces = np.array([[p.vertices[i] for i in range(3)] for p in body_mesh.polygons])

        # Save numpy arrays
        np.save(output_path + "_vertices.npy", vertices)
        np.save(output_path + "_faces.npy", faces)

        body_eval.to_mesh_clear()

        print(f"✓ Exported: {os.path.basename(obj_path)}")
        print(f"  Vertices: {len(vertices)}, Faces: {len(faces)}")

    def generate_random_parameters(self):
        """Generate random shape and pose parameters."""
        # Random pose
        pose = np.zeros(72)
        pose[0:3] = np.random.randn(3) * 0.1      # Pelvis
        pose[9:12] = np.random.randn(3) * 0.2     # Spine
        pose[36:39] = np.random.randn(3) * 0.3    # Neck
        pose[12:18] = np.random.randn(6) * 0.5    # Legs
        pose[48:72] = np.random.randn(24) * 0.4   # Arms

        return {
            'betas': np.random.randn(self.num_betas) * 0.5,
            'pose': pose,
        }

    def cleanup(self):
        """Clean up scene."""
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete()


def main():
    """Main pipeline."""
    parser = argparse.ArgumentParser(description='Generate Fitted Clothed 3D Models')
    parser.add_argument('--star_model', type=str, required=True)
    parser.add_argument('--clothing_mesh', type=str, nargs='+')
    parser.add_argument('--output_dir', type=str, default='../outputs/fitted_clothed_models')
    parser.add_argument('--num_samples', type=int, default=5)
    parser.add_argument('--cloth_frames', type=int, default=30,
                       help='Cloth simulation frames')

    # Parse args
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    args = parser.parse_args(argv)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "="*70)
    print("Generate Fitted Clothed 3D Models with Weight Transfer")
    print("="*70)
    print(f"\nSTAR model: {args.star_model}")
    print(f"Clothing: {args.clothing_mesh}")
    print(f"Output dir: {args.output_dir}")
    print(f"Num samples: {args.num_samples}")
    print()

    # Generate samples
    for i in range(args.num_samples):
        print(f"\n[{i+1}/{args.num_samples}] Generating fitted clothed model...")

        # Initialize generator
        generator = FittedClothedModelGenerator(
            star_model_path=args.star_model,
            clothing_mesh_paths=args.clothing_mesh
        )

        # Load STAR model
        generator.load_star_model()

        # Generate random parameters
        params = generator.generate_random_parameters()

        # Apply shape
        generator.apply_shape_parameters(params['betas'])

        # Load clothing with automatic weight transfer
        if args.clothing_mesh:
            for clothing_path in args.clothing_mesh:
                generator.load_clothing_with_weight_transfer(clothing_path)

            # Apply pose (clothing will follow automatically now!)
            generator.apply_pose_parameters(params['pose'])

            # Run cloth simulation for natural draping
            generator.apply_cloth_simulation(frames=args.cloth_frames)

        # Export clothed model
        output_path = os.path.join(args.output_dir, f"fitted_{i+1:04d}")
        generator.export_clothed_model(output_path)

        # Save parameters
        param_path = os.path.join(args.output_dir, f"fitted_{i+1:04d}_params.pkl")
        with open(param_path, 'wb') as f:
            pickle.dump(params, f)

        # Cleanup
        generator.cleanup()

    print("\n" + "="*70)
    print("✓ All fitted clothed 3D models generated!")
    print("="*70)
    print(f"\nOutput: {args.output_dir}")
    print("\nKey improvement: Clothing now has proper skinning weights")
    print("  → Clothing follows body pose and shape correctly")
    print()


if __name__ == "__main__":
    main()
