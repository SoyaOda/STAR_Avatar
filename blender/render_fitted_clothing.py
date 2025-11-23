"""
Blender Script: Render Fitted Clothing (Simplified)

This script is MUCH simpler than the previous version because:
1. Clothing is already fitted to body shape (no weight transfer needed)
2. No rigging needed (static meshes)
3. Just import OBJ files and render

Usage:
    blender --background --python render_fitted_clothing.py -- \
        --data_dir ../outputs/fitted_clothing_data \
        --output_dir ../outputs/rendered_clothing \
        --num_samples 5
"""

import bpy
import numpy as np
import os
import sys
import argparse
from mathutils import Vector, Euler
import random


class SimpleFittedClothingRenderer:
    """
    Simple renderer for pre-fitted clothing meshes.

    No rigging, no weight transfer - just load and render!
    """

    def __init__(self, hdri_dir=None):
        """Initialize renderer."""
        self.hdri_dir = hdri_dir
        self.hdri_files = []

        if hdri_dir and os.path.exists(hdri_dir):
            self.hdri_files = [
                f for f in os.listdir(hdri_dir)
                if f.lower().endswith(('.hdr', '.exr'))
            ]
            print(f"✓ Found {len(self.hdri_files)} HDRI files")

        self._setup_scene()

    def _setup_scene(self):
        """Setup clean Blender scene with camera and lighting."""
        # Clear scene
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete()

        # Setup camera
        bpy.ops.object.camera_add(location=(0, -3.0, 1.2))
        camera = bpy.context.object
        camera.rotation_euler = Euler((np.radians(75), 0, 0), 'XYZ')
        bpy.context.scene.camera = camera

        # Setup render settings (高品質化)
        bpy.context.scene.render.engine = 'CYCLES'
        bpy.context.scene.cycles.samples = 128  # 64 → 128 (より滑らか)
        bpy.context.scene.render.resolution_x = 1024  # 512 → 1024
        bpy.context.scene.render.resolution_y = 1024  # 512 → 1024

        # Enable denoising for cleaner output
        bpy.context.scene.cycles.use_denoising = True

        print("✓ Scene setup complete")

    def load_sample(self, sample_dir):
        """
        Load body and clothing meshes from sample directory.

        Args:
            sample_dir: Directory containing body.obj and clothing.obj
        """
        # Clear existing mesh objects (but keep camera and lights)
        bpy.ops.object.select_all(action='DESELECT')
        for obj in bpy.context.scene.objects:
            if obj.type == 'MESH':
                obj.select_set(True)
        bpy.ops.object.delete()

        body_path = os.path.join(sample_dir, "body.obj")

        # Support both "clothing.obj" and "pants.obj"
        clothing_path = os.path.join(sample_dir, "clothing.obj")
        if not os.path.exists(clothing_path):
            clothing_path = os.path.join(sample_dir, "pants.obj")

        if not os.path.exists(body_path):
            raise FileNotFoundError(f"Body mesh not found: {body_path}")

        # Load body (Blender 5.0+ compatible)
        # Clear selection first
        bpy.ops.object.select_all(action='DESELECT')

        # Import using new API (Blender 5.0+)
        try:
            bpy.ops.wm.obj_import(filepath=body_path)
        except AttributeError:
            # Fallback for older Blender versions
            bpy.ops.import_scene.obj(filepath=body_path)

        body_obj = bpy.context.selected_objects[0]
        body_obj.name = "Body"

        # Apply smooth shading to body
        bpy.context.view_layer.objects.active = body_obj
        bpy.ops.object.shade_smooth()

        # Create skin material
        body_mat = bpy.data.materials.new(name="Skin")
        body_mat.use_nodes = True
        nodes = body_mat.node_tree.nodes
        nodes.clear()

        # Principled BSDF for skin
        bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
        bsdf.inputs['Base Color'].default_value = (0.8, 0.6, 0.5, 1.0)  # Skin tone

        # Try to set subsurface scattering (Blender version compatibility)
        try:
            bsdf.inputs['Subsurface Weight'].default_value = 0.1
            bsdf.inputs['Subsurface Radius'].default_value = (0.9, 0.6, 0.5)
        except KeyError:
            # Older Blender versions
            try:
                bsdf.inputs['Subsurface'].default_value = 0.1
                bsdf.inputs['Subsurface Color'].default_value = (0.9, 0.6, 0.5, 1.0)
            except KeyError:
                pass  # Skip if not available

        bsdf.inputs['Roughness'].default_value = 0.4

        output = nodes.new(type='ShaderNodeOutputMaterial')
        body_mat.node_tree.links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])

        body_obj.data.materials.append(body_mat)

        print(f"✓ Loaded body: {body_path}")

        # Load clothing if it exists
        if os.path.exists(clothing_path):
            # Clear selection
            bpy.ops.object.select_all(action='DESELECT')

            # Import using new API (Blender 5.0+)
            try:
                bpy.ops.wm.obj_import(filepath=clothing_path)
            except AttributeError:
                # Fallback for older Blender versions
                bpy.ops.import_scene.obj(filepath=clothing_path)

            clothing_obj = bpy.context.selected_objects[0]
            clothing_obj.name = "Clothing"

            # Apply smooth shading to clothing
            bpy.context.view_layer.objects.active = clothing_obj
            bpy.ops.object.shade_smooth()

            # Create fabric material
            clothing_mat = bpy.data.materials.new(name="Fabric")
            clothing_mat.use_nodes = True
            nodes = clothing_mat.node_tree.nodes
            nodes.clear()

            # Principled BSDF for fabric
            bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')

            # Random fabric colors
            fabric_colors = [
                (0.1, 0.1, 0.2, 1.0),   # Dark blue
                (0.8, 0.1, 0.1, 1.0),   # Red
                (0.1, 0.1, 0.1, 1.0),   # Black
                (0.9, 0.9, 0.9, 1.0),   # White
                (0.2, 0.6, 0.3, 1.0),   # Green
            ]

            bsdf.inputs['Base Color'].default_value = random.choice(fabric_colors)
            bsdf.inputs['Roughness'].default_value = 0.7

            output = nodes.new(type='ShaderNodeOutputMaterial')
            clothing_mat.node_tree.links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])

            clothing_obj.data.materials.append(clothing_mat)

            print(f"✓ Loaded clothing: {clothing_path}")
        else:
            print(f"⚠️  No clothing mesh found, rendering body only")

    def setup_hdri_lighting(self, hdri_path=None):
        """
        Setup HDRI environment lighting.

        Args:
            hdri_path: Path to HDRI file (if None, use random from hdri_dir)
        """
        # Get HDRI path
        if hdri_path is None and self.hdri_files:
            hdri_file = random.choice(self.hdri_files)
            hdri_path = os.path.join(self.hdri_dir, hdri_file)

        if hdri_path and os.path.exists(hdri_path):
            # Setup world environment
            world = bpy.context.scene.world
            world.use_nodes = True
            nodes = world.node_tree.nodes
            nodes.clear()

            # Environment texture
            env_tex = nodes.new(type='ShaderNodeTexEnvironment')
            env_tex.image = bpy.data.images.load(hdri_path)

            # Background
            background = nodes.new(type='ShaderNodeBackground')
            background.inputs['Strength'].default_value = 1.0

            # Output
            output = nodes.new(type='ShaderNodeOutputWorld')

            # Connect nodes
            world.node_tree.links.new(env_tex.outputs['Color'], background.inputs['Color'])
            world.node_tree.links.new(background.outputs['Background'], output.inputs['Surface'])

            print(f"✓ HDRI lighting: {os.path.basename(hdri_path)}")
        else:
            # Improved 3-point lighting (instead of single sun)
            print("✓ Using 3-point lighting setup (no HDRI)")

            # Key light (main light)
            bpy.ops.object.light_add(type='AREA', location=(2, -3, 3))
            key_light = bpy.context.object
            key_light.data.energy = 300
            key_light.data.size = 2.0
            key_light.rotation_euler = (np.radians(60), 0, np.radians(-30))

            # Fill light (softer, opposite side)
            bpy.ops.object.light_add(type='AREA', location=(-2, -2, 2))
            fill_light = bpy.context.object
            fill_light.data.energy = 150
            fill_light.data.size = 2.5
            fill_light.rotation_euler = (np.radians(45), 0, np.radians(30))

            # Rim/back light (separation from background)
            bpy.ops.object.light_add(type='AREA', location=(0, 2, 2))
            rim_light = bpy.context.object
            rim_light.data.energy = 100
            rim_light.data.size = 1.5
            rim_light.rotation_euler = (np.radians(120), 0, 0)

            # Set background to lighter gray
            world = bpy.context.scene.world
            if not world.use_nodes:
                world.use_nodes = True
            bg_node = world.node_tree.nodes.get('Background')
            if bg_node:
                bg_node.inputs['Color'].default_value = (0.25, 0.25, 0.25, 1.0)  # Lighter gray

    def render(self, output_path):
        """
        Render the scene.

        Args:
            output_path: Output image path
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        bpy.context.scene.render.filepath = output_path
        bpy.ops.render.render(write_still=True)
        print(f"✓ Rendered: {output_path}")


def parse_args():
    """Parse command line arguments."""
    # Blender passes args after '--'
    argv = sys.argv
    if '--' in argv:
        argv = argv[argv.index('--') + 1:]
    else:
        argv = []

    parser = argparse.ArgumentParser(
        description='Render fitted clothing meshes in Blender'
    )
    parser.add_argument(
        '--data_dir', type=str, required=True,
        help='Directory containing sample subdirectories (with body.obj and clothing.obj)'
    )
    parser.add_argument(
        '--output_dir', type=str, required=True,
        help='Output directory for rendered images'
    )
    parser.add_argument(
        '--num_samples', type=int, default=None,
        help='Number of samples to render (default: all)'
    )
    parser.add_argument(
        '--hdri_dir', type=str, default=None,
        help='Directory containing HDRI files for lighting'
    )

    return parser.parse_args(argv)


def main():
    """Main rendering loop."""
    args = parse_args()

    print("\n" + "="*70)
    print("Render Fitted Clothing - Simplified Pipeline")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Data dir: {args.data_dir}")
    print(f"  Output dir: {args.output_dir}")
    print(f"  HDRI dir: {args.hdri_dir if args.hdri_dir else 'None (simple lighting)'}")

    # Initialize renderer
    renderer = SimpleFittedClothingRenderer(hdri_dir=args.hdri_dir)

    # Get list of sample directories
    if not os.path.exists(args.data_dir):
        print(f"❌ Data directory not found: {args.data_dir}")
        return

    sample_dirs = sorted([
        os.path.join(args.data_dir, d)
        for d in os.listdir(args.data_dir)
        if os.path.isdir(os.path.join(args.data_dir, d)) and d.startswith('sample_')
    ])

    if args.num_samples:
        sample_dirs = sample_dirs[:args.num_samples]

    print(f"\nFound {len(sample_dirs)} samples to render")

    # Render each sample
    for i, sample_dir in enumerate(sample_dirs):
        print(f"\n{'='*70}")
        print(f"Rendering Sample {i+1}/{len(sample_dirs)}")
        print(f"{'='*70}")

        try:
            # Load sample
            renderer.load_sample(sample_dir)

            # Setup lighting
            renderer.setup_hdri_lighting()

            # Render
            sample_name = os.path.basename(sample_dir)
            output_path = os.path.join(args.output_dir, f"{sample_name}.png")
            renderer.render(output_path)

        except Exception as e:
            print(f"❌ Error rendering sample {i+1}: {e}")
            continue

    print("\n" + "="*70)
    print("✓ Rendering Complete!")
    print("="*70)
    print(f"\nOutput directory: {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
