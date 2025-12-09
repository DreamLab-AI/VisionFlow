"""
Blender Validation Script for ComfyUI Text-to-3D Skill

This script is executed via Blender MCP to validate generated 3D models
by rendering from multiple orbit camera angles.

Usage: Execute via Blender MCP execute_script tool
"""

import bpy
import math
import os

def setup_scene():
    """Clear scene and setup basic environment"""
    # Delete all existing objects
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

    # Create new world for environment lighting
    if not bpy.context.scene.world:
        bpy.ops.world.new()

    world = bpy.context.scene.world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links

    # Clear existing nodes
    nodes.clear()

    # Create background node with soft gray
    bg_node = nodes.new('ShaderNodeBackground')
    bg_node.inputs['Color'].default_value = (0.9, 0.9, 0.9, 1.0)
    bg_node.inputs['Strength'].default_value = 1.0

    # Create output node
    output_node = nodes.new('ShaderNodeOutputWorld')
    output_node.location = (300, 0)

    # Link background to output
    links.new(bg_node.outputs['Background'], output_node.inputs['Surface'])

def import_glb(file_path):
    """Import GLB/GLTF model"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Model file not found: {file_path}")

    bpy.ops.import_scene.gltf(filepath=file_path)

    # Get imported objects
    imported = [obj for obj in bpy.context.selected_objects]

    if not imported:
        raise RuntimeError("No objects were imported from GLB file")

    return imported

def center_and_scale_model(objects):
    """Center model at origin and scale to fit viewport"""
    # Find bounding box of all imported objects
    min_coords = [float('inf')] * 3
    max_coords = [float('-inf')] * 3

    for obj in objects:
        if obj.type == 'MESH':
            for vertex in obj.bound_box:
                world_vertex = obj.matrix_world @ mathutils.Vector(vertex)
                for i in range(3):
                    min_coords[i] = min(min_coords[i], world_vertex[i])
                    max_coords[i] = max(max_coords[i], world_vertex[i])

    # Calculate center and size
    center = [(min_coords[i] + max_coords[i]) / 2 for i in range(3)]
    size = max(max_coords[i] - min_coords[i] for i in range(3))

    # Create empty at center as parent
    bpy.ops.object.empty_add(type='PLAIN_AXES', location=(0, 0, 0))
    parent_empty = bpy.context.active_object
    parent_empty.name = "ModelParent"

    # Parent all imported objects and center
    for obj in objects:
        obj.parent = parent_empty
        obj.location.x -= center[0]
        obj.location.y -= center[1]
        obj.location.z -= center[2]

    # Scale to fit (target size ~2 units)
    if size > 0:
        scale_factor = 2.0 / size
        parent_empty.scale = (scale_factor, scale_factor, scale_factor)

    return parent_empty

def setup_lighting():
    """Setup three-point lighting for product visualization"""
    # Key light (main)
    bpy.ops.object.light_add(type='AREA', location=(3, -3, 4))
    key_light = bpy.context.active_object
    key_light.name = "KeyLight"
    key_light.data.energy = 500
    key_light.data.size = 2
    key_light.rotation_euler = (math.radians(45), 0, math.radians(45))

    # Fill light (softer, opposite side)
    bpy.ops.object.light_add(type='AREA', location=(-3, -2, 2))
    fill_light = bpy.context.active_object
    fill_light.name = "FillLight"
    fill_light.data.energy = 200
    fill_light.data.size = 3
    fill_light.rotation_euler = (math.radians(60), 0, math.radians(-45))

    # Rim light (back)
    bpy.ops.object.light_add(type='AREA', location=(0, 4, 3))
    rim_light = bpy.context.active_object
    rim_light.name = "RimLight"
    rim_light.data.energy = 300
    rim_light.data.size = 2
    rim_light.rotation_euler = (math.radians(-45), 0, math.radians(180))

def create_orbit_cameras(radius=5, height=2, num_angles=8):
    """Create cameras in orbit around origin"""
    cameras = []

    for i in range(num_angles):
        angle = (2 * math.pi * i) / num_angles
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        z = height

        bpy.ops.object.camera_add(location=(x, y, z))
        cam = bpy.context.active_object
        cam.name = f"OrbitCam_{i:02d}"

        # Point camera at origin
        direction = mathutils.Vector((0, 0, 0)) - cam.location
        rot_quat = direction.to_track_quat('-Z', 'Y')
        cam.rotation_euler = rot_quat.to_euler()

        cameras.append(cam)

    return cameras

def render_all_angles(cameras, output_dir, resolution=(1920, 1080)):
    """Render from all camera angles"""
    scene = bpy.context.scene

    # Setup render settings
    scene.render.resolution_x = resolution[0]
    scene.render.resolution_y = resolution[1]
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'

    # Use Cycles for better quality (or EEVEE for speed)
    scene.render.engine = 'BLENDER_EEVEE'
    scene.eevee.taa_render_samples = 64

    os.makedirs(output_dir, exist_ok=True)

    renders = []
    for i, cam in enumerate(cameras):
        scene.camera = cam
        output_path = os.path.join(output_dir, f"render_{i:02d}.png")
        scene.render.filepath = output_path
        bpy.ops.render.render(write_still=True)
        renders.append(output_path)
        print(f"Rendered angle {i+1}/{len(cameras)}: {output_path}")

    return renders

def validate_model(glb_path, output_dir="/tmp/blender_validation"):
    """
    Main validation function - import model, setup scene, render from multiple angles

    Args:
        glb_path: Path to the GLB file to validate
        output_dir: Directory to save rendered images

    Returns:
        List of rendered image paths
    """
    import mathutils  # Import here to ensure it's available

    # Setup clean scene
    setup_scene()

    # Import model
    imported_objects = import_glb(glb_path)

    # Center and scale
    model_parent = center_and_scale_model(imported_objects)

    # Setup lighting
    setup_lighting()

    # Create orbit cameras (8 angles)
    cameras = create_orbit_cameras(radius=5, height=2, num_angles=8)

    # Render all angles
    renders = render_all_angles(cameras, output_dir)

    return renders

# Example usage (when called from Blender MCP):
# result = validate_model("/path/to/mesh.glb", "/tmp/renders")
# print(f"Rendered {len(result)} images")
