---
name: ComfyUI Text-to-3D
description: Generate 3D models from text prompts using FLUX2 image generation and SAM3D reconstruction, with Blender validation
---

# ComfyUI Text-to-3D Skill

Generate production-ready 3D models from text descriptions using a two-phase pipeline: FLUX2 for high-quality image generation, followed by SAM3D for 3D reconstruction with texture baking.

## Capabilities

- Expand simple prompts into detailed FLUX2-optimized descriptions
- Generate images with optimal 3D reconstruction settings (3/4 angle, white background, no shadow)
- Automatic background removal using BiRefNetRMBG
- SAM3D depth estimation, sparse generation, and mesh decoding
- Texture baking with nvdiffrast rendering engine
- Import generated mesh.glb into Blender for validation
- Multi-angle orbit camera renders for quality assessment
- Automatic retry with improved prompts on failed renders

## When to Use This Skill

Use this skill when you need to:
- Create 3D models from text descriptions
- Generate game assets or architectural models
- Produce textured meshes for visualization
- Automate text-to-3D pipelines

## Prerequisites

- ComfyUI running at `http://192.168.0.51:8188` with:
  - FLUX2 models (flux2_dev_fp8mixed, mistral_3_small_flux2_bf16, flux2-vae)
  - SAM3D custom nodes
  - BiRefNetRMBG for background removal
- Blender with MCP addon active on port 9876 (for validation)
- GPU with 48GB+ VRAM (workflows are split for memory management)

## Pipeline Architecture

### Phase 1: Image Generation (FLUX2)
```
Text Prompt → FLUX2 Prompt Expansion → Image Generation (1248x832)
                                                ↓
                                      Save Image for Phase 2
```

### Phase 2: 3D Reconstruction (SAM3D)
```
Load Image → BiRefNetRMBG → Resize to 1024x1024 → Depth Estimation
                                    ↓
                            Sparse Structure Gen
                                    ↓
                              SLAT Generation
                                    ↓
              ┌─────────────────────┴─────────────────────┐
        Gaussian Decode                              Mesh Decode
        (gaussian.ply)                               (mesh.glb)
                                    ↓
                            Texture Bake → mesh_textured.glb
```

### Phase 3: Validation (Blender)
```
Import mesh.glb → Setup Lighting → Orbit Camera Renders → Quality Assessment
                                                               ↓
                                               Retry with improved prompt if needed
```

## FLUX2 Prompt Expansion Rules

When expanding user prompts, follow these guidelines:

### Mandatory Elements (Always Include)
- **3/4 angle view**: "viewed from three-quarter angle" or "3/4 perspective view"
- **White background**: "isolated on clean white background" or "pure white backdrop"
- **No shadows**: "without cast shadows" or "shadowless presentation"
- **Infinite depth of field**: "infinite depth of field, everything in sharp focus"

### Recommended Elements
- **Camera/Lens**: "shot on Hasselblad X2D 100C, 45mm f/3.5 lens"
- **Lighting**: "soft studio lighting" or "even diffused lighting"
- **Subject position**: "centered composition"

### FLUX2 Best Practices
- Use Subject + Action + Style + Context framework
- Include HEX colors for specific tones (e.g., "oxidized copper #4A7C59")
- Target 30-80 words optimal length
- No negative prompts (FLUX2 doesn't support them)
- Avoid quality tags (unnecessary for FLUX2)

### Example Expansion

**User Input**: "vintage sports car"

**Expanded FLUX2 Prompt**:
```
Sleek vintage 1960s sports car with chrome bumpers and wire-spoke wheels,
classic Italian racing red #C41E3A exterior with cream leather interior visible
through open windows. Elegant curves of the bodywork catching soft studio light,
polished chrome grille and dual exhausts, shot on Hasselblad X2D 100C with 45mm
f/3.5 lens, three-quarter angle view, infinite depth of field with everything
in sharp focus, centered composition isolated on clean white background without
cast shadows
```

## Image Dimensions

| Orientation | FLUX2 Generation | SAM3D Processing |
|-------------|------------------|------------------|
| Landscape (default) | 1248 x 832 | 1024 x 1024 |
| Portrait | 832 x 1248 | 1024 x 1024 |

## Segmentation Prompt Distillation

The SAM3 segmentation prompt should be a simple noun capturing the whole object:

| FLUX2 Prompt Subject | Segmentation Prompt |
|---------------------|---------------------|
| "vintage 1960s sports car with chrome bumpers..." | "car" |
| "retro futurist skyscraper with gothic elements..." | "building" |
| "ornate victorian armchair with velvet upholstery..." | "chair" |
| "detailed fantasy sword with dragon hilt..." | "sword" |

## Workflow Execution

### Step 1: Submit Phase 1 (FLUX2)
```bash
# Load and customize flux2-phase1-generate.json template
# Set prompt in node 95 (PrimitiveString)
# Set dimensions in node 79 (EmptyFlux2LatentImage)
# Submit to ComfyUI API
```

### Step 2: Wait and Transfer
```bash
# Wait for image generation to complete
# Copy output image to storage-input for Phase 2
sudo cp /path/to/output/image.png /path/to/storage-input/
```

### Step 3: Submit Phase 2 (SAM3D)
```bash
# Load flux2-phase2-sam3d-rmbg.json template
# Update node 88 (LoadImage) with generated image filename
# Submit to ComfyUI API
```

### Step 4: Retrieve Outputs
Generated files in ComfyUI output:
- `gaussian.ply` - Gaussian splatting representation
- `mesh.glb` - Base 3D mesh
- `mesh_textured.glb` - Textured mesh (from SAM3DTextureBake)
- `pointcloud.ply` - Dense point cloud

### Step 5: Blender Validation
```python
# Import mesh into Blender via MCP
import_model(file_path="/path/to/mesh_textured.glb", format="gltf")

# Setup lighting
execute_script("""
import bpy
# Delete default objects
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# Add environment lighting
bpy.ops.world.new()
world = bpy.data.worlds[-1]
bpy.context.scene.world = world
world.use_nodes = True
nodes = world.node_tree.nodes
bg = nodes.get('Background')
bg.inputs['Strength'].default_value = 0.5
""")

# Create orbit camera positions
execute_script("""
import bpy
import math

# Create camera
bpy.ops.object.camera_add(location=(5, -5, 3))
cam = bpy.context.active_object
cam.name = 'OrbitCam'

# Point at origin
constraint = cam.constraints.new('TRACK_TO')
constraint.target = bpy.data.objects.get('imported_mesh')
constraint.track_axis = 'TRACK_NEGATIVE_Z'
constraint.up_axis = 'UP_Y'
""")

# Render from multiple angles
render_scene(output_path="/tmp/render_front.png", resolution_x=1920, resolution_y=1080)
```

## Error Handling

### GPU Memory Issues
- Phase 1 and Phase 2 are split to prevent OOM
- Restart ComfyUI between phases if needed
- Monitor GPU memory: `nvidia-smi`

### Segmentation Failures
If BiRefNetRMBG fails to isolate the subject:
1. Check if subject is clearly distinguishable from background
2. Ensure FLUX2 prompt specified white background
3. Try a simpler object shape

### SAM3D Reconstruction Issues
If mesh quality is poor:
1. Verify input image has clear subject at 3/4 angle
2. Check that background was properly removed
3. Adjust SAM3D inference steps (default: 25)

## Templates Directory

```
/skills/comfyui-3d/templates/
├── flux2-phase1-generate.json      # FLUX2 image generation
├── flux2-phase2-sam3d-rmbg.json    # SAM3D with BiRefNet background removal
└── blender-validation.py           # Blender orbit render script
```

## Integration with Blender MCP

This skill works with the Blender MCP addon on port 9876. Ensure:
1. Blender is running with the addon active
2. MCP server shows "Connected" status
3. File paths are accessible from Blender's context

## Example Usage

```
Create a 3D model of a medieval castle tower with stone walls and a conical roof
```

**Skill Execution**:
1. Expands prompt with FLUX2 best practices
2. Generates image at 1248x832
3. Processes through SAM3D at 1024x1024
4. Produces mesh.glb with baked textures
5. Imports to Blender for orbit validation renders
6. Returns render previews for quality assessment

## Performance Notes

| Phase | GPU VRAM | Duration |
|-------|----------|----------|
| FLUX2 Generation | ~41GB | 30-60s |
| SAM3D Reconstruction | ~23GB | 2-5min |
| Blender Render | ~2GB | 10-30s per angle |

## Advanced Configuration

### SAM3D Parameters
- `stage1_inference_steps`: 25 (sparse structure)
- `stage1_cfg_strength`: 7
- `stage2_inference_steps`: 25 (SLAT generation)
- `stage2_cfg_strength`: 5
- `texture_size`: 1024
- `simplify`: 0.95

### FLUX2 Parameters
- `steps`: 28
- `guidance`: 4
- `sampler`: euler
- `scheduler`: simple
