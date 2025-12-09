# Example: Vintage Sports Car

This example demonstrates the complete text-to-3D pipeline for generating a vintage sports car model.

## User Input
```
vintage 1960s sports car
```

## Step 1: Prompt Expansion (FLUX2)

**Expanded Prompt:**
```
Sleek vintage 1960s sports car with chrome bumpers and wire-spoke wheels, classic
Italian racing red #C41E3A exterior with cream leather interior visible through
open windows, elegant curves of the bodywork catching soft studio light, polished
chrome grille and dual exhausts, shot on Hasselblad X2D 100C with 45mm f/3.5 lens,
three-quarter angle view, infinite depth of field with everything in sharp focus,
centered composition isolated on clean white background without cast shadows
```

**Segmentation Prompt:** `car`

## Step 2: Phase 1 Workflow (FLUX2)

Configuration:
- Resolution: 1248 x 832 (landscape)
- Steps: 28
- Guidance: 4
- Sampler: euler
- Scheduler: simple

Output: `Vintage_Car_00001_.png`

## Step 3: Phase 2 Workflow (SAM3D)

Processing:
1. BiRefNetRMBG background removal
2. Resize to 1024x1024 with white padding
3. SAM3D depth estimation
4. Sparse structure generation (25 steps, CFG 7)
5. SLAT generation (25 steps, CFG 5)
6. Mesh decode with 0.95 simplification
7. Gaussian splatting decode
8. Texture baking at 1024px

Outputs:
- `gaussian.ply` (~60MB)
- `mesh.glb` (~1MB)
- `mesh_textured.glb` (~5MB)
- `pointcloud.ply` (~70MB)

## Step 4: Blender Validation

Camera positions (8 angles, radius 5, height 2):
- Front (0°)
- Front-Right (45°)
- Right (90°)
- Back-Right (135°)
- Back (180°)
- Back-Left (225°)
- Left (270°)
- Front-Left (315°)

Lighting:
- Key light: Area, 500W, position (3, -3, 4)
- Fill light: Area, 200W, position (-3, -2, 2)
- Rim light: Area, 300W, position (0, 4, 3)

Render: EEVEE, 1920x1080, 64 samples

## Expected Results

Quality indicators for successful generation:
- [x] Complete car body with all panels
- [x] Visible wheels and tires
- [x] Chrome details preserved
- [x] Interior visible through windows
- [x] Clean silhouette in all angles
- [x] Consistent texture across views

## If Retry Needed

Common issues and fixes:

| Issue | Prompt Addition |
|-------|-----------------|
| Missing wheels | "complete vehicle with all four wheels clearly visible" |
| Incomplete body | "fully detailed complete car body from hood to trunk" |
| Poor chrome | "highly reflective chrome trim and bumpers" |
| Texture issues | "consistent glossy paint finish throughout" |

## CLI Execution

```bash
# Run full pipeline
./tools/text-to-3d-pipeline.js "vintage 1960s sports car"

# With portrait orientation
./tools/text-to-3d-pipeline.js "vintage 1960s sports car" --portrait

# Skip Blender validation
./tools/text-to-3d-pipeline.js "vintage 1960s sports car" --no-blender

# Get improvement suggestions
./tools/prompt-improver.js "vintage 1960s sports car" --feedback "wheels are missing"
```
