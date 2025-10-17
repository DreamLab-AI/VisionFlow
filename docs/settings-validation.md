# Settings Validation Rules

## Overview

The settings validation system ensures all configuration values are safe, valid, and within acceptable ranges before being persisted to the database. Validation prevents GPU kernel crashes, physics instabilities, and UI rendering errors.

## Validation Architecture

### Validation Pipeline

```
Incoming Settings
      ↓
[1] Type Checking
      ↓
[2] Format Validation (regex)
      ↓
[3] Range Validation
      ↓
[4] Cross-Field Validation
      ↓
[5] GPU Safety Checks
      ↓
Validated Settings → Database
```

## Validation Rules by Category

### Rendering Settings

#### `ambientLightIntensity`
- **Type:** `f32`
- **Range:** `0.0` to `10.0`
- **Default:** `0.5`
- **Description:** Ambient light intensity for scene illumination
- **Error:** "Ambient light intensity must be between 0.0 and 10.0"

#### `directionalLightIntensity`
- **Type:** `f32`
- **Range:** `0.0` to `10.0`
- **Default:** `0.4`
- **Description:** Directional light intensity
- **Error:** "Directional light intensity must be between 0.0 and 10.0"

#### `environmentIntensity`
- **Type:** `f32`
- **Range:** `0.0` to `10.0`
- **Default:** `0.3`
- **Description:** Environment map intensity
- **Error:** "Environment intensity must be between 0.0 and 10.0"

#### `backgroundColor`
- **Type:** `String`
- **Format:** Hex color (`#RRGGBB` or `#RRGGBBAA`)
- **Regex:** `^#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{8})$`
- **Default:** `"#000000"`
- **Error:** "Background color must be a valid hex color (#RRGGBB or #RRGGBBAA)"

#### `enableAmbientOcclusion`
- **Type:** `bool`
- **Default:** `false`
- **Description:** Enable screen-space ambient occlusion (SSAO)

#### `enableAntialiasing`
- **Type:** `bool`
- **Default:** `true`
- **Description:** Enable multi-sample anti-aliasing (MSAA)

#### `enableShadows`
- **Type:** `bool`
- **Default:** `false`
- **Description:** Enable shadow mapping

#### `shadowMapSize`
- **Type:** `String`
- **Enum:** `["512", "1024", "2048", "4096"]`
- **Default:** `"2048"`
- **Error:** "Shadow map size must be 512, 1024, 2048, or 4096"

#### `shadowBias`
- **Type:** `f32`
- **Range:** `0.0` to `0.01`
- **Default:** `0.0001`
- **Error:** "Shadow bias must be between 0.0 and 0.01"

---

### Animation Settings

#### `enableMotionBlur`
- **Type:** `bool`
- **Default:** `true`

#### `motionBlurStrength`
- **Type:** `f32`
- **Range:** `0.0` to `1.0`
- **Default:** `0.1`
- **Error:** "Motion blur strength must be between 0.0 and 1.0"

#### `enableNodeAnimations`
- **Type:** `bool`
- **Default:** `true`

#### `pulseEnabled`
- **Type:** `bool`
- **Default:** `true`

#### `pulseSpeed`
- **Type:** `f32`
- **Range:** `0.1` to `10.0`
- **Default:** `1.2`
- **Error:** "Pulse speed must be between 0.1 and 10.0"

#### `pulseStrength`
- **Type:** `f32`
- **Range:** `0.0` to `2.0`
- **Default:** `0.8`
- **Error:** "Pulse strength must be between 0.0 and 2.0"

#### `waveSpeed`
- **Type:** `f32`
- **Range:** `0.1` to `10.0`
- **Default:** `0.5`
- **Error:** "Wave speed must be between 0.1 and 10.0"

---

### Glow Settings

Glow settings are critical for GPU stability. Invalid values can crash shaders.

#### `enabled`
- **Type:** `bool`
- **Default:** `false`

#### `intensity`
- **Type:** `f32`
- **Range:** `0.0` to `10.0`
- **Default:** `1.2`
- **Must be finite:** No NaN or Infinity
- **Error:** "Glow intensity must be between 0.0 and 10.0 and finite"

#### `radius`
- **Type:** `f32`
- **Range:** `0.0` to `10.0`
- **Default:** `1.2`
- **Must be finite:** No NaN or Infinity
- **Error:** "Glow radius must be between 0.0 and 10.0 and finite"

#### `threshold`
- **Type:** `f32`
- **Range:** `0.0` to `1.0`
- **Default:** `0.39`
- **Must be finite:** No NaN or Infinity
- **Error:** "Glow threshold must be between 0.0 and 1.0 and finite"

#### `opacity`
- **Type:** `f32`
- **Range:** `0.0` to `1.0`
- **Default:** `0.6`
- **Error:** "Glow opacity must be between 0.0 and 1.0"

#### `baseColor`
- **Type:** `String`
- **Format:** Hex color
- **Default:** `"#00ffff"`
- **Error:** "Base color must be a valid hex color"

#### `emissionColor`
- **Type:** `String`
- **Format:** Hex color
- **Default:** `"#00e5ff"`
- **Error:** "Emission color must be a valid hex color"

#### Cross-Field Validation
- If `enabled = true`, all numeric fields must be finite (no NaN/Infinity)
- Color fields must be valid hex format when glow is enabled

---

### Bloom Settings

Similar to glow, bloom settings require strict validation for GPU safety.

#### `enabled`
- **Type:** `bool`
- **Default:** `true`

#### `intensity`
- **Type:** `f32`
- **Range:** `0.0` to `10.0`
- **Must be finite**
- **Error:** "Bloom intensity must be between 0.0 and 10.0 and finite"

#### `radius`
- **Type:** `f32`
- **Range:** `0.0` to `10.0`
- **Must be finite**
- **Error:** "Bloom radius must be between 0.0 and 10.0 and finite"

#### `threshold`
- **Type:** `f32`
- **Range:** `0.0` to `1.0`
- **Must be finite**
- **Error:** "Bloom threshold must be between 0.0 and 1.0 and finite"

#### `strength`
- **Type:** `f32`
- **Range:** `0.0` to `1.0`
- **Error:** "Bloom strength must be between 0.0 and 1.0"

#### `knee`
- **Type:** `f32`
- **Range:** `0.0` to `2.0`
- **Error:** "Bloom knee must be between 0.0 and 2.0"

---

### Node Settings

#### `baseColor`
- **Type:** `String`
- **Format:** Hex color
- **Error:** "Node base color must be a valid hex color"

#### `metalness`
- **Type:** `f32`
- **Range:** `0.0` to `1.0`
- **Error:** "Metalness must be between 0.0 and 1.0"

#### `opacity`
- **Type:** `f32`
- **Range:** `0.0` to `1.0`
- **Error:** "Opacity must be between 0.0 and 1.0"

#### `roughness`
- **Type:** `f32`
- **Range:** `0.0` to `1.0`
- **Error:** "Roughness must be between 0.0 and 1.0"

#### `nodeSize`
- **Type:** `f32`
- **Range:** `0.1` to `100.0`
- **Default:** `1.7`
- **Error:** "Node size must be between 0.1 and 100.0"

#### `quality`
- **Type:** `String`
- **Enum:** `["low", "medium", "high", "ultra"]`
- **Default:** `"high"`
- **Error:** "Quality must be low, medium, high, or ultra"

---

### Edge Settings

#### `arrowSize`
- **Type:** `f32`
- **Range:** `0.01` to `5.0`
- **Default:** `0.02`
- **Error:** "Arrow size must be between 0.01 and 5.0"

#### `baseWidth`
- **Type:** `f32`
- **Range:** `0.01` to `5.0`
- **Default:** `0.5`
- **Error:** "Base width must be between 0.01 and 5.0"

#### `color`
- **Type:** `String`
- **Format:** Hex color
- **Error:** "Edge color must be a valid hex color"

#### `opacity`
- **Type:** `f32`
- **Range:** `0.0` to `1.0`
- **Error:** "Edge opacity must be between 0.0 and 1.0"

#### `widthRange`
- **Type:** `Vec<f32>` (exactly 2 elements)
- **Constraints:**
  - Length must be exactly 2
  - `widthRange[0] < widthRange[1]`
  - Both values in range `0.1` to `10.0`
- **Default:** `[0.3, 1.5]`
- **Errors:**
  - "Width range must have exactly 2 elements"
  - "Width range minimum must be less than maximum"
  - "Width range values must be between 0.1 and 10.0"

---

### Physics Settings

Physics parameters are validated to prevent simulation instabilities.

#### `springK`
- **Type:** `f32`
- **Range:** `0.1` to `100.0`
- **Default:** `10.0`
- **Error:** "Spring constant must be between 0.1 and 100.0"

#### `repelK`
- **Type:** `f32`
- **Range:** `1.0` to `1000.0`
- **Default:** `100.0`
- **Error:** "Repulsion constant must be between 1.0 and 1000.0"

#### `damping`
- **Type:** `f32`
- **Range:** `0.0` to `1.0`
- **Default:** `0.1`
- **Error:** "Damping must be between 0.0 and 1.0"

#### `dt`
- **Type:** `f32`
- **Range:** `0.001` to `0.1`
- **Default:** `0.016`
- **Error:** "Timestep must be between 0.001 and 0.1"

#### `maxVelocity`
- **Type:** `f32`
- **Range:** `1.0` to `1000.0`
- **Default:** `50.0`
- **Error:** "Max velocity must be between 1.0 and 1000.0"

#### `maxForce`
- **Type:** `f32`
- **Range:** `1.0` to `10000.0`
- **Default:** `500.0`
- **Error:** "Max force must be between 1.0 and 10000.0"

#### `iterations`
- **Type:** `u32`
- **Range:** `1` to `1000`
- **Default:** `50`
- **Error:** "Iterations must be between 1 and 1000"

#### `massScale`
- **Type:** `f32`
- **Range:** `0.1` to `10.0`
- **Default:** `1.0`
- **Error:** "Mass scale must be between 0.1 and 10.0"

#### `boundaryDamping`
- **Type:** `f32`
- **Range:** `0.0` to `1.0`
- **Default:** `0.95`
- **Error:** "Boundary damping must be between 0.0 and 1.0"

#### `boundsSize`
- **Type:** `f32`
- **Range:** `100.0` to `10000.0`
- **Default:** `1000.0`
- **Error:** "Bounds size must be between 100.0 and 10000.0"

#### `separationRadius`
- **Type:** `f32`
- **Range:** `0.1` to `50.0`
- **Default:** `2.0`
- **Error:** "Separation radius must be between 0.1 and 50.0"

#### `restLength`
- **Type:** `f32`
- **Range:** `1.0` to `1000.0`
- **Default:** `50.0`
- **Error:** "Rest length must be between 1.0 and 1000.0"

#### `repulsionCutoff`
- **Type:** `f32`
- **Range:** `5.0` to `200.0`
- **Default:** `50.0`
- **Error:** "Repulsion cutoff must be between 5.0 and 200.0"

#### `repulsionSofteningEpsilon`
- **Type:** `f32`
- **Range:** `0.000001` to `1.0`
- **Default:** `0.0001`
- **Error:** "Repulsion softening must be between 0.000001 and 1.0"

#### `centerGravityK`
- **Type:** `f32`
- **Range:** `0.0` to `0.1`
- **Default:** `0.0`
- **Error:** "Center gravity must be between 0.0 and 0.1"

#### `gridCellSize`
- **Type:** `f32`
- **Range:** `1.0` to `100.0`
- **Default:** `50.0`
- **Error:** "Grid cell size must be between 1.0 and 100.0"

#### `warmupIterations`
- **Type:** `u32`
- **Range:** `0` to `1000`
- **Default:** `100`
- **Error:** "Warmup iterations must be between 0 and 1000"

#### `coolingRate`
- **Type:** `f32`
- **Range:** `0.0001` to `0.1`
- **Default:** `0.001`
- **Error:** "Cooling rate must be between 0.0001 and 0.1"

#### `warmupCurve`
- **Type:** `String`
- **Enum:** `["linear", "exponential", "logarithmic"]`
- **Default:** `"exponential"`
- **Error:** "Warmup curve must be linear, exponential, or logarithmic"

#### `computeMode`
- **Type:** `u32`
- **Enum:** `[0, 1, 2]`
- **Values:** `0` = CPU, `1` = GPU, `2` = Hybrid
- **Default:** `1`
- **Error:** "Compute mode must be 0 (CPU), 1 (GPU), or 2 (Hybrid)"

---

### Auto-Balance Configuration

#### `stabilityVarianceThreshold`
- **Type:** `f32`
- **Range:** `1.0` to `1000.0`
- **Default:** `100.0`

#### `stabilityFrameCount`
- **Type:** `u32`
- **Range:** `10` to `600`
- **Default:** `180`

#### `clusteringDistanceThreshold`
- **Type:** `f32`
- **Range:** `1.0` to `100.0`
- **Default:** `20.0`

#### `bouncingNodePercentage`
- **Type:** `f32`
- **Range:** `0.0` to `1.0`
- **Default:** `0.33`

#### `parameterAdjustmentRate`
- **Type:** `f32`
- **Range:** `0.01` to `1.0`
- **Default:** `0.1`

---

### Auto-Pause Configuration

#### `enabled`
- **Type:** `bool`
- **Default:** `true`

#### `equilibriumVelocityThreshold`
- **Type:** `f32`
- **Range:** `0.0` to `10.0`
- **Default:** `0.1`

#### `equilibriumCheckFrames`
- **Type:** `u32`
- **Range:** `1` to `300`
- **Default:** `30`

#### `equilibriumEnergyThreshold`
- **Type:** `f32`
- **Range:** `0.0` to `1.0`
- **Default:** `0.01`

---

### Network Settings

#### `port`
- **Type:** `u16`
- **Range:** `1` to `65535`
- **Default:** `4000`
- **Error:** "Port must be between 1 and 65535"

#### `bindAddress`
- **Type:** `String`
- **Format:** IP address
- **Default:** `"0.0.0.0"`

#### `domain`
- **Type:** `String`
- **Format:** Domain name
- **Regex:** `^[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?(\.[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?)*$`
- **Error:** "Domain must be a valid domain name"

#### `maxRequestSize`
- **Type:** `usize`
- **Range:** `1024` to `104857600` (1KB to 100MB)
- **Default:** `10485760` (10MB)

---

### WebSocket Settings

#### `updateRate`
- **Type:** `u32`
- **Range:** `1` to `120`
- **Default:** `60`
- **Error:** "Update rate must be between 1 and 120 Hz"

#### `maxConnections`
- **Type:** `u32`
- **Range:** `1` to `10000`
- **Default:** `100`

#### `heartbeatInterval`
- **Type:** `u32` (milliseconds)
- **Range:** `1000` to `60000`
- **Default:** `10000`

#### `maxMessageSize`
- **Type:** `usize` (bytes)
- **Range:** `1024` to `104857600`
- **Default:** `10485760`

---

## Cross-Field Validation Rules

### 1. Glow & Bloom Interaction
- If both `glow.enabled` and `bloom.enabled` are `true`, total intensity should not exceed `15.0`:
  ```
  (glow.intensity + bloom.intensity) <= 15.0
  ```
- **Error:** "Combined glow and bloom intensity too high (max 15.0)"

### 2. Physics Stability
- `maxVelocity` must be greater than `restLength / dt`:
  ```
  maxVelocity > (restLength / dt)
  ```
- **Error:** "Max velocity too low for physics timestep and rest length"

### 3. Boundary Consistency
- If `enableBounds = true`, `boundsSize` must be greater than `restLength * 2`:
  ```
  boundsSize > (restLength * 2)
  ```
- **Error:** "Bounds size too small for rest length"

### 4. Grid Cell Size vs Repulsion Cutoff
- `gridCellSize` should be at least `repulsionCutoff`:
  ```
  gridCellSize >= repulsionCutoff
  ```
- **Warning:** "Grid cell size smaller than repulsion cutoff (performance impact)"

---

## GPU Safety Validations

### Finite Number Check
All floating-point values sent to GPU must be finite:

```rust
fn validate_finite(value: f32, field_name: &str) -> Result<(), ValidationError> {
    if !value.is_finite() {
        return Err(ValidationError {
            field: field_name.to_string(),
            message: format!("{} must be finite (not NaN or Infinity)", field_name),
        });
    }
    Ok(())
}
```

Applied to:
- All glow settings
- All bloom settings
- All physics force parameters

### Division by Zero Protection
- `dt` must not be zero
- `mass_scale` must not be zero
- `repulsion_softening_epsilon` must be > 0

---

## Custom Validation Functions

### Hex Color Validation

```rust
fn validate_hex_color(color: &str) -> Result<(), ValidationError> {
    let hex_regex = Regex::new(r"^#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{8})$")?;
    if !hex_regex.is_match(color) {
        return Err(ValidationError::new("invalid_hex_color"));
    }
    Ok(())
}
```

### Width Range Validation

```rust
fn validate_width_range(range: &[f32]) -> Result<(), ValidationError> {
    if range.len() != 2 {
        return Err(ValidationError::new("width_range_length"));
    }
    if range[0] >= range[1] {
        return Err(ValidationError::new("width_range_order"));
    }
    Ok(())
}
```

---

## Validation Error Response Format

```json
{
  "valid": false,
  "errors": [
    {
      "field": "visualisation.rendering.ambientLightIntensity",
      "message": "Value must be between 0.0 and 10.0",
      "receivedValue": 15.0,
      "constraint": "range(0.0, 10.0)",
      "suggestion": "Use a value between 0.0 (dark) and 10.0 (very bright)"
    }
  ]
}
```

---

## Adding New Validation Rules

### Step 1: Define Validation Function

```rust
pub fn validate_my_setting(value: &MyType) -> Result<(), ValidationError> {
    if !is_valid(value) {
        return Err(ValidationError::new("my_setting_invalid"));
    }
    Ok(())
}
```

### Step 2: Add to Struct Validator

```rust
#[derive(Validate)]
struct MySettings {
    #[validate(custom(function = "validate_my_setting"))]
    pub my_setting: MyType,
}
```

### Step 3: Update Validation Service

Add field-specific validation in `ValidationService::validate_settings()`.

### Step 4: Document the Rule

Add entry to this documentation with:
- Type
- Range/format
- Default value
- Error message
- Example valid/invalid values

---

## Related Documentation

- [Settings System Architecture](./settings-system.md)
- [Settings API Reference](./settings-api.md)
- [Database Schema](./settings-schema.md)
- [Migration Guide](./settings-migration-guide.md)
