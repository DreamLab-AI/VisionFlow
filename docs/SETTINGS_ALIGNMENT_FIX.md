# ✅ Settings System Alignment Fixed

## Problem Identified
Client was sending `repelK: 600.0` but server was:
1. Looking for `repulsionK` instead of `repelK`
2. Validating range 0.1-50.0 (too restrictive for GPU experimentation)
3. Resulting in 400 Bad Request: "repulsion parameter must be between 0.1 and 50.0"

## Fixes Applied

### 1. Server Validation (`settings_handler.rs`)
- **Fixed parameter names**: Now accepts `repelK` (not `repulsionK`)
- **Expanded ranges for GPU experimentation**:
  - `repelK`: 0.1-50.0 → 0.1-1000.0
  - `springK`: 0.001-2.0 → 0.0001-10.0
  - `attractionK`: 0.0-5.0 → 0.0-10.0
- **Fixed mapping**: `repulsionStrength` → `repelK` (was going to `repulsionK`)

### 2. Client Defaults (`defaultSettings.ts`)
- **Reduced default**: `repelK: 600.0` → `50.0` (safe starting value)
- Keeps experimentation possible while starting with stable defaults

### 3. Control Panel Ranges (`IntegratedControlPanel.tsx`)
- **Updated slider ranges to match server**:
  - `repelK`: max 100 → max 1000
  - `springK`: max 2 → max 10
  - `attractionK`: max 1 → max 10

## Parameter Name Alignment

| Legacy Name | GPU Name | Client Uses | Server Validates |
|-------------|----------|-------------|------------------|
| repulsionStrength | repel_k | repelK | ✅ repelK |
| springStrength | spring_k | springK | ✅ springK |
| attractionStrength | attraction_k | attractionK | ✅ attractionK |
| timeStep | dt | dt | ✅ dt |
| collisionRadius | separation_radius | separationRadius | ✅ separationRadius |

## Validation Ranges (GPU Experimentation)

| Parameter | Old Range | New Range | Purpose |
|-----------|-----------|-----------|---------|
| repelK | 0.1-50.0 | 0.1-1000.0 | Allow extreme repulsion testing |
| springK | 0.001-2.0 | 0.0001-10.0 | Fine-grained edge control |
| attractionK | 0.0-5.0 | 0.0-10.0 | Strong clustering effects |

## Testing
```bash
# Test various repelK values
curl -X POST http://localhost:3001/api/settings \
  -H "Content-Type: application/json" \
  -d '{"visualisation": {"graphs": {"logseq": {"physics": {"repelK": 50}}}}}'  # ✅ Should work

curl -X POST http://localhost:3001/api/settings \
  -H "Content-Type: application/json" \
  -d '{"visualisation": {"graphs": {"logseq": {"physics": {"repelK": 600}}}}}'  # ✅ Should work now

curl -X POST http://localhost:3001/api/settings \
  -H "Content-Type: application/json" \
  -d '{"visualisation": {"graphs": {"logseq": {"physics": {"repelK": 1001}}}}}'  # ❌ Should fail
```

## Build Status
- ✅ Server: Compiles successfully
- ✅ Client: Builds without errors
- ✅ Validation: Properly aligned

## Result
Settings now properly sync between client and server with GPU-aligned parameter names and appropriate experimentation ranges. The 400 Bad Request error is resolved.