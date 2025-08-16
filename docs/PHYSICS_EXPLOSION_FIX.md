# Physics Explosion Fix - Root Cause Analysis & Solution

## Problem Identified
The knowledge graph is exploding and bouncing due to extreme physics parameters in the settings.yaml file.

## Critical Parameter Issues

| Parameter | Current Value | Problem | Safe Range | Recommended |
|-----------|--------------|---------|------------|-------------|
| repel_k | **784.57** | Way too high - causes explosion | 10-200 | 50 |
| damping | **0.107** | Too low - no energy dissipation | 0.5-0.99 | 0.85 |
| max_velocity | **94.48** | Allows extreme speeds | 1-10 | 5.0 |
| dt | **0.070** | Too large - numerical instability | 0.001-0.02 | 0.016 |
| iterations | **436** | Excessive - amplifies problems | 10-100 | 50 |
| spring_k | 1.28 | Slightly high | 0.001-1.0 | 0.005 |

## Root Causes

1. **Extreme Repulsion Force (repel_k: 784.57)**
   - Pushes nodes apart with massive force
   - Creates explosive behavior

2. **Low Damping (0.107)**
   - Energy doesn't dissipate
   - Nodes continue bouncing indefinitely
   - Should be at least 0.5 for stability

3. **High Max Velocity (94.48)**
   - Allows nodes to fly off at extreme speeds
   - Should be capped at 5-10 for stable visualization

4. **Large Time Step (dt: 0.070)**
   - Causes numerical integration errors
   - Creates unstable physics simulation

## Immediate Fix

Update the physics parameters in settings.yaml:

```yaml
physics:
  repel_k: 50.0        # Reduced from 784.57
  damping: 0.85        # Increased from 0.107
  max_velocity: 5.0    # Reduced from 94.48
  dt: 0.016           # Reduced from 0.070
  iterations: 50      # Reduced from 436
  spring_k: 0.005     # Reduced from 1.28
  attraction_k: 0.01  # Reduced from 0.139
```

## Additional Issues Found

The logs also show a mismatch between:
- What the graph_actor reports: `repel_k: 784.57`
- What GPU compute receives: `repel_k: 100` (clamped)

This indicates the GPU is applying safety limits but the values are still problematic.

## Verification Steps

1. Apply the fixed parameters
2. Restart the application
3. Monitor logs for stable values
4. Verify smooth graph animation

## Prevention

1. Add validation ranges in the UI
2. Implement server-side parameter clamping
3. Add physics preset configurations
4. Log warnings for extreme values