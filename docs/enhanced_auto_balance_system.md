# Enhanced Auto-Balance System for CUDA Kernel Parameters

## Overview

The enhanced auto-balance system has been updated to handle the new CUDA kernel parameters introduced with the spatial hashing approach. This system now intelligently adjusts parameters to handle clustering issues, spreading/explosion issues, and numerical stability problems.

## New CUDA Parameters Handled

### 1. grid_cell_size
- **Purpose**: Controls the spatial hashing grid resolution
- **Auto-tuning**: Adjusted based on average inter-node distance
- **Optimal Value**: ~2x average inter-node distance
- **Range**: 1.0 - 50.0

### 2. repulsion_cutoff
- **Purpose**: Maximum distance for repulsion calculations
- **Auto-tuning**: Set to 1.5x optimal grid cell size
- **Range**: 5.0 - 200.0

### 3. repulsion_softening_epsilon
- **Purpose**: Prevents division by zero in repulsion calculations
- **Auto-tuning**: Increased when numerical instability detected
- **Range**: 1e-6 - 1.0

### 4. center_gravity_k
- **Purpose**: Pulls nodes toward the origin to prevent spreading
- **Auto-tuning**: Increased when nodes spread beyond threshold
- **Range**: 0.0 - 0.1

## Enhanced Detection Logic

### Spatial Hashing Issues Detection
```rust
fn detect_spatial_hashing_issues(&self, positions: &[(f32, f32, f32)], config: &AutoBalanceConfig) -> (bool, f32)
```
- Analyzes spatial distribution efficiency
- Calculates efficiency score based on grid cell utilization
- Detects excessive clustering that reduces spatial hash effectiveness

### Numerical Instability Detection
```rust
fn detect_numerical_instability(&self, positions: &[(f32, f32, f32)], config: &AutoBalanceConfig) -> bool
```
- Detects NaN or infinite positions
- Monitors exponential growth in kinetic energy
- Triggers emergency parameter adjustments

## Auto-Balance Priority System

The system now processes issues in the following priority order:

1. **Numerical Instability (Critical)**
   - Emergency parameter adjustment
   - Increases repulsion softening
   - Reduces time step and forces
   - Increases damping for stability

2. **Spatial Hashing Inefficiency**
   - Adjusts grid cell size based on node distribution
   - Optimizes repulsion cutoff for spatial hash performance
   - Adjusts repulsion softening for clustering

3. **Traditional Issues** (Bouncing, Spreading, Clustering)
   - Enhanced with new parameter adjustments
   - Improved containment using center gravity

## Configuration

### New AutoBalanceConfig Parameters

```rust
pub struct AutoBalanceConfig {
    // Existing parameters...
    
    // New CUDA kernel parameter tuning ranges
    pub grid_cell_size_min: f32,              // 1.0
    pub grid_cell_size_max: f32,              // 50.0
    pub repulsion_cutoff_min: f32,            // 5.0
    pub repulsion_cutoff_max: f32,            // 200.0
    pub repulsion_softening_min: f32,         // 1e-6
    pub repulsion_softening_max: f32,         // 1.0
    pub center_gravity_min: f32,              // 0.0
    pub center_gravity_max: f32,              // 0.1
    
    // Detection thresholds
    pub spatial_hash_efficiency_threshold: f32,  // 0.3
    pub cluster_density_threshold: f32,          // 50.0
    pub numerical_instability_threshold: f32,   // 1e-3
}
```

## Usage Examples

### Enabling Enhanced Auto-Balance

The enhanced system works automatically when auto-balance is enabled:

```json
{
  "visualisation": {
    "graphs": {
      "logseq": {
        "physics": {
          "auto_balance": true,
          "auto_balance_interval_ms": 500,
          "auto_balance_config": {
            "spatial_hash_efficiency_threshold": 0.3,
            "cluster_density_threshold": 50.0,
            "numerical_instability_threshold": 0.001
          }
        }
      }
    }
  }
}
```

### Monitoring Auto-Balance Actions

The system provides detailed notifications:

- **Spatial Hashing**: "Optimizing spatial hashing efficiency"
- **Numerical Issues**: "Emergency - Fixing numerical instability" 
- **Containment**: "Increasing containment forces to prevent spreading"
- **Traditional**: "Stabilizing bouncing nodes", "Expanding clustered nodes"

## Implementation Details

### Parameter Mapping

Since the CUDA kernel uses different parameter names, the system maps parameters:

- `grid_cell_size` → Approximated using `max_repulsion_dist`
- `repulsion_softening_epsilon` → Mapped to `cooling_rate`
- `center_gravity_k` → Mapped to enhanced `attraction_k` and `cooling_rate`

### Smooth Transitions

All parameter changes use exponential smoothing to prevent jarring adjustments:

```rust
fn smooth_transition_params(&mut self) {
    let rate = self.param_transition_rate;
    self.simulation_params.max_repulsion_dist = 
        self.simulation_params.max_repulsion_dist * (1.0 - rate) + 
        self.target_params.max_repulsion_dist * rate;
    // ... other parameters
}
```

## Benefits

1. **Improved Stability**: Better numerical stability through proactive detection
2. **Enhanced Performance**: Optimal spatial hashing parameters for better GPU utilization
3. **Automatic Containment**: Smart center gravity adjustments prevent spreading
4. **Reduced Manual Tuning**: System handles complex parameter interactions automatically

## Future Enhancements

1. **Machine Learning Integration**: Use neural networks to predict optimal parameters
2. **Historical Analysis**: Learn from past successful configurations
3. **Multi-Graph Optimization**: Optimize parameters across different graph types
4. **Real-time Metrics**: Provide detailed performance metrics for parameter effectiveness