# GPU Compute Architecture

*This file redirects to the comprehensive GPU documentation.*

See [GPU Compute Integration](gpu-compute.md) for detailed GPU architecture and implementation.

## GPU Acceleration Overview

VisionFlow leverages CUDA acceleration for:
- Physics simulation (force-directed layouts)
- Parallel graph processing
- Real-time analytics
- Visual effects computation

## Quick Links

- [GPU Compute Integration](gpu-compute.md) - Complete GPU implementation
- [GPU Analytics Algorithms](gpu-analytics-algorithms.md) - Clustering and analysis
- [GPU Compute Improvements](gpu-compute-improvements.md) - Performance optimizations
- [GPU Modular System](gpu-modular-system.md) - Architecture overview

## Key Features

### Unified CUDA Kernel
- Single kernel handles all physics modes
- Structure of Arrays (SoA) memory layout
- Optimized for maximum GPU utilization

### Four Compute Modes
- **Mode 0**: Basic force-directed layout
- **Mode 1**: Dual graph processing
- **Mode 2**: Constraint-enhanced physics
- **Mode 3**: Visual analytics with clustering

### Performance Characteristics
- 60 FPS physics simulation
- 100,000+ node capacity
- Sub-millisecond computation times
- Graceful CPU fallback

---

[← Back to Architecture Documentation](README.md)