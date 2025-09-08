/**
 * RENDERING MATERIALS INDEX
 * 
 * Consolidated materials library for the dual-pipeline post-processing architecture.
 * All materials are designed to work seamlessly with the SelectiveBloom component.
 * 
 * ARCHITECTURE:
 * - BloomStandardMaterial: Simple emissive materials for post-processing effects
 * - HologramNodeMaterial: Complex shader materials with unique visual effects
 * 
 * USAGE:
 * - Import materials from this index for consistent API
 * - Use presets for common configurations
 * - Assign objects to appropriate layers (1 for graph, 2 for environment)
 */

// Core material classes
export { BloomStandardMaterial, BloomStandardPresets } from './BloomStandardMaterial';
export { HologramNodeMaterial, HologramNodePresets } from './HologramNodeMaterial';

// Re-export for compatibility
export default {
  BloomStandardMaterial: BloomStandardMaterial,
  HologramNodeMaterial: HologramNodeMaterial
};