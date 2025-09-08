/**
 * RENDERING SYSTEM INDEX
 * 
 * Central export point for the new rendering architecture.
 * 
 * ARCHITECTURE OVERVIEW:
 * - SelectiveBloom: The main post-processing component using R3F
 * - Materials: Specialized materials optimized for bloom effects
 * 
 * LAYER SYSTEM:
 * - Layer 0: Default geometry and background elements
 * - Layer 1: Graph elements (nodes, edges) - affected by bloom pipeline
 * - Layer 2: Environment elements (hologram, particles) - affected by glow pipeline
 */

// Core post-processing pipeline
export { SelectiveBloom } from './SelectiveBloom';
export { useBloom } from './SelectiveBloom';

// Material library
export * from './materials';

// Re-export default
export default {
  SelectiveBloom: SelectiveBloom
};