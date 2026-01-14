/**
 * Three.js WebXR Components
 *
 * VR-optimized visualization components for Meta Quest 3.
 * All components target 72fps with aggressive LOD and performance optimizations.
 */

// Core VR components
export { VRGraphCanvas } from './VRGraphCanvas';
export { VRInteractionManager } from './VRInteractionManager';

// Agent action visualization
export {
  VRActionConnectionsLayer,
  VRImpactRing,
} from './VRActionConnectionsLayer';

export { VRAgentActionScene } from './VRAgentActionScene';

// Default export
export { default } from './VRGraphCanvas';
