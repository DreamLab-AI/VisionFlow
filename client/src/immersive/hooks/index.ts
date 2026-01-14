/**
 * Immersive Mode Hooks
 *
 * React hooks for VR/XR functionality.
 */

// LOD management
export {
  useVRConnectionsLOD,
  calculateOptimalThresholds,
  getLODDistribution,
} from './useVRConnectionsLOD';
export type { LODLevel, VRConnectionsLODConfig } from './useVRConnectionsLOD';

// Hand tracking
export {
  useVRHandTracking,
  xrControllerToHandState,
  agentsToTargetNodes,
} from './useVRHandTracking';
export type {
  HandState,
  TargetNode,
  VRHandTrackingConfig,
  VRHandTrackingResult,
} from './useVRHandTracking';

// Re-export existing hooks if present
export * from './useImmersiveData';
