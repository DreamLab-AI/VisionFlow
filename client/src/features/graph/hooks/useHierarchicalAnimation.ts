import { useRef, useCallback, useEffect } from 'react';
import * as THREE from 'three';
import { useFrame } from '@react-three/fiber';
import { createLogger } from '../../../utils/loggerConfig';

const logger = createLogger('HierarchicalAnimation');

interface AnimationState {
  active: boolean;
  startTime: number;
  duration: number;
  fromPositions: Map<string, THREE.Vector3>;
  toPositions: Map<string, THREE.Vector3>;
  fromScales: Map<string, number>;
  toScales: Map<string, number>;
  nodeIds: string[];
}

/**
 * Hook for managing smooth expand/collapse animations in hierarchical view
 */
export const useHierarchicalAnimation = (duration: number = 800) => {
  const animationStateRef = useRef<AnimationState>({
    active: false,
    startTime: 0,
    duration,
    fromPositions: new Map(),
    toPositions: new Map(),
    fromScales: new Map(),
    toScales: new Map(),
    nodeIds: [],
  });

  const currentProgressRef = useRef(0);

  /**
   * Start expand animation for a class
   */
  const startExpandAnimation = useCallback(
    (
      collapsedPosition: THREE.Vector3,
      expandedPositions: Map<string, THREE.Vector3>,
      nodeIds: string[]
    ) => {
      const state = animationStateRef.current;

      state.active = true;
      state.startTime = performance.now();
      state.nodeIds = nodeIds;

      // Set initial positions (all at collapsed center)
      state.fromPositions.clear();
      nodeIds.forEach((id) => {
        state.fromPositions.set(id, collapsedPosition.clone());
      });

      // Set target positions (expanded layout)
      state.toPositions = new Map(expandedPositions);

      // Scales: small to normal
      state.fromScales.clear();
      state.toScales.clear();
      nodeIds.forEach((id) => {
        state.fromScales.set(id, 0.3);
        state.toScales.set(id, 1.0);
      });

      logger.info('Started expand animation', {
        nodeCount: nodeIds.length,
        duration: state.duration,
      });
    },
    []
  );

  /**
   * Start collapse animation for a class
   */
  const startCollapseAnimation = useCallback(
    (
      expandedPositions: Map<string, THREE.Vector3>,
      collapsedPosition: THREE.Vector3,
      nodeIds: string[]
    ) => {
      const state = animationStateRef.current;

      state.active = true;
      state.startTime = performance.now();
      state.nodeIds = nodeIds;

      // Set initial positions (expanded layout)
      state.fromPositions = new Map(expandedPositions);

      // Set target position (all to collapsed center)
      state.toPositions.clear();
      nodeIds.forEach((id) => {
        state.toPositions.set(id, collapsedPosition.clone());
      });

      // Scales: normal to small
      state.fromScales.clear();
      state.toScales.clear();
      nodeIds.forEach((id) => {
        state.fromScales.set(id, 1.0);
        state.toScales.set(id, 0.3);
      });

      logger.info('Started collapse animation', {
        nodeCount: nodeIds.length,
        duration: state.duration,
      });
    },
    []
  );

  /**
   * Update animation frame
   */
  useFrame(() => {
    const state = animationStateRef.current;
    if (!state.active) return;

    const elapsed = performance.now() - state.startTime;
    const progress = Math.min(elapsed / state.duration, 1);

    currentProgressRef.current = easeInOutCubic(progress);

    // Animation complete
    if (progress >= 1) {
      state.active = false;
      currentProgressRef.current = 1;
      logger.debug('Animation completed');
    }
  });

  /**
   * Get current animated position for a node
   */
  const getAnimatedPosition = useCallback((nodeId: string): THREE.Vector3 | null => {
    const state = animationStateRef.current;
    if (!state.active) return null;

    const fromPos = state.fromPositions.get(nodeId);
    const toPos = state.toPositions.get(nodeId);

    if (!fromPos || !toPos) return null;

    const progress = currentProgressRef.current;
    return new THREE.Vector3().lerpVectors(fromPos, toPos, progress);
  }, []);

  /**
   * Get current animated scale for a node
   */
  const getAnimatedScale = useCallback((nodeId: string): number | null => {
    const state = animationStateRef.current;
    if (!state.active) return null;

    const fromScale = state.fromScales.get(nodeId);
    const toScale = state.toScales.get(nodeId);

    if (fromScale === undefined || toScale === undefined) return null;

    const progress = currentProgressRef.current;
    return THREE.MathUtils.lerp(fromScale, toScale, progress);
  }, []);

  /**
   * Check if animation is active
   */
  const isAnimating = useCallback(() => {
    return animationStateRef.current.active;
  }, []);

  /**
   * Stop current animation
   */
  const stopAnimation = useCallback(() => {
    animationStateRef.current.active = false;
    currentProgressRef.current = 0;
    logger.debug('Animation stopped');
  }, []);

  return {
    startExpandAnimation,
    startCollapseAnimation,
    getAnimatedPosition,
    getAnimatedScale,
    isAnimating,
    stopAnimation,
    currentProgress: currentProgressRef.current,
  };
};

/**
 * Easing function for smooth animations
 */
const easeInOutCubic = (t: number): number => {
  return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
};

export default useHierarchicalAnimation;
