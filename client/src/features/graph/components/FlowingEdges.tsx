import React, { useRef, useMemo, useEffect, forwardRef, useImperativeHandle } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import { EdgeSettings } from '../../settings/config/settings';
import { useSettingsStore } from '../../../store/settingsStore';
import { registerEdgeObject, unregisterEdgeObject } from '../../visualisation/hooks/bloomRegistry';

interface FlowingEdgesProps {
  points: number[];
  settings: EdgeSettings;
  colorOverride?: string; // Per-mode edge color override
  edgeData?: Array<{
    source: string;
    target: string;
    weight?: number;
    active?: boolean;
  }>;
}

/** Imperative handle for direct buffer updates from useFrame (avoids React state). */
export interface FlowingEdgesHandle {
  updatePoints(points: number[]): void;
}

// Minimum edge opacity - allows very low transparency while preventing truly invisible edges
const MIN_EDGE_OPACITY = 0.01;

export const FlowingEdges = forwardRef<FlowingEdgesHandle, FlowingEdgesProps>(({ points, settings: propSettings, colorOverride, edgeData }, ref) => {
  const globalSettings = useSettingsStore((state) => state.settings);
  const edgeBloomStrength = globalSettings?.visualisation?.glow?.edgeGlowStrength ?? 0.5;
  const lineRef = useRef<THREE.LineSegments>(null);
  const materialRef = useRef<THREE.LineBasicMaterial>(null);
  const positionAttrRef = useRef<THREE.BufferAttribute | null>(null);
  const geometryRef = useRef<THREE.BufferGeometry | null>(null);

  // Imperative handle: lets parent push edge points directly without React state
  useImperativeHandle(ref, () => ({
    updatePoints(newPts: number[]) {
      const geo = geometryRef.current;
      if (!geo) return;

      if (newPts.length < 6) {
        if (lineRef.current) lineRef.current.visible = false;
        return;
      }

      const newPositions = new Float32Array(newPts);
      const existingAttr = positionAttrRef.current;
      if (existingAttr && existingAttr.array.length >= newPositions.length) {
        (existingAttr.array as Float32Array).set(newPositions);
        existingAttr.needsUpdate = true;
        geo.setDrawRange(0, newPositions.length / 3);
      } else {
        const bufferSize = Math.ceil(newPositions.length * 1.5);
        const buffer = new Float32Array(bufferSize);
        buffer.set(newPositions);
        const attr = new THREE.BufferAttribute(buffer, 3);
        attr.setUsage(THREE.DynamicDrawUsage);
        positionAttrRef.current = attr;
        geo.setAttribute('position', attr);
        geo.setDrawRange(0, newPositions.length / 3);
      }
      geo.computeBoundingSphere();
      if (lineRef.current) lineRef.current.visible = true;
    }
  }), []);

  // Stable geometry ref - created once, buffer updated in-place.
  // This avoids the dispose/recreate race that caused edges to vanish.
  const geometry = useMemo(() => {
    const geo = new THREE.BufferGeometry();
    const positions = new Float32Array(points);
    const attr = new THREE.BufferAttribute(positions, 3);
    attr.setUsage(THREE.DynamicDrawUsage);
    geo.setAttribute('position', attr);
    geo.computeBoundingSphere();
    positionAttrRef.current = attr;
    geometryRef.current = geo;
    return geo;
    // Intentionally depend only on initial mount - updates happen via the effect below
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Update geometry buffer in-place when points change (after initial mount)
  const isFirstRender = useRef(true);
  useEffect(() => {
    if (isFirstRender.current) {
      isFirstRender.current = false;
      return;
    }

    if (points.length < 6) {
      if (lineRef.current) lineRef.current.visible = false;
      return;
    }

    const geo = geometry;
    const newPositions = new Float32Array(points);

    const existingAttr = positionAttrRef.current;
    if (existingAttr && existingAttr.array.length >= newPositions.length) {
      // Reuse buffer - copy in-place
      (existingAttr.array as Float32Array).set(newPositions);
      existingAttr.needsUpdate = true;
      geo.setDrawRange(0, newPositions.length / 3);
    } else {
      // Need larger buffer - allocate with headroom
      const bufferSize = Math.ceil(newPositions.length * 1.5);
      const buffer = new Float32Array(bufferSize);
      buffer.set(newPositions);
      const attr = new THREE.BufferAttribute(buffer, 3);
      attr.setUsage(THREE.DynamicDrawUsage);
      positionAttrRef.current = attr;
      geo.setAttribute('position', attr);
      geo.setDrawRange(0, newPositions.length / 3);
    }

    geo.computeBoundingSphere();

    if (lineRef.current) {
      lineRef.current.visible = true;
    }
  }, [points, geometry]);

  // Cleanup geometry on unmount only
  useEffect(() => {
    return () => {
      geometry.dispose();
    };
  }, [geometry]);

  // Resolve effective opacity: ensure edges are always visible
  const effectiveOpacity = Math.max(MIN_EDGE_OPACITY, propSettings.opacity ?? 0.6);

  const edgeColor = colorOverride || propSettings.color || '#FF5722';

  const material = useMemo(() => {
    const color = new THREE.Color(edgeColor);

    // Additive glow boost ensures edges remain visible regardless of bloom
    const bloomBoost = 1 + (edgeBloomStrength * 0.5);
    const bloomAdjustedColor = color.clone().multiplyScalar(bloomBoost);

    const mat = new THREE.LineBasicMaterial({
      color: bloomAdjustedColor,
      transparent: true,
      opacity: Math.min(1.0, effectiveOpacity),
      linewidth: propSettings.baseWidth || 2,
      depthWrite: false,
      depthTest: true,
      alphaTest: 0.01,
      toneMapped: false,
    });

    return mat;
  }, [edgeColor, effectiveOpacity, propSettings.baseWidth, edgeBloomStrength]);

  useEffect(() => {
    return () => { material?.dispose(); };
  }, [material]);

  useEffect(() => {
    materialRef.current = material;
  }, [material]);

  // Register with bloom layer system
  useEffect(() => {
    const obj = lineRef.current as any;
    if (obj) {
      if (!obj.layers) {
        obj.layers = new THREE.Layers();
      }
      obj.layers.set(0);
      obj.layers.enable(1);
      obj.layers.disable(2);
      registerEdgeObject(obj);
    }
    return () => {
      if (obj) unregisterEdgeObject(obj);
    };
  }, []);

  // Animate flow effect
  useFrame((state) => {
    if (materialRef.current && (propSettings as any).enableFlowEffect) {
      const flowIntensity = Math.sin(state.clock.elapsedTime * ((propSettings as any).flowSpeed || 1.0)) * 0.3 + 0.7;
      materialRef.current.opacity = effectiveOpacity * flowIntensity;
    }
  });

  if (points.length < 6) {
    return null;
  }

  return (
    <lineSegments
      ref={lineRef}
      geometry={geometry}
      material={material}
      renderOrder={5}
      frustumCulled={false}
    />
  );
});
