import React, { useMemo, useCallback, useEffect, useRef, forwardRef, useImperativeHandle } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import {
  createGlassEdgeMaterial,
  createGlassEdgeGeometry,
} from '../../../rendering/materials/GlassEdgeMaterial';

const MAX_EDGES = 10_000;

interface GlassEdgesProps {
  points: number[];
  settings: any;
  colorOverride?: string;
}

export interface GlassEdgesHandle {
  updatePoints(points: number[]): void;
}

/** Pre-allocated temp objects for matrix composition -- avoids per-frame GC. */
const tmpMat = new THREE.Matrix4();
const tmpPos = new THREE.Vector3();
const tmpSrc = new THREE.Vector3();
const tmpTgt = new THREE.Vector3();
const tmpUp = new THREE.Vector3(0, 1, 0);
const tmpQuat = new THREE.Quaternion();
const tmpDir = new THREE.Vector3();
const tmpScale = new THREE.Vector3();

/** Compute up to `limit` edge matrices. Returns total edge count. */
function computeInstanceMatrices(mesh: THREE.InstancedMesh, pts: number[], limit?: number): number {
  const edgeCount = Math.min(Math.floor(pts.length / 6), MAX_EDGES);
  const renderCount = limit !== undefined ? Math.min(limit, edgeCount) : edgeCount;
  for (let i = 0; i < renderCount; i++) {
    const off = i * 6;
    tmpSrc.set(pts[off], pts[off + 1], pts[off + 2]);
    tmpTgt.set(pts[off + 3], pts[off + 4], pts[off + 5]);

    // Midpoint
    tmpPos.addVectors(tmpSrc, tmpTgt).multiplyScalar(0.5);

    // Direction and length
    tmpDir.subVectors(tmpTgt, tmpSrc);
    const len = tmpDir.length();
    if (len < 1e-6) {
      tmpMat.makeScale(0, 0, 0);
      mesh.setMatrixAt(i, tmpMat);
      continue;
    }
    tmpDir.normalize();

    // Quaternion: rotate unit-Y cylinder to align with edge direction
    // Guard against anti-parallel vectors (dot ~ -1) which cause NaN
    const dot = tmpUp.dot(tmpDir);
    if (dot < -0.9999) {
      // Anti-parallel: 180-degree rotation around X axis
      tmpQuat.set(1, 0, 0, 0);
    } else {
      tmpQuat.setFromUnitVectors(tmpUp, tmpDir);
    }

    // Compose: translate to midpoint, rotate, scale Y by length
    tmpScale.set(1, len, 1);
    tmpMat.compose(tmpPos, tmpQuat, tmpScale);
    mesh.setMatrixAt(i, tmpMat);
  }

  mesh.count = renderCount;
  mesh.instanceMatrix.needsUpdate = true;
  return edgeCount;
}

export const GlassEdges = forwardRef<GlassEdgesHandle, GlassEdgesProps>(
  ({ points, settings, colorOverride }, ref) => {
    const meshRef = useRef<THREE.InstancedMesh | null>(null);
    const edgeRevealRef = useRef(0);
    const totalEdgesRef = useRef(0);
    const edgeDataHashRef = useRef('');
    const EDGE_REVEAL_BATCH = 80;

    const { mesh, uniforms } = useMemo(() => {
      // Resolve initial edge color: prefer override, then settings, then default
      const initialColor = colorOverride || settings?.color || undefined;
      const geo = createGlassEdgeGeometry(settings?.edgeRadius ?? 0.03);
      const result = createGlassEdgeMaterial(initialColor);

      // Apply initial opacity from settings if provided
      if (settings?.opacity !== undefined) {
        (result.material as THREE.MeshPhysicalMaterial).opacity = settings.opacity;
      }

      const m = new THREE.InstancedMesh(geo, result.material, MAX_EDGES);
      m.frustumCulled = false;
      m.count = 0;

      // Initial population — first batch only, rest via progressive reveal
      if (points.length >= 6) {
        computeInstanceMatrices(m, points, EDGE_REVEAL_BATCH);
      }

      meshRef.current = m;
      return { mesh: m, uniforms: result.uniforms };
      // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    // Update material color and opacity when settings change
    useEffect(() => {
      const mat = mesh.material as THREE.MeshPhysicalMaterial;
      const targetColor = colorOverride || settings?.color;
      if (targetColor && mat.color) {
        mat.color.set(targetColor);
      }
      if (settings?.opacity !== undefined) {
        mat.opacity = settings.opacity;
      }
      mat.needsUpdate = true;
    }, [colorOverride, settings?.color, settings?.opacity, mesh]);

    // Recompute when points prop changes -- reset progressive reveal and dirty flag
    useEffect(() => {
      if (points.length >= 6) {
        totalEdgesRef.current = Math.min(Math.floor(points.length / 6), MAX_EDGES);
        edgeRevealRef.current = 0; // Reset for progressive reveal in useFrame
        edgeDataHashRef.current = ''; // Force recompute on next reveal cycle
      } else {
        mesh.count = 0;
        totalEdgesRef.current = 0;
        edgeDataHashRef.current = '';
        mesh.instanceMatrix.needsUpdate = true;
      }
    }, [points, mesh]);

    // NOTE: No disposal useEffect here. React StrictMode double-invokes useMemo
    // and runs effect cleanup between mount cycles, which disposes the GPU buffers
    // of the ONLY mesh instance. WebGPU cannot recover disposed geometry/material.
    // Three.js / GC handles cleanup when the component fully unmounts.

    // Imperative path for hot-loop updates from useFrame callers.
    // Uses dirty-flag hash to skip redundant GPU uploads when edge data
    // hasn't actually changed (the common case for static graphs).
    const updatePoints = useCallback(
      (newPts: number[]) => {
        if (newPts.length < 6) {
          if (edgeDataHashRef.current !== '') {
            edgeDataHashRef.current = '';
            mesh.count = 0;
            mesh.instanceMatrix.needsUpdate = true;
          }
          return;
        }
        const hash = `${newPts.length}-${newPts[0]}-${newPts[newPts.length - 1]}`;
        if (hash === edgeDataHashRef.current) return;
        edgeDataHashRef.current = hash;
        computeInstanceMatrices(mesh, newPts);
      },
      [mesh],
    );

    useImperativeHandle(ref, () => ({ updatePoints }), [updatePoints]);

    // Subtle emissive pulse on edges
    useFrame(({ clock }) => {
      const mat = meshRef.current?.material as THREE.MeshPhysicalMaterial | undefined;
      if (mat) {
        mat.emissiveIntensity = 0.15 + Math.sin(clock.elapsedTime * 0.8) * 0.08;
      }

      // Progressive edge reveal: ramp up each frame
      if (edgeRevealRef.current < totalEdgesRef.current && points.length >= 6) {
        edgeRevealRef.current = Math.min(
          edgeRevealRef.current + EDGE_REVEAL_BATCH,
          totalEdgesRef.current,
        );
        computeInstanceMatrices(mesh, points, edgeRevealRef.current);
        // Update hash after progressive reveal completes so imperative path
        // can detect unchanged data and skip redundant GPU uploads.
        if (edgeRevealRef.current >= totalEdgesRef.current) {
          const pts = points;
          edgeDataHashRef.current = pts.length >= 6
            ? `${pts.length}-${pts[0]}-${pts[pts.length - 1]}`
            : '';
        }
      }
    });

    return <primitive object={mesh} />;
  },
);

GlassEdges.displayName = 'GlassEdges';

export default GlassEdges;
