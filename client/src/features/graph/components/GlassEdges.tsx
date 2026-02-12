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

function computeInstanceMatrices(mesh: THREE.InstancedMesh, pts: number[]): void {
  const edgeCount = Math.min(Math.floor(pts.length / 6), MAX_EDGES);
  for (let i = 0; i < edgeCount; i++) {
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

  mesh.count = edgeCount;
  mesh.instanceMatrix.needsUpdate = true;
}

export const GlassEdges = forwardRef<GlassEdgesHandle, GlassEdgesProps>(
  ({ points, settings, colorOverride }, ref) => {
    const meshRef = useRef<THREE.InstancedMesh | null>(null);

    const { mesh, uniforms } = useMemo(() => {
      const geo = createGlassEdgeGeometry(settings?.edgeRadius ?? 0.03);
      const result = createGlassEdgeMaterial();

      if (colorOverride) {
        // Override base color via material property when specified
        const mat = result.material as THREE.MeshPhysicalMaterial;
        if (mat.color) mat.color = new THREE.Color(colorOverride);
      }

      const m = new THREE.InstancedMesh(geo, result.material, MAX_EDGES);
      m.frustumCulled = false;
      m.count = 0;

      // Initial population
      if (points.length >= 6) {
        computeInstanceMatrices(m, points);
      }

      meshRef.current = m;
      return { mesh: m, uniforms: result.uniforms };
      // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    // Recompute when points prop changes (declarative path)
    useEffect(() => {
      if (points.length >= 6) {
        computeInstanceMatrices(mesh, points);
      } else {
        mesh.count = 0;
        mesh.instanceMatrix.needsUpdate = true;
      }
    }, [points, mesh]);

    // Dispose GPU resources on unmount
    useEffect(() => {
      return () => {
        if (meshRef.current) {
          meshRef.current.geometry.dispose();
          (meshRef.current.material as THREE.Material).dispose();
        }
      };
    }, []);

    // Imperative path for hot-loop updates from useFrame callers
    const updatePoints = useCallback(
      (newPts: number[]) => {
        if (newPts.length < 6) {
          mesh.count = 0;
          mesh.instanceMatrix.needsUpdate = true;
          return;
        }
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
    });

    return <primitive object={mesh} />;
  },
);

GlassEdges.displayName = 'GlassEdges';

export default GlassEdges;
