import React, { useRef, useMemo, useState } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import { ConvexGeometry } from 'three/examples/jsm/geometries/ConvexGeometry.js';

// ============================================================================
// Types
// ============================================================================

interface ClusterHullsProps {
  nodes: Array<{
    id: string;
    metadata?: any;
    position?: { x: number; y: number; z: number };
  }>;
  nodePositionsRef: React.RefObject<Float32Array | null>;
  nodeIdToIndexMap: Map<string, number>;
  settings: any;
}

// ============================================================================
// Constants
// ============================================================================

const DOMAIN_COLORS: Record<string, string> = {
  'AI': '#4FC3F7',
  'BC': '#81C784',
  'RB': '#FFB74D',
  'MV': '#CE93D8',
  'TC': '#FFD54F',
  'DT': '#EF5350',
  'NGM': '#4DB6AC',
};

const DEFAULT_COLOR = '#90A4AE';

const MIN_CLUSTER_SIZE = 4;
const TICK_INTERVAL = 30;

// ============================================================================
// Helpers
// ============================================================================

function getClusterKey(node: { metadata?: any }): string {
  return (
    node.metadata?.source_domain ??
    node.metadata?.domain ??
    node.metadata?.cluster ??
    'unknown'
  );
}

function getDomainHullColor(domain: string): string {
  return DOMAIN_COLORS[domain] ?? DEFAULT_COLOR;
}

/**
 * Given a set of 3D points, compute padded points offset from centroid,
 * then return a ConvexGeometry. Returns null if fewer than 4 valid points.
 */
function buildHullGeometry(
  points: THREE.Vector3[],
  padding: number,
): ConvexGeometry | null {
  if (points.length < MIN_CLUSTER_SIZE) return null;

  // Compute centroid
  const centroid = new THREE.Vector3();
  for (const p of points) {
    centroid.add(p);
  }
  centroid.divideScalar(points.length);

  // Offset each point away from centroid by (1 + padding)
  const paddedPoints = points.map((p) => {
    const dir = new THREE.Vector3().subVectors(p, centroid);
    return centroid.clone().add(dir.multiplyScalar(1 + padding));
  });

  try {
    return new ConvexGeometry(paddedPoints);
  } catch {
    // ConvexGeometry can throw on degenerate point sets
    return null;
  }
}

// ============================================================================
// Component
// ============================================================================

export const ClusterHulls: React.FC<ClusterHullsProps> = ({
  nodes,
  nodePositionsRef,
  nodeIdToIndexMap,
  settings,
}) => {
  const groupRef = useRef<THREE.Group>(null);
  const frameCounter = useRef(0);
  const [tick, setTick] = useState(0);

  // ---- Early exit if feature is disabled ----
  // Respect both the visual clusterHulls.enabled toggle AND qualityGates.showClusters
  const clusterHullsEnabled = settings?.visualisation?.clusterHulls?.enabled;
  const showClusters = settings?.qualityGates?.showClusters;
  // If qualityGates.showClusters is explicitly false, disable hulls even if clusterHulls.enabled is true
  const enabled = showClusters === false ? false : clusterHullsEnabled;
  const opacity = settings?.visualisation?.clusterHulls?.opacity ?? 0.08;
  const padding = settings?.visualisation?.clusterHulls?.padding ?? 0.15;

  // ---- Increment tick every TICK_INTERVAL frames ----
  useFrame(() => {
    frameCounter.current += 1;
    if (frameCounter.current >= TICK_INTERVAL) {
      frameCounter.current = 0;
      setTick((t) => t + 1);
    }
  });

  // ---- Group nodes into clusters ----
  const clusterMap = useMemo(() => {
    const map = new Map<string, string[]>();
    for (const node of nodes) {
      const key = getClusterKey(node);
      let arr = map.get(key);
      if (!arr) {
        arr = [];
        map.set(key, arr);
      }
      arr.push(node.id);
    }
    return map;
  }, [nodes]);

  // ---- Build hull geometries from current positions ----
  const hullEntries = useMemo(() => {
    if (!enabled) return [];

    const positions = nodePositionsRef.current;
    if (!positions) return [];

    const entries: Array<{
      domain: string;
      geometry: ConvexGeometry;
    }> = [];

    clusterMap.forEach((nodeIds, domain) => {
      if (nodeIds.length < MIN_CLUSTER_SIZE) return;

      const points: THREE.Vector3[] = [];

      for (const id of nodeIds) {
        const idx = nodeIdToIndexMap.get(id);
        if (idx === undefined) continue;

        const base = idx * 3;
        if (base + 2 >= positions.length) continue;

        const x = positions[base];
        const y = positions[base + 1];
        const z = positions[base + 2];

        // Skip zero/NaN positions
        if (Number.isNaN(x) || Number.isNaN(y) || Number.isNaN(z)) continue;

        points.push(new THREE.Vector3(x, y, z));
      }

      const geometry = buildHullGeometry(points, padding);
      if (geometry) {
        entries.push({ domain, geometry });
      }
    });

    return entries;
    // tick drives periodic recomputation
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [enabled, clusterMap, nodeIdToIndexMap, nodePositionsRef, padding, tick]);

  // ---- Cleanup previous geometries when entries change ----
  const prevGeometries = useRef<THREE.BufferGeometry[]>([]);
  useMemo(() => {
    for (const geo of prevGeometries.current) {
      geo.dispose();
    }
    prevGeometries.current = hullEntries.map((e) => e.geometry);
  }, [hullEntries]);

  if (!enabled) return null;

  return (
    <group ref={groupRef} renderOrder={1}>
      {hullEntries.map(({ domain, geometry }) => (
        <mesh
          key={`hull-${domain}`}
          geometry={geometry}
          renderOrder={1}
        >
          <meshBasicMaterial
            color={getDomainHullColor(domain)}
            transparent={true}
            opacity={opacity}
            side={THREE.DoubleSide}
            depthWrite={false}
          />
        </mesh>
      ))}
    </group>
  );
};
