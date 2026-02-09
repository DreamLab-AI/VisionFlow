import React, { useRef, useMemo, useCallback, useEffect, forwardRef, useImperativeHandle } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import type { GraphVisualMode } from './GraphManager';
import type { Node as GraphNode } from '../managers/graphDataManager';
import { createGemNodeMaterial, createGemGeometry, createTslGemMaterial } from '../../../rendering/materials/GemNodeMaterial';
import { createCrystalOrbMaterial, createCrystalOrbGeometry } from '../../../rendering/materials/CrystalOrbMaterial';
import { createAgentCapsuleMaterial, createAgentCapsuleGeometry } from '../../../rendering/materials/AgentCapsuleMaterial';

export interface GemNodesProps {
  nodes: GraphNode[];
  edges: any[];
  graphMode: GraphVisualMode;
  perNodeVisualModeMap: Map<string, GraphVisualMode>;
  nodePositionsRef: React.MutableRefObject<Float32Array | null>;
  connectionCountMap: Map<string, number>;
  hierarchyMap: Map<string, any>;
  nodeIdToIndexMap: Map<string, number>;
  settings: any;
  ssspResult: any;
  onPointerDown: (event: any) => void;
  onPointerMove: (event: any) => void;
  onPointerUp: (event: any) => void;
  onPointerMissed: () => void;
  onDoubleClick: (event: any) => void;
  selectedNodeId: string | null;
}

export interface GemNodesHandle {
  getMesh: () => THREE.InstancedMesh | null;
  getColorArray: () => Float32Array | null;
}

/** Round up to next power of 2 (minimum 1). */
const nextPowerOf2 = (n: number): number => Math.pow(2, Math.ceil(Math.log2(Math.max(n, 1))));

// Mode-aware node scale (mirrors GraphManager getNodeScale)
const getNodeScale = (
  node: GraphNode, conns: Map<string, number>,
  mode: GraphVisualMode, hier?: Map<string, any>,
): number => {
  const base = node.metadata?.size || 1.0;
  const id = String(node.id);
  if (mode === 'ontology') {
    const depth = hier?.get(id)?.depth ?? (node.metadata?.depth ?? 0);
    const ic = parseInt(node.metadata?.instanceCount || '0', 10);
    return base * Math.max(0.4, 1.0 - depth * 0.15) * (1 + Math.log(ic + 1) * 0.1);
  }
  if (mode === 'agent') {
    const w = node.metadata?.workload ?? 0;
    const t = node.metadata?.tokenRate ?? 0;
    return base * (1 + w * 0.3 + Math.min(t / 100, 0.5));
  }
  const auth = node.metadata?.authority ?? node.metadata?.authorityScore ?? 0;
  const cc = conns.get(id) || 0;
  return base * (1 + Math.log(cc + 1) * 0.4) * (1 + auth * 0.5) * 2.5;
};

const getDominantMode = (
  nodes: GraphNode[], global: GraphVisualMode, perNode: Map<string, GraphVisualMode>,
): GraphVisualMode => {
  if (perNode.size === 0) return global;
  const c: Record<string, number> = { knowledge_graph: 0, ontology: 0, agent: 0 };
  for (const n of nodes) c[perNode.get(String(n.id)) || global]++;
  let best = global, max = -1;
  for (const [m, v] of Object.entries(c)) if (v > max) { max = v; best = m as GraphVisualMode; }
  return best;
};

const _mat = new THREE.Matrix4();
const _col = new THREE.Color();
const ONTOLOGY_SPECTRUM = ['#FF6B6B', '#FFD93D', '#4ECDC4', '#AA96DA', '#95E1D3'];
const AGENT_STATUS_MAP: Record<string, string> = {
  active: '#2ECC71', busy: '#F39C12', idle: '#95A5A6', error: '#E74C3C',
};

/** Map a lastModified timestamp to 0-1 recency (1 = recent, 0 = stale). */
const computeRecency = (lastModified: string | number | undefined): number => {
  if (!lastModified) return 0.3;
  const ms = typeof lastModified === 'number' ? lastModified : Date.parse(String(lastModified));
  if (isNaN(ms)) return 0.3;
  const ageSec = (Date.now() - ms) / 1000;
  return Math.max(0.01, Math.exp(-ageSec / 3600)); // 0s->1.0, 1h->~0.37, 4h->~0.02
};

const GemNodesInner: React.ForwardRefRenderFunction<GemNodesHandle, GemNodesProps> = (props, ref) => {
  const {
    nodes, graphMode, perNodeVisualModeMap, nodePositionsRef,
    connectionCountMap, hierarchyMap, settings, ssspResult,
    onPointerDown, onPointerMove, onPointerUp, onPointerMissed, onDoubleClick,
    selectedNodeId,
  } = props;

  const meshRef = useRef<THREE.InstancedMesh | null>(null);
  const metaTexRef = useRef<THREE.DataTexture | null>(null);
  const prevMetaHashRef = useRef('');
  const dominant = getDominantMode(nodes, graphMode, perNodeVisualModeMap);

  const { mesh, uniforms } = useMemo(() => {
    const count = Math.max(nextPowerOf2(nodes.length), 64); // Over-allocate to avoid re-creation
    const [geo, matResult] = dominant === 'ontology'
      ? [createCrystalOrbGeometry(), createCrystalOrbMaterial()] as const
      : dominant === 'agent'
        ? [createAgentCapsuleGeometry(), createAgentCapsuleMaterial()] as const
        : [createGemGeometry(), createGemNodeMaterial()] as const;

    const inst = new THREE.InstancedMesh(geo, matResult.material, count);
    inst.frustumCulled = false;
    inst.count = 0; // Start invisible -- useFrame sets actual count
    const id = new THREE.Matrix4();
    for (let i = 0; i < count; i++) {
      inst.setMatrixAt(i, id);
      inst.setColorAt(i, _col.set(0.5, 0.5, 0.5));
    }
    inst.instanceMatrix.needsUpdate = true;
    if (inst.instanceColor) inst.instanceColor.needsUpdate = true;

    // Per-instance metadata texture for TSL (RGBA float: quality, authority, connections, recency)
    const texData = new Float32Array(count * 4);
    const metaTex = new THREE.DataTexture(texData, count, 1, THREE.RGBAFormat, THREE.FloatType);
    metaTex.minFilter = THREE.NearestFilter;
    metaTex.magFilter = THREE.NearestFilter;
    metaTex.needsUpdate = true;
    metaTexRef.current = metaTex;

    meshRef.current = inst;
    return { mesh: inst, uniforms: matResult.uniforms };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dominant, nodes.length]);

  // Dispose GPU resources when mesh is recreated or component unmounts
  useEffect(() => {
    return () => {
      if (meshRef.current) {
        meshRef.current.geometry.dispose();
        (meshRef.current.material as THREE.Material).dispose();
      }
      metaTexRef.current?.dispose();
    };
  }, [dominant, nodes.length]);

  useImperativeHandle(ref, () => ({
    getMesh: () => meshRef.current,
    getColorArray: () => meshRef.current?.instanceColor?.array as Float32Array | null ?? null,
  }), [mesh]);

  // Attempt TSL material upgrade (WebGPU only, knowledge_graph mode)
  useEffect(() => {
    if (dominant === 'ontology' || dominant === 'agent') return;
    const tex = metaTexRef.current;
    if (!tex) return;
    let cancelled = false;
    createTslGemMaterial(tex, tex.image.width).then((tslMat) => {
      if (cancelled || !tslMat || !meshRef.current) return;
      const old = meshRef.current.material;
      meshRef.current.material = tslMat;
      if (old && typeof (old as any).dispose === 'function') (old as any).dispose();
      console.log('[GemNodes] TSL metadata material activated');
    });
    return () => { cancelled = true; };
  }, [dominant]);

  const computeColor = useCallback((node: GraphNode, mode: GraphVisualMode): THREE.Color => {
    if (ssspResult) {
      const d = ssspResult.distances?.[node.id];
      if (node.id === ssspResult.sourceNodeId) return _col.set('#00FFFF');
      if (!isFinite(d)) return _col.set('#666666');
      const nd = ssspResult.normalizedDistances?.[node.id] || 0;
      return _col.setRGB(Math.min(1, nd * 1.2), Math.min(1, (1 - nd) * 1.2), 0.1);
    }
    if (mode === 'agent') {
      return _col.set(AGENT_STATUS_MAP[node.metadata?.status?.toLowerCase() || 'active'] || '#2ECC71');
    }
    if (mode === 'ontology') {
      const depth = hierarchyMap?.get(node.id)?.depth ?? (node.metadata?.depth ?? 0);
      return _col.set(ONTOLOGY_SPECTRUM[Math.min(depth, ONTOLOGY_SPECTRUM.length - 1)]);
    }
    const auth = node.metadata?.authority ?? node.metadata?.authorityScore ?? 0;
    _col.set('#90A4AE');
    if (auth > 0) _col.offsetHSL(0, auth * 0.06, auth * 0.3);
    return _col;
  }, [ssspResult, hierarchyMap]);

  useFrame(({ clock }) => {
    const inst = meshRef.current;
    if (!inst || nodes.length === 0) return;
    if (uniforms.time) uniforms.time.value = clock.elapsedTime;

    const positions = nodePositionsRef.current;
    const baseScale = (settings?.visualisation?.nodes?.nodeSize ?? 0.5) / 0.5;
    const texBuf = metaTexRef.current?.image?.data as Float32Array | undefined;

    // Subtle emissive pulse for agent capsule material
    const u = uniforms as any;
    if (dominant === 'agent' && u.activityLevel) {
      const currentMat = inst.material as any;
      if (currentMat.emissiveIntensity !== undefined) {
        const pulse = Math.pow((Math.sin(clock.elapsedTime * Math.PI) + 1) * 0.5, 4);
        currentMat.emissiveIntensity = 0.15 + pulse * u.activityLevel.value * 0.2;
      }
    }

    for (let i = 0; i < nodes.length; i++) {
      const node = nodes[i];
      const mode = perNodeVisualModeMap.get(String(node.id)) || graphMode;
      let s = getNodeScale(node, connectionCountMap, mode, hierarchyMap) * baseScale;
      if (selectedNodeId && String(node.id) === selectedNodeId) {
        s *= 1 + Math.sin(clock.elapsedTime * 3) * 0.15;
      }

      const i3 = i * 3;
      let x: number, y: number, z: number;
      if (positions && i3 + 2 < positions.length) {
        x = positions[i3]; y = positions[i3 + 1]; z = positions[i3 + 2];
      } else {
        const p = node.position || { x: 0, y: 0, z: 0 };
        x = p.x; y = p.y; z = p.z;
      }
      _mat.makeScale(s, s, s);
      _mat.setPosition(x, y, z);
      inst.setMatrixAt(i, _mat);

      // Per-instance color via Three.js managed instanceColor
      const c = computeColor(node, mode);
      inst.setColorAt(i, c);
    }
    inst.count = nodes.length;
    inst.instanceMatrix.needsUpdate = true;

    // Only flag instanceColor when SSSP mode or graph mode changes trigger recoloring
    // (colors are always written above, but the upload is the expensive part)
    if (inst.instanceColor) inst.instanceColor.needsUpdate = true;

    // Dirty-flag metadata texture: only upload when inputs structurally change
    if (texBuf) {
      const metaHash = `${nodes.length}-${connectionCountMap.size}-${selectedNodeId}`;
      if (metaHash !== prevMetaHashRef.current) {
        for (let i = 0; i < nodes.length; i++) {
          const node = nodes[i];
          const i4 = i * 4;
          const nid = String(node.id);
          texBuf[i4]     = node.metadata?.quality ?? node.metadata?.authorityScore ?? 0.5;
          texBuf[i4 + 1] = node.metadata?.authority ?? node.metadata?.authorityScore ?? 0;
          const cc = connectionCountMap.get(nid) || 0;
          texBuf[i4 + 2] = Math.min(cc / 20, 1.0);
          texBuf[i4 + 3] = computeRecency(node.metadata?.lastModified ?? node.metadata?.updatedAt);
        }
        if (metaTexRef.current) metaTexRef.current.needsUpdate = true;
        prevMetaHashRef.current = metaHash;
      }
    }
  });

  return (
    <primitive
      object={mesh}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      onPointerMissed={onPointerMissed}
      onDoubleClick={onDoubleClick}
    />
  );
};

export const GemNodes = forwardRef(GemNodesInner);
export default GemNodes;
