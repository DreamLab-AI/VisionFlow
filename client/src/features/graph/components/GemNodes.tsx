import React, { useRef, useMemo, useCallback, forwardRef, useImperativeHandle } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import type { GraphVisualMode } from './GraphManager';
import type { Node as GraphNode } from '../managers/graphDataManager';
import { createGemNodeMaterial, createGemGeometry } from '../../../rendering/materials/GemNodeMaterial';
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

  // Allocate a large buffer (4096 instances) so the mesh is created ONCE and
  // never recreated when nodes.length grows from 0→N on data load.
  // Only recreate when the visual mode (dominant) changes.
  // useFrame sets inst.count to the actual node count each frame.
  const { mesh, uniforms } = useMemo(() => {
    const count = 4096;
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
  }, [dominant]);

  // NOTE: No disposal useEffect here. React StrictMode double-invokes useMemo
  // and runs effect cleanup between mount cycles, which disposes the GPU buffers
  // of the ONLY mesh instance. WebGPU cannot recover disposed geometry/material.
  // Three.js / GC handles cleanup when the component fully unmounts.

  useImperativeHandle(ref, () => ({
    getMesh: () => meshRef.current,
    getColorArray: () => meshRef.current?.instanceColor?.array as Float32Array | null ?? null,
  }), [mesh]);

  // TSL DISABLED: Adding emissiveNode/opacityNode to MeshPhysicalMaterial triggers
  // shader recompilation that breaks InstancedMesh draw calls on WebGPU r182.
  // Visual quality achieved through standard PBR properties + per-instance color +
  // per-frame emissive modulation in useFrame instead.

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

  // Progressive reveal: ramp up visible instance count over frames so nodes
  // appear in waves (~120 nodes/frame at 60fps → full 1090 in ~0.15s).
  const revealedRef = useRef(0);
  const prevNodeCountRef = useRef(0);
  const REVEAL_BATCH = 120;

  const diagLoggedRef = useRef(false);
  const frameCountRef = useRef(0);
  useFrame(({ clock, camera, scene }) => {
    const inst = meshRef.current;
    if (!inst || nodes.length === 0) return;

    // Workaround: R3F <primitive> sometimes fails to attach InstancedMesh to scene.
    // If the mesh has no parent after mount, attach it directly.
    if (!inst.parent && scene) {
      scene.add(inst);
      console.log('[GemNodes] manually attached mesh to scene (R3F primitive workaround)');
    }

    if (uniforms.time) uniforms.time.value = clock.elapsedTime;

    // Reset progressive reveal when node count changes (new data loaded)
    if (nodes.length !== prevNodeCountRef.current) {
      revealedRef.current = 0;
      prevNodeCountRef.current = nodes.length;
    }

    const positions = nodePositionsRef.current;
    frameCountRef.current++;

    // Delayed diagnostic — fires at frame 60 when positions are loaded
    if (!diagLoggedRef.current && frameCountRef.current >= 60) {
      diagLoggedRef.current = true;
      const mat = inst.material as any;
      // Sample first 3 instance matrices
      const tempMat = new THREE.Matrix4();
      const tempVec = new THREE.Vector3();
      const tempScale = new THREE.Vector3();
      const matSamples: any[] = [];
      for (let si = 0; si < Math.min(3, inst.count); si++) {
        inst.getMatrixAt(si, tempMat);
        tempVec.setFromMatrixPosition(tempMat);
        tempScale.setFromMatrixScale(tempMat);
        matSamples.push({ i: si, pos: { x: +tempVec.x.toFixed(1), y: +tempVec.y.toFixed(1), z: +tempVec.z.toFixed(1) }, scale: +tempScale.x.toFixed(2) });
      }
      // Compute bounding box from first 20 instances
      const bbox = new THREE.Box3();
      for (let bi = 0; bi < Math.min(20, inst.count); bi++) {
        inst.getMatrixAt(bi, tempMat);
        tempVec.setFromMatrixPosition(tempMat);
        bbox.expandByPoint(tempVec);
      }
      const bboxSize = new THREE.Vector3();
      bbox.getSize(bboxSize);
      console.log('[GemNodes] DIAG frame60:', {
        nodeCount: nodes.length,
        instCount: inst.count,
        hasPositions: !!positions,
        posLen: positions?.length ?? 0,
        visible: inst.visible,
        hasParent: !!inst.parent,
        parentType: inst.parent?.type,
        frustumCulled: inst.frustumCulled,
        matType: mat?.type,
        matTransmission: mat?.transmission,
        matOpacity: mat?.opacity,
        matTransparent: mat?.transparent,
        matDepthWrite: mat?.depthWrite,
        matSide: mat?.side,
        hasOpacityNode: !!mat?.opacityNode,
        hasEmissiveNode: !!mat?.emissiveNode,
        hasColorNode: !!mat?.colorNode,
        matSamples,
        bboxSize: { x: +bboxSize.x.toFixed(1), y: +bboxSize.y.toFixed(1), z: +bboxSize.z.toFixed(1) },
        cameraPos: { x: +camera.position.x.toFixed(1), y: +camera.position.y.toFixed(1), z: +camera.position.z.toFixed(1) },
        dominant,
      });
    }
    const baseScale = (settings?.visualisation?.nodes?.nodeSize ?? 0.5) / 0.5;
    const texBuf = metaTexRef.current?.image?.data as Float32Array | undefined;

    // Per-frame emissive modulation (replaces TSL which breaks InstancedMesh on WebGPU).
    // Gentle breathing pulse on the shared material — all instances share it but
    // per-instance color variation comes from instanceColor.
    const currentMat = inst.material as any;
    if (currentMat.emissiveIntensity !== undefined) {
      const u = uniforms as any;
      if (dominant === 'agent' && u.activityLevel) {
        const pulse = Math.pow((Math.sin(clock.elapsedTime * Math.PI) + 1) * 0.5, 4);
        currentMat.emissiveIntensity = 0.15 + pulse * u.activityLevel.value * 0.2;
      } else {
        // Knowledge graph / ontology: subtle breathing emissive
        const breath = (Math.sin(clock.elapsedTime * 0.8) + 1) * 0.5;
        currentMat.emissiveIntensity = 0.3 + breath * 0.3;
      }
    }

    // Progressive reveal: ramp up visible count each frame
    if (revealedRef.current < nodes.length) {
      revealedRef.current = Math.min(revealedRef.current + REVEAL_BATCH, nodes.length);
    }
    const visCount = revealedRef.current;

    for (let i = 0; i < visCount; i++) {
      const node = nodes[i];
      const mode = perNodeVisualModeMap.get(String(node.id)) || graphMode;
      let s = getNodeScale(node, connectionCountMap, mode, hierarchyMap) * baseScale;
      if (selectedNodeId && String(node.id) === selectedNodeId) {
        s *= 1 + Math.sin(clock.elapsedTime * 3) * 0.15;
      }

      // Map from visibleNodes index to graphData.nodes index for correct position lookup
      const srcIdx = props.nodeIdToIndexMap.get(String(node.id));
      const posIdx = srcIdx !== undefined ? srcIdx : i;
      const i3 = posIdx * 3;
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
    inst.count = visCount;
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
      key={mesh.uuid}
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
