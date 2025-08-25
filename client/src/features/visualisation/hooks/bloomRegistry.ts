// Bloom registry to manage object groups for selective bloom
import * as THREE from 'three';

type Obj = THREE.Object3D;

const env = new Set<Obj>();
const nodes = new Set<Obj>();
const edges = new Set<Obj>();

function safeAdd(set: Set<Obj>, obj?: Obj | null) {
  if (obj) set.add(obj);
}
function safeDelete(set: Set<Obj>, obj?: Obj | null) {
  if (obj) set.delete(obj);
}

export function registerEnvObject(obj?: Obj | null) { safeAdd(env, obj); }
export function unregisterEnvObject(obj?: Obj | null) { safeDelete(env, obj); }
export function getEnvSelection(): Obj[] { return Array.from(env); }

export function registerNodeObject(obj?: Obj | null) { safeAdd(nodes, obj); }
export function unregisterNodeObject(obj?: Obj | null) { safeDelete(nodes, obj); }
export function getNodeSelection(): Obj[] { return Array.from(nodes); }

export function registerEdgeObject(obj?: Obj | null) { safeAdd(edges, obj); }
export function unregisterEdgeObject(obj?: Obj | null) { safeDelete(edges, obj); }
export function getEdgeSelection(): Obj[] { return Array.from(edges); }

export function clearBloomRegistry() {
  env.clear();
  nodes.clear();
  edges.clear();
}