

import * as THREE from 'three';
import { createLogger } from './loggerConfig';

const logger = createLogger('GraphOptimizations');

// Frustum culling helper
export class FrustumCuller {
  private frustum = new THREE.Frustum();
  private matrix = new THREE.Matrix4();
  private _reusableSphere = new THREE.Sphere();

  public updateFrustum(camera: THREE.Camera) {
    this.matrix.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
    this.frustum.setFromProjectionMatrix(this.matrix);
  }

  public isNodeVisible(position: THREE.Vector3, radius: number = 1): boolean {
    this._reusableSphere.center.copy(position);
    this._reusableSphere.radius = radius;
    return this.frustum.intersectsSphere(this._reusableSphere);
  }

  public cullNodes(nodes: Array<{ position: THREE.Vector3; radius?: number }>) {
    return nodes.filter(node => this.isNodeVisible(node.position, node.radius));
  }
}

// Level of Detail (LOD) manager
export class LODManager {
  private camera: THREE.Camera | null = null;
  private geometryCache = new Map<string, THREE.BufferGeometry>();

  constructor(camera?: THREE.Camera) {
    this.camera = camera || null;
  }

  public setCamera(camera: THREE.Camera) {
    this.camera = camera;
  }

  public getLODLevel(nodePosition: THREE.Vector3): 'high' | 'medium' | 'low' | 'hidden' {
    if (!this.camera) return 'high';

    const distance = this.camera.position.distanceTo(nodePosition);

    if (distance < 20) return 'high';
    if (distance < 50) return 'medium';
    if (distance < 100) return 'low';
    return 'hidden';
  }

  public getGeometryForLOD(level: 'high' | 'medium' | 'low'): THREE.BufferGeometry {
    const cached = this.geometryCache.get(level);
    if (cached) return cached;

    let geometry: THREE.BufferGeometry;
    switch (level) {
      case 'high':
        geometry = new THREE.SphereGeometry(0.5, 32, 32);
        break;
      case 'medium':
        geometry = new THREE.SphereGeometry(0.5, 16, 16);
        break;
      case 'low':
      default:
        geometry = new THREE.SphereGeometry(0.5, 8, 8);
        break;
    }
    this.geometryCache.set(level, geometry);
    return geometry;
  }

  public shouldRenderNode(nodePosition: THREE.Vector3, minDistance: number = 150): boolean {
    if (!this.camera) return true;
    return this.camera.position.distanceTo(nodePosition) < minDistance;
  }

  public disposeGeometries(): void {
    this.geometryCache.forEach(geometry => geometry.dispose());
    this.geometryCache.clear();
  }
}

// Enhanced instanced rendering manager
export class InstancedRenderingManager {
  private geometryPool = new Map<string, THREE.BufferGeometry>();
  private materialPool = new Map<string, THREE.Material>();
  private maxInstances: number;
  
  constructor(maxInstances: number = 5000) {
    this.maxInstances = maxInstances;
  }
  
  public createInstancedMesh(
    geometryKey: string,
    materialKey: string,
    instanceCount: number,
    geometryFactory: () => THREE.BufferGeometry,
    materialFactory: () => THREE.Material
  ): THREE.InstancedMesh {
    
    if (!this.geometryPool.has(geometryKey)) {
      const geometry = geometryFactory();
      this.geometryPool.set(geometryKey, geometry);
    }
    
    
    if (!this.materialPool.has(materialKey)) {
      const material = materialFactory();
      this.materialPool.set(materialKey, material);
    }
    
    const geometry = this.geometryPool.get(geometryKey)!;
    const material = this.materialPool.get(materialKey)!;
    
    const count = Math.min(instanceCount, this.maxInstances);
    const mesh = new THREE.InstancedMesh(geometry, material, count);
    
    
    mesh.frustumCulled = true;
    
    return mesh;
  }
  
  // Pre-allocated reusable objects for updateInstancedMesh
  private _matrix = new THREE.Matrix4();
  private _defaultColor = new THREE.Color(0x00ffff);
  private _position = new THREE.Vector3();
  private _quaternion = new THREE.Quaternion();
  private _scaleVec = new THREE.Vector3();

  public updateInstancedMesh(
    mesh: THREE.InstancedMesh,
    nodes: Array<{ position: THREE.Vector3; scale?: number; color?: THREE.Color }>,
    lodManager?: LODManager,
    frustumCuller?: FrustumCuller
  ): { renderedCount: number; culledCount: number } {
    const matrix = this._matrix;
    let renderedCount = 0;
    let culledCount = 0;
    const maxCount = mesh.instanceMatrix.count;


    // Use native setColorAt path — InstancedBufferAttribute causes
    // drawIndexed(Infinity) crash on WebGPU backend.
    if (!mesh.instanceColor) {
      const _c = new THREE.Color();
      for (let ci = 0; ci < maxCount; ci++) mesh.setColorAt(ci, _c);
    }

    const colorAttribute = mesh.instanceColor as THREE.InstancedBufferAttribute;
    const nodeCount = Math.min(nodes.length, maxCount);

    // Compact visible instances to the front of the buffer
    for (let i = 0; i < nodeCount; i++) {
      const node = nodes[i];


      let shouldRender = true;

      if (lodManager && !lodManager.shouldRenderNode(node.position)) {
        shouldRender = false;
      }

      if (frustumCuller && !frustumCuller.isNodeVisible(node.position)) {
        shouldRender = false;
      }

      if (shouldRender) {

        const scale = node.scale || 1;
        this._position.copy(node.position);
        this._scaleVec.set(scale, scale, scale);
        matrix.compose(this._position, this._quaternion, this._scaleVec);
        mesh.setMatrixAt(renderedCount, matrix);


        const nodeColor = node.color || this._defaultColor;
        colorAttribute.setXYZ(renderedCount, nodeColor.r, nodeColor.g, nodeColor.b);

        renderedCount++;
      } else {
        culledCount++;
      }
    }


    mesh.instanceMatrix.needsUpdate = true;
    colorAttribute.needsUpdate = true;
    mesh.count = renderedCount;

    return { renderedCount, culledCount };
  }
  
  public dispose() {
    this.geometryPool.forEach(geometry => geometry.dispose());
    this.materialPool.forEach(material => material.dispose());
    this.geometryPool.clear();
    this.materialPool.clear();
  }
}

// SharedArrayBuffer communication for workers (if supported)
export class SharedBufferCommunication {
  private sharedBuffer: SharedArrayBuffer | null = null;
  private positionArray: Float32Array | null = null;
  private metadataArray: Int32Array | null = null;
  private supported: boolean;
  
  constructor() {
    try {
      new SharedArrayBuffer(1);
      this.supported = true;
    } catch {
      this.supported = false;
      logger.warn('SharedArrayBuffer not supported or blocked by security policy, falling back to message passing');
    }
  }
  
  public isSupported(): boolean {
    return this.supported;
  }
  
  public initializeBuffer(nodeCount: number): boolean {
    if (!this.supported) return false;
    
    try {
      
      const positionBytes = nodeCount * 3 * 4; 
      const metadataBytes = nodeCount * 4; 
      const totalBytes = positionBytes + metadataBytes;
      
      this.sharedBuffer = new SharedArrayBuffer(totalBytes);
      
      
      this.positionArray = new Float32Array(this.sharedBuffer, 0, nodeCount * 3);
      this.metadataArray = new Int32Array(this.sharedBuffer, positionBytes, nodeCount);
      
      logger.info(`SharedArrayBuffer initialized: ${totalBytes} bytes for ${nodeCount} nodes`);
      return true;
    } catch (error) {
      logger.error('Failed to initialize SharedArrayBuffer:', error);
      return false;
    }
  }
  
  public getSharedBuffer(): SharedArrayBuffer | null {
    return this.sharedBuffer;
  }
  
  public getPositionArray(): Float32Array | null {
    return this.positionArray;
  }
  
  public getMetadataArray(): Int32Array | null {
    return this.metadataArray;
  }
  
  public updatePositions(positions: Float32Array): boolean {
    if (!this.positionArray || positions.length > this.positionArray.length) {
      return false;
    }
    
    this.positionArray.set(positions);
    return true;
  }
  
  public updateMetadata(metadata: Int32Array): boolean {
    if (!this.metadataArray || metadata.length > this.metadataArray.length) {
      return false;
    }
    
    this.metadataArray.set(metadata);
    return true;
  }
  
  public dispose() {
    this.sharedBuffer = null;
    this.positionArray = null;
    this.metadataArray = null;
  }
}

// Spatial partitioning using Octree for large node counts
export class SpatialOctree {
  private root: OctreeNode | null = null;
  private bounds: THREE.Box3;
  private maxDepth: number;
  private maxObjectsPerNode: number;
  
  constructor(bounds: THREE.Box3, maxDepth: number = 8, maxObjectsPerNode: number = 10) {
    this.bounds = bounds.clone();
    this.maxDepth = maxDepth;
    this.maxObjectsPerNode = maxObjectsPerNode;
  }
  
  public insert(object: { position: THREE.Vector3; id: string; [key: string]: any }) {
    if (!this.root) {
      this.root = new OctreeNode(this.bounds, 0);
    }
    
    this.root.insert(object, this.maxDepth, this.maxObjectsPerNode);
  }
  
  public query(frustum: THREE.Frustum): Array<{ position: THREE.Vector3; id: string; [key: string]: any }> {
    if (!this.root) return [];
    
    const results: Array<{ position: THREE.Vector3; id: string; [key: string]: any }> = [];
    this.root.query(frustum, results);
    return results;
  }
  
  public queryRadius(center: THREE.Vector3, radius: number): Array<{ position: THREE.Vector3; id: string; [key: string]: any }> {
    if (!this.root) return [];
    
    const results: Array<{ position: THREE.Vector3; id: string; [key: string]: any }> = [];
    const sphere = new THREE.Sphere(center, radius);
    this.root.queryRadius(sphere, results);
    return results;
  }
  
  public clear() {
    this.root = null;
  }
}

class OctreeNode {
  private bounds: THREE.Box3;
  private depth: number;
  private objects: Array<{ position: THREE.Vector3; id: string; [key: string]: any }> = [];
  private children: OctreeNode[] | null = null;
  
  constructor(bounds: THREE.Box3, depth: number) {
    this.bounds = bounds.clone();
    this.depth = depth;
  }
  
  public insert(object: { position: THREE.Vector3; id: string; [key: string]: any }, maxDepth: number, maxObjects: number) {
    if (!this.bounds.containsPoint(object.position)) {
      return false;
    }
    
    if (this.objects.length < maxObjects || this.depth >= maxDepth) {
      this.objects.push(object);
      return true;
    }
    
    if (!this.children) {
      this.subdivide();
    }
    
    for (const child of this.children!) {
      if (child.insert(object, maxDepth, maxObjects)) {
        return true;
      }
    }
    
    
    this.objects.push(object);
    return true;
  }
  
  public query(frustum: THREE.Frustum, results: Array<{ position: THREE.Vector3; id: string; [key: string]: any }>) {
    if (!frustum.intersectsBox(this.bounds)) {
      return;
    }
    
    for (const object of this.objects) {
      results.push(object);
    }
    
    if (this.children) {
      for (const child of this.children) {
        child.query(frustum, results);
      }
    }
  }
  
  public queryRadius(sphere: THREE.Sphere, results: Array<{ position: THREE.Vector3; id: string; [key: string]: any }>) {
    if (!sphere.intersectsBox(this.bounds)) {
      return;
    }
    
    for (const object of this.objects) {
      if (sphere.containsPoint(object.position)) {
        results.push(object);
      }
    }
    
    if (this.children) {
      for (const child of this.children) {
        child.queryRadius(sphere, results);
      }
    }
  }
  
  private subdivide() {
    const center = this.bounds.getCenter(new THREE.Vector3());
    const min = this.bounds.min;
    const max = this.bounds.max;
    
    this.children = [
      new OctreeNode(new THREE.Box3(new THREE.Vector3(min.x, min.y, min.z), new THREE.Vector3(center.x, center.y, center.z)), this.depth + 1),
      new OctreeNode(new THREE.Box3(new THREE.Vector3(center.x, min.y, min.z), new THREE.Vector3(max.x, center.y, center.z)), this.depth + 1),
      new OctreeNode(new THREE.Box3(new THREE.Vector3(min.x, center.y, min.z), new THREE.Vector3(center.x, max.y, center.z)), this.depth + 1),
      new OctreeNode(new THREE.Box3(new THREE.Vector3(center.x, center.y, min.z), new THREE.Vector3(max.x, max.y, center.z)), this.depth + 1),
      new OctreeNode(new THREE.Box3(new THREE.Vector3(min.x, min.y, center.z), new THREE.Vector3(center.x, center.y, max.z)), this.depth + 1),
      new OctreeNode(new THREE.Box3(new THREE.Vector3(center.x, min.y, center.z), new THREE.Vector3(max.x, center.y, max.z)), this.depth + 1),
      new OctreeNode(new THREE.Box3(new THREE.Vector3(min.x, center.y, center.z), new THREE.Vector3(center.x, max.y, max.z)), this.depth + 1),
      new OctreeNode(new THREE.Box3(new THREE.Vector3(center.x, center.y, center.z), new THREE.Vector3(max.x, max.y, max.z)), this.depth + 1),
    ];
  }
}

// Performance optimization helper
export class GraphOptimizer {
  private frustumCuller = new FrustumCuller();
  private lodManager = new LODManager();
  private instancedManager = new InstancedRenderingManager();
  private sharedBuffer = new SharedBufferCommunication();
  private octree: SpatialOctree | null = null;
  
  public initializeOptimizations(camera: THREE.Camera, renderer: THREE.WebGLRenderer) {
    this.lodManager.setCamera(camera);
    
    
    const bounds = new THREE.Box3(
      new THREE.Vector3(-100, -100, -100),
      new THREE.Vector3(100, 100, 100)
    );
    this.octree = new SpatialOctree(bounds);
    
    logger.info('Graph optimizations initialized', {
      sharedBufferSupported: this.sharedBuffer.isSupported(),
      rendererCapabilities: {
        maxTextures: renderer.capabilities.maxTextures,
        maxVertexUniforms: renderer.capabilities.maxVertexUniforms,
        maxFragmentUniforms: renderer.capabilities.maxFragmentUniforms
      }
    });
  }
  
  public optimizeFrame(camera: THREE.Camera) {
    this.frustumCuller.updateFrustum(camera);
    this.lodManager.setCamera(camera);
  }
  
  public getFrustumCuller(): FrustumCuller {
    return this.frustumCuller;
  }
  
  public getLODManager(): LODManager {
    return this.lodManager;
  }
  
  public getInstancedManager(): InstancedRenderingManager {
    return this.instancedManager;
  }
  
  public getSharedBuffer(): SharedBufferCommunication {
    return this.sharedBuffer;
  }
  
  public getOctree(): SpatialOctree | null {
    return this.octree;
  }
  
  public dispose() {
    this.instancedManager.dispose();
    this.lodManager.disposeGeometries();
    this.sharedBuffer.dispose();
    this.octree?.clear();
  }
}

// Singleton instance
export const graphOptimizer = new GraphOptimizer();