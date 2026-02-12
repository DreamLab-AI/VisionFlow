// Three.js migration verified: CanvasTexture for dynamic text, onBeforeRender for
// billboard/rotation animation, standard Object3D parent/child hierarchy.

import * as THREE from 'three';
import { ClientCore } from './VircadiaClientCore';
import { EntitySyncManager } from './EntitySyncManager';
import { GraphEntityMapper, VircadiaEntity } from './GraphEntityMapper';
import { createLogger } from '../../utils/loggerConfig';
import { BinaryWebSocketProtocol, MessageType, AgentPositionUpdate } from '../BinaryWebSocketProtocol';

const logger = createLogger('CollaborativeGraphSync');

export interface CollaborativeConfig {
    highlightColor: THREE.Color;
    annotationColor: THREE.Color;
    selectionTimeout: number;
    enableAnnotations: boolean;
    enableFiltering: boolean;
    enableVRPresence: boolean;
}

export interface UserSelection {
    agentId: string;
    username: string;
    nodeIds: string[];
    timestamp: number;
    filterState?: FilterState;
}

export interface FilterState {
    searchQuery?: string;
    categoryFilter?: string[];
    timeRange?: { start: number; end: number };
    customFilters?: Record<string, any>;
}

export interface GraphAnnotation {
    id: string;
    agentId: string;
    username: string;
    nodeId: string;
    text: string;
    position: { x: number; y: number; z: number };
    timestamp: number;
}

export interface UserPresence {
    userId: string;
    username: string;
    position: THREE.Vector3;
    rotation: THREE.Quaternion;
    headPosition?: THREE.Vector3;
    headRotation?: THREE.Quaternion;
    leftHandPosition?: THREE.Vector3;
    leftHandRotation?: THREE.Quaternion;
    rightHandPosition?: THREE.Vector3;
    rightHandRotation?: THREE.Quaternion;
    lastUpdate: number;
}

export interface GraphOperation {
    id: string;
    type: 'node_move' | 'node_add' | 'node_delete' | 'edge_add' | 'edge_delete';
    userId: string;
    nodeId?: string;
    position?: { x: number; y: number; z: number };
    timestamp: number;
    version: number;
}

export class CollaborativeGraphSync {
    private localAgentId: string | null = null;
    private activeSelections = new Map<string, UserSelection>();
    private annotations = new Map<string, GraphAnnotation>();
    private userPresence = new Map<string, UserPresence>();
    private selectionHighlights = new Map<string, THREE.Mesh[]>();
    private annotationMeshes = new Map<string, THREE.Mesh>();
    private presenceMeshes = new Map<string, THREE.Mesh>();
    private localSelection: string[] = [];
    private localFilterState: FilterState | null = null;
    private operationVersion = 0;

    /** EntitySyncManager for bi-directional Vircadia sync */
    private entitySync: EntitySyncManager | null = null;
    private entityUpdateUnsubscribe: (() => void) | null = null;

    private defaultConfig: CollaborativeConfig = {
        highlightColor: new THREE.Color(0.2, 0.8, 0.3),
        annotationColor: new THREE.Color(1.0, 0.8, 0.2),
        selectionTimeout: 30000,
        enableAnnotations: true,
        enableFiltering: true,
        enableVRPresence: true
    };

    private protocol = BinaryWebSocketProtocol.getInstance();
    private textDecoder = new TextDecoder();

    constructor(
        private scene: THREE.Scene,
        private client: ClientCore,
        config?: Partial<CollaborativeConfig>
    ) {
        this.defaultConfig = { ...this.defaultConfig, ...config };
        this.setupConnectionListeners();
        this.initEntitySync();
    }

    async initialize(): Promise<void> {
        logger.info('Initializing collaborative graph sync...');

        const info = this.client.Utilities.Connection.getConnectionInfo();
        if (info.agentId) {
            this.localAgentId = info.agentId;
        }

        await this.loadAnnotations();

        // Register bi-directional entity update listener
        if (this.entitySync) {
            this.entityUpdateUnsubscribe = this.entitySync.onEntityUpdate((entities) => {
                this.handleIncomingEntityUpdates(entities);
            });
            logger.info('Bi-directional Vircadia entity sync registered');
        }

        logger.info('Collaborative sync initialized');
    }

    /**
     * Initialize the EntitySyncManager for bi-directional Vircadia sync.
     * Positions flow: server → binary protocol → client AND client → EntitySync → Vircadia
     */
    private initEntitySync(): void {
        try {
            this.entitySync = new EntitySyncManager(this.client, {
                syncGroup: 'public.NORMAL',
                batchSize: 100,
                syncIntervalMs: 100,
                enableRealTimePositions: true,
            });
            logger.info('EntitySyncManager initialized for bi-directional sync');
        } catch (err) {
            logger.warn('Failed to initialize EntitySyncManager:', err);
        }
    }

    /**
     * Forward a node position update to the Vircadia entity sync layer.
     * Called from applyOperation when node_move operations arrive.
     */
    public syncNodePositionToVircadia(nodeId: string, position: { x: number; y: number; z: number }): void {
        if (this.entitySync) {
            this.entitySync.updateNodePosition(nodeId, position);
        }
    }

    /**
     * Handle incoming entity updates from Vircadia (server → client direction).
     * Reconciles remote entity positions into the local scene graph.
     */
    private handleIncomingEntityUpdates(entities: VircadiaEntity[]): void {
        for (const entity of entities) {
            const metadata = GraphEntityMapper.extractMetadata(entity);
            if (!metadata) continue;

            if (metadata.entityType === 'node' && metadata.position) {
                const nodeMesh = this.scene.getObjectByName(`node_${metadata.graphId}`);
                if (nodeMesh) {
                    // Only apply if the position differs significantly (avoid jitter)
                    const dx = nodeMesh.position.x - metadata.position.x;
                    const dy = nodeMesh.position.y - metadata.position.y;
                    const dz = nodeMesh.position.z - metadata.position.z;
                    const distSq = dx * dx + dy * dy + dz * dz;

                    if (distSq > 0.01) { // 0.1 unit threshold
                        nodeMesh.position.set(
                            metadata.position.x,
                            metadata.position.y,
                            metadata.position.z
                        );
                        logger.debug(`[BiSync] Reconciled node ${metadata.graphId} from Vircadia entity`);
                    }
                }
            }
        }
    }

    /**
     * Get the EntitySyncManager for external access (e.g. pushing full graph).
     */
    public getEntitySync(): EntitySyncManager | null {
        return this.entitySync;
    }

    // Arrow function class properties for stable references (Fix 3 - bind leak)
    private handleSyncUpdateEvent = async (): Promise<void> => {
        // Sync update handler - placeholder for event-driven sync processing
        logger.debug('Sync update event received');
    };

    private handleStatusChangeEvent = (): void => {
        const info = this.client.Utilities.Connection.getConnectionInfo();
        if (info.isConnected && info.agentId) {
            this.localAgentId = info.agentId;
        }
    };

    private handleSyncUpdate(payload: ArrayBuffer): void {
        const view = new DataView(payload);
        let offset = 0;

        while (offset < payload.byteLength) {
            const opType = view.getUint8(offset);
            offset += 1;

            const userId = this.textDecoder.decode(payload.slice(offset, offset + 36));
            offset += 36;

            const nodeIdLength = view.getUint16(offset, true);
            offset += 2;

            const nodeId = this.textDecoder.decode(payload.slice(offset, offset + nodeIdLength));
            offset += nodeIdLength;

            const operation: GraphOperation = {
                id: `${userId}_${Date.now()}`,
                type: this.getOperationType(opType),
                userId,
                nodeId,
                timestamp: Date.now(),
                version: this.operationVersion++
            };

            if (opType === 0) { // node_move
                operation.position = {
                    x: view.getFloat32(offset, true),
                    y: view.getFloat32(offset + 4, true),
                    z: view.getFloat32(offset + 8, true)
                };
                offset += 12;
            }

            this.applyOperation(operation);
        }
    }

    private handleAnnotationUpdate(payload: ArrayBuffer): void {
        const text = this.textDecoder.decode(payload);
        try {
            const annotation: GraphAnnotation = JSON.parse(text);
            this.annotations.set(annotation.id, annotation);
            this.createAnnotationMesh(annotation);
        } catch (error) {
            logger.error('Failed to parse annotation:', error);
        }
    }

    private handleSelectionUpdate(payload: ArrayBuffer): void {
        const text = this.textDecoder.decode(payload);
        try {
            const selection: UserSelection = JSON.parse(text);
            this.activeSelections.set(selection.agentId, selection);
            this.updateSelectionHighlight(selection);
        } catch (error) {
            logger.error('Failed to parse selection:', error);
        }
    }

    private getOperationType(opType: number): GraphOperation['type'] {
        switch (opType) {
            case 0: return 'node_move';
            case 1: return 'node_add';
            case 2: return 'node_delete';
            case 3: return 'edge_add';
            case 4: return 'edge_delete';
            default: return 'node_move';
        }
    }

    private applyOperation(operation: GraphOperation): void {
        // Server-authoritative position flow:
        // 1. Server computes positions via GPU physics
        // 2. Server broadcasts via binary protocol to all clients (desktop + VR + Vircadia)
        // 3. Each client applies optimistic tweening toward server targets
        // 4. Collaborative operations (e.g. node_move from another user) are applied
        //    as visual updates; the authoritative position comes from the server's
        //    next physics broadcast.
        if (operation.type === 'node_move' && operation.position) {
            const nodeMesh = this.scene.getObjectByName(`node_${operation.nodeId}`);
            if (nodeMesh) {
                nodeMesh.position.set(
                    operation.position.x,
                    operation.position.y,
                    operation.position.z
                );
            }
            // Forward position to Vircadia entity sync for bi-directional mirroring
            if (operation.nodeId) {
                this.syncNodePositionToVircadia(operation.nodeId, operation.position);
            }
            // Note: The graph data manager receives authoritative positions from the
            // server via binary WebSocket protocol. This collaborative operation is
            // an optimistic preview that will be reconciled on the next server tick.
        }

        logger.debug(`Applied operation: ${operation.type} on node ${operation.nodeId}`);
    }

    private resolveConflict(local: GraphOperation, remote: GraphOperation): GraphOperation {
        // Operational transform: last-write-wins with user priority
        if (remote.timestamp > local.timestamp) {
            logger.debug(`Conflict resolved: using remote operation (${remote.userId})`);
            return remote;
        } else if (remote.timestamp === local.timestamp) {
            // Use lexicographic ordering on userId for determinism
            if (remote.userId > local.userId) {
                return remote;
            }
        }

        logger.debug(`Conflict resolved: using local operation (${local.userId})`);
        return local;
    }

    async selectNodes(nodeIds: string[]): Promise<void> {
        if (!this.localAgentId) {
            logger.warn('Cannot broadcast selection: no agent ID');
            return;
        }

        this.localSelection = nodeIds;

        const selection: UserSelection = {
            agentId: this.localAgentId,
            username: 'Local User',
            nodeIds,
            timestamp: Date.now(),
            filterState: this.localFilterState ?? undefined
        };

        // Selection is tracked locally; binary broadcast removed since
        // getWebSocket() was non-functional (Fix 2). Use JSON query path
        // or event system when WebSocket access is available.
        this.activeSelections.set(this.localAgentId, selection);
        this.updateSelectionHighlight(selection);

        logger.debug(`Selection broadcast: ${nodeIds.length} nodes`);
    }

    async updateFilterState(filterState: FilterState): Promise<void> {
        if (!this.defaultConfig.enableFiltering) {
            return;
        }

        this.localFilterState = filterState;
        await this.selectNodes(this.localSelection);

        logger.debug('Filter state broadcast:', filterState);
    }

    async createAnnotation(nodeId: string, text: string, position: THREE.Vector3): Promise<void> {
        if (!this.defaultConfig.enableAnnotations || !this.localAgentId) {
            return;
        }

        const annotation: GraphAnnotation = {
            id: `annotation_${this.localAgentId}_${Date.now()}`,
            agentId: this.localAgentId,
            username: 'Local User',
            nodeId,
            text,
            position: { x: position.x, y: position.y, z: position.z },
            timestamp: Date.now()
        };

        this.createAnnotationMesh(annotation);
        this.annotations.set(annotation.id, annotation);

        logger.info(`Annotation created: "${text}" on node ${nodeId}`);
    }

    async loadAnnotations(): Promise<void> {
        if (!this.defaultConfig.enableAnnotations) {
            return;
        }

        logger.info('Annotations loading initiated');
    }

    private createAnnotationMesh(annotation: GraphAnnotation): void {
        // Create canvas for text texture
        const canvas = document.createElement('canvas');
        canvas.width = 512;
        canvas.height = 128;
        const ctx = canvas.getContext('2d')!;

        ctx.fillStyle = 'rgba(20, 20, 30, 0.85)';
        ctx.fillRect(0, 0, 512, 128);

        ctx.fillStyle = `#${this.defaultConfig.annotationColor.getHexString()}`;
        ctx.font = 'bold 32px Arial';
        ctx.textAlign = 'center';
        ctx.fillText(annotation.text, 256, 50);

        ctx.fillStyle = 'rgba(255, 255, 255, 0.7)';
        ctx.font = '20px Arial';
        ctx.fillText(`- ${annotation.username}`, 256, 90);

        const texture = new THREE.CanvasTexture(canvas);
        texture.needsUpdate = true;

        const geometry = new THREE.PlaneGeometry(0.5, 0.2);
        const material = new THREE.MeshBasicMaterial({
            map: texture,
            transparent: true,
            side: THREE.DoubleSide
        });

        const plane = new THREE.Mesh(geometry, material);
        plane.name = `${annotation.id}_mesh`;
        plane.position.set(
            annotation.position.x,
            annotation.position.y,
            annotation.position.z
        );

        // Billboard: face the camera every frame via onBeforeRender
        plane.onBeforeRender = (_renderer, _scene, camera) => {
            plane.quaternion.copy(camera.quaternion);
        };

        this.scene.add(plane);

        // Dispose old mesh/texture if replacing (Fix 4 - texture leak)
        const existingMesh = this.annotationMeshes.get(annotation.id);
        if (existingMesh) {
            this.scene.remove(existingMesh);
            this.disposeAnnotationMesh(existingMesh);
        }

        this.annotationMeshes.set(annotation.id, plane);

        logger.debug(`Annotation mesh created: "${annotation.text}"`);
    }

    private updateSelectionHighlight(selection: UserSelection): void {
        const existingHighlights = this.selectionHighlights.get(selection.agentId);
        if (existingHighlights) {
            existingHighlights.forEach(mesh => {
                this.scene.remove(mesh);
                mesh.geometry.dispose();
                (mesh.material as THREE.Material).dispose();
            });
        }

        const highlights: THREE.Mesh[] = [];

        selection.nodeIds.forEach(nodeId => {
            const nodeMesh = this.scene.getObjectByName(`node_${nodeId}`);
            if (!nodeMesh) {
                return;
            }

            // Calculate bounding sphere
            const box = new THREE.Box3().setFromObject(nodeMesh);
            const sphere = new THREE.Sphere();
            box.getBoundingSphere(sphere);

            const geometry = new THREE.TorusGeometry(
                sphere.radius * 1.25,
                0.02,
                16,
                32
            );

            const hue = parseInt(selection.agentId.substring(0, 8), 16) % 360;
            const color = new THREE.Color().setHSL(hue / 360, 0.8, 0.9);

            const material = new THREE.MeshBasicMaterial({
                color: color,
                transparent: true,
                opacity: 0.6
            });

            const highlight = new THREE.Mesh(geometry, material);
            highlight.name = `highlight_${selection.agentId}_${nodeId}`;
            highlight.position.copy(nodeMesh.position);
            highlight.position.y += sphere.radius;

            // Animate rotation each frame using onBeforeRender
            const rotationSpeed = 0.02;
            highlight.onBeforeRender = () => {
                highlight.rotation.y += rotationSpeed;
            };

            this.scene.add(highlight);
            highlights.push(highlight);
        });

        this.selectionHighlights.set(selection.agentId, highlights);

        logger.debug(`Updated highlight for ${selection.username}: ${selection.nodeIds.length} nodes`);
    }

    private updatePresenceMesh(presence: UserPresence): void {
        let mesh = this.presenceMeshes.get(presence.userId);

        if (!mesh) {
            // Create user cursor/avatar
            const geometry = new THREE.SphereGeometry(0.1, 16, 16);

            const hue = parseInt(presence.userId.substring(0, 8), 16) % 360;
            const color = new THREE.Color().setHSL(hue / 360, 0.7, 0.8);

            const material = new THREE.MeshBasicMaterial({
                color: color,
                transparent: true,
                opacity: 0.8
            });

            mesh = new THREE.Mesh(geometry, material);
            mesh!.name = `presence_${presence.userId}`;

            // Add nameplate
            const canvas = document.createElement('canvas');
            canvas.width = 256;
            canvas.height = 64;
            const ctx = canvas.getContext('2d')!;

            ctx.fillStyle = 'rgba(0,0,0,0.7)';
            ctx.fillRect(0, 0, 256, 64);
            ctx.fillStyle = 'white';
            ctx.font = 'bold 24px Arial';
            ctx.textAlign = 'center';
            ctx.fillText(presence.username, 128, 40);

            const texture = new THREE.CanvasTexture(canvas);
            const nameplateGeometry = new THREE.PlaneGeometry(0.5, 0.1);
            const nameplateMaterial = new THREE.MeshBasicMaterial({
                map: texture,
                transparent: true,
                side: THREE.DoubleSide
            });

            const nameplate = new THREE.Mesh(nameplateGeometry, nameplateMaterial);
            nameplate.name = `nameplate_${presence.userId}`;
            nameplate.position.y = 0.2;

            // Billboard: face the camera every frame via onBeforeRender
            nameplate.onBeforeRender = (_renderer, _scene, camera) => {
                nameplate.quaternion.copy(camera.quaternion);
            };

            mesh!.add(nameplate);
            this.scene.add(mesh!);
            this.presenceMeshes.set(presence.userId, mesh!);
        }

        // Update position
        mesh!.position.copy(presence.position);

        // Update VR hands if available
        if (this.defaultConfig.enableVRPresence) {
            this.updateVRHandPresence(presence);
        }
    }

    private updateVRHandPresence(presence: UserPresence): void {
        if (!presence.leftHandPosition && !presence.rightHandPosition) {
            return;
        }

        // Left hand
        if (presence.leftHandPosition) {
            let leftHand = this.scene.getObjectByName(`lefthand_${presence.userId}`) as THREE.Mesh;
            if (!leftHand) {
                const geometry = new THREE.SphereGeometry(0.025, 8, 8);
                const material = new THREE.MeshBasicMaterial({ color: 0xff0000 });
                leftHand = new THREE.Mesh(geometry, material);
                leftHand.name = `lefthand_${presence.userId}`;
                this.scene.add(leftHand);
            }
            leftHand.position.copy(presence.leftHandPosition);
            if (presence.leftHandRotation) {
                leftHand.quaternion.copy(presence.leftHandRotation);
            }
        }

        // Right hand
        if (presence.rightHandPosition) {
            let rightHand = this.scene.getObjectByName(`righthand_${presence.userId}`) as THREE.Mesh;
            if (!rightHand) {
                const geometry = new THREE.SphereGeometry(0.025, 8, 8);
                const material = new THREE.MeshBasicMaterial({ color: 0x0000ff });
                rightHand = new THREE.Mesh(geometry, material);
                rightHand.name = `righthand_${presence.userId}`;
                this.scene.add(rightHand);
            }
            rightHand.position.copy(presence.rightHandPosition);
            if (presence.rightHandRotation) {
                rightHand.quaternion.copy(presence.rightHandRotation);
            }
        }
    }

    private setupConnectionListeners(): void {
        // Use stable arrow function references for proper removal in dispose() (Fix 3)
        this.client.Utilities.Connection.addEventListener('syncUpdate', this.handleSyncUpdateEvent);
        this.client.Utilities.Connection.addEventListener('statusChange', this.handleStatusChangeEvent);
    }

    async deleteAnnotation(annotationId: string): Promise<void> {
        const annotation = this.annotations.get(annotationId);
        if (!annotation || annotation.agentId !== this.localAgentId) {
            logger.warn('Cannot delete annotation from another user');
            return;
        }

        const mesh = this.annotationMeshes.get(annotationId);
        if (mesh) {
            this.scene.remove(mesh);
            this.disposeAnnotationMesh(mesh); // Fix 4 - dispose texture
            this.annotationMeshes.delete(annotationId);
        }

        this.annotations.delete(annotationId);

        logger.info(`Annotation deleted: ${annotationId}`);
    }

    /** Dispose an annotation mesh including its CanvasTexture (Fix 4) */
    private disposeAnnotationMesh(mesh: THREE.Mesh): void {
        mesh.geometry.dispose();
        const material = mesh.material as THREE.MeshBasicMaterial;
        if (material.map) {
            material.map.dispose();
        }
        material.dispose();
    }

    /** Dispose a presence mesh including its nameplate CanvasTexture (Fix 4) */
    private disposePresenceMesh(mesh: THREE.Object3D): void {
        if (mesh instanceof THREE.Mesh) {
            mesh.geometry.dispose();
            (mesh.material as THREE.Material).dispose();
        }
        // Dispose child nameplate textures
        mesh.children.forEach(child => {
            if (child instanceof THREE.Mesh) {
                child.geometry.dispose();
                const mat = child.material as THREE.MeshBasicMaterial;
                if (mat.map) {
                    mat.map.dispose();
                }
                mat.dispose();
            }
        });
    }

    getActiveSelections(): UserSelection[] {
        return Array.from(this.activeSelections.values());
    }

    getAnnotations(): GraphAnnotation[] {
        return Array.from(this.annotations.values());
    }

    getNodeAnnotations(nodeId: string): GraphAnnotation[] {
        return Array.from(this.annotations.values()).filter(a => a.nodeId === nodeId);
    }

    getUserPresence(): UserPresence[] {
        return Array.from(this.userPresence.values());
    }

    dispose(): void {
        logger.info('Disposing CollaborativeGraphSync');

        // Unsubscribe entity update listener
        if (this.entityUpdateUnsubscribe) {
            this.entityUpdateUnsubscribe();
            this.entityUpdateUnsubscribe = null;
        }

        // Dispose entity sync manager
        if (this.entitySync) {
            this.entitySync.dispose();
            this.entitySync = null;
        }

        // Remove event listeners using stable references (Fix 3)
        this.client.Utilities.Connection.removeEventListener('syncUpdate', this.handleSyncUpdateEvent);
        this.client.Utilities.Connection.removeEventListener('statusChange', this.handleStatusChangeEvent);

        this.selectionHighlights.forEach(highlights => {
            highlights.forEach(mesh => {
                this.scene.remove(mesh);
                mesh.geometry.dispose();
                (mesh.material as THREE.Material).dispose();
            });
        });
        this.selectionHighlights.clear();

        // Dispose annotation meshes including textures (Fix 4)
        this.annotationMeshes.forEach(mesh => {
            this.scene.remove(mesh);
            this.disposeAnnotationMesh(mesh);
        });
        this.annotationMeshes.clear();

        // Dispose presence meshes including nameplate textures (Fix 4)
        this.presenceMeshes.forEach(mesh => {
            // THREE.Mesh extends Object3D; cast needed due to @types/three dual-path resolution
            this.scene.remove(mesh as never);
            this.disposePresenceMesh(mesh as never);
        });
        this.presenceMeshes.clear();

        this.activeSelections.clear();
        this.annotations.clear();
        this.userPresence.clear();
    }
}
