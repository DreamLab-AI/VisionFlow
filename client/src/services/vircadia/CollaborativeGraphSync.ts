// TODO: Migrated from Babylon.js to Three.js
// Some features may require additional Three.js implementations:
// - DynamicTexture equivalent (use CanvasTexture)
// - Billboard mode (use sprites or manual camera-facing)
// - Mesh parent/child relationships work differently

import * as THREE from 'three';
import { ClientCore } from './VircadiaClientCore';
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
    private presenceMeshes = new Map<string, THREE.Object3D>();
    private localSelection: string[] = [];
    private localFilterState: FilterState | null = null;
    private operationVersion = 0;
    private pendingOperations: GraphOperation[] = [];

    private defaultConfig: CollaborativeConfig = {
        highlightColor: new THREE.Color(0.2, 0.8, 0.3),
        annotationColor: new THREE.Color(1.0, 0.8, 0.2),
        selectionTimeout: 30000,
        enableAnnotations: true,
        enableFiltering: true,
        enableVRPresence: true
    };

    private protocol = BinaryWebSocketProtocol.getInstance();
    private ws: WebSocket | null = null;

    constructor(
        private scene: THREE.Scene,
        private client: ClientCore,
        config?: Partial<CollaborativeConfig>
    ) {
        this.defaultConfig = { ...this.defaultConfig, ...config };
        this.setupConnectionListeners();
    }

    async initialize(): Promise<void> {
        logger.info('Initializing collaborative graph sync with WebSocket...');

        const info = this.client.Utilities.Connection.getConnectionInfo();
        if (info.agentId) {
            this.localAgentId = info.agentId;
        }

        await this.initializeWebSocketSync();
        await this.loadAnnotations();

        logger.info('Collaborative sync initialized');
    }

    private async initializeWebSocketSync(): Promise<void> {
        // Get WebSocket connection from client core
        // @ts-ignore - getWebSocket may not be exposed in public interface
        this.ws = (this.client.Utilities.Connection as any).getWebSocket?.() ?? null;

        if (!this.ws) {
            logger.warn('No WebSocket connection available');
            return;
        }

        // Listen for sync updates via WebSocket
        this.ws.addEventListener('message', this.handleWebSocketMessage.bind(this));

        // Subscribe to sync events
        this.sendSubscription('graph_sync');
        this.sendSubscription('user_presence');
        this.sendSubscription('annotations');

        logger.info('WebSocket sync initialized');
    }

    private handleWebSocketMessage(event: MessageEvent): void {
        if (!(event.data instanceof ArrayBuffer)) {
            return;
        }

        const header = this.protocol.parseHeader(event.data);
        if (!header) {
            return;
        }

        const payload = this.protocol.extractPayload(event.data, header);

        switch (header.type) {
            case MessageType.POSITION_UPDATE:
                this.handleUserPositions(payload);
                break;
            case 0x10: // SYNC_UPDATE (new message type)
                this.handleSyncUpdate(payload);
                break;
            case 0x11: // ANNOTATION_UPDATE
                this.handleAnnotationUpdate(payload);
                break;
            case 0x12: // SELECTION_UPDATE
                this.handleSelectionUpdate(payload);
                break;
        }
    }

    private handleUserPositions(payload: ArrayBuffer): void {
        const updates = this.protocol.decodePositionUpdates(payload);

        for (const update of updates) {
            const userId = update.agentId.toString();
            if (userId === this.localAgentId) continue;

            const presence: UserPresence = {
                userId,
                username: `User ${userId}`,
                position: new THREE.Vector3(
                    update.position.x,
                    update.position.y,
                    update.position.z
                ),
                rotation: new THREE.Quaternion(),
                lastUpdate: update.timestamp
            };

            this.userPresence.set(userId, presence);
            this.updatePresenceMesh(presence);
        }
    }

    private handleSyncUpdate(payload: ArrayBuffer): void {
        const view = new DataView(payload);
        let offset = 0;

        while (offset < payload.byteLength) {
            const opType = view.getUint8(offset);
            offset += 1;

            const userId = new TextDecoder().decode(payload.slice(offset, offset + 36));
            offset += 36;

            const nodeIdLength = view.getUint16(offset, true);
            offset += 2;

            const nodeId = new TextDecoder().decode(payload.slice(offset, offset + nodeIdLength));
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
        const text = new TextDecoder().decode(payload);
        try {
            const annotation: GraphAnnotation = JSON.parse(text);
            this.annotations.set(annotation.id, annotation);
            this.createAnnotationMesh(annotation);
        } catch (error) {
            logger.error('Failed to parse annotation:', error);
        }
    }

    private handleSelectionUpdate(payload: ArrayBuffer): void {
        const text = new TextDecoder().decode(payload);
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
        // Operational transform: resolve conflicts
        const conflictingOps = this.pendingOperations.filter(
            op => op.nodeId === operation.nodeId && op.timestamp > operation.timestamp - 1000
        );

        if (conflictingOps.length > 0) {
            const resolved = this.resolveConflict(operation, conflictingOps[0]);
            operation = resolved;
        }

        // Apply the operation to the graph
        if (operation.type === 'node_move' && operation.position) {
            const nodeMesh = this.scene.getObjectByName(`node_${operation.nodeId}`);
            if (nodeMesh) {
                nodeMesh.position.set(
                    operation.position.x,
                    operation.position.y,
                    operation.position.z
                );
            }
        }

        logger.debug(`Applied operation: ${operation.type} on node ${operation.nodeId}`);
    }

    private resolveConflict(local: GraphOperation, remote: GraphOperation): GraphOperation {
        // Operational transform: last-write-wins with user priority
        // In a real implementation, use vector clocks or CRDTs
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

        // Send via WebSocket instead of polling
        this.broadcastSelection(selection);

        logger.debug(`Selection broadcast: ${nodeIds.length} nodes`);
    }

    private broadcastSelection(selection: UserSelection): void {
        if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
            return;
        }

        const message = JSON.stringify(selection);
        const payload = new TextEncoder().encode(message).buffer;
        const packet = this.protocol.createMessage(0x12 as MessageType, payload);

        this.ws.send(packet);
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
        this.broadcastAnnotation(annotation);

        logger.info(`Annotation created: "${text}" on node ${nodeId}`);
    }

    private broadcastAnnotation(annotation: GraphAnnotation): void {
        if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
            return;
        }

        const message = JSON.stringify(annotation);
        const payload = new TextEncoder().encode(message).buffer;
        const packet = this.protocol.createMessage(0x11 as MessageType, payload);

        this.ws.send(packet);
    }

    async loadAnnotations(): Promise<void> {
        if (!this.defaultConfig.enableAnnotations) {
            return;
        }

        // Request annotations via WebSocket subscription
        this.sendSubscription('annotations');
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

        // TODO: Billboard mode - need to implement camera-facing in render loop
        // plane.lookAt(camera.position) should be called each frame

        this.scene.add(plane);
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

            // TODO: Rotation animation needs to be handled in render loop
            // Add userData for animation
            highlight.userData.rotationSpeed = 0.02;

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

            // @ts-ignore - THREE.js type mismatch between Mesh and Object3D
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
            // TODO: Billboard mode for nameplate

            mesh!.add(nameplate);
            // @ts-ignore - THREE.js type mismatch between Object3D variants
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

    broadcastUserPosition(position: THREE.Vector3, rotation: THREE.Quaternion): void {
        if (!this.localAgentId || !this.ws || this.ws.readyState !== WebSocket.OPEN) {
            return;
        }

        const update: AgentPositionUpdate = {
            agentId: parseInt(this.localAgentId) || 0,
            position: { x: position.x, y: position.y, z: position.z },
            timestamp: Date.now(),
            flags: 0
        };

        const packet = this.protocol.encodePositionUpdates([update]);
        if (packet) {
            this.ws.send(packet);
        }
    }

    broadcastVRPresence(
        headPos: THREE.Vector3,
        headRot: THREE.Quaternion,
        leftHandPos?: THREE.Vector3,
        leftHandRot?: THREE.Quaternion,
        rightHandPos?: THREE.Vector3,
        rightHandRot?: THREE.Quaternion
    ): void {
        if (!this.localAgentId || !this.ws || this.ws.readyState !== WebSocket.OPEN) {
            return;
        }

        // Encode VR presence data (29 bytes for head + 29*2 for hands = 87 bytes total)
        const buffer = new ArrayBuffer(87);
        const view = new DataView(buffer);
        let offset = 0;

        // Head position (12 bytes)
        view.setFloat32(offset, headPos.x, true); offset += 4;
        view.setFloat32(offset, headPos.y, true); offset += 4;
        view.setFloat32(offset, headPos.z, true); offset += 4;

        // Head rotation (16 bytes)
        view.setFloat32(offset, headRot.x, true); offset += 4;
        view.setFloat32(offset, headRot.y, true); offset += 4;
        view.setFloat32(offset, headRot.z, true); offset += 4;
        view.setFloat32(offset, headRot.w, true); offset += 4;

        // User ID (1 byte flag: 0 = no hands, 1 = left only, 2 = right only, 3 = both)
        let handFlags = 0;
        if (leftHandPos) handFlags |= 1;
        if (rightHandPos) handFlags |= 2;
        view.setUint8(offset, handFlags); offset += 1;

        // Left hand (29 bytes if present)
        if (leftHandPos && leftHandRot) {
            view.setFloat32(offset, leftHandPos.x, true); offset += 4;
            view.setFloat32(offset, leftHandPos.y, true); offset += 4;
            view.setFloat32(offset, leftHandPos.z, true); offset += 4;
            view.setFloat32(offset, leftHandRot.x, true); offset += 4;
            view.setFloat32(offset, leftHandRot.y, true); offset += 4;
            view.setFloat32(offset, leftHandRot.z, true); offset += 4;
            view.setFloat32(offset, leftHandRot.w, true); offset += 4;
            offset += 1; // padding
        } else {
            offset += 29;
        }

        // Right hand (29 bytes if present)
        if (rightHandPos && rightHandRot) {
            view.setFloat32(offset, rightHandPos.x, true); offset += 4;
            view.setFloat32(offset, rightHandPos.y, true); offset += 4;
            view.setFloat32(offset, rightHandPos.z, true); offset += 4;
            view.setFloat32(offset, rightHandRot.x, true); offset += 4;
            view.setFloat32(offset, rightHandRot.y, true); offset += 4;
            view.setFloat32(offset, rightHandRot.z, true); offset += 4;
            view.setFloat32(offset, rightHandRot.w, true); offset += 4;
        }

        const packet = this.protocol.createMessage(0x10 as MessageType, buffer);
        this.ws.send(packet);
    }

    private sendSubscription(channel: string): void {
        if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
            return;
        }

        const message = JSON.stringify({ type: 'subscribe', channel });
        const payload = new TextEncoder().encode(message).buffer;
        const packet = this.protocol.createMessage(0xFF as MessageType, payload);

        this.ws.send(packet);
        logger.debug(`Subscribed to channel: ${channel}`);
    }

    private setupConnectionListeners(): void {
        this.client.Utilities.Connection.addEventListener('statusChange', () => {
            const info = this.client.Utilities.Connection.getConnectionInfo();
            if (info.isConnected && info.agentId) {
                this.localAgentId = info.agentId;
                this.initializeWebSocketSync();
            }
        });
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
            mesh.geometry.dispose();
            (mesh.material as THREE.Material).dispose();
            this.annotationMeshes.delete(annotationId);
        }

        this.annotations.delete(annotationId);

        // Broadcast deletion via WebSocket
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            const message = JSON.stringify({ type: 'delete_annotation', id: annotationId });
            const payload = new TextEncoder().encode(message).buffer;
            const packet = this.protocol.createMessage(0x11 as MessageType, payload);
            this.ws.send(packet);
        }

        logger.info(`Annotation deleted: ${annotationId}`);
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

        if (this.ws) {
            this.ws.removeEventListener('message', this.handleWebSocketMessage.bind(this));
        }

        this.selectionHighlights.forEach(highlights => {
            highlights.forEach(mesh => {
                this.scene.remove(mesh);
                mesh.geometry.dispose();
                (mesh.material as THREE.Material).dispose();
            });
        });
        this.selectionHighlights.clear();

        this.annotationMeshes.forEach(mesh => {
            this.scene.remove(mesh);
            mesh.geometry.dispose();
            (mesh.material as THREE.Material).dispose();
        });
        this.annotationMeshes.clear();

        this.presenceMeshes.forEach(mesh => {
            // @ts-ignore - THREE.js type mismatch between Object3D variants
            this.scene.remove(mesh);
            if (mesh instanceof THREE.Mesh) {
                mesh.geometry.dispose();
                (mesh.material as THREE.Material).dispose();
            }
        });
        this.presenceMeshes.clear();

        this.activeSelections.clear();
        this.annotations.clear();
        this.userPresence.clear();
    }
}
