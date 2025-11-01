

import * as BABYLON from '@babylonjs/core';
import { ClientCore } from '../../services/vircadia/VircadiaClientCore';
import { EntitySyncManager } from '../../services/vircadia/EntitySyncManager';
import { GraphEntityMapper, VircadiaEntity, VircadiaEntityMetadata } from '../../services/vircadia/GraphEntityMapper';
import { createLogger } from '../../utils/loggerConfig';

const logger = createLogger('VircadiaSceneBridge');

export interface SceneBridgeConfig {
    enableRealTimeSync: boolean;
    instancedRendering: boolean;
    enableLOD: boolean;
    maxRenderDistance: number;
}

export class VircadiaSceneBridge {
    private syncManager: EntitySyncManager;
    private entityMeshes = new Map<string, BABYLON.TransformNode>();
    private masterNodeMesh: BABYLON.Mesh | null = null;
    private masterEdgeMaterial: BABYLON.LinesMesh | null = null;
    private unsubscribeEntityUpdates: (() => void) | null = null;

    private config: SceneBridgeConfig = {
        enableRealTimeSync: true,
        instancedRendering: true,
        enableLOD: true,
        maxRenderDistance: 50
    };

    constructor(
        private scene: BABYLON.Scene,
        private client: ClientCore,
        config?: Partial<SceneBridgeConfig>
    ) {
        this.config = { ...this.config, ...config };
        this.syncManager = new EntitySyncManager(client, {
            syncGroup: 'public.NORMAL',
            batchSize: 100,
            syncIntervalMs: 100,
            enableRealTimePositions: this.config.enableRealTimeSync
        });

        this.initializeMasterMeshes();
        this.setupEntityUpdateListener();

        logger.info('VircadiaSceneBridge initialized', this.config);
    }

    
    private initializeMasterMeshes(): void {
        
        this.masterNodeMesh = BABYLON.MeshBuilder.CreateSphere(
            'master_node_sphere',
            { diameter: 1, segments: 16 },
            this.scene
        );
        this.masterNodeMesh.isVisible = false;

        
        const nodeMaterial = new BABYLON.StandardMaterial('master_node_material', this.scene);
        nodeMaterial.emissiveColor = new BABYLON.Color3(0.1, 0.2, 0.5);
        nodeMaterial.diffuseColor = new BABYLON.Color3(0.3, 0.5, 0.8);
        this.masterNodeMesh.material = nodeMaterial;

        logger.debug('Master meshes created for instancing');
    }

    
    private createMeshFromEntity(entity: VircadiaEntity): BABYLON.TransformNode | null {
        const metadata = GraphEntityMapper.extractMetadata(entity);
        if (!metadata) {
            logger.warn('Entity has no metadata, skipping', entity.general__entity_name);
            return null;
        }

        if (metadata.entityType === 'node') {
            return this.createNodeMesh(entity, metadata);
        } else if (metadata.entityType === 'edge') {
            return this.createEdgeMesh(entity, metadata);
        }

        return null;
    }

    
    private createNodeMesh(entity: VircadiaEntity, metadata: VircadiaEntityMetadata): BABYLON.TransformNode {
        let mesh: BABYLON.TransformNode;

        if (this.config.instancedRendering && this.masterNodeMesh) {
            
            mesh = this.masterNodeMesh.createInstance(entity.general__entity_name);
        } else {
            
            mesh = BABYLON.MeshBuilder.CreateSphere(
                entity.general__entity_name,
                { diameter: 1, segments: 16 },
                this.scene
            );
        }

        
        if (metadata.position) {
            mesh.position = new BABYLON.Vector3(
                metadata.position.x,
                metadata.position.y,
                metadata.position.z
            );
        }

        if (metadata.scale) {
            mesh.scaling = new BABYLON.Vector3(
                metadata.scale.x,
                metadata.scale.y,
                metadata.scale.z
            );
        }

        
        if (metadata.color && mesh instanceof BABYLON.Mesh) {
            const color = BABYLON.Color3.FromHexString(metadata.color);
            const material = new BABYLON.StandardMaterial(`${entity.general__entity_name}_mat`, this.scene);
            material.emissiveColor = color.scale(0.3);
            material.diffuseColor = color;
            mesh.material = material;
        }

        
        if (metadata.label) {
            this.createNodeLabel(mesh, metadata.label);
        }

        
        if (this.config.enableLOD && mesh instanceof BABYLON.Mesh) {
            this.setupLOD(mesh);
        }

        logger.debug(`Created node mesh: ${entity.general__entity_name}`, metadata);
        return mesh;
    }

    
    private createEdgeMesh(entity: VircadiaEntity, metadata: VircadiaEntityMetadata): BABYLON.TransformNode {
        const sourcePos = metadata.position || { x: 0, y: 0, z: 0 };
        const targetPos = (metadata.visualProperties?.targetPosition as { x: number; y: number; z: number }) || sourcePos;

        const points = [
            new BABYLON.Vector3(sourcePos.x, sourcePos.y, sourcePos.z),
            new BABYLON.Vector3(targetPos.x, targetPos.y, targetPos.z)
        ];

        const line = BABYLON.MeshBuilder.CreateLines(
            entity.general__entity_name,
            { points },
            this.scene
        );

        
        if (metadata.color) {
            line.color = BABYLON.Color3.FromHexString(metadata.color);
        } else {
            line.color = new BABYLON.Color3(0.4, 0.4, 0.5);
        }

        logger.debug(`Created edge mesh: ${entity.general__entity_name}`, metadata);
        return line;
    }

    
    private createNodeLabel(node: BABYLON.TransformNode, text: string): void {
        const plane = BABYLON.MeshBuilder.CreatePlane(
            `${node.name}_label`,
            { width: 0.3, height: 0.1 },
            this.scene
        );

        plane.parent = node;
        plane.position.y = 0.15;
        plane.billboardMode = BABYLON.Mesh.BILLBOARDMODE_ALL;

        
        const dynamicTexture = new BABYLON.DynamicTexture(
            `${node.name}_label_texture`,
            { width: 512, height: 128 },
            this.scene
        );

        const ctx = dynamicTexture.getContext();
        ctx.fillStyle = 'rgba(0, 0, 0, 0.6)';
        ctx.fillRect(0, 0, 512, 128);
        ctx.fillStyle = 'white';
        ctx.font = 'bold 48px Arial';
        ctx.textAlign = 'center';
        ctx.fillText(text, 256, 80);
        dynamicTexture.update();

        const labelMaterial = new BABYLON.StandardMaterial(`${node.name}_label_mat`, this.scene);
        labelMaterial.diffuseTexture = dynamicTexture;
        labelMaterial.emissiveTexture = dynamicTexture;
        labelMaterial.opacityTexture = dynamicTexture;
        labelMaterial.backFaceCulling = false;

        plane.material = labelMaterial;
    }

    
    private setupLOD(mesh: BABYLON.Mesh): void {
        const highDetail = mesh;
        const mediumDetail = mesh.clone(`${mesh.name}_medium`);
        mediumDetail.isVisible = false;
        const lowDetail = BABYLON.MeshBuilder.CreateSphere(
            `${mesh.name}_low`,
            { diameter: 1, segments: 8 },
            this.scene
        );
        lowDetail.isVisible = false;

        mesh.addLODLevel(15, mediumDetail);
        mesh.addLODLevel(30, lowDetail);
        mesh.addLODLevel(this.config.maxRenderDistance, null); 
    }

    
    private setupEntityUpdateListener(): void {
        this.unsubscribeEntityUpdates = this.syncManager.onEntityUpdate((entities) => {
            logger.debug(`Received ${entities.length} entity updates from Vircadia`);

            entities.forEach(entity => {
                this.updateOrCreateMesh(entity);
            });
        });
    }

    
    private updateOrCreateMesh(entity: VircadiaEntity): void {
        let mesh = this.entityMeshes.get(entity.general__entity_name);

        if (!mesh) {
            
            mesh = this.createMeshFromEntity(entity);
            if (mesh) {
                this.entityMeshes.set(entity.general__entity_name, mesh);
            }
        } else {
            
            const metadata = GraphEntityMapper.extractMetadata(entity);
            if (metadata?.position) {
                mesh.position = new BABYLON.Vector3(
                    metadata.position.x,
                    metadata.position.y,
                    metadata.position.z
                );
            }
        }
    }

    
    async loadGraphFromVircadia(): Promise<void> {
        logger.info('Loading graph from Vircadia...');

        try {
            const graphData = await this.syncManager.pullGraphFromVircadia();
            logger.info(`Loaded ${graphData.nodes.length} nodes and ${graphData.edges.length} edges`);

            
            this.clearScene();

            
            const entities = new GraphEntityMapper().mapGraphToEntities(graphData);
            entities.forEach(entity => {
                const mesh = this.createMeshFromEntity(entity);
                if (mesh) {
                    this.entityMeshes.set(entity.general__entity_name, mesh);
                }
            });

            logger.info('Graph rendered in Babylon scene');

        } catch (error) {
            logger.error('Failed to load graph from Vircadia:', error);
            throw error;
        }
    }

    
    async pushGraphToVircadia(graphData: { nodes: any[]; edges: any[] }): Promise<void> {
        logger.info('Pushing graph to Vircadia...');

        try {
            await this.syncManager.pushGraphToVircadia(graphData);
            logger.info('Graph pushed to Vircadia successfully');
        } catch (error) {
            logger.error('Failed to push graph to Vircadia:', error);
            throw error;
        }
    }

    
    updateNodePosition(nodeId: string, position: BABYLON.Vector3): void {
        const entityName = `node_${nodeId}`;
        const mesh = this.entityMeshes.get(entityName);

        if (mesh) {
            mesh.position = position;
        }

        
        this.syncManager.updateNodePosition(nodeId, {
            x: position.x,
            y: position.y,
            z: position.z
        });
    }

    
    clearScene(): void {
        logger.info('Clearing scene');

        this.entityMeshes.forEach(mesh => {
            mesh.dispose();
        });
        this.entityMeshes.clear();
    }

    
    getStats() {
        return this.syncManager.getStats();
    }

    
    dispose(): void {
        logger.info('Disposing VircadiaSceneBridge');

        if (this.unsubscribeEntityUpdates) {
            this.unsubscribeEntityUpdates();
        }

        this.clearScene();

        if (this.masterNodeMesh) {
            this.masterNodeMesh.dispose();
        }

        this.syncManager.dispose();
    }
}
