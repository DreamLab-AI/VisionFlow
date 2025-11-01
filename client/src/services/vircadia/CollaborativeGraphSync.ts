

import * as BABYLON from '@babylonjs/core';
import { ClientCore } from './VircadiaClientCore';
import { createLogger } from '../../utils/loggerConfig';

const logger = createLogger('CollaborativeGraphSync');

export interface CollaborativeConfig {
    highlightColor: BABYLON.Color3;
    annotationColor: BABYLON.Color3;
    selectionTimeout: number;
    enableAnnotations: boolean;
    enableFiltering: boolean;
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

export class CollaborativeGraphSync {
    private localAgentId: string | null = null;
    private activeSelections = new Map<string, UserSelection>();
    private annotations = new Map<string, GraphAnnotation>();
    private selectionHighlights = new Map<string, BABYLON.Mesh[]>();
    private annotationMeshes = new Map<string, BABYLON.Mesh>();
    private syncInterval: ReturnType<typeof setInterval> | null = null;
    private localSelection: string[] = [];
    private localFilterState: FilterState | null = null;

    private defaultConfig: CollaborativeConfig = {
        highlightColor: new BABYLON.Color3(0.2, 0.8, 0.3),
        annotationColor: new BABYLON.Color3(1.0, 0.8, 0.2),
        selectionTimeout: 30000, 
        enableAnnotations: true,
        enableFiltering: true
    };

    constructor(
        private scene: BABYLON.Scene,
        private client: ClientCore,
        config?: Partial<CollaborativeConfig>
    ) {
        this.defaultConfig = { ...this.defaultConfig, ...config };
        this.setupConnectionListeners();
    }

    
    async initialize(): Promise<void> {
        logger.info('Initializing collaborative graph sync...');

        
        const info = this.client.Utilities.Connection.getConnectionInfo();
        if (info.agentId) {
            this.localAgentId = info.agentId;
        }

        
        this.startSyncInterval();

        
        await this.loadAnnotations();

        logger.info('Collaborative sync initialized');
    }

    
    private setupConnectionListeners(): void {
        
        this.client.Utilities.Connection.addEventListener('syncUpdate', async () => {
            await this.fetchRemoteSelections();
            if (this.defaultConfig.enableAnnotations) {
                await this.fetchAnnotations();
            }
        });

        
        this.client.Utilities.Connection.addEventListener('statusChange', () => {
            const info = this.client.Utilities.Connection.getConnectionInfo();
            if (info.isConnected && info.agentId) {
                this.localAgentId = info.agentId;
            }
        });
    }

    
    async selectNodes(nodeIds: string[]): Promise<void> {
        if (!this.localAgentId) {
            logger.warn('Cannot broadcast selection: no agent ID');
            return;
        }

        this.localSelection = nodeIds;

        try {
            const query = `
                INSERT INTO entity.entities (
                    general__entity_name,
                    general__semantic_version,
                    group__sync,
                    meta__data
                ) VALUES (
                    'selection_${this.localAgentId}',
                    '1.0.0',
                    'public.NORMAL',
                    '${JSON.stringify({
                        type: 'selection',
                        agentId: this.localAgentId,
                        nodeIds,
                        filterState: this.localFilterState,
                        timestamp: Date.now()
                    })}'::jsonb
                )
                ON CONFLICT (general__entity_name)
                DO UPDATE SET meta__data = EXCLUDED.meta__data
            `;

            await this.client.Utilities.Connection.query({ query, timeoutMs: 2000 });

            logger.debug(`Selection broadcast: ${nodeIds.length} nodes`);

        } catch (error) {
            logger.error('Failed to broadcast selection:', error);
        }
    }

    
    async updateFilterState(filterState: FilterState): Promise<void> {
        if (!this.defaultConfig.enableFiltering) {
            return;
        }

        this.localFilterState = filterState;

        
        await this.selectNodes(this.localSelection);

        logger.debug('Filter state broadcast:', filterState);
    }

    
    async createAnnotation(nodeId: string, text: string, position: BABYLON.Vector3): Promise<void> {
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

        try {
            const query = `
                INSERT INTO entity.entities (
                    general__entity_name,
                    general__semantic_version,
                    group__sync,
                    meta__data
                ) VALUES (
                    '${annotation.id}',
                    '1.0.0',
                    'public.NORMAL',
                    '${JSON.stringify({
                        type: 'annotation',
                        ...annotation
                    })}'::jsonb
                )
            `;

            await this.client.Utilities.Connection.query({ query, timeoutMs: 3000 });

            
            this.createAnnotationMesh(annotation);

            this.annotations.set(annotation.id, annotation);

            logger.info(`Annotation created: "${text}" on node ${nodeId}`);

        } catch (error) {
            logger.error('Failed to create annotation:', error);
        }
    }

    
    private async fetchRemoteSelections(): Promise<void> {
        try {
            const query = `
                SELECT * FROM entity.entities
                WHERE general__entity_name LIKE 'selection_%'
                AND general__entity_name != 'selection_${this.localAgentId}'
                AND general__created_at > NOW() - INTERVAL '${this.defaultConfig.selectionTimeout / 1000} seconds'
            `;

            const result = await this.client.Utilities.Connection.query<{ result: any[] }>({
                query,
                timeoutMs: 3000
            });

            if (!result?.result) {
                return;
            }

            const selections = result.result as any[];

            
            this.activeSelections.clear();

            for (const selection of selections) {
                const metadata = selection.meta__data;
                const agentId = metadata.agentId;

                const userSelection: UserSelection = {
                    agentId,
                    username: metadata.username || `User ${agentId.substring(0, 8)}`,
                    nodeIds: metadata.nodeIds || [],
                    timestamp: metadata.timestamp || Date.now(),
                    filterState: metadata.filterState
                };

                this.activeSelections.set(agentId, userSelection);

                
                this.updateSelectionHighlight(userSelection);
            }

        } catch (error) {
            logger.debug('Failed to fetch remote selections:', error);
        }
    }

    
    private updateSelectionHighlight(selection: UserSelection): void {
        
        const existingHighlights = this.selectionHighlights.get(selection.agentId);
        if (existingHighlights) {
            existingHighlights.forEach(mesh => mesh.dispose());
        }

        const highlights: BABYLON.Mesh[] = [];

        
        selection.nodeIds.forEach(nodeId => {
            const nodeMesh = this.scene.getMeshByName(`node_${nodeId}`);
            if (!nodeMesh) {
                return;
            }

            
            const highlight = BABYLON.MeshBuilder.CreateTorus(
                `highlight_${selection.agentId}_${nodeId}`,
                {
                    diameter: nodeMesh.getBoundingInfo().boundingSphere.radiusWorld * 2.5,
                    thickness: 0.02,
                    tessellation: 32
                },
                this.scene
            );

            highlight.position = nodeMesh.position.clone();
            highlight.position.y += nodeMesh.getBoundingInfo().boundingSphere.radiusWorld;

            
            this.scene.onBeforeRenderObservable.add(() => {
                highlight.rotation.y += 0.02;
            });

            
            const hue = parseInt(selection.agentId.substring(0, 8), 16) % 360;
            const color = BABYLON.Color3.FromHSV(hue, 0.8, 0.9);

            const material = new BABYLON.StandardMaterial(`highlight_mat_${selection.agentId}`, this.scene);
            material.emissiveColor = color;
            material.disableLighting = true;
            material.alpha = 0.6;

            highlight.material = material;

            highlights.push(highlight);
        });

        this.selectionHighlights.set(selection.agentId, highlights);

        logger.debug(`Updated highlight for ${selection.username}: ${selection.nodeIds.length} nodes`);
    }

    
    private async fetchAnnotations(): Promise<void> {
        try {
            const query = `
                SELECT * FROM entity.entities
                WHERE general__entity_name LIKE 'annotation_%'
                AND general__created_at > NOW() - INTERVAL '1 hour'
            `;

            const result = await this.client.Utilities.Connection.query<{ result: any[] }>({
                query,
                timeoutMs: 3000
            });

            if (!result?.result) {
                return;
            }

            const annotationEntities = result.result as any[];

            for (const entity of annotationEntities) {
                const metadata = entity.meta__data;

                if (metadata.type !== 'annotation') {
                    continue;
                }

                const annotation: GraphAnnotation = {
                    id: entity.general__entity_name,
                    agentId: metadata.agentId,
                    username: metadata.username,
                    nodeId: metadata.nodeId,
                    text: metadata.text,
                    position: metadata.position,
                    timestamp: metadata.timestamp
                };

                if (!this.annotations.has(annotation.id)) {
                    this.annotations.set(annotation.id, annotation);
                    this.createAnnotationMesh(annotation);
                }
            }

        } catch (error) {
            logger.debug('Failed to fetch annotations:', error);
        }
    }

    
    private async loadAnnotations(): Promise<void> {
        if (!this.defaultConfig.enableAnnotations) {
            return;
        }

        await this.fetchAnnotations();
        logger.info(`Loaded ${this.annotations.size} annotations`);
    }

    
    private createAnnotationMesh(annotation: GraphAnnotation): void {
        
        const plane = BABYLON.MeshBuilder.CreatePlane(
            `${annotation.id}_mesh`,
            { width: 0.5, height: 0.2 },
            this.scene
        );

        plane.position = new BABYLON.Vector3(
            annotation.position.x,
            annotation.position.y,
            annotation.position.z
        );
        plane.billboardMode = BABYLON.Mesh.BILLBOARDMODE_ALL;

        
        const dynamicTexture = new BABYLON.DynamicTexture(
            `${annotation.id}_texture`,
            { width: 512, height: 128 },
            this.scene
        );

        const ctx = dynamicTexture.getContext();
        ctx.fillStyle = 'rgba(20, 20, 30, 0.85)';
        ctx.fillRect(0, 0, 512, 128);

        
        ctx.fillStyle = this.defaultConfig.annotationColor.toHexString();
        ctx.font = 'bold 32px Arial';
        ctx.textAlign = 'center';
        ctx.fillText(annotation.text, 256, 50);

        
        ctx.fillStyle = 'rgba(255, 255, 255, 0.7)';
        ctx.font = '20px Arial';
        ctx.fillText(`- ${annotation.username}`, 256, 90);

        dynamicTexture.update();

        const material = new BABYLON.StandardMaterial(`${annotation.id}_mat`, this.scene);
        material.diffuseTexture = dynamicTexture;
        material.emissiveTexture = dynamicTexture;
        material.opacityTexture = dynamicTexture;
        material.backFaceCulling = false;

        plane.material = material;

        this.annotationMeshes.set(annotation.id, plane);

        logger.debug(`Annotation mesh created: "${annotation.text}"`);
    }

    
    async deleteAnnotation(annotationId: string): Promise<void> {
        const annotation = this.annotations.get(annotationId);
        if (!annotation) {
            return;
        }

        
        if (annotation.agentId !== this.localAgentId) {
            logger.warn('Cannot delete annotation from another user');
            return;
        }

        try {
            const query = `
                DELETE FROM entity.entities
                WHERE general__entity_name = '${annotationId}'
            `;

            await this.client.Utilities.Connection.query({ query, timeoutMs: 2000 });

            
            const mesh = this.annotationMeshes.get(annotationId);
            if (mesh) {
                mesh.dispose();
                this.annotationMeshes.delete(annotationId);
            }

            this.annotations.delete(annotationId);

            logger.info(`Annotation deleted: ${annotationId}`);

        } catch (error) {
            logger.error('Failed to delete annotation:', error);
        }
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

    
    private startSyncInterval(): void {
        if (this.syncInterval) {
            return;
        }

        logger.info('Starting collaborative sync interval');

        this.syncInterval = setInterval(async () => {
            await this.fetchRemoteSelections();

            if (this.defaultConfig.enableAnnotations) {
                await this.fetchAnnotations();
            }
        }, 1000); 
    }

    
    private stopSyncInterval(): void {
        if (this.syncInterval) {
            clearInterval(this.syncInterval);
            this.syncInterval = null;
            logger.info('Stopped collaborative sync interval');
        }
    }

    
    dispose(): void {
        logger.info('Disposing CollaborativeGraphSync');

        this.stopSyncInterval();

        
        this.selectionHighlights.forEach(highlights => {
            highlights.forEach(mesh => mesh.dispose());
        });
        this.selectionHighlights.clear();

        
        this.annotationMeshes.forEach(mesh => mesh.dispose());
        this.annotationMeshes.clear();

        this.activeSelections.clear();
        this.annotations.clear();
    }
}
