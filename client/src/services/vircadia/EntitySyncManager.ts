

import { ClientCore, QueryResult } from './VircadiaClientCore';
import { GraphEntityMapper, GraphData, GraphNode, GraphEdge, VircadiaEntity } from './GraphEntityMapper';
import { createLogger } from '../../utils/loggerConfig';

const logger = createLogger('EntitySyncManager');

export interface SyncConfig {
    syncGroup: string;
    batchSize: number;
    syncIntervalMs: number;
    enableRealTimePositions: boolean;
}

export interface SyncStats {
    totalEntities: number;
    syncedNodes: number;
    syncedEdges: number;
    lastSyncTime: number;
    pendingUpdates: number;
    errors: number;
}

export class EntitySyncManager {
    private mapper: GraphEntityMapper;
    private syncInterval: ReturnType<typeof setInterval> | null = null;
    private pendingPositionUpdates = new Map<string, { x: number; y: number; z: number }>();
    private stats: SyncStats = {
        totalEntities: 0,
        syncedNodes: 0,
        syncedEdges: 0,
        lastSyncTime: 0,
        pendingUpdates: 0,
        errors: 0
    };

    private config: SyncConfig = {
        syncGroup: 'public.NORMAL',
        batchSize: 100,
        syncIntervalMs: 100, 
        enableRealTimePositions: true
    };

    constructor(
        private client: ClientCore,
        config?: Partial<SyncConfig>
    ) {
        this.config = { ...this.config, ...config };
        this.mapper = new GraphEntityMapper({
            syncGroup: this.config.syncGroup,
            loadPriority: 0,
            createdBy: 'visionflow-xr'
        });

        
        this.setupEventListeners();
    }

    private setupEventListeners(): void {
        
        this.client.Utilities.Connection.addEventListener('syncUpdate', () => {
            logger.debug('Received sync update from server');
            
        });

        
        this.client.Utilities.Connection.addEventListener('statusChange', () => {
            const info = this.client.Utilities.Connection.getConnectionInfo();
            if (info.isConnected) {
                logger.info('Connected - starting sync manager');
                this.startRealTimeSync();
            } else {
                logger.warn('Disconnected - stopping sync manager');
                this.stopRealTimeSync();
            }
        });
    }

    
    async pushGraphToVircadia(graphData: GraphData): Promise<void> {
        logger.info(`Pushing graph to Vircadia: ${graphData.nodes.length} nodes, ${graphData.edges.length} edges`);

        const entities = this.mapper.mapGraphToEntities(graphData);
        this.stats.totalEntities = entities.length;

        
        for (let i = 0; i < entities.length; i += this.config.batchSize) {
            const batch = entities.slice(i, i + this.config.batchSize);
            await this.insertEntitiesBatch(batch);
        }

        this.stats.syncedNodes = graphData.nodes.length;
        this.stats.syncedEdges = graphData.edges.length;
        this.stats.lastSyncTime = Date.now();

        logger.info('Graph push complete', this.stats);
    }

    
    async pullGraphFromVircadia(): Promise<GraphData> {
        logger.info('Pulling graph from Vircadia');

        try {
            const query = `
                SELECT * FROM entity.entities
                WHERE group__sync = $1
                AND (general__entity_name LIKE 'node_%'
                   OR general__entity_name LIKE 'edge_%')
                ORDER BY group__load_priority ASC
            `;

            const result = await this.client.Utilities.Connection.query<{ result: VircadiaEntity[] }>({
                query,
                parameters: [this.config.syncGroup],
                timeoutMs: 30000
            });

            if (!result || !result.result) {
                logger.warn('No entities found in Vircadia');
                return { nodes: [], edges: [] };
            }

            const entities = result.result as unknown as VircadiaEntity[];
            const graphData = GraphEntityMapper.entitiesToGraph(entities);

            logger.info(`Pulled graph from Vircadia: ${graphData.nodes.length} nodes, ${graphData.edges.length} edges`);
            return graphData;

        } catch (error) {
            logger.error('Failed to pull graph from Vircadia:', error);
            this.stats.errors++;
            throw error;
        }
    }

    
    updateNodePosition(nodeId: string, position: { x: number; y: number; z: number }): void {
        if (!this.config.enableRealTimePositions) {
            return;
        }

        const entityName = `node_${nodeId}`;
        this.pendingPositionUpdates.set(entityName, position);
        this.stats.pendingUpdates = this.pendingPositionUpdates.size;
    }

    
    private async flushPositionUpdates(): Promise<void> {
        if (this.pendingPositionUpdates.size === 0) {
            return;
        }

        const updates = Array.from(this.pendingPositionUpdates.entries());
        this.pendingPositionUpdates.clear();

        try {
            const parameterizedUpdates = updates.map(([entityName, position]) =>
                this.mapper.generatePositionUpdateSQL(entityName, position)
            );

            for (const { query, parameters } of parameterizedUpdates) {
                await this.client.Utilities.Connection.query({
                    query,
                    parameters,
                    timeoutMs: 5000
                });
            }

            logger.debug(`Flushed ${updates.length} position updates to Vircadia`);
            this.stats.pendingUpdates = 0;

        } catch (error) {
            logger.error('Failed to flush position updates:', error);
            this.stats.errors++;

            updates.forEach(([entityName, position]) => {
                this.pendingPositionUpdates.set(entityName, position);
            });
        }
    }

    
    private async insertEntitiesBatch(entities: VircadiaEntity[]): Promise<void> {
        try {
            const { queries } = this.mapper.generateBatchInsertSQL(entities);

            for (const { query, parameters } of queries) {
                await this.client.Utilities.Connection.query({
                    query,
                    parameters,
                    timeoutMs: 10000
                });
            }

            logger.debug(`Inserted batch of ${entities.length} entities`);

        } catch (error) {
            logger.error('Failed to insert entity batch:', error);
            this.stats.errors++;
            throw error;
        }
    }

    
    private startRealTimeSync(): void {
        if (this.syncInterval) {
            return;
        }

        logger.info(`Starting real-time sync (interval: ${this.config.syncIntervalMs}ms)`);

        this.syncInterval = setInterval(() => {
            this.flushPositionUpdates();
        }, this.config.syncIntervalMs);
    }

    
    private stopRealTimeSync(): void {
        if (this.syncInterval) {
            clearInterval(this.syncInterval);
            this.syncInterval = null;
            logger.info('Stopped real-time sync');
        }
    }

    
    onEntityUpdate(callback: (entities: VircadiaEntity[]) => void): () => void {
        const handler = async () => {
            try {
                
                const query = `
                    SELECT * FROM entity.entities
                    WHERE group__sync = $1
                    AND general__updated_at > NOW() - INTERVAL '1 second'
                    ORDER BY general__updated_at DESC
                `;

                const result = await this.client.Utilities.Connection.query<{ result: VircadiaEntity[] }>({
                    query,
                    parameters: [this.config.syncGroup],
                    timeoutMs: 5000
                });

                if (result?.result) {
                    const entities = result.result as unknown as VircadiaEntity[];
                    callback(entities);
                }

            } catch (error) {
                logger.error('Failed to query entity updates:', error);
            }
        };

        this.client.Utilities.Connection.addEventListener('syncUpdate', handler);

        
        return () => {
            this.client.Utilities.Connection.removeEventListener('syncUpdate', handler);
        };
    }

    
    async deleteEntity(entityName: string): Promise<void> {
        try {
            const query = `
                DELETE FROM entity.entities
                WHERE general__entity_name = $1
            `;

            await this.client.Utilities.Connection.query({
                query,
                parameters: [entityName],
                timeoutMs: 5000
            });

            logger.debug(`Deleted entity: ${entityName}`);

        } catch (error) {
            logger.error(`Failed to delete entity ${entityName}:`, error);
            this.stats.errors++;
            throw error;
        }
    }

    
    async clearGraph(): Promise<void> {
        logger.warn('Clearing all graph entities from Vircadia');

        try {
            const query = `
                DELETE FROM entity.entities
                WHERE group__sync = $1
                AND (general__entity_name LIKE 'node_%' OR general__entity_name LIKE 'edge_%')
            `;

            await this.client.Utilities.Connection.query({
                query,
                parameters: [this.config.syncGroup],
                timeoutMs: 10000
            });

            this.stats.totalEntities = 0;
            this.stats.syncedNodes = 0;
            this.stats.syncedEdges = 0;

            logger.info('Graph cleared from Vircadia');

        } catch (error) {
            logger.error('Failed to clear graph:', error);
            this.stats.errors++;
            throw error;
        }
    }

    
    getStats(): SyncStats {
        return { ...this.stats };
    }

    
    dispose(): void {
        this.stopRealTimeSync();
        this.pendingPositionUpdates.clear();
        logger.info('EntitySyncManager disposed');
    }
}
