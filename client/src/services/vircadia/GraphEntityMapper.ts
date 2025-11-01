

import { createLogger } from '../../utils/loggerConfig';

const logger = createLogger('GraphEntityMapper');

// VisionFlow graph types
export interface GraphNode {
    id: string;
    label: string;
    type?: string;
    color?: string;
    size?: number;
    x?: number;
    y?: number;
    z?: number;
    metadata?: Record<string, unknown>;
}

export interface GraphEdge {
    id: string;
    source: string;
    target: string;
    label?: string;
    color?: string;
    weight?: number;
    metadata?: Record<string, unknown>;
}

export interface GraphData {
    nodes: GraphNode[];
    edges: GraphEdge[];
}

// Vircadia entity types (from Vircadia schema)
export interface VircadiaEntity {
    general__entity_name: string;
    general__semantic_version: string;
    general__created_by?: string;
    general__updated_by?: string;
    group__sync: string;
    group__load_priority: number;
    meta__data?: Record<string, unknown>;
}

export interface VircadiaEntityMetadata {
    entityType: 'node' | 'edge';
    graphId: string;
    position?: { x: number; y: number; z: number };
    rotation?: { x: number; y: number; z: number };
    scale?: { x: number; y: number; z: number };
    color?: string;
    label?: string;
    visualProperties?: Record<string, unknown>;
    sourceId?: string;  
    targetId?: string;  
}

export interface EntitySyncOptions {
    syncGroup: string;
    loadPriority: number;
    createdBy: string;
}

export class GraphEntityMapper {
    private defaultOptions: EntitySyncOptions = {
        syncGroup: 'public.NORMAL',
        loadPriority: 0,
        createdBy: 'visionflow'
    };

    constructor(private options: Partial<EntitySyncOptions> = {}) {
        this.defaultOptions = { ...this.defaultOptions, ...options };
    }

    
    mapNodeToEntity(node: GraphNode): VircadiaEntity {
        const entityName = `node_${node.id}`;

        
        const position = {
            x: node.x ?? 0,
            y: node.y ?? 0,
            z: node.z ?? 0
        };

        
        const scale = {
            x: node.size ?? 0.1,
            y: node.size ?? 0.1,
            z: node.size ?? 0.1
        };

        const metadata: VircadiaEntityMetadata = {
            entityType: 'node',
            graphId: node.id,
            position,
            rotation: { x: 0, y: 0, z: 0 },
            scale,
            color: node.color ?? '#3b82f6',
            label: node.label,
            visualProperties: {
                type: node.type,
                originalMetadata: node.metadata
            }
        };

        const entity: VircadiaEntity = {
            general__entity_name: entityName,
            general__semantic_version: '1.0.0',
            general__created_by: this.defaultOptions.createdBy,
            group__sync: this.defaultOptions.syncGroup,
            group__load_priority: this.defaultOptions.loadPriority,
            meta__data: metadata
        };

        logger.debug(`Mapped node ${node.id} to entity ${entityName}`, entity);
        return entity;
    }

    
    mapEdgeToEntity(edge: GraphEdge, nodePositions: Map<string, { x: number; y: number; z: number }>): VircadiaEntity {
        const entityName = `edge_${edge.id}`;

        const sourcePos = nodePositions.get(edge.source);
        const targetPos = nodePositions.get(edge.target);

        if (!sourcePos || !targetPos) {
            logger.warn(`Cannot map edge ${edge.id}: missing node positions`, {
                source: edge.source,
                target: edge.target,
                hasSource: !!sourcePos,
                hasTarget: !!targetPos
            });
        }

        const metadata: VircadiaEntityMetadata = {
            entityType: 'edge',
            graphId: edge.id,
            sourceId: edge.source,
            targetId: edge.target,
            color: edge.color ?? '#6b7280',
            label: edge.label,
            position: sourcePos || { x: 0, y: 0, z: 0 },
            visualProperties: {
                weight: edge.weight,
                targetPosition: targetPos || { x: 0, y: 0, z: 0 },
                originalMetadata: edge.metadata
            }
        };

        const entity: VircadiaEntity = {
            general__entity_name: entityName,
            general__semantic_version: '1.0.0',
            general__created_by: this.defaultOptions.createdBy,
            group__sync: this.defaultOptions.syncGroup,
            group__load_priority: this.defaultOptions.loadPriority + 1, 
            meta__data: metadata
        };

        logger.debug(`Mapped edge ${edge.id} to entity ${entityName}`, entity);
        return entity;
    }

    
    mapGraphToEntities(graphData: GraphData): VircadiaEntity[] {
        logger.info(`Mapping graph with ${graphData.nodes.length} nodes and ${graphData.edges.length} edges`);

        const entities: VircadiaEntity[] = [];

        
        const nodePositions = new Map<string, { x: number; y: number; z: number }>();
        graphData.nodes.forEach(node => {
            nodePositions.set(node.id, {
                x: node.x ?? 0,
                y: node.y ?? 0,
                z: node.z ?? 0
            });
        });

        
        graphData.nodes.forEach(node => {
            entities.push(this.mapNodeToEntity(node));
        });

        
        graphData.edges.forEach(edge => {
            entities.push(this.mapEdgeToEntity(edge, nodePositions));
        });

        logger.info(`Mapped ${entities.length} total entities`);
        return entities;
    }

    
    generateEntityInsertSQL(entity: VircadiaEntity): string {
        const columns = [
            'general__entity_name',
            'general__semantic_version',
            'general__created_by',
            'group__sync',
            'group__load_priority',
            'meta__data'
        ];

        const values = [
            `'${entity.general__entity_name}'`,
            `'${entity.general__semantic_version}'`,
            `'${entity.general__created_by}'`,
            `'${entity.group__sync}'`,
            entity.group__load_priority,
            `'${JSON.stringify(entity.meta__data)}'::jsonb`
        ];

        const sql = `
INSERT INTO entity.entities (${columns.join(', ')})
VALUES (${values.join(', ')})
ON CONFLICT (general__entity_name)
DO UPDATE SET
    meta__data = EXCLUDED.meta__data,
    general__updated_at = CURRENT_TIMESTAMP;
        `.trim();

        return sql;
    }

    
    generateBatchInsertSQL(entities: VircadiaEntity[]): string {
        const statements = entities.map(entity => this.generateEntityInsertSQL(entity));
        return statements.join('\n\n');
    }

    
    static extractMetadata(entity: VircadiaEntity): VircadiaEntityMetadata | null {
        if (!entity.meta__data) {
            return null;
        }
        return entity.meta__data as VircadiaEntityMetadata;
    }

    
    static entityToGraphNode(entity: VircadiaEntity): GraphNode | null {
        const metadata = GraphEntityMapper.extractMetadata(entity);
        if (!metadata || metadata.entityType !== 'node') {
            return null;
        }

        const node: GraphNode = {
            id: metadata.graphId,
            label: metadata.label || metadata.graphId,
            type: (metadata.visualProperties?.type as string) || 'default',
            color: metadata.color,
            size: metadata.scale?.x,
            x: metadata.position?.x,
            y: metadata.position?.y,
            z: metadata.position?.z,
            metadata: metadata.visualProperties?.originalMetadata as Record<string, unknown>
        };

        return node;
    }

    
    static entityToGraphEdge(entity: VircadiaEntity): GraphEdge | null {
        const metadata = GraphEntityMapper.extractMetadata(entity);
        if (!metadata || metadata.entityType !== 'edge') {
            return null;
        }

        const edge: GraphEdge = {
            id: metadata.graphId,
            source: metadata.sourceId || '',
            target: metadata.targetId || '',
            label: metadata.label,
            color: metadata.color,
            weight: metadata.visualProperties?.weight as number,
            metadata: metadata.visualProperties?.originalMetadata as Record<string, unknown>
        };

        return edge;
    }

    
    static entitiesToGraph(entities: VircadiaEntity[]): GraphData {
        const nodes: GraphNode[] = [];
        const edges: GraphEdge[] = [];

        entities.forEach(entity => {
            const node = GraphEntityMapper.entityToGraphNode(entity);
            if (node) {
                nodes.push(node);
                return;
            }

            const edge = GraphEntityMapper.entityToGraphEdge(entity);
            if (edge) {
                edges.push(edge);
            }
        });

        logger.info(`Converted ${entities.length} entities to ${nodes.length} nodes and ${edges.length} edges`);

        return { nodes, edges };
    }

    
    updateEntityPosition(
        entity: VircadiaEntity,
        position: { x: number; y: number; z: number }
    ): VircadiaEntity {
        const metadata = GraphEntityMapper.extractMetadata(entity);
        if (!metadata) {
            logger.warn(`Cannot update position: entity has no metadata`, entity);
            return entity;
        }

        metadata.position = position;

        return {
            ...entity,
            meta__data: metadata
        };
    }

    
    generatePositionUpdateSQL(
        entityName: string,
        position: { x: number; y: number; z: number }
    ): string {
        return `
UPDATE entity.entities
SET meta__data = jsonb_set(
    jsonb_set(
        jsonb_set(
            meta__data,
            '{position,x}', '${position.x}'
        ),
        '{position,y}', '${position.y}'
    ),
    '{position,z}', '${position.z}'
)
WHERE general__entity_name = '${entityName}';
        `.trim();
    }
}
