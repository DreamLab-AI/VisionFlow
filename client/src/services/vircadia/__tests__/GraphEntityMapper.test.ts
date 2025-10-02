/**
 * GraphEntityMapper Tests - Phase 5: Unit Testing
 */

import { describe, it, expect } from 'vitest';
import { GraphEntityMapper, GraphNode, GraphEdge } from '../GraphEntityMapper';

describe('GraphEntityMapper', () => {
    let mapper: GraphEntityMapper;

    beforeEach(() => {
        mapper = new GraphEntityMapper();
    });

    describe('Node Mapping', () => {
        it('should map graph node to Vircadia entity', () => {
            const node: GraphNode = {
                id: 'node-1',
                label: 'Test Node',
                x: 1.5,
                y: 2.0,
                z: 3.5,
                size: 0.2,
                color: '#ff0000'
            };

            const entity = mapper.mapNodeToEntity(node);

            expect(entity.general__entity_name).toBe('node_node-1');
            expect(entity.general__semantic_version).toBe('1.0.0');
            expect(entity.group__sync).toBe('public.NORMAL');
            expect(entity.meta__data.entityType).toBe('node');
            expect(entity.meta__data.position).toEqual({ x: 1.5, y: 2.0, z: 3.5 });
            expect(entity.meta__data.scale).toEqual({ x: 0.2, y: 0.2, z: 0.2 });
            expect(entity.meta__data.color).toBe('#ff0000');
            expect(entity.meta__data.label).toBe('Test Node');
        });

        it('should use default values for missing node properties', () => {
            const node: GraphNode = {
                id: 'node-2',
                label: 'Minimal Node'
            };

            const entity = mapper.mapNodeToEntity(node);

            expect(entity.meta__data.position).toEqual({ x: 0, y: 0, z: 0 });
            expect(entity.meta__data.scale).toEqual({ x: 0.1, y: 0.1, z: 0.1 });
            expect(entity.meta__data.color).toBe('#3b82f6');
        });
    });

    describe('Edge Mapping', () => {
        it('should map graph edge to Vircadia entity', () => {
            const edge: GraphEdge = {
                id: 'edge-1',
                source: 'node-1',
                target: 'node-2',
                sourcePosition: { x: 1.0, y: 1.0, z: 1.0 },
                targetPosition: { x: 2.0, y: 2.0, z: 2.0 },
                color: '#00ff00'
            };

            const entity = mapper.mapEdgeToEntity(edge);

            expect(entity.general__entity_name).toBe('edge_edge-1');
            expect(entity.meta__data.entityType).toBe('edge');
            expect(entity.meta__data.source).toBe('node-1');
            expect(entity.meta__data.target).toBe('node-2');
            expect(entity.meta__data.color).toBe('#00ff00');
        });

        it('should use default positions if not provided', () => {
            const edge: GraphEdge = {
                id: 'edge-2',
                source: 'node-1',
                target: 'node-2'
            };

            const entity = mapper.mapEdgeToEntity(edge);

            expect(entity.meta__data.position).toEqual({ x: 0, y: 0, z: 0 });
        });
    });

    describe('Graph to Entities', () => {
        it('should map entire graph to entities array', () => {
            const graphData = {
                nodes: [
                    { id: 'node-1', label: 'Node 1', x: 1, y: 1, z: 1 },
                    { id: 'node-2', label: 'Node 2', x: 2, y: 2, z: 2 }
                ],
                edges: [
                    { id: 'edge-1', source: 'node-1', target: 'node-2' }
                ]
            };

            const entities = mapper.mapGraphToEntities(graphData);

            expect(entities).toHaveLength(3); // 2 nodes + 1 edge
            expect(entities[0].general__entity_name).toBe('node_node-1');
            expect(entities[1].general__entity_name).toBe('node_node-2');
            expect(entities[2].general__entity_name).toBe('edge_edge-1');
        });
    });

    describe('Entity to Graph', () => {
        it('should convert Vircadia entity back to graph node', () => {
            const entity = {
                general__entity_name: 'node_node-1',
                general__semantic_version: '1.0.0',
                group__sync: 'public.NORMAL',
                meta__data: {
                    entityType: 'node',
                    graphId: 'node-1',
                    position: { x: 1.5, y: 2.0, z: 3.5 },
                    scale: { x: 0.2, y: 0.2, z: 0.2 },
                    color: '#ff0000',
                    label: 'Test Node'
                }
            };

            const node = mapper.mapEntityToNode(entity);

            expect(node.id).toBe('node-1');
            expect(node.label).toBe('Test Node');
            expect(node.x).toBe(1.5);
            expect(node.y).toBe(2.0);
            expect(node.z).toBe(3.5);
            expect(node.size).toBe(0.2);
            expect(node.color).toBe('#ff0000');
        });

        it('should convert Vircadia entity back to graph edge', () => {
            const entity = {
                general__entity_name: 'edge_edge-1',
                general__semantic_version: '1.0.0',
                group__sync: 'public.NORMAL',
                meta__data: {
                    entityType: 'edge',
                    graphId: 'edge-1',
                    source: 'node-1',
                    target: 'node-2',
                    color: '#00ff00'
                }
            };

            const edge = mapper.mapEntityToEdge(entity);

            expect(edge.id).toBe('edge-1');
            expect(edge.source).toBe('node-1');
            expect(edge.target).toBe('node-2');
            expect(edge.color).toBe('#00ff00');
        });
    });

    describe('SQL Generation', () => {
        it('should generate INSERT SQL for entity', () => {
            const entity = mapper.mapNodeToEntity({
                id: 'node-1',
                label: 'Test Node',
                x: 1, y: 1, z: 1
            });

            const sql = mapper.generateEntityInsertSQL(entity);

            expect(sql).toContain('INSERT INTO entity.entities');
            expect(sql).toContain('node_node-1');
            expect(sql).toContain('ON CONFLICT');
        });

        it('should generate position UPDATE SQL', () => {
            const sql = mapper.generatePositionUpdateSQL('node_node-1', {
                x: 1.5, y: 2.0, z: 3.5
            });

            expect(sql).toContain('UPDATE entity.entities');
            expect(sql).toContain('jsonb_set');
            expect(sql).toContain('node_node-1');
            expect(sql).toContain('1.5');
            expect(sql).toContain('2.0');
            expect(sql).toContain('3.5');
        });
    });

    describe('Metadata Extraction', () => {
        it('should extract metadata from entity', () => {
            const entity = {
                general__entity_name: 'node_node-1',
                meta__data: {
                    entityType: 'node',
                    position: { x: 1, y: 1, z: 1 },
                    color: '#ff0000'
                }
            };

            const metadata = GraphEntityMapper.extractMetadata(entity);

            expect(metadata).toBeDefined();
            expect(metadata?.entityType).toBe('node');
            expect(metadata?.position).toEqual({ x: 1, y: 1, z: 1 });
            expect(metadata?.color).toBe('#ff0000');
        });

        it('should return null for entity without metadata', () => {
            const entity = {
                general__entity_name: 'invalid',
                meta__data: null
            };

            const metadata = GraphEntityMapper.extractMetadata(entity as any);
            expect(metadata).toBeNull();
        });
    });
});
