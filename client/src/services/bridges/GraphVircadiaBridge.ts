// Three.js migration verified: all scene, mesh, vector, and material APIs use THREE.*

import * as THREE from 'three';
import { ClientCore } from '../vircadia/VircadiaClientCore';
import { CollaborativeGraphSync, type FilterState } from '../vircadia/CollaborativeGraphSync';
import { EntitySyncManager } from '../vircadia/EntitySyncManager';
import { GraphEntityMapper } from '../vircadia/GraphEntityMapper';
import { createLogger } from '../../utils/loggerConfig';

const logger = createLogger('GraphVircadiaBridge');

export interface GraphNode {
  id: string;
  label: string;
  position: { x: number; y: number; z: number };
  type?: string;
  metadata?: Record<string, any>;
}

export interface GraphEdge {
  source: string;
  target: string;
  type?: string;
}

export interface UserSelectionEvent {
  userId: string;
  username: string;
  nodeIds: string[];
}

export interface AnnotationEvent {
  id: string;
  userId: string;
  username: string;
  nodeId: string;
  text: string;
  position: { x: number; y: number; z: number };
}

export class GraphVircadiaBridge {
  private nodeEntityMap = new Map<string, string>();
  private nodePositionMap = new Map<string, { x: number; y: number; z: number }>();
  private localSelectionCallback?: (nodeIds: string[]) => void;
  private remoteSelectionCallback?: (event: UserSelectionEvent) => void;
  private annotationCallback?: (event: AnnotationEvent) => void;
  private isActive = false;
  private entitySync: EntitySyncManager | null = null;
  private mapper: GraphEntityMapper;

  constructor(
    private scene: THREE.Scene,
    private client: ClientCore,
    private collab: CollaborativeGraphSync
  ) {
    this.mapper = new GraphEntityMapper({
      syncGroup: 'public.NORMAL',
      loadPriority: 0,
      createdBy: 'visionflow-bridge'
    });
  }


  async initialize(): Promise<void> {
    logger.info('Initializing GraphVircadiaBridge...');

    if (!this.client.Utilities.Connection.getConnectionInfo().isConnected) {
      throw new Error('Vircadia client must be connected before initializing bridge');
    }

    await this.collab.initialize();

    // Initialize EntitySyncManager for pushing graph entities to Vircadia
    this.entitySync = new EntitySyncManager(this.client, {
      syncGroup: 'public.NORMAL',
      batchSize: 100,
      syncIntervalMs: 100,
      enableRealTimePositions: true,
    });

    // CollaborativeGraphSync has no EventEmitter interface.
    // Remote selection / annotation / filter events are received via
    // the binary WebSocket protocol and processed internally by
    // CollaborativeGraphSync. The bridge reads state via polling
    // (getActiveSelections, getAnnotations) instead.

    this.isActive = true;
    logger.info('GraphVircadiaBridge initialized successfully');
  }


  syncGraphToVircadia(nodes: GraphNode[], edges: GraphEdge[]): void {
    if (!this.isActive) return;

    try {

      nodes.forEach(node => {
        this.syncNodeToEntity(node);
      });


      edges.forEach(edge => {
        this.syncEdgeToEntity(edge);
      });

      logger.debug(`Synced ${nodes.length} nodes and ${edges.length} edges to Vircadia`);
    } catch (error) {
      logger.error('Failed to sync graph to Vircadia:', error);
    }
  }


  private syncNodeToEntity(node: GraphNode): void {
    const entityName = `node_${node.id}`;
    this.nodeEntityMap.set(node.id, entityName);
    this.nodePositionMap.set(node.id, node.position);

    if (!this.entitySync) {
      logger.warn(`syncNodeToEntity(${entityName}): EntitySyncManager not initialized`);
      return;
    }

    // Map bridge GraphNode to mapper GraphNode format and create entity
    const mapperNode = {
      id: node.id,
      label: node.label,
      type: node.type,
      x: node.position.x,
      y: node.position.y,
      z: node.position.z,
      metadata: node.metadata,
    };
    const entity = this.mapper.mapNodeToEntity(mapperNode);

    // Use EntitySyncManager's position update for real-time sync
    this.entitySync.updateNodePosition(node.id, node.position);

    logger.debug(`Synced node ${node.id} to entity ${entityName}`);
  }

  private syncEdgeToEntity(edge: GraphEdge): void {
    const sourceEntityName = this.nodeEntityMap.get(edge.source);
    const targetEntityName = this.nodeEntityMap.get(edge.target);

    if (!sourceEntityName || !targetEntityName) {
      logger.debug(`syncEdgeToEntity(${edge.source}->${edge.target}): missing node entities, skipping`);
      return;
    }

    if (!this.entitySync) {
      logger.warn(`syncEdgeToEntity(${edge.source}->${edge.target}): EntitySyncManager not initialized`);
      return;
    }

    // Map to mapper format using tracked node positions
    const mapperEdge = {
      id: `${edge.source}_${edge.target}`,
      source: edge.source,
      target: edge.target,
      label: edge.type,
    };
    this.mapper.mapEdgeToEntity(mapperEdge, this.nodePositionMap);

    logger.debug(`Synced edge ${edge.source}->${edge.target} to Vircadia`);
  }


  broadcastLocalSelection(nodeIds: string[]): void {
    if (!this.isActive) return;

    try {
      this.collab.selectNodes(nodeIds);
      logger.debug(`Broadcasted selection of ${nodeIds.length} nodes`);
    } catch (error) {
      logger.error('Failed to broadcast selection:', error);
    }
  }


  async addAnnotation(
    nodeId: string,
    text: string,
    position: { x: number; y: number; z: number }
  ): Promise<string> {
    if (!this.isActive) {
      throw new Error('Bridge not active');
    }

    try {
      const threePosition = new THREE.Vector3(position.x, position.y, position.z);
      await this.collab.createAnnotation(nodeId, text, threePosition);
      // createAnnotation generates the ID internally; retrieve the latest for this node
      const nodeAnnotations = this.collab.getNodeAnnotations(nodeId);
      const annotationId = nodeAnnotations.length > 0
        ? nodeAnnotations[nodeAnnotations.length - 1].id
        : '';

      logger.info(`Added annotation ${annotationId} to node ${nodeId}`);
      return annotationId;
    } catch (error) {
      logger.error('Failed to add annotation:', error);
      throw error;
    }
  }


  async removeAnnotation(annotationId: string): Promise<void> {
    if (!this.isActive) return;

    try {
      await this.collab.deleteAnnotation(annotationId);
      logger.info(`Removed annotation ${annotationId}`);
    } catch (error) {
      logger.error('Failed to remove annotation:', error);
    }
  }


  broadcastFilterState(filterState: {
    searchQuery?: string;
    categoryFilter?: string[];
    timeRange?: { start: number; end: number };
    customFilters?: Record<string, any>;
  }): void {
    if (!this.isActive) return;

    try {
      const collabFilterState: FilterState = {
        searchQuery: filterState.searchQuery,
        categoryFilter: filterState.categoryFilter,
        timeRange: filterState.timeRange,
        customFilters: filterState.customFilters,
      };
      this.collab.updateFilterState(collabFilterState);
      logger.debug('Broadcasted filter state');
    } catch (error) {
      logger.error('Failed to broadcast filter state:', error);
    }
  }


  private handleRemoteSelection(event: {
    agentId: string;
    username: string;
    nodeIds: string[];
  }): void {
    logger.debug(`Remote user ${event.username} selected ${event.nodeIds.length} nodes`);

    if (this.remoteSelectionCallback) {
      this.remoteSelectionCallback({
        userId: event.agentId,
        username: event.username,
        nodeIds: event.nodeIds
      });
    }
  }


  private handleRemoteAnnotation(annotation: {
    id: string;
    agentId: string;
    username: string;
    nodeId: string;
    text: string;
    position: { x: number; y: number; z: number };
  }): void {
    logger.info(`Remote annotation added by ${annotation.username} on node ${annotation.nodeId}`);

    if (this.annotationCallback) {
      this.annotationCallback({
        id: annotation.id,
        userId: annotation.agentId,
        username: annotation.username,
        nodeId: annotation.nodeId,
        text: annotation.text,
        position: annotation.position
      });
    }
  }


  private handleAnnotationRemoved(annotationId: string): void {
    logger.debug(`Annotation ${annotationId} removed`);
  }


  private handleFilterStateChanged(event: {
    agentId: string;
    username: string;
    filterState: any;
  }): void {
    logger.debug(`Remote user ${event.username} changed filter state`);

  }


  onLocalSelection(callback: (nodeIds: string[]) => void): void {
    this.localSelectionCallback = callback;
  }


  onRemoteSelection(callback: (event: UserSelectionEvent) => void): void {
    this.remoteSelectionCallback = callback;
  }


  onAnnotation(callback: (event: AnnotationEvent) => void): void {
    this.annotationCallback = callback;
  }


  getActiveUsers(): Array<{
    userId: string;
    username: string;
    selectedNodes: string[];
  }> {
    if (!this.isActive) return [];

    return this.collab.getActiveSelections().map(selection => ({
      userId: selection.agentId,
      username: selection.username,
      selectedNodes: selection.nodeIds
    }));
  }


  getAnnotations(): AnnotationEvent[] {
    if (!this.isActive) return [];

    return this.collab.getAnnotations().map(ann => ({
      id: ann.id,
      userId: ann.agentId,
      username: ann.username,
      nodeId: ann.nodeId,
      text: ann.text,
      position: ann.position
    }));
  }


  dispose(): void {
    this.isActive = false;
    this.nodeEntityMap.clear();
    this.nodePositionMap.clear();
    this.localSelectionCallback = undefined;
    this.remoteSelectionCallback = undefined;
    this.annotationCallback = undefined;
    if (this.entitySync) {
      this.entitySync.dispose();
      this.entitySync = null;
    }
    this.collab.dispose();
    logger.info('GraphVircadiaBridge disposed');
  }
}
