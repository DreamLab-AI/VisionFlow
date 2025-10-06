/**
 * GraphVircadiaBridge - Synchronize knowledge graph with Vircadia collaborative features
 *
 * This bridge enables multi-user graph exploration by synchronizing node selections,
 * annotations, and filter states across all connected Vircadia clients.
 */

import * as BABYLON from '@babylonjs/core';
import { ClientCore } from '../vircadia/VircadiaClientCore';
import { CollaborativeGraphSync } from '../vircadia/CollaborativeGraphSync';
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
  private nodeEntityMap = new Map<string, string>(); // nodeId -> entityId
  private localSelectionCallback?: (nodeIds: string[]) => void;
  private remoteSelectionCallback?: (event: UserSelectionEvent) => void;
  private annotationCallback?: (event: AnnotationEvent) => void;
  private isActive = false;

  constructor(
    private scene: BABYLON.Scene,
    private client: ClientCore,
    private collab: CollaborativeGraphSync
  ) {}

  /**
   * Initialize the bridge
   */
  async initialize(): Promise<void> {
    logger.info('Initializing GraphVircadiaBridge...');

    if (!this.client.Utilities.Connection.getConnectionInfo().isConnected) {
      throw new Error('Vircadia client must be connected before initializing bridge');
    }

    await this.collab.initialize();

    // Listen for remote user selections
    this.collab.on('user-selection', this.handleRemoteSelection.bind(this));

    // Listen for annotations from other users
    this.collab.on('annotation-added', this.handleRemoteAnnotation.bind(this));
    this.collab.on('annotation-removed', this.handleAnnotationRemoved.bind(this));

    // Listen for filter state changes
    this.collab.on('filter-state-changed', this.handleFilterStateChanged.bind(this));

    this.isActive = true;
    logger.info('GraphVircadiaBridge initialized successfully');
  }

  /**
   * Synchronize graph nodes to Vircadia
   */
  syncGraphToVircadia(nodes: GraphNode[], edges: GraphEdge[]): void {
    if (!this.isActive) return;

    try {
      // Sync nodes as collaborative entities
      nodes.forEach(node => {
        this.syncNodeToEntity(node);
      });

      // Sync edges
      edges.forEach(edge => {
        this.syncEdgeToEntity(edge);
      });

      logger.debug(`Synced ${nodes.length} nodes and ${edges.length} edges to Vircadia`);
    } catch (error) {
      logger.error('Failed to sync graph to Vircadia:', error);
    }
  }

  /**
   * Sync a single node to Vircadia entity
   */
  private syncNodeToEntity(node: GraphNode): void {
    const entityId = `graph-node-${node.id}`;

    // Store mapping
    this.nodeEntityMap.set(node.id, entityId);

    // Nodes are managed by CollaborativeGraphSync
    // Just track the mapping for selection synchronization
  }

  /**
   * Sync edge to Vircadia
   */
  private syncEdgeToEntity(edge: GraphEdge): void {
    const sourceEntityId = this.nodeEntityMap.get(edge.source);
    const targetEntityId = this.nodeEntityMap.get(edge.target);

    if (sourceEntityId && targetEntityId) {
      // Edges are visualized in the graph, we just track them
      // for collaborative features
    }
  }

  /**
   * Broadcast local user's node selection to all connected users
   */
  broadcastLocalSelection(nodeIds: string[]): void {
    if (!this.isActive) return;

    try {
      this.collab.setLocalSelection(nodeIds);
      logger.debug(`Broadcasted selection of ${nodeIds.length} nodes`);
    } catch (error) {
      logger.error('Failed to broadcast selection:', error);
    }
  }

  /**
   * Add annotation to a node (visible to all users)
   */
  async addAnnotation(
    nodeId: string,
    text: string,
    position: { x: number; y: number; z: number }
  ): Promise<string> {
    if (!this.isActive) {
      throw new Error('Bridge not active');
    }

    try {
      const annotationId = await this.collab.addAnnotation({
        nodeId,
        text,
        position
      });

      logger.info(`Added annotation ${annotationId} to node ${nodeId}`);
      return annotationId;
    } catch (error) {
      logger.error('Failed to add annotation:', error);
      throw error;
    }
  }

  /**
   * Remove annotation
   */
  async removeAnnotation(annotationId: string): Promise<void> {
    if (!this.isActive) return;

    try {
      await this.collab.removeAnnotation(annotationId);
      logger.info(`Removed annotation ${annotationId}`);
    } catch (error) {
      logger.error('Failed to remove annotation:', error);
    }
  }

  /**
   * Broadcast filter state to all users
   */
  broadcastFilterState(filterState: {
    searchQuery?: string;
    categoryFilter?: string[];
    timeRange?: { start: number; end: number };
    customFilters?: Record<string, any>;
  }): void {
    if (!this.isActive) return;

    try {
      this.collab.setLocalFilterState(filterState);
      logger.debug('Broadcasted filter state');
    } catch (error) {
      logger.error('Failed to broadcast filter state:', error);
    }
  }

  /**
   * Handle remote user selection event
   */
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

  /**
   * Handle remote annotation added
   */
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

  /**
   * Handle annotation removed
   */
  private handleAnnotationRemoved(annotationId: string): void {
    logger.debug(`Annotation ${annotationId} removed`);
  }

  /**
   * Handle filter state changed by another user
   */
  private handleFilterStateChanged(event: {
    agentId: string;
    username: string;
    filterState: any;
  }): void {
    logger.debug(`Remote user ${event.username} changed filter state`);
    // Optionally sync filter state to local UI
  }

  /**
   * Register callback for local selection changes
   */
  onLocalSelection(callback: (nodeIds: string[]) => void): void {
    this.localSelectionCallback = callback;
  }

  /**
   * Register callback for remote selection changes
   */
  onRemoteSelection(callback: (event: UserSelectionEvent) => void): void {
    this.remoteSelectionCallback = callback;
  }

  /**
   * Register callback for annotations
   */
  onAnnotation(callback: (event: AnnotationEvent) => void): void {
    this.annotationCallback = callback;
  }

  /**
   * Get all active users and their selections
   */
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

  /**
   * Get all annotations
   */
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

  /**
   * Cleanup and disconnect
   */
  dispose(): void {
    this.isActive = false;
    this.nodeEntityMap.clear();
    this.localSelectionCallback = undefined;
    this.remoteSelectionCallback = undefined;
    this.annotationCallback = undefined;
    this.collab.dispose();
    logger.info('GraphVircadiaBridge disposed');
  }
}
