/**
 * BotsVircadiaBridge - Synchronize agent swarm with Vircadia multi-user entities
 *
 * This bridge connects the VisionFlow agent swarm system with Vircadia's multi-user
 * entity synchronization, allowing multiple users to see the same agent positions
 * and states in real-time.
 */

import { ClientCore } from '../vircadia/VircadiaClientCore';
import { EntitySyncManager } from '../vircadia/EntitySyncManager';
import { AvatarManager } from '../vircadia/AvatarManager';
import type { BotsAgent, BotsEdge } from '../../features/bots/types/BotsTypes';
import { createLogger } from '../../utils/loggerConfig';

const logger = createLogger('BotsVircadiaBridge');

export interface BridgeConfig {
  syncPositions: boolean;
  syncMetadata: boolean;
  syncEdges: boolean;
  updateInterval: number;
  enableAvatars: boolean;
}

export interface VircadiaEntity {
  id: string;
  type: string;
  position: { x: number; y: number; z: number };
  rotation?: { x: number; y: number; z: number; w: number };
  scale?: { x: number; y: number; z: number };
  metadata?: Record<string, any>;
}

export class BotsVircadiaBridge {
  private syncInterval: ReturnType<typeof setInterval> | null = null;
  private agentEntityMap = new Map<string, string>(); // agentId -> entityId
  private lastSyncedAgents = new Map<string, BotsAgent>();
  private isActive = false;

  private defaultConfig: BridgeConfig = {
    syncPositions: true,
    syncMetadata: true,
    syncEdges: true,
    updateInterval: 100, // 100ms for smooth updates
    enableAvatars: true
  };

  constructor(
    private client: ClientCore,
    private entitySync: EntitySyncManager,
    private avatars: AvatarManager | null,
    config?: Partial<BridgeConfig>
  ) {
    this.defaultConfig = { ...this.defaultConfig, ...config };
  }

  /**
   * Initialize the bridge and start synchronization
   */
  async initialize(): Promise<void> {
    logger.info('Initializing BotsVircadiaBridge...');

    if (!this.client.Utilities.Connection.getConnectionInfo().isConnected) {
      throw new Error('Vircadia client must be connected before initializing bridge');
    }

    // Listen for entity updates from Vircadia (other users' changes)
    this.entitySync.on('entity-updated', this.handleVircadiaEntityUpdate.bind(this));
    this.entitySync.on('entity-deleted', this.handleVircadiaEntityDeleted.bind(this));

    this.isActive = true;
    logger.info('BotsVircadiaBridge initialized successfully');
  }

  /**
   * Synchronize agents from VisionFlow to Vircadia
   */
  syncAgentsToVircadia(agents: BotsAgent[], edges: BotsEdge[]): void {
    if (!this.isActive) return;

    try {
      // Sync each agent as a Vircadia entity
      agents.forEach(agent => {
        this.syncAgentToEntity(agent);
      });

      // Remove entities for agents that no longer exist
      this.cleanupStaleEntities(agents);

      // Sync edges if enabled
      if (this.defaultConfig.syncEdges) {
        this.syncEdgesToVircadia(edges);
      }

      logger.debug(`Synced ${agents.length} agents and ${edges.length} edges to Vircadia`);
    } catch (error) {
      logger.error('Failed to sync agents to Vircadia:', error);
    }
  }

  /**
   * Sync a single agent to Vircadia entity
   */
  private syncAgentToEntity(agent: BotsAgent): void {
    // Check if agent data has changed
    const lastSynced = this.lastSyncedAgents.get(agent.id);
    if (lastSynced && this.isAgentUnchanged(agent, lastSynced)) {
      return; // Skip if unchanged
    }

    const entityId = this.agentEntityMap.get(agent.id) || `agent-${agent.id}`;

    // Convert agent position to Vircadia world coordinates
    const position = this.convertAgentPosition(agent.position);

    // Create entity data
    const entityData: VircadiaEntity = {
      id: entityId,
      type: 'agent-avatar',
      position,
      scale: { x: 1, y: 1, z: 1 },
      metadata: this.defaultConfig.syncMetadata ? {
        agentName: agent.name,
        agentType: agent.type,
        health: agent.health,
        status: agent.status,
        capabilities: agent.capabilities,
        currentTask: agent.currentTask,
        tokenUsage: agent.tokenUsage,
        isActive: agent.status === 'active',
        color: this.getAgentColor(agent)
      } : undefined
    };

    // Update or create entity
    this.entitySync.updateEntity(entityData);

    // Store mapping and last synced state
    this.agentEntityMap.set(agent.id, entityId);
    this.lastSyncedAgents.set(agent.id, { ...agent });
  }

  /**
   * Convert agent position to Vircadia world coordinates
   */
  private convertAgentPosition(position: { x: number; y: number; z: number }): { x: number; y: number; z: number } {
    // VisionFlow uses different coordinate system - adjust as needed
    // Y is up in Vircadia, might be Z in VisionFlow
    return {
      x: position.x * 10, // Scale up for better visibility
      y: position.z * 10, // Swap Y/Z if needed
      z: position.y * 10
    };
  }

  /**
   * Get agent color for visualization
   */
  private getAgentColor(agent: BotsAgent): string {
    const typeColors: Record<string, string> = {
      'researcher': '#4A90E2',
      'coder': '#50E3C2',
      'analyst': '#F5A623',
      'optimizer': '#D0021B',
      'coordinator': '#7ED321'
    };

    return typeColors[agent.type] || '#9013FE';
  }

  /**
   * Check if agent data has changed
   */
  private isAgentUnchanged(agent: BotsAgent, lastSynced: BotsAgent): boolean {
    return (
      agent.position.x === lastSynced.position.x &&
      agent.position.y === lastSynced.position.y &&
      agent.position.z === lastSynced.position.z &&
      agent.health === lastSynced.health &&
      agent.status === lastSynced.status
    );
  }

  /**
   * Remove entities for agents that no longer exist
   */
  private cleanupStaleEntities(currentAgents: BotsAgent[]): void {
    const currentAgentIds = new Set(currentAgents.map(a => a.id));
    const staleAgentIds: string[] = [];

    this.agentEntityMap.forEach((entityId, agentId) => {
      if (!currentAgentIds.has(agentId)) {
        staleAgentIds.push(agentId);
        this.entitySync.deleteEntity(entityId);
      }
    });

    staleAgentIds.forEach(agentId => {
      this.agentEntityMap.delete(agentId);
      this.lastSyncedAgents.delete(agentId);
    });

    if (staleAgentIds.length > 0) {
      logger.debug(`Cleaned up ${staleAgentIds.length} stale agent entities`);
    }
  }

  /**
   * Sync edges (communication links) to Vircadia
   */
  private syncEdgesToVircadia(edges: BotsEdge[]): void {
    // Create line entities for edges
    edges.forEach(edge => {
      const entityId = `edge-${edge.source}-${edge.target}`;
      const sourceEntity = this.agentEntityMap.get(edge.source);
      const targetEntity = this.agentEntityMap.get(edge.target);

      if (sourceEntity && targetEntity) {
        this.entitySync.updateEntity({
          id: entityId,
          type: 'communication-link',
          position: { x: 0, y: 0, z: 0 }, // Lines don't need position
          metadata: {
            source: sourceEntity,
            target: targetEntity,
            type: edge.type,
            active: true
          }
        });
      }
    });
  }

  /**
   * Handle entity updates from Vircadia (other users' changes)
   */
  private handleVircadiaEntityUpdate(entity: VircadiaEntity): void {
    // Check if this is an agent entity we should respond to
    if (entity.type !== 'agent-avatar') return;

    // Find corresponding agent
    const agentId = Array.from(this.agentEntityMap.entries())
      .find(([_, entityId]) => entityId === entity.id)?.[0];

    if (!agentId) {
      logger.debug('Received update for unknown agent entity:', entity.id);
      return;
    }

    // In a full implementation, you might update the local agent state
    // based on Vircadia changes (bidirectional sync)
    logger.debug(`Entity ${entity.id} updated by another user`);
  }

  /**
   * Handle entity deletion from Vircadia
   */
  private handleVircadiaEntityDeleted(entityId: string): void {
    // Find and remove from our tracking
    const agentId = Array.from(this.agentEntityMap.entries())
      .find(([_, eid]) => eid === entityId)?.[0];

    if (agentId) {
      this.agentEntityMap.delete(agentId);
      this.lastSyncedAgents.delete(agentId);
      logger.debug(`Agent entity ${entityId} deleted`);
    }
  }

  /**
   * Enable automatic periodic synchronization
   */
  startAutoSync(
    getAgentsCallback: () => { agents: BotsAgent[]; edges: BotsEdge[] }
  ): void {
    if (this.syncInterval) {
      this.stopAutoSync();
    }

    this.syncInterval = setInterval(() => {
      const { agents, edges } = getAgentsCallback();
      this.syncAgentsToVircadia(agents, edges);
    }, this.defaultConfig.updateInterval);

    logger.info(`Auto-sync started with ${this.defaultConfig.updateInterval}ms interval`);
  }

  /**
   * Stop automatic synchronization
   */
  stopAutoSync(): void {
    if (this.syncInterval) {
      clearInterval(this.syncInterval);
      this.syncInterval = null;
      logger.info('Auto-sync stopped');
    }
  }

  /**
   * Cleanup and disconnect
   */
  dispose(): void {
    this.stopAutoSync();
    this.isActive = false;
    this.agentEntityMap.clear();
    this.lastSyncedAgents.clear();
    logger.info('BotsVircadiaBridge disposed');
  }
}
