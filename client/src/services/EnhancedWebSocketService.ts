/**
 * Enhanced WebSocket Service with Real-Time Events
 * Extends the existing WebSocketService with typed message system and comprehensive event handling
 */

import { createLogger } from '../utils/loggerConfig';
import { createErrorMetadata } from '../utils/loggerConfig';
import { debugState } from '../utils/clientDebugState';
import { webSocketService, WebSocketService } from './WebSocketService';
import {
  WebSocketMessage,
  WebSocketEventHandlers,
  WebSocketConfig,
  WebSocketConnectionState,
  WebSocketError,
  WebSocketStatistics,
  Subscription,
  SubscriptionFilters,
  MessageHandler,
  // Specific message types
  WorkspaceUpdateMessage,
  WorkspaceDeletedMessage,
  WorkspaceCollaborationMessage,
  AnalysisProgressMessage,
  AnalysisCompleteMessage,
  AnalysisErrorMessage,
  OptimizationUpdateMessage,
  OptimizationResultMessage,
  ExportProgressMessage,
  ExportReadyMessage,
  ShareCreatedMessage,
  ShareAccessMessage,
  ConnectionStatusMessage,
  SystemNotificationMessage,
  UserActivityMessage,
  PerformanceMetricsMessage,
  ServerHealthMessage
} from '../types/websocketTypes';

const logger = createLogger('EnhancedWebSocketService');

class EnhancedWebSocketService {
  private static instance: EnhancedWebSocketService;

  // Enhanced state management
  private subscriptions: Map<string, Subscription> = new Map();
  private subscriptionCounter: number = 0;
  private statistics: WebSocketStatistics;
  private config: WebSocketConfig;
  private enhancedConnectionState: WebSocketConnectionState;
  private messageBuffer: Map<string, WebSocketMessage[]> = new Map();
  private reconnectionStrategy: 'exponential' | 'linear' | 'fibonacci' = 'exponential';

  private constructor() {
    this.initializeStatistics();
    this.initializeConfig();
    this.initializeConnectionState();
    this.setupLegacyEventBridge();
  }

  public static getInstance(): EnhancedWebSocketService {
    if (!EnhancedWebSocketService.instance) {
      EnhancedWebSocketService.instance = new EnhancedWebSocketService();
    }
    return EnhancedWebSocketService.instance;
  }

  private initializeStatistics(): void {
    this.statistics = {
      messagesReceived: 0,
      messagesSent: 0,
      bytesReceived: 0,
      bytesSent: 0,
      connectionTime: 0,
      reconnections: 0,
      averageLatency: 0,
      messagesByType: {},
      errors: 0,
      lastActivity: Date.now()
    };
  }

  private initializeConfig(): void {
    this.config = {
      reconnect: {
        maxAttempts: 15,
        baseDelay: 1000,
        maxDelay: 30000,
        backoffFactor: 1.5
      },
      heartbeat: {
        interval: 30000,
        timeout: 10000
      },
      compression: true,
      binaryProtocol: true
    };
  }

  private initializeConnectionState(): void {
    this.enhancedConnectionState = {
      status: 'disconnected',
      reconnectAttempts: 0,
      serverFeatures: [],
      latency: undefined
    };
  }

  private setupLegacyEventBridge(): void {
    // Bridge legacy message handler to new typed system
    webSocketService.onMessage((message) => {
      if (this.isTypedMessage(message)) {
        this.handleTypedMessage(message as WebSocketMessage);
      }
    });

    // Bridge connection status changes
    webSocketService.onConnectionStatusChange((connected) => {
      this.enhancedConnectionState.status = connected ? 'connected' : 'disconnected';
      if (connected) {
        this.enhancedConnectionState.lastConnected = Date.now();
        this.enhancedConnectionState.reconnectAttempts = 0;
      }
    });
  }

  private isTypedMessage(message: any): boolean {
    return message &&
           typeof message.timestamp === 'number' &&
           typeof message.type === 'string';
  }

  // Core subscription methods
  public subscribe<K extends keyof WebSocketEventHandlers>(
    type: K,
    handler: WebSocketEventHandlers[K],
    options?: {
      once?: boolean;
      filters?: SubscriptionFilters;
      buffer?: boolean;
    }
  ): () => void {
    const id = `${type}_${++this.subscriptionCounter}`;

    const subscription: Subscription = {
      id,
      type,
      handler: handler as MessageHandler,
      options: {
        once: options?.once,
        filter: options?.filters ? this.createFilterFunction(options.filters) : undefined
      }
    };

    this.subscriptions.set(id, subscription);

    // Handle buffered messages if requested
    if (options?.buffer && this.messageBuffer.has(type)) {
      const bufferedMessages = this.messageBuffer.get(type) || [];
      bufferedMessages.forEach(message => {
        if (!subscription.options?.filter || subscription.options.filter(message)) {
          handler(message as any);
        }
      });
    }

    if (debugState.isEnabled()) {
      logger.info(`Subscribed to ${type} events with ID: ${id}`, { filters: options?.filters });
    }

    return () => {
      this.subscriptions.delete(id);
      if (debugState.isEnabled()) {
        logger.info(`Unsubscribed from ${type} events with ID: ${id}`);
      }
    };
  }

  public subscribeOnce<K extends keyof WebSocketEventHandlers>(
    type: K,
    handler: WebSocketEventHandlers[K],
    filters?: SubscriptionFilters
  ): () => void {
    return this.subscribe(type, handler, { once: true, filters });
  }

  public unsubscribeAll(type?: keyof WebSocketEventHandlers): void {
    if (type) {
      const keysToDelete = Array.from(this.subscriptions.keys()).filter(key =>
        this.subscriptions.get(key)?.type === type
      );
      keysToDelete.forEach(key => this.subscriptions.delete(key));
      logger.info(`Unsubscribed from all ${type} events`);
    } else {
      this.subscriptions.clear();
      logger.info('Unsubscribed from all events');
    }
  }

  private handleTypedMessage(message: WebSocketMessage): void {
    if (debugState.isDataDebugEnabled()) {
      logger.debug(`Received typed message: ${message.type}`, { data: message.data });
    }

    // Update statistics
    this.statistics.messagesReceived++;
    this.statistics.messagesByType[message.type] = (this.statistics.messagesByType[message.type] || 0) + 1;
    this.statistics.lastActivity = Date.now();

    // Handle special system messages
    this.handleSystemMessages(message);

    // Buffer message for late subscribers if needed
    this.bufferMessage(message);

    // Notify typed subscribers
    this.notifySubscribers(message);
  }

  private handleSystemMessages(message: WebSocketMessage): void {
    switch (message.type) {
      case 'connection_status':
        this.handleConnectionStatusMessage(message as ConnectionStatusMessage);
        break;
      case 'server_health':
        this.handleServerHealthMessage(message as ServerHealthMessage);
        break;
      case 'performance_metrics':
        this.updatePerformanceMetrics(message as PerformanceMetricsMessage);
        break;
    }
  }

  private handleConnectionStatusMessage(message: ConnectionStatusMessage): void {
    const { status, serverLoad, latency, features } = message.data;

    this.enhancedConnectionState.serverFeatures = features;
    this.enhancedConnectionState.latency = latency;

    if (debugState.isEnabled()) {
      logger.info('Connection status updated:', { status, serverLoad, latency, features });
    }
  }

  private handleServerHealthMessage(message: ServerHealthMessage): void {
    const { status, services, load } = message.data;

    if (status === 'unhealthy') {
      logger.warn('Server health degraded:', { services, load });
    }
  }

  private updatePerformanceMetrics(message: PerformanceMetricsMessage): void {
    const { metrics } = message.data;

    // Update latency statistics
    if (metrics.renderTime) {
      const currentLatency = this.statistics.averageLatency;
      this.statistics.averageLatency = currentLatency === 0
        ? metrics.renderTime
        : (currentLatency + metrics.renderTime) / 2;
    }
  }

  private bufferMessage(message: WebSocketMessage): void {
    const bufferableTypes = ['workspace_update', 'analysis_complete', 'optimization_result'];

    if (bufferableTypes.includes(message.type)) {
      if (!this.messageBuffer.has(message.type)) {
        this.messageBuffer.set(message.type, []);
      }

      const buffer = this.messageBuffer.get(message.type)!;
      buffer.push(message);

      // Keep only last 10 messages
      if (buffer.length > 10) {
        buffer.shift();
      }
    }
  }

  private notifySubscribers(message: WebSocketMessage): void {
    const relevantSubscriptions = Array.from(this.subscriptions.values()).filter(sub => {
      if (sub.type !== message.type) return false;
      if (sub.options?.filter && !sub.options.filter(message)) return false;
      return true;
    });

    relevantSubscriptions.forEach(subscription => {
      try {
        subscription.handler(message);

        // Remove one-time subscriptions
        if (subscription.options?.once) {
          this.subscriptions.delete(subscription.id);
        }
      } catch (error) {
        logger.error(`Error in typed message handler for ${message.type}:`, createErrorMetadata(error));
        this.statistics.errors++;
      }
    });
  }

  private createFilterFunction(filters: SubscriptionFilters): (message: WebSocketMessage) => boolean {
    return (message: WebSocketMessage) => {
      const data = message.data as any;

      if (filters.workspaceId && data.workspaceId !== filters.workspaceId) return false;
      if (filters.userId && data.userId !== filters.userId) return false;
      if (filters.graphId && data.graphId !== filters.graphId) return false;
      if (filters.analysisId && data.analysisId !== filters.analysisId) return false;
      if (filters.optimizationId && data.optimizationId !== filters.optimizationId) return false;
      if (filters.exportId && data.exportId !== filters.exportId) return false;

      return true;
    };
  }

  // Message sending methods
  public sendTypedMessage(message: Omit<WebSocketMessage, 'timestamp' | 'clientId'>): void {
    const fullMessage: WebSocketMessage = {
      ...message,
      timestamp: Date.now(),
      clientId: this.generateClientId()
    } as WebSocketMessage;

    // Use legacy service for actual sending
    webSocketService.sendMessage(fullMessage.type, fullMessage.data);

    // Update statistics
    this.statistics.messagesSent++;
    this.statistics.messagesByType[message.type] = (this.statistics.messagesByType[message.type] || 0) + 1;

    if (debugState.isDataDebugEnabled()) {
      logger.debug(`Sent typed message: ${message.type}`, fullMessage);
    }
  }

  // Workspace-specific methods
  public broadcastWorkspaceUpdate(workspaceId: string, changes: any, operation: 'create' | 'update' | 'delete' | 'favorite' | 'archive'): void {
    const message: Omit<WorkspaceUpdateMessage, 'timestamp' | 'clientId'> = {
      type: 'workspace_update',
      data: {
        workspaceId,
        changes,
        operation,
        userId: this.getCurrentUserId()
      }
    };

    this.sendTypedMessage(message);
  }

  public notifyWorkspaceDeleted(workspaceId: string): void {
    const message: Omit<WorkspaceDeletedMessage, 'timestamp' | 'clientId'> = {
      type: 'workspace_deleted',
      data: {
        workspaceId,
        userId: this.getCurrentUserId()
      }
    };

    this.sendTypedMessage(message);
  }

  // Analysis-specific methods
  public reportAnalysisProgress(analysisId: string, progress: number, stage: string, currentOperation: string, metrics?: any): void {
    const message: Omit<AnalysisProgressMessage, 'timestamp' | 'clientId'> = {
      type: 'analysis_progress',
      data: {
        analysisId,
        progress,
        stage,
        currentOperation,
        metrics
      }
    };

    this.sendTypedMessage(message);
  }

  public reportAnalysisComplete(analysisId: string, results: any): void {
    const message: Omit<AnalysisCompleteMessage, 'timestamp' | 'clientId'> = {
      type: 'analysis_complete',
      data: {
        analysisId,
        results,
        success: true
      }
    };

    this.sendTypedMessage(message);
  }

  // Optimization-specific methods
  public reportOptimizationUpdate(optimizationId: string, progress: number, algorithm: string, metrics: any): void {
    const message: Omit<OptimizationUpdateMessage, 'timestamp' | 'clientId'> = {
      type: 'optimization_update',
      data: {
        optimizationId,
        progress,
        algorithm,
        currentIteration: 0,
        totalIterations: 100,
        metrics
      }
    };

    this.sendTypedMessage(message);
  }

  public reportOptimizationResult(optimizationId: string, results: any): void {
    const message: Omit<OptimizationResultMessage, 'timestamp' | 'clientId'> = {
      type: 'optimization_result',
      data: {
        optimizationId,
        ...results,
        success: true
      }
    };

    this.sendTypedMessage(message);
  }

  // Export-specific methods
  public reportExportProgress(exportId: string, format: string, progress: number, stage: any): void {
    const message: Omit<ExportProgressMessage, 'timestamp' | 'clientId'> = {
      type: 'export_progress',
      data: {
        exportId,
        format,
        progress,
        stage
      }
    };

    this.sendTypedMessage(message);
  }

  public reportExportReady(exportId: string, format: string, downloadUrl: string, size: number, metadata: any): void {
    const message: Omit<ExportReadyMessage, 'timestamp' | 'clientId'> = {
      type: 'export_ready',
      data: {
        exportId,
        format,
        downloadUrl,
        size,
        expiresAt: new Date(Date.now() + 24 * 60 * 60 * 1000).toISOString(), // 24 hours
        metadata
      }
    };

    this.sendTypedMessage(message);
  }

  public reportShareCreated(shareId: string, shareUrl: string, permissions: string[], expiresAt?: string): void {
    const message: Omit<ShareCreatedMessage, 'timestamp' | 'clientId'> = {
      type: 'share_created',
      data: {
        shareId,
        shareUrl,
        expiresAt,
        passwordProtected: false,
        permissions
      }
    };

    this.sendTypedMessage(message);
  }

  // Notification methods
  public showSystemNotification(level: 'info' | 'warning' | 'error', title: string, message: string, persistent?: boolean): void {
    const notification: Omit<SystemNotificationMessage, 'timestamp' | 'clientId'> = {
      type: 'system_notification',
      data: {
        level,
        title,
        message,
        persistent
      }
    };

    this.sendTypedMessage(notification);
  }

  // Utility methods
  public getStatistics(): WebSocketStatistics {
    return { ...this.statistics };
  }

  public getConnectionState(): WebSocketConnectionState {
    return { ...this.enhancedConnectionState };
  }

  public updateConfig(config: Partial<WebSocketConfig>): void {
    this.config = { ...this.config, ...config };
    logger.info('Enhanced WebSocket configuration updated', config);
  }

  public isConnected(): boolean {
    return webSocketService.isReady();
  }

  public getActiveSubscriptions(): string[] {
    return Array.from(this.subscriptions.values()).map(sub => `${sub.type}:${sub.id}`);
  }

  public clearMessageBuffer(type?: keyof WebSocketEventHandlers): void {
    if (type) {
      this.messageBuffer.delete(type);
    } else {
      this.messageBuffer.clear();
    }
  }

  private generateClientId(): string {
    return `client_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  private getCurrentUserId(): string {
    // This would typically come from an auth service
    return 'current_user_id';
  }

  // Legacy compatibility methods
  public connect(): Promise<void> {
    return webSocketService.connect();
  }

  public close(): void {
    webSocketService.close();
  }

  public forceReconnect(): void {
    webSocketService.forceReconnect();
  }
}

// Create and export singleton instance
export const enhancedWebSocketService = EnhancedWebSocketService.getInstance();
export default EnhancedWebSocketService;