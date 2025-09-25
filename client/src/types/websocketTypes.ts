/**
 * WebSocket Message Types for Real-Time Updates
 * Comprehensive typing for all real-time features
 */

// Base WebSocket message structure
export interface BaseWebSocketMessage {
  type: string;
  timestamp: number;
  clientId?: string;
  sessionId?: string;
}

// Workspace-related messages
export interface WorkspaceUpdateMessage extends BaseWebSocketMessage {
  type: 'workspace_update';
  data: {
    workspaceId: string;
    changes: Partial<{
      name: string;
      description: string;
      status: 'active' | 'archived';
      memberCount: number;
      favorite: boolean;
      lastAccessed: string;
      settings: Record<string, any>;
    }>;
    userId?: string;
    operation: 'create' | 'update' | 'delete' | 'favorite' | 'archive';
  };
}

export interface WorkspaceDeletedMessage extends BaseWebSocketMessage {
  type: 'workspace_deleted';
  data: {
    workspaceId: string;
    userId?: string;
  };
}

export interface WorkspaceCollaborationMessage extends BaseWebSocketMessage {
  type: 'workspace_collaboration';
  data: {
    workspaceId: string;
    action: 'user_joined' | 'user_left' | 'permission_changed';
    userId: string;
    userName?: string;
    permissions?: string[];
  };
}

// Analysis-related messages
export interface AnalysisProgressMessage extends BaseWebSocketMessage {
  type: 'analysis_progress';
  data: {
    analysisId: string;
    graphId?: string;
    progress: number; // 0-100
    stage: string;
    estimatedTimeRemaining?: number;
    currentOperation: string;
    metrics?: {
      nodesProcessed: number;
      edgesProcessed: number;
      clustersFound?: number;
      similarityScore?: number;
    };
  };
}

export interface AnalysisCompleteMessage extends BaseWebSocketMessage {
  type: 'analysis_complete';
  data: {
    analysisId: string;
    graphId?: string;
    results: {
      similarity: {
        overall: number;
        structural: number;
        semantic: number;
      };
      matches: number;
      differences: number;
      clusters: number;
      centrality: {
        betweenness: number;
        closeness: number;
        eigenvector: number;
      };
      processing_time: number;
    };
    success: boolean;
    error?: string;
  };
}

export interface AnalysisErrorMessage extends BaseWebSocketMessage {
  type: 'analysis_error';
  data: {
    analysisId: string;
    graphId?: string;
    error: string;
    stage: string;
    retryable: boolean;
  };
}

// Optimization-related messages
export interface OptimizationUpdateMessage extends BaseWebSocketMessage {
  type: 'optimization_update';
  data: {
    optimizationId: string;
    graphId?: string;
    progress: number; // 0-100
    algorithm: string;
    currentIteration: number;
    totalIterations: number;
    metrics: {
      performanceGain: number;
      confidence: number;
      energyLevel?: number;
      convergence?: number;
    };
    recommendations?: Array<{
      type: string;
      priority: 'low' | 'medium' | 'high';
      description: string;
    }>;
  };
}

export interface OptimizationResultMessage extends BaseWebSocketMessage {
  type: 'optimization_result';
  data: {
    optimizationId: string;
    graphId?: string;
    algorithm: string;
    confidence: number;
    performanceGain: number;
    clusters: number;
    recommendations: Array<{
      type: string;
      priority: 'low' | 'medium' | 'high';
      description: string;
    }>;
    layoutChanges?: {
      nodesRepositioned: number;
      clustersFormed: number;
      edgesOptimized: number;
    };
    success: boolean;
    error?: string;
  };
}

// Export-related messages
export interface ExportProgressMessage extends BaseWebSocketMessage {
  type: 'export_progress';
  data: {
    exportId: string;
    graphId?: string;
    format: string;
    progress: number; // 0-100
    stage: 'preparing' | 'processing' | 'rendering' | 'finalizing' | 'uploading';
    size?: number;
    estimatedTimeRemaining?: number;
  };
}

export interface ExportReadyMessage extends BaseWebSocketMessage {
  type: 'export_ready';
  data: {
    exportId: string;
    graphId?: string;
    format: string;
    downloadUrl: string;
    size: number;
    expiresAt: string;
    metadata: {
      resolution?: string;
      compressionUsed: boolean;
      includedMetadata: boolean;
    };
  };
}

export interface ShareCreatedMessage extends BaseWebSocketMessage {
  type: 'share_created';
  data: {
    shareId: string;
    graphId?: string;
    shareUrl: string;
    expiresAt?: string;
    passwordProtected: boolean;
    permissions: string[];
    description?: string;
  };
}

export interface ShareAccessMessage extends BaseWebSocketMessage {
  type: 'share_access';
  data: {
    shareId: string;
    action: 'accessed' | 'downloaded' | 'expired';
    userId?: string;
    ipAddress?: string;
    userAgent?: string;
  };
}

// System and connection messages
export interface ConnectionStatusMessage extends BaseWebSocketMessage {
  type: 'connection_status';
  data: {
    status: 'connected' | 'disconnected' | 'reconnecting' | 'error';
    serverLoad?: number;
    latency?: number;
    features: string[];
  };
}

export interface SystemNotificationMessage extends BaseWebSocketMessage {
  type: 'system_notification';
  data: {
    level: 'info' | 'warning' | 'error';
    title: string;
    message: string;
    actions?: Array<{
      label: string;
      action: string;
    }>;
    persistent?: boolean;
  };
}

export interface UserActivityMessage extends BaseWebSocketMessage {
  type: 'user_activity';
  data: {
    userId: string;
    userName?: string;
    action: string;
    resource: string;
    resourceId?: string;
    metadata?: Record<string, any>;
  };
}

// Performance and metrics messages
export interface PerformanceMetricsMessage extends BaseWebSocketMessage {
  type: 'performance_metrics';
  data: {
    metrics: {
      cpu: number;
      memory: number;
      network: number;
      renderTime: number;
      frameRate: number;
    };
    graphId?: string;
    nodeCount?: number;
    edgeCount?: number;
  };
}

export interface ServerHealthMessage extends BaseWebSocketMessage {
  type: 'server_health';
  data: {
    status: 'healthy' | 'degraded' | 'unhealthy';
    services: Array<{
      name: string;
      status: 'up' | 'down' | 'degraded';
      latency?: number;
    }>;
    load: {
      cpu: number;
      memory: number;
      activeConnections: number;
    };
  };
}

// Graph Interaction messages
export interface GraphProcessingProgressMessage extends BaseWebSocketMessage {
  type: 'graph_processing_progress';
  data: {
    taskId: string;
    graphId?: string;
    progress: number; // 0-100
    stage: string;
    currentOperation: string;
    estimatedTimeRemaining?: number;
    metrics?: {
      stepsProcessed: number;
      totalSteps: number;
      currentStep: string;
      operationsCompleted: number;
    };
  };
}

export interface GraphProcessingCompleteMessage extends BaseWebSocketMessage {
  type: 'graph_processing_complete';
  data: {
    taskId: string;
    graphId?: string;
    success: boolean;
    results?: any;
    processedSteps?: number;
    totalTime?: number;
    error?: string;
  };
}

export interface GraphProcessingErrorMessage extends BaseWebSocketMessage {
  type: 'graph_processing_error';
  data: {
    taskId: string;
    graphId?: string;
    error: string;
    stage: string;
    retryable: boolean;
  };
}

export interface TimeTraverseProgressMessage extends BaseWebSocketMessage {
  type: 'time_traverse_progress';
  data: {
    taskId: string;
    graphId?: string;
    progress: number; // 0-100
    stage: string;
    currentStep: number;
    totalSteps: number;
    stepName?: string;
    estimatedTimeRemaining?: number;
  };
}

export interface TimeTraverseCompleteMessage extends BaseWebSocketMessage {
  type: 'time_traverse_complete';
  data: {
    taskId: string;
    graphId?: string;
    success: boolean;
    totalSteps: number;
    timeline?: any;
    processingTime?: number;
    error?: string;
  };
}

export interface CollaborationSessionMessage extends BaseWebSocketMessage {
  type: 'collaboration_session';
  data: {
    sessionId: string;
    action: 'created' | 'user_joined' | 'user_left' | 'ended';
    userId?: string;
    userName?: string;
    participantCount: number;
    shareUrl?: string;
  };
}

export interface VRARModeMessage extends BaseWebSocketMessage {
  type: 'vr_ar_mode';
  data: {
    sessionId: string;
    mode: 'vr' | 'ar' | 'disabled';
    userId?: string;
    features: {
      handTracking: boolean;
      hapticFeedback: boolean;
      spatialAudio?: boolean;
    };
  };
}

export interface ExplorationTourMessage extends BaseWebSocketMessage {
  type: 'exploration_tour';
  data: {
    tourId: string;
    action: 'created' | 'updated' | 'waypoint_added' | 'completed';
    waypoints?: Array<{
      nodeId: string;
      description: string;
      order: number;
    }>;
    progress?: number;
  };
}

// Union type of all possible WebSocket messages
export type WebSocketMessage =
  | WorkspaceUpdateMessage
  | WorkspaceDeletedMessage
  | WorkspaceCollaborationMessage
  | AnalysisProgressMessage
  | AnalysisCompleteMessage
  | AnalysisErrorMessage
  | OptimizationUpdateMessage
  | OptimizationResultMessage
  | ExportProgressMessage
  | ExportReadyMessage
  | ShareCreatedMessage
  | ShareAccessMessage
  | ConnectionStatusMessage
  | SystemNotificationMessage
  | UserActivityMessage
  | PerformanceMetricsMessage
  | ServerHealthMessage
  | GraphProcessingProgressMessage
  | GraphProcessingCompleteMessage
  | GraphProcessingErrorMessage
  | TimeTraverseProgressMessage
  | TimeTraverseCompleteMessage
  | CollaborationSessionMessage
  | VRARModeMessage
  | ExplorationTourMessage;

// Event handler types
export type MessageHandler<T extends WebSocketMessage = WebSocketMessage> = (message: T) => void;

export interface WebSocketEventHandlers {
  // Workspace events
  workspace_update: MessageHandler<WorkspaceUpdateMessage>;
  workspace_deleted: MessageHandler<WorkspaceDeletedMessage>;
  workspace_collaboration: MessageHandler<WorkspaceCollaborationMessage>;

  // Analysis events
  analysis_progress: MessageHandler<AnalysisProgressMessage>;
  analysis_complete: MessageHandler<AnalysisCompleteMessage>;
  analysis_error: MessageHandler<AnalysisErrorMessage>;

  // Optimization events
  optimization_update: MessageHandler<OptimizationUpdateMessage>;
  optimization_result: MessageHandler<OptimizationResultMessage>;

  // Export events
  export_progress: MessageHandler<ExportProgressMessage>;
  export_ready: MessageHandler<ExportReadyMessage>;
  share_created: MessageHandler<ShareCreatedMessage>;
  share_access: MessageHandler<ShareAccessMessage>;

  // System events
  connection_status: MessageHandler<ConnectionStatusMessage>;
  system_notification: MessageHandler<SystemNotificationMessage>;
  user_activity: MessageHandler<UserActivityMessage>;
  performance_metrics: MessageHandler<PerformanceMetricsMessage>;
  server_health: MessageHandler<ServerHealthMessage>;

  // Graph Interaction events
  graph_processing_progress: MessageHandler<GraphProcessingProgressMessage>;
  graph_processing_complete: MessageHandler<GraphProcessingCompleteMessage>;
  graph_processing_error: MessageHandler<GraphProcessingErrorMessage>;
  time_traverse_progress: MessageHandler<TimeTraverseProgressMessage>;
  time_traverse_complete: MessageHandler<TimeTraverseCompleteMessage>;
  collaboration_session: MessageHandler<CollaborationSessionMessage>;
  vr_ar_mode: MessageHandler<VRARModeMessage>;
  exploration_tour: MessageHandler<ExplorationTourMessage>;
}

// Configuration interfaces
export interface WebSocketConfig {
  url?: string;
  protocols?: string[];
  reconnect: {
    maxAttempts: number;
    baseDelay: number;
    maxDelay: number;
    backoffFactor: number;
  };
  heartbeat: {
    interval: number;
    timeout: number;
  };
  compression: boolean;
  binaryProtocol: boolean;
}

export interface WebSocketConnectionState {
  status: 'disconnected' | 'connecting' | 'connected' | 'reconnecting' | 'failed';
  lastConnected?: number;
  lastError?: string;
  reconnectAttempts: number;
  serverFeatures: string[];
  latency?: number;
}

// Subscription management
export interface Subscription {
  id: string;
  type: keyof WebSocketEventHandlers;
  handler: MessageHandler;
  options?: {
    once?: boolean;
    filter?: (message: WebSocketMessage) => boolean;
  };
}

export interface SubscriptionFilters {
  workspaceId?: string;
  userId?: string;
  graphId?: string;
  analysisId?: string;
  optimizationId?: string;
  exportId?: string;
}

// Error types
export interface WebSocketError {
  code: string;
  message: string;
  type: 'connection' | 'protocol' | 'auth' | 'rate_limit' | 'server' | 'client';
  retryable: boolean;
  retryAfter?: number;
  context?: Record<string, any>;
}

// Statistics and monitoring
export interface WebSocketStatistics {
  messagesReceived: number;
  messagesSent: number;
  bytesReceived: number;
  bytesSent: number;
  connectionTime: number;
  reconnections: number;
  averageLatency: number;
  messagesByType: Record<string, number>;
  errors: number;
  lastActivity: number;
}