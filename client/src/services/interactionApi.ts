

import { createLogger } from '../utils/loggerConfig';
import { createErrorMetadata } from '../utils/loggerConfig';
import { debugState } from '../utils/clientDebugState';
import { unifiedApiClient } from './api/UnifiedApiClient';
import { webSocketService } from '../store/websocketStore';
import type {
  GraphProcessingProgressMessage,
  GraphProcessingCompleteMessage,
  GraphProcessingErrorMessage,
  TimeTraverseProgressMessage,
  TimeTraverseCompleteMessage
} from '../types/websocketTypes';

const logger = createLogger('InteractionApi');

export interface GraphProcessingOptions {
  mode: 'time_travel' | 'collaboration' | 'vr_ar' | 'exploration';
  params: Record<string, any>;
}

export interface GraphProcessingProgress {
  taskId: string;
  progress: number;
  stage: string;
  currentOperation: string;
  estimatedTimeRemaining?: number;
  metrics?: {
    stepsProcessed: number;
    totalSteps: number;
    currentStep: string;
    operationsCompleted: number;
  };
}

export interface GraphProcessingResult {
  taskId: string;
  success: boolean;
  data?: any;
  error?: string;
  processedSteps?: number;
  totalTime?: number;
}

export interface ProgressCallback {
  onProgress: (progress: GraphProcessingProgress) => void;
  onComplete: (result: GraphProcessingResult) => void;
  onError: (error: { taskId: string; error: string; stage: string; retryable: boolean }) => void;
}

class InteractionApi {
  private static instance: InteractionApi;
  private activeProcesses: Map<string, ProgressCallback> = new Map();
  private wsUnsubscribers: (() => void)[] = [];

  private constructor() {
    this.setupWebSocketListeners();
  }

  public static getInstance(): InteractionApi {
    if (!InteractionApi.instance) {
      InteractionApi.instance = new InteractionApi();
    }
    return InteractionApi.instance;
  }

  private setupWebSocketListeners(): void {
    
    const progressUnsub = webSocketService.on('graph_processing_progress', (data: any) => {
      const message = data as GraphProcessingProgressMessage;
      const callback = this.activeProcesses.get(message.data.taskId);

      if (callback) {
        callback.onProgress({
          taskId: message.data.taskId,
          progress: message.data.progress,
          stage: message.data.stage,
          currentOperation: message.data.currentOperation,
          estimatedTimeRemaining: message.data.estimatedTimeRemaining,
          metrics: message.data.metrics
        });
      }
    });

    
    const completeUnsub = webSocketService.on('graph_processing_complete', (data: any) => {
      const message = data as GraphProcessingCompleteMessage;
      const callback = this.activeProcesses.get(message.data.taskId);

      if (callback) {
        callback.onComplete({
          taskId: message.data.taskId,
          success: message.data.success,
          data: message.data.results,
          processedSteps: message.data.processedSteps,
          totalTime: message.data.totalTime
        });
        this.activeProcesses.delete(message.data.taskId);
      }
    });

    
    const errorUnsub = webSocketService.on('graph_processing_error', (data: any) => {
      const message = data as GraphProcessingErrorMessage;
      const callback = this.activeProcesses.get(message.data.taskId);

      if (callback) {
        callback.onError({
          taskId: message.data.taskId,
          error: message.data.error,
          stage: message.data.stage,
          retryable: message.data.retryable
        });
      }
    });

    
    const timeTraverseProgressUnsub = webSocketService.on('time_traverse_progress', (data: any) => {
      const message = data as TimeTraverseProgressMessage;
      const callback = this.activeProcesses.get(message.data.taskId);

      if (callback) {
        callback.onProgress({
          taskId: message.data.taskId,
          progress: message.data.progress,
          stage: message.data.stage,
          currentOperation: `Step ${message.data.currentStep}/${message.data.totalSteps}`,
          estimatedTimeRemaining: message.data.estimatedTimeRemaining,
          metrics: {
            stepsProcessed: message.data.currentStep,
            totalSteps: message.data.totalSteps,
            currentStep: message.data.stepName || `Step ${message.data.currentStep}`,
            operationsCompleted: message.data.currentStep
          }
        });
      }
    });

    
    const timeTraverseCompleteUnsub = webSocketService.on('time_traverse_complete', (data: any) => {
      const message = data as TimeTraverseCompleteMessage;
      const callback = this.activeProcesses.get(message.data.taskId);

      if (callback) {
        callback.onComplete({
          taskId: message.data.taskId,
          success: message.data.success,
          data: {
            totalSteps: message.data.totalSteps,
            timeline: message.data.timeline
          },
          processedSteps: message.data.totalSteps,
          totalTime: message.data.processingTime
        });
        this.activeProcesses.delete(message.data.taskId);
      }
    });

    this.wsUnsubscribers = [
      progressUnsub,
      completeUnsub,
      errorUnsub,
      timeTraverseProgressUnsub,
      timeTraverseCompleteUnsub
    ];
  }

  
  public async startGraphProcessing(
    graphData: any,
    options: GraphProcessingOptions,
    callbacks: ProgressCallback
  ): Promise<string> {
    try {
      if (debugState.isEnabled()) {
        logger.info(`Starting graph processing with mode: ${options.mode}`);
      }

      const response = await unifiedApiClient.postData<{ taskId: string; success: boolean }>('/graph/interaction/start', {
        graphData,
        mode: options.mode,
        params: options.params
      });

      if (!response.success) {
        throw new Error('Failed to start graph processing');
      }

      const taskId = response.taskId;
      this.activeProcesses.set(taskId, callbacks);

      if (debugState.isEnabled()) {
        logger.info(`Graph processing started with task ID: ${taskId}`);
      }

      return taskId;
    } catch (error) {
      logger.error('Failed to start graph processing:', createErrorMetadata(error));
      throw error;
    }
  }

  
  public async getProcessingStatus(taskId: string): Promise<GraphProcessingProgress | null> {
    try {
      const response = await unifiedApiClient.getData<{
        success: boolean;
        status?: GraphProcessingProgress;
      }>(`/graph/interaction/status/${taskId}`);

      return response.success ? response.status || null : null;
    } catch (error) {
      logger.error(`Failed to get processing status for task ${taskId}:`, createErrorMetadata(error));
      return null;
    }
  }

  
  public async cancelProcessing(taskId: string): Promise<boolean> {
    try {
      if (debugState.isEnabled()) {
        logger.info(`Cancelling graph processing task: ${taskId}`);
      }

      const response = await unifiedApiClient.postData<{ success: boolean }>(`/graph/interaction/cancel/${taskId}`, {});

      if (response.success) {
        this.activeProcesses.delete(taskId);
      }

      return response.success;
    } catch (error) {
      logger.error(`Failed to cancel processing task ${taskId}:`, createErrorMetadata(error));
      return false;
    }
  }

  
  public async getProcessingResults(taskId: string): Promise<GraphProcessingResult | null> {
    try {
      const response = await unifiedApiClient.getData<{
        success: boolean;
        result?: GraphProcessingResult;
      }>(`/graph/interaction/results/${taskId}`);

      return response.success ? response.result || null : null;
    } catch (error) {
      logger.error(`Failed to get processing results for task ${taskId}:`, createErrorMetadata(error));
      return null;
    }
  }

  
  public async initializeTimeTravel(
    graphId: string,
    callbacks: ProgressCallback
  ): Promise<string> {
    try {
      if (debugState.isEnabled()) {
        logger.info(`Initializing time travel for graph: ${graphId}`);
      }

      const response = await unifiedApiClient.postData<{ taskId: string; success: boolean }>('/graph/time-travel/initialize', {
        graphId
      });

      if (!response.success) {
        throw new Error('Failed to initialize time travel');
      }

      const taskId = response.taskId;
      this.activeProcesses.set(taskId, callbacks);

      return taskId;
    } catch (error) {
      logger.error('Failed to initialize time travel:', createErrorMetadata(error));
      throw error;
    }
  }

  
  public async navigateToStep(taskId: string, step: number): Promise<boolean> {
    try {
      const response = await unifiedApiClient.postData<{ success: boolean }>(`/graph/time-travel/navigate/${taskId}`, {
        step
      });

      return response.success;
    } catch (error) {
      logger.error(`Failed to navigate to step ${step} for task ${taskId}:`, createErrorMetadata(error));
      return false;
    }
  }

  
  public async startPlayback(taskId: string, speed: number = 1): Promise<boolean> {
    try {
      const response = await unifiedApiClient.postData<{ success: boolean }>(`/graph/time-travel/playback/${taskId}`, {
        action: 'start',
        speed
      });

      return response.success;
    } catch (error) {
      logger.error(`Failed to start playback for task ${taskId}:`, createErrorMetadata(error));
      return false;
    }
  }

  
  public async stopPlayback(taskId: string): Promise<boolean> {
    try {
      const response = await unifiedApiClient.postData<{ success: boolean }>(`/graph/time-travel/playback/${taskId}`, {
        action: 'stop'
      });

      return response.success;
    } catch (error) {
      logger.error(`Failed to stop playback for task ${taskId}:`, createErrorMetadata(error));
      return false;
    }
  }

  
  public async createCollaborationSession(graphId: string): Promise<{
    sessionId: string;
    shareUrl: string;
    success: boolean;
  }> {
    try {
      const response = await unifiedApiClient.postData<{
        sessionId: string;
        shareUrl: string;
        success: boolean;
      }>('/graph/collaboration/create', {
        graphId
      });

      return response;
    } catch (error) {
      logger.error('Failed to create collaboration session:', createErrorMetadata(error));
      throw error;
    }
  }

  
  public async joinCollaborationSession(sessionId: string): Promise<boolean> {
    try {
      const response = await unifiedApiClient.postData<{ success: boolean }>(`/graph/collaboration/join/${sessionId}`, {});
      return response.success;
    } catch (error) {
      logger.error(`Failed to join collaboration session ${sessionId}:`, createErrorMetadata(error));
      return false;
    }
  }

  
  public async leaveCollaborationSession(sessionId: string): Promise<boolean> {
    try {
      const response = await unifiedApiClient.postData<{ success: boolean }>(`/graph/collaboration/leave/${sessionId}`, {});
      return response.success;
    } catch (error) {
      logger.error(`Failed to leave collaboration session ${sessionId}:`, createErrorMetadata(error));
      return false;
    }
  }

  
  public cleanup(): void {
    this.wsUnsubscribers.forEach(unsub => unsub());
    this.wsUnsubscribers = [];
    this.activeProcesses.clear();
  }
}

export const interactionApi = InteractionApi.getInstance();