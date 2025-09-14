import { createLogger } from './logger';
import { Vec3, BinaryNodeData } from '../types/binaryProtocol';

const logger = createLogger('BatchQueue');

export interface BatchQueueConfig {
  batchSize: number;           // Maximum items per batch (50-100)
  flushIntervalMs: number;     // Flush interval in milliseconds (200ms for 5Hz)
  maxQueueSize: number;        // Maximum queue size before forcing flush
  priorityField?: string;      // Optional field for priority sorting
}

export interface QueueItem<T> {
  data: T;
  timestamp: number;
  priority: number;
  retryCount?: number;
}

export interface BatchProcessor<T> {
  processBatch: (batch: T[]) => Promise<void>;
  onError?: (error: Error, batch: T[]) => void;
  onSuccess?: (batch: T[]) => void;
}

/**
 * Generic batch queue for efficiently batching updates
 * Supports priority queuing, automatic flushing, and retry logic
 */
export class BatchQueue<T> {
  private queue: QueueItem<T>[] = [];
  private flushTimer: NodeJS.Timeout | null = null;
  private isProcessing = false;
  private config: BatchQueueConfig;
  private processor: BatchProcessor<T>;
  private metrics = {
    totalBatches: 0,
    totalItems: 0,
    failedBatches: 0,
    droppedItems: 0,
  };

  constructor(config: BatchQueueConfig, processor: BatchProcessor<T>) {
    this.config = {
      batchSize: 50,
      flushIntervalMs: 200,
      maxQueueSize: 1000,
      ...config
    };
    this.processor = processor;
    
    logger.info('BatchQueue initialized:', this.config);
  }

  /**
   * Add an item to the queue with optional priority
   */
  enqueue(data: T, priority: number = 0): void {
    // Check queue size limit
    if (this.queue.length >= this.config.maxQueueSize) {
      // Drop oldest low-priority items if queue is full
      this.dropOldestLowPriorityItems(1);
      this.metrics.droppedItems++;
      logger.warn('Queue full, dropped oldest low-priority item');
    }

    const item: QueueItem<T> = {
      data,
      timestamp: Date.now(),
      priority,
      retryCount: 0
    };

    // Insert item in priority order (higher priority first)
    const insertIndex = this.queue.findIndex(q => q.priority < priority);
    if (insertIndex === -1) {
      this.queue.push(item);
    } else {
      this.queue.splice(insertIndex, 0, item);
    }

    // Check if we should flush immediately
    if (this.queue.length >= this.config.batchSize) {
      this.flush();
    } else {
      this.scheduleFlush();
    }
  }

  /**
   * Add multiple items to the queue
   */
  enqueueBatch(items: T[], priority: number = 0): void {
    items.forEach(item => this.enqueue(item, priority));
  }

  /**
   * Schedule a flush operation
   */
  private scheduleFlush(): void {
    if (this.flushTimer) {
      return; // Already scheduled
    }

    this.flushTimer = setTimeout(() => {
      this.flush();
    }, this.config.flushIntervalMs);
  }

  /**
   * Force immediate flush of the queue
   */
  async flush(): Promise<void> {
    // Clear any pending flush timer
    if (this.flushTimer) {
      clearTimeout(this.flushTimer);
      this.flushTimer = null;
    }

    // Don't process if already processing or queue is empty
    if (this.isProcessing || this.queue.length === 0) {
      return;
    }

    this.isProcessing = true;

    try {
      // Process in batches
      while (this.queue.length > 0) {
        // Take up to batchSize items from the queue
        const batch = this.queue.splice(0, this.config.batchSize);
        const batchData = batch.map(item => item.data);

        try {
          await this.processor.processBatch(batchData);
          
          // Update metrics
          this.metrics.totalBatches++;
          this.metrics.totalItems += batchData.length;

          // Call success callback
          if (this.processor.onSuccess) {
            this.processor.onSuccess(batchData);
          }

          logger.debug(`Processed batch of ${batchData.length} items`);
        } catch (error) {
          logger.error('Batch processing failed:', error);
          this.metrics.failedBatches++;

          // Handle retry logic
          this.handleFailedBatch(batch, error as Error);

          // Call error callback
          if (this.processor.onError) {
            this.processor.onError(error as Error, batchData);
          }
        }
      }
    } finally {
      this.isProcessing = false;

      // Schedule next flush if there are items in queue
      if (this.queue.length > 0) {
        this.scheduleFlush();
      }
    }
  }

  /**
   * Handle failed batch with retry logic
   */
  private handleFailedBatch(batch: QueueItem<T>[], error: Error): void {
    const MAX_RETRIES = 3;

    batch.forEach(item => {
      item.retryCount = (item.retryCount || 0) + 1;

      if (item.retryCount < MAX_RETRIES) {
        // Re-queue with increased priority for retry
        this.enqueue(item.data, item.priority + 10);
        logger.info(`Re-queued item for retry (attempt ${item.retryCount}/${MAX_RETRIES})`);
      } else {
        this.metrics.droppedItems++;
        logger.error(`Dropped item after ${MAX_RETRIES} retries`);
      }
    });
  }

  /**
   * Drop oldest low-priority items to make room
   */
  private dropOldestLowPriorityItems(count: number): void {
    // Sort by priority (ascending) then by timestamp (ascending)
    const sorted = [...this.queue].sort((a, b) => {
      if (a.priority !== b.priority) {
        return a.priority - b.priority; // Lower priority first
      }
      return a.timestamp - b.timestamp; // Older first
    });

    // Remove the specified number of items
    for (let i = 0; i < Math.min(count, sorted.length); i++) {
      const indexToRemove = this.queue.indexOf(sorted[i]);
      if (indexToRemove !== -1) {
        this.queue.splice(indexToRemove, 1);
      }
    }
  }

  /**
   * Clear all items from the queue
   */
  clear(): void {
    this.queue = [];
    if (this.flushTimer) {
      clearTimeout(this.flushTimer);
      this.flushTimer = null;
    }
    logger.info('Queue cleared');
  }

  /**
   * Get current queue size
   */
  size(): number {
    return this.queue.length;
  }

  /**
   * Get queue metrics
   */
  getMetrics() {
    return {
      ...this.metrics,
      currentQueueSize: this.queue.length,
      isProcessing: this.isProcessing
    };
  }

  /**
   * Destroy the queue and clean up resources
   */
  destroy(): void {
    this.clear();
    logger.info('BatchQueue destroyed');
  }
}

/**
 * Specialized BatchQueue for node position updates
 */
export class NodePositionBatchQueue extends BatchQueue<BinaryNodeData> {
  constructor(processor: BatchProcessor<BinaryNodeData>) {
    super(
      {
        batchSize: 50,        // 50 nodes per batch
        flushIntervalMs: 200, // 5Hz update rate
        maxQueueSize: 500,    // Max 500 nodes in queue
        priorityField: 'nodeId'
      },
      processor
    );
  }

  /**
   * Enqueue position update with deduplication
   * Newer updates for the same node replace older ones
   */
  enqueuePositionUpdate(nodeData: BinaryNodeData, priority: number = 0): void {
    // Remove any existing update for this node
    this.deduplicateNode(nodeData.nodeId);
    
    // Add the new update
    this.enqueue(nodeData, priority);
  }

  /**
   * Remove existing updates for a specific node
   */
  private deduplicateNode(nodeId: number): void {
    const queue = (this as any).queue as QueueItem<BinaryNodeData>[];
    const index = queue.findIndex(item => item.data.nodeId === nodeId);
    if (index !== -1) {
      queue.splice(index, 1);
      logger.debug(`Deduplicated update for node ${nodeId}`);
    }
  }
}

/**
 * Create a batch processor for WebSocket binary data
 */
export function createWebSocketBatchProcessor(
  sendFunction: (data: ArrayBuffer) => void
): BatchProcessor<BinaryNodeData> {
  return {
    processBatch: async (batch: BinaryNodeData[]) => {
      // Convert to binary format
      const { createBinaryNodeData } = await import('../types/binaryProtocol');
      const binaryData = createBinaryNodeData(batch);
      
      // Send via WebSocket
      sendFunction(binaryData);
      
      logger.debug(`Sent batch of ${batch.length} node updates (${binaryData.byteLength} bytes)`);
    },
    onError: (error, batch) => {
      logger.error(`Failed to send batch of ${batch.length} nodes:`, error);
    },
    onSuccess: (batch) => {
      logger.debug(`Successfully sent ${batch.length} node updates`);
    }
  };
}