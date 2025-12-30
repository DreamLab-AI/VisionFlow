import { createLogger } from './loggerConfig';
import { Vec3, BinaryNodeData } from '../types/binaryProtocol';

const logger = createLogger('BatchQueue');

export interface BatchQueueConfig {
  batchSize: number;           
  flushIntervalMs: number;     
  maxQueueSize: number;        
  priorityField?: string;      
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
      batchSize: config.batchSize,
      flushIntervalMs: config.flushIntervalMs,
      maxQueueSize: config.maxQueueSize,
      priorityField: config.priorityField
    };
    this.processor = processor;
    
    logger.info('BatchQueue initialized:', this.config);
  }

  
  enqueue(data: T, priority: number = 0): void {
    
    if (this.queue.length >= this.config.maxQueueSize) {
      
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

    
    const insertIndex = this.queue.findIndex(q => q.priority < priority);
    if (insertIndex === -1) {
      this.queue.push(item);
    } else {
      this.queue.splice(insertIndex, 0, item);
    }

    
    if (this.queue.length >= this.config.batchSize) {
      this.flush();
    } else {
      this.scheduleFlush();
    }
  }

  
  enqueueBatch(items: T[], priority: number = 0): void {
    items.forEach(item => this.enqueue(item, priority));
  }

  
  private scheduleFlush(): void {
    if (this.flushTimer) {
      return; 
    }

    this.flushTimer = setTimeout(() => {
      this.flush();
    }, this.config.flushIntervalMs);
  }

  
  async flush(): Promise<void> {
    
    if (this.flushTimer) {
      clearTimeout(this.flushTimer);
      this.flushTimer = null;
    }

    
    if (this.isProcessing || this.queue.length === 0) {
      return;
    }

    this.isProcessing = true;

    try {
      
      while (this.queue.length > 0) {
        
        const batch = this.queue.splice(0, this.config.batchSize);
        const batchData = batch.map(item => item.data);

        try {
          await this.processor.processBatch(batchData);
          
          
          this.metrics.totalBatches++;
          this.metrics.totalItems += batchData.length;

          
          if (this.processor.onSuccess) {
            this.processor.onSuccess(batchData);
          }

          logger.debug(`Processed batch of ${batchData.length} items`);
        } catch (error) {
          logger.error('Batch processing failed:', error);
          this.metrics.failedBatches++;

          
          this.handleFailedBatch(batch, error as Error);

          
          if (this.processor.onError) {
            this.processor.onError(error as Error, batchData);
          }
        }
      }
    } finally {
      this.isProcessing = false;

      
      if (this.queue.length > 0) {
        this.scheduleFlush();
      }
    }
  }

  
  private handleFailedBatch(batch: QueueItem<T>[], error: Error): void {
    const MAX_RETRIES = 3;

    batch.forEach(item => {
      item.retryCount = (item.retryCount || 0) + 1;

      if (item.retryCount < MAX_RETRIES) {
        
        this.enqueue(item.data, item.priority + 10);
        logger.info(`Re-queued item for retry (attempt ${item.retryCount}/${MAX_RETRIES})`);
      } else {
        this.metrics.droppedItems++;
        logger.error(`Dropped item after ${MAX_RETRIES} retries`);
      }
    });
  }

  
  private dropOldestLowPriorityItems(count: number): void {
    
    const sorted = [...this.queue].sort((a, b) => {
      if (a.priority !== b.priority) {
        return a.priority - b.priority; 
      }
      return a.timestamp - b.timestamp; 
    });

    
    for (let i = 0; i < Math.min(count, sorted.length); i++) {
      const indexToRemove = this.queue.indexOf(sorted[i]);
      if (indexToRemove !== -1) {
        this.queue.splice(indexToRemove, 1);
      }
    }
  }

  
  clear(): void {
    this.queue = [];
    if (this.flushTimer) {
      clearTimeout(this.flushTimer);
      this.flushTimer = null;
    }
    logger.info('Queue cleared');
  }

  
  size(): number {
    return this.queue.length;
  }

  
  getMetrics() {
    return {
      ...this.metrics,
      currentQueueSize: this.queue.length,
      isProcessing: this.isProcessing
    };
  }

  
  destroy(): void {
    this.clear();
    logger.info('BatchQueue destroyed');
  }
}


export class NodePositionBatchQueue extends BatchQueue<BinaryNodeData> {
  constructor(processor: BatchProcessor<BinaryNodeData>) {
    super(
      {
        batchSize: 50,        
        flushIntervalMs: 200, 
        maxQueueSize: 500,    
        priorityField: 'nodeId'
      },
      processor
    );
  }

  
  enqueuePositionUpdate(nodeData: BinaryNodeData, priority: number = 0): void {
    
    this.deduplicateNode(nodeData.nodeId);
    
    
    this.enqueue(nodeData, priority);
  }

  
  private deduplicateNode(nodeId: number): void {
    const queue = (this as any).queue as QueueItem<BinaryNodeData>[];
    const index = queue.findIndex(item => item.data.nodeId === nodeId);
    if (index !== -1) {
      queue.splice(index, 1);
      logger.debug(`Deduplicated update for node ${nodeId}`);
    }
  }
}


export function createWebSocketBatchProcessor(
  sendFunction: (data: ArrayBuffer) => void
): BatchProcessor<BinaryNodeData> {
  return {
    processBatch: async (batch: BinaryNodeData[]) => {
      
      const { createBinaryNodeData } = await import('../types/binaryProtocol');
      const binaryData = createBinaryNodeData(batch);
      
      
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