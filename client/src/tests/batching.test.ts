import { BatchQueue, BatchQueueConfig, BatchProcessor } from '../utils/BatchQueue';
import { validateNodePositions, sanitizeNodeData } from '../utils/validation';
import { BinaryNodeData } from '../types/binaryProtocol';

describe('BatchQueue', () => {
  let processedBatches: any[][] = [];
  let mockProcessor: BatchProcessor<any>;
  
  beforeEach(() => {
    processedBatches = [];
    mockProcessor = {
      processBatch: jest.fn(async (batch) => {
        processedBatches.push(batch);
      }),
      onError: jest.fn(),
      onSuccess: jest.fn()
    };
  });

  test('should batch items according to batchSize', async () => {
    const config: BatchQueueConfig = {
      batchSize: 3,
      flushIntervalMs: 50,
      maxQueueSize: 100
    };
    
    const queue = new BatchQueue(config, mockProcessor);
    
    // Add 5 items
    for (let i = 0; i < 5; i++) {
      queue.enqueue({ id: i });
    }
    
    // First batch should be sent immediately (3 items)
    expect(processedBatches).toHaveLength(1);
    expect(processedBatches[0]).toHaveLength(3);
    
    // Wait for flush interval
    await new Promise(resolve => setTimeout(resolve, 100));
    
    // Second batch should be sent (2 items)
    expect(processedBatches).toHaveLength(2);
    expect(processedBatches[1]).toHaveLength(2);
    
    queue.destroy();
  });

  test('should respect priority ordering', async () => {
    const config: BatchQueueConfig = {
      batchSize: 5,
      flushIntervalMs: 50,
      maxQueueSize: 100
    };
    
    const queue = new BatchQueue(config, mockProcessor);
    
    // Add items with different priorities
    queue.enqueue({ id: 1 }, 0);  // Low priority
    queue.enqueue({ id: 2 }, 10); // High priority
    queue.enqueue({ id: 3 }, 5);  // Medium priority
    queue.enqueue({ id: 4 }, 10); // High priority
    queue.enqueue({ id: 5 }, 0);  // Low priority
    
    // Should process immediately due to batch size
    expect(processedBatches).toHaveLength(1);
    
    // Check priority ordering (high priority first)
    const batch = processedBatches[0];
    expect(batch[0].id).toBe(2); // Priority 10
    expect(batch[1].id).toBe(4); // Priority 10
    expect(batch[2].id).toBe(3); // Priority 5
    expect(batch[3].id).toBe(1); // Priority 0
    expect(batch[4].id).toBe(5); // Priority 0
    
    queue.destroy();
  });

  test('should handle flush() correctly', async () => {
    const config: BatchQueueConfig = {
      batchSize: 10,
      flushIntervalMs: 1000, // Long interval
      maxQueueSize: 100
    };
    
    const queue = new BatchQueue(config, mockProcessor);
    
    // Add items that won't trigger immediate batch
    queue.enqueue({ id: 1 });
    queue.enqueue({ id: 2 });
    
    expect(processedBatches).toHaveLength(0);
    
    // Force flush
    await queue.flush();
    
    expect(processedBatches).toHaveLength(1);
    expect(processedBatches[0]).toHaveLength(2);
    
    queue.destroy();
  });
});

describe('Validation', () => {
  test('should validate node positions correctly', () => {
    const validNodes: BinaryNodeData[] = [
      {
        nodeId: 1,
        position: { x: 100, y: 200, z: 300 },
        velocity: { x: 1, y: 2, z: 3 }
      },
      {
        nodeId: 2,
        position: { x: -100, y: -200, z: -300 },
        velocity: { x: -1, y: -2, z: -3 }
      }
    ];

    const result = validateNodePositions(validNodes);
    expect(result.valid).toBe(true);
    expect(result.errors).toBeUndefined();
  });

  test('should reject invalid positions', () => {
    const invalidNodes: BinaryNodeData[] = [
      {
        nodeId: 1,
        position: { x: NaN, y: 200, z: 300 },
        velocity: { x: 1, y: 2, z: 3 }
      },
      {
        nodeId: 2,
        position: { x: Infinity, y: -200, z: -300 },
        velocity: { x: -1, y: -2, z: -3 }
      }
    ];

    const result = validateNodePositions(invalidNodes);
    expect(result.valid).toBe(false);
    expect(result.errors).toBeDefined();
    expect(result.errors!.length).toBeGreaterThan(0);
  });

  test('should detect duplicate node IDs', () => {
    const duplicateNodes: BinaryNodeData[] = [
      {
        nodeId: 1,
        position: { x: 100, y: 200, z: 300 },
        velocity: { x: 1, y: 2, z: 3 }
      },
      {
        nodeId: 1, // Duplicate ID
        position: { x: -100, y: -200, z: -300 },
        velocity: { x: -1, y: -2, z: -3 }
      }
    ];

    const result = validateNodePositions(duplicateNodes);
    expect(result.valid).toBe(false);
    expect(result.errors).toBeDefined();
    expect(result.errors![0]).toContain('Duplicate node IDs');
  });

  test('should sanitize invalid data', () => {
    const invalidNode: BinaryNodeData = {
      nodeId: -1,
      position: { x: NaN, y: 20000, z: -20000 },
      velocity: { x: 2000, y: 0, z: 0 } // Exceeds max velocity
    };

    const sanitized = sanitizeNodeData(invalidNode, {
      maxCoordinate: 10000,
      minCoordinate: -10000,
      maxVelocity: 1000
    });

    expect(sanitized.nodeId).toBe(0); // Negative ID fixed
    expect(sanitized.position.x).toBe(0); // NaN becomes 0
    expect(sanitized.position.y).toBe(10000); // Clamped to max
    expect(sanitized.position.z).toBe(-10000); // Clamped to min
    
    // Velocity should be scaled down
    const speed = Math.sqrt(
      sanitized.velocity.x ** 2 + 
      sanitized.velocity.y ** 2 + 
      sanitized.velocity.z ** 2
    );
    expect(speed).toBeLessThanOrEqual(1000);
  });
});

describe('Integration: BatchQueue with Validation', () => {
  test('should validate and batch node updates', async () => {
    const processedNodes: BinaryNodeData[][] = [];
    
    const processor: BatchProcessor<BinaryNodeData> = {
      processBatch: async (batch) => {
        // Validate batch before processing
        const validation = validateNodePositions(batch);
        if (validation.valid) {
          processedNodes.push(batch);
        } else {
          throw new Error('Validation failed');
        }
      }
    };

    const config: BatchQueueConfig = {
      batchSize: 2,
      flushIntervalMs: 50,
      maxQueueSize: 100
    };
    
    const queue = new BatchQueue<BinaryNodeData>(config, processor);
    
    // Add valid nodes
    queue.enqueue({
      nodeId: 1,
      position: { x: 100, y: 200, z: 300 },
      velocity: { x: 1, y: 2, z: 3 }
    });
    
    queue.enqueue({
      nodeId: 2,
      position: { x: -100, y: -200, z: -300 },
      velocity: { x: -1, y: -2, z: -3 }
    });
    
    // Should process immediately due to batch size
    expect(processedNodes).toHaveLength(1);
    expect(processedNodes[0]).toHaveLength(2);
    
    queue.destroy();
  });
});