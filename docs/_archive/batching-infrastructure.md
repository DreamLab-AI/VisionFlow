# Client-Side Batching Infrastructure

This document describes the client-side batching infrastructure implemented to improve performance by batching updates before sending them to the server.

## Overview

The batching infrastructure addresses performance issues caused by sending updates one-by-one to the server. It implements:

1. **BatchQueue Utility**: A generic batching queue that collects updates and sends them in batches
2. **Throttled Position Updates**: Updates are batched and sent at 5Hz (200ms intervals)
3. **Client-Side Validation**: Pre-validation prevents rejected updates from reaching the server
4. **Batch Update Endpoints**: REST API endpoints for batch operations

## Key Components

### 1. BatchQueue (`/client/src/utils/BatchQueue.ts`)

A generic utility class for batching any type of updates:

```typescript
const queue = new BatchQueue<T>(config, processor);

// Configuration options
const config: BatchQueueConfig = {
  batchSize: 50,           // Max items per batch
  flushIntervalMs: 200,    // Flush interval (5Hz)
  maxQueueSize: 1000,      // Max queue size
  priorityField: 'nodeId'  // Optional priority field
};
```

Features:
- Automatic batching based on size or time interval
- Priority queuing (higher priority items processed first)
- Retry logic with exponential backoff
- Metrics tracking
- Deduplication support

### 2. Validation (`/client/src/utils/validation.ts`)

Pre-validation ensures data integrity before sending:

```typescript
// Validate node positions
const result = validateNodePositions(nodes, {
  maxNodes: 10000,
  maxCoordinate: 10000,
  minCoordinate: -10000,
  maxVelocity: 1000,
  allowNaN: false,
  allowInfinity: false
});

// Sanitize invalid data
const sanitized = sanitizeNodeData(invalidNode, config);
```

Validation checks:
- NaN and Infinity values
- Coordinate bounds
- Velocity limits
- Duplicate node IDs
- Array size limits

### 3. WebSocket Integration

The WebSocketService now includes batching for position updates:

```typescript
// Send position updates (automatically batched)
webSocketService.sendNodePositionUpdates([
  {
    nodeId: 1,
    position: { x: 100, y: 200, z: 300 },
    velocity: { x: 1, y: 2, z: 3 }
  }
]);

// Force immediate flush
await webSocketService.flushPositionUpdates();

// Get queue metrics
const metrics = webSocketService.getPositionQueueMetrics();
```

### 4. Batch Update API (`/client/src/api/batchUpdateApi.ts`)

REST API endpoints for batch operations:

```typescript
// Batch update node positions
await batchUpdateApi.updateNodePositions(updates);

// Batch update settings
await batchUpdateApi.updateSettings([
  { path: 'visualisation.glow.intensity', value: 1.5 },
  { path: 'visualisation.glow.radius', value: 0.8 }
]);

// Batch create/delete nodes
await batchUpdateApi.createNodes(nodes);
await batchUpdateApi.deleteNodes(nodeIds);
```

## Usage Examples

### Basic Position Updates

```typescript
// In your graph interaction handler
const handleNodeDrag = (nodeId: number, position: Vec3) => {
  // Single update - automatically batched with others
  webSocketService.sendNodePositionUpdates([{
    nodeId,
    position,
    velocity: { x: 0, y: 0, z: 0 }
  }]);
};
```

### Batch Multiple Updates

```typescript
// Update multiple nodes at once
const updates = selectedNodes.map(node => ({
  nodeId: node.id,
  position: node.position,
  velocity: node.velocity || { x: 0, y: 0, z: 0 }
}));

webSocketService.sendNodePositionUpdates(updates);
```

### Priority Updates

```typescript
// Agent nodes get higher priority
const priority = isAgentNode(nodeId) ? 10 : 0;
positionBatchQueue.enqueuePositionUpdate(nodeData, priority);
```

### Custom Batch Processor

```typescript
const customProcessor: BatchProcessor<MyData> = {
  processBatch: async (batch) => {
    // Custom processing logic
    await myApi.sendBatch(batch);
  },
  onError: (error, batch) => {
    console.error('Batch failed:', error);
  },
  onSuccess: (batch) => {
    console.log('Batch processed:', batch.length);
  }
};

const queue = new BatchQueue(config, customProcessor);
```

## Performance Benefits

1. **Reduced Network Traffic**: 50-100 updates batched into single requests
2. **5Hz Update Rate**: Consistent 200ms intervals prevent overwhelming the server
3. **Pre-Validation**: Invalid updates caught before sending
4. **Priority Processing**: Important updates (e.g., agent nodes) processed first
5. **Automatic Retry**: Failed batches retry with exponential backoff

## Configuration

### Batch Size
- Default: 50 nodes per batch
- Adjustable based on network conditions
- Larger batches = fewer requests but higher latency

### Update Frequency
- Default: 5Hz (200ms intervals)
- Balances responsiveness with server load
- Adjustable via `flushIntervalMs`

### Validation Limits
```typescript
const DEFAULT_VALIDATION_CONFIG = {
  maxNodes: 10000,        // Maximum nodes per batch
  maxCoordinate: 10000,   // Maximum coordinate value
  minCoordinate: -10000,  // Minimum coordinate value
  maxVelocity: 1000,      // Maximum velocity magnitude
  allowNaN: false,        // Reject NaN values
  allowInfinity: false    // Reject Infinity values
};
```

## Best Practices

1. **Always Validate**: Use validation before sending updates
2. **Batch Related Updates**: Group updates that occur together
3. **Use Priority**: Mark important updates with higher priority
4. **Monitor Metrics**: Track queue size and processing stats
5. **Handle Errors**: Implement error callbacks for failed batches
6. **Flush on Navigation**: Force flush before page unload

## Testing

Run the test suite:

```bash
npm test batching.test.ts
```

The tests cover:
- Basic batching functionality
- Priority ordering
- Validation logic
- Sanitization
- Integration scenarios

## Migration Guide

To migrate existing code to use batching:

1. Replace direct WebSocket sends with `sendNodePositionUpdates()`
2. Remove manual batching logic (now handled automatically)
3. Add validation before sending updates
4. Use batch API endpoints for REST operations
5. Monitor queue metrics during development

## Troubleshooting

### Updates Not Sending
- Check WebSocket connection status
- Verify validation is passing
- Check queue metrics for pending items
- Force flush if needed

### High Latency
- Reduce batch size for lower latency
- Decrease flush interval
- Check network conditions

### Validation Failures
- Enable debug logging to see validation errors
- Use sanitizeNodeData() for automatic fixing
- Adjust validation limits if needed