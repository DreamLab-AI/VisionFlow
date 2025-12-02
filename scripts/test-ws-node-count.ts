import WebSocket from 'ws';

// Test: Connect to WebSocket and check node count in initialGraphLoad
const ws = new WebSocket('ws://192.168.0.51:8080/ws');

let receivedInitialLoad = false;

ws.on('open', () => {
  console.log('[WS] Connected');
});

ws.on('message', (data: Buffer | string) => {
  // Check if binary
  if (Buffer.isBuffer(data)) {
    console.log(`[WS] Binary: ${data.length} bytes`);
    return;
  }

  try {
    const msg = JSON.parse(data.toString());
    console.log(`[WS] Message type: ${msg.type}`);

    if (msg.type === 'initialGraphLoad') {
      receivedInitialLoad = true;
      const nodeCount = msg.nodes?.length || 0;
      const edgeCount = msg.edges?.length || 0;
      console.log(`\n========================================`);
      console.log(`=== INITIAL GRAPH LOAD RESULTS ===`);
      console.log(`========================================`);
      console.log(`Nodes: ${nodeCount}`);
      console.log(`Edges: ${edgeCount}`);

      if (nodeCount <= 200 && nodeCount > 0) {
        console.log(`\n✓ SUCCESS: Fresh client received sparse data (${nodeCount} nodes)`);
        console.log(`  Expected: ~200 nodes (limit-based sparse initial load)`);
      } else if (nodeCount === 0) {
        console.log(`\n✗ FAIL: Received 0 nodes - filter too strict or data missing`);
      } else {
        console.log(`\n✗ FAIL: Received full graph (${nodeCount} nodes)`);
        console.log(`  Filter not applied - should be ~200 nodes max`);
      }

      ws.close();
    } else if (msg.type === 'connection_established') {
      console.log(`[WS] Connection established, client_id: ${msg.client_id}`);
    } else if (msg.type === 'state_sync') {
      console.log(`[WS] State sync received`);
    }
  } catch (e) {
    // Not JSON
    console.log(`[WS] Text: ${data.toString().substring(0, 100)}`);
  }
});

ws.on('error', (err) => {
  console.error('[WS] Error:', err.message);
});

ws.on('close', () => {
  console.log('[WS] Connection closed');
  if (!receivedInitialLoad) {
    console.log('\n✗ FAIL: Did not receive initialGraphLoad message');
  }
  process.exit(receivedInitialLoad ? 0 : 1);
});

// Timeout after 30 seconds
setTimeout(() => {
  if (!receivedInitialLoad) {
    console.log('\nTimeout - no initialGraphLoad received after 30s');
    ws.close();
    process.exit(1);
  }
}, 30000);
