import { chromium } from 'playwright';

/**
 * Filter test that directly sends WebSocket message
 * This bypasses the React UI entirely and sends the filter message directly
 */
async function testFilterWebSocket() {
  console.log('=== WebSocket Filter Test ===\n');

  const browser = await chromium.launch({
    headless: false,
    args: ['--no-sandbox', '--disable-setuid-sandbox']
  });

  const page = await browser.newPage();
  page.setDefaultTimeout(20000);

  // Track all WebSocket connections
  const wsUrls: string[] = [];

  // Intercept WebSocket creation
  await page.addInitScript(() => {
    // Store reference to original WebSocket
    const OriginalWebSocket = window.WebSocket;
    const webSockets: WebSocket[] = [];
    (window as any).__webSockets = webSockets;

    // Wrap WebSocket constructor
    (window as any).WebSocket = function(url: string, protocols?: string | string[]) {
      console.log('[WebSocket Interceptor] New connection to:', url);
      const ws = new OriginalWebSocket(url, protocols);
      webSockets.push(ws);
      return ws;
    };
    (window as any).WebSocket.prototype = OriginalWebSocket.prototype;
    (window as any).WebSocket.CONNECTING = OriginalWebSocket.CONNECTING;
    (window as any).WebSocket.OPEN = OriginalWebSocket.OPEN;
    (window as any).WebSocket.CLOSING = OriginalWebSocket.CLOSING;
    (window as any).WebSocket.CLOSED = OriginalWebSocket.CLOSED;
  });

  // Capture filter-related console messages
  page.on('console', msg => {
    const text = msg.text();
    if (text.includes('filter') || text.includes('Filter') ||
        text.includes('1009') || text.includes('WebSocket') ||
        text.includes('nodeFilter') || text.includes('SendInitialGraphLoad')) {
      console.log(`[Browser] ${text}`);
    }
  });

  try {
    console.log('1. Navigating to VisionFlow...');
    await page.goto('http://192.168.0.51:3001', { waitUntil: 'domcontentloaded', timeout: 30000 });
    await page.waitForTimeout(12000); // Wait for graph to fully load and WebSocket to connect

    // Dismiss overlays
    for (let i = 0; i < 5; i++) {
      await page.keyboard.press('Escape');
      await page.waitForTimeout(100);
    }

    // Remove overlay elements
    await page.evaluate(() => {
      document.querySelectorAll('.fixed').forEach(el => {
        const text = el.textContent || '';
        if (text.includes('SpaceMouse') || text.includes('Secure')) {
          (el as HTMLElement).style.display = 'none';
        }
      });
    });

    // Get initial node count
    const initialCount = await page.evaluate(() => {
      const match = document.body.innerText.match(/Nodes[:\s]+(\d[\d,]*)/i);
      return match ? parseInt(match[1].replace(/,/g, ''), 10) : null;
    });
    console.log(`\n2. Initial node count: ${initialCount}`);

    // Check WebSocket status and send filter message
    console.log('\n3. Checking WebSocket connections...');
    const wsInfo = await page.evaluate(() => {
      // @ts-ignore
      const webSockets = window.__webSockets || [];
      const wsStatus = webSockets.map((ws: WebSocket, i: number) => ({
        index: i,
        url: ws.url,
        readyState: ws.readyState,
        readyStateLabel: ['CONNECTING', 'OPEN', 'CLOSING', 'CLOSED'][ws.readyState]
      }));
      return {
        count: webSockets.length,
        sockets: wsStatus
      };
    });
    console.log(`   WebSocket info: ${JSON.stringify(wsInfo, null, 2)}`);

    // Send filter message directly through WebSocket
    console.log('\n4. Sending filter_update message...');

    // First, set up a message listener for the response
    await page.evaluate(() => {
      // @ts-ignore
      window.__lastWsMessages = [];
      // @ts-ignore
      const webSockets = window.__webSockets || [];
      const dataWs = webSockets.find((ws: WebSocket) =>
        ws.readyState === WebSocket.OPEN && ws.url.includes('/wss')
      );
      if (dataWs) {
        const originalOnMessage = dataWs.onmessage;
        dataWs.onmessage = (event: MessageEvent) => {
          // @ts-ignore
          window.__lastWsMessages.push(event.data);
          console.log('[WebSocket Response]', typeof event.data === 'string' ? event.data.substring(0, 200) : 'binary data');
          if (originalOnMessage) originalOnMessage.call(dataWs, event);
        };
      }
    });

    const sendResult = await page.evaluate(() => {
      // @ts-ignore
      const webSockets = window.__webSockets || [];

      // Find the /wss WebSocket (the data connection, not vite-hmr)
      const dataWs = webSockets.find((ws: WebSocket) =>
        ws.readyState === WebSocket.OPEN && ws.url.includes('/wss')
      );
      const openWs = dataWs;

      if (!openWs) {
        return { success: false, error: 'No open WebSocket found' };
      }

      // Send the filter update message
      // Server expects 'data' or 'filter' key, not 'payload'
      const filterMessage = JSON.stringify({
        type: 'filter_update',
        data: {
          enabled: true,
          quality_threshold: 0.8,
          authority_threshold: 0.5,
          filter_by_quality: true,
          filter_by_authority: false,
          filter_mode: 'or'
        }
      });

      try {
        openWs.send(filterMessage);
        console.log('[WebSocket Interceptor] Sent filter_update:', filterMessage);
        return {
          success: true,
          wsUrl: openWs.url,
          message: filterMessage
        };
      } catch (e) {
        return { success: false, error: String(e) };
      }
    });
    console.log(`   Send result: ${JSON.stringify(sendResult, null, 2)}`);

    // Wait a moment for response
    await page.waitForTimeout(2000);

    // Check what messages were received
    const wsResponses = await page.evaluate(() => {
      // @ts-ignore
      const messages = window.__lastWsMessages || [];
      return messages.filter((m: string) =>
        typeof m === 'string' && (m.includes('filter') || m.includes('initial_graph'))
      );
    });
    console.log(`\n5. WebSocket responses received: ${wsResponses.length}`);
    for (const resp of wsResponses) {
      console.log(`   Response: ${resp.substring(0, 300)}`);
    }

    console.log('\n5. Waiting 15s for server to process filter and send new data...');
    await page.waitForTimeout(15000);

    // Get final node count
    const finalCount = await page.evaluate(() => {
      const match = document.body.innerText.match(/Nodes[:\s]+(\d[\d,]*)/i);
      return match ? parseInt(match[1].replace(/,/g, ''), 10) : null;
    });

    console.log('\n========================================');
    console.log('=== RESULTS ===');
    console.log('========================================');
    console.log(`Initial: ${initialCount}`);
    console.log(`Final:   ${finalCount}`);

    if (initialCount && finalCount && finalCount < initialCount) {
      console.log(`\n SUCCESS: Reduced by ${initialCount - finalCount} nodes (${((initialCount - finalCount) / initialCount * 100).toFixed(1)}%)`);
    } else {
      console.log('\n No reduction observed');

      // Debug: Check what messages were received
      console.log('\n6. Debugging server response...');

      // Try to check if the UI shows any filter state
      const filterState = await page.evaluate(() => {
        // Check if there's a visible "Nodes: X" that might have changed
        const nodeText = document.body.innerText;
        const allMatches = nodeText.match(/Nodes[:\s]+\d[\d,]*/gi);

        // Also check for any "filtered" indicators
        const hasFiltered = nodeText.toLowerCase().includes('filtered');
        const hasQuality = nodeText.toLowerCase().includes('quality');

        return {
          allNodeMatches: allMatches,
          hasFilteredText: hasFiltered,
          hasQualityText: hasQuality
        };
      });
      console.log(`   Filter state check: ${JSON.stringify(filterState)}`);
    }

  } catch (error) {
    console.error('Test error:', error);
  } finally {
    await browser.close();
  }

  console.log('\nTest completed.');
}

testFilterWebSocket().catch(console.error);
