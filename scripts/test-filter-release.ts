import { chromium } from 'playwright';

/**
 * Test filter with release binary - sends filter_update via WebSocket
 * Expects node count to drop from 1009 to ~190 with quality threshold 0.8
 */
async function testFilterRelease() {
  console.log('=== Filter Test (Release Binary) ===\n');

  const browser = await chromium.launch({
    headless: false,
    args: ['--no-sandbox', '--disable-setuid-sandbox']
  });

  const page = await browser.newPage();
  page.setDefaultTimeout(60000);

  // Track WebSocket messages
  let filterSuccessReceived = false;
  let graphLoadNodes = 0;

  await page.addInitScript(() => {
    const OriginalWebSocket = window.WebSocket;
    (window as any).__wsSocket = null;
    (window as any).__wsResponses = [];

    (window as any).WebSocket = function(url: string, protocols?: string | string[]) {
      console.log('[WS] New connection:', url);
      const ws = new OriginalWebSocket(url, protocols);

      if (url.includes('/wss')) {
        (window as any).__wsSocket = ws;
      }

      ws.addEventListener('message', (event) => {
        if (typeof event.data === 'string') {
          try {
            const parsed = JSON.parse(event.data);
            (window as any).__wsResponses.push(parsed);
            console.log('[WS Response]', parsed.type,
              parsed.type === 'initialGraphLoad' ? `(${parsed.nodes?.length} nodes)` : '');
          } catch {}
        }
      });

      return ws;
    };
    (window as any).WebSocket.prototype = OriginalWebSocket.prototype;
    Object.assign((window as any).WebSocket, OriginalWebSocket);
  });

  // Log browser console
  page.on('console', msg => {
    const text = msg.text();
    if (text.includes('WS') || text.includes('Filter') || text.includes('190') ||
        text.includes('1009') || text.includes('filter') || text.includes('nodes')) {
      console.log(`[Browser] ${text}`);
    }
  });

  try {
    console.log('1. Navigating to VisionFlow...');
    await page.goto('http://192.168.0.51:3001', { waitUntil: 'domcontentloaded' });

    // Wait for initial load
    console.log('2. Waiting for graph data (15s)...');
    await page.waitForTimeout(15000);

    // Dismiss overlays
    for (let i = 0; i < 3; i++) {
      await page.keyboard.press('Escape');
      await page.waitForTimeout(100);
    }

    // Get initial node count from UI
    const initialCount = await page.evaluate(() => {
      const match = document.body.innerText.match(/Nodes[:\s]+([\d,]+)/i);
      return match ? parseInt(match[1].replace(/,/g, ''), 10) : null;
    });
    console.log(`\n3. Initial node count: ${initialCount}`);

    // Send filter_update via WebSocket
    console.log('\n4. Sending filter_update message via WebSocket...');
    const sendResult = await page.evaluate(() => {
      const ws = (window as any).__wsSocket;
      if (!ws || ws.readyState !== 1) {
        return { error: 'No open WebSocket', readyState: ws?.readyState };
      }

      const filterMsg = {
        type: 'filter_update',
        enabled: true,
        filter_by_quality: true,
        filter_by_authority: false,
        quality_threshold: 0.8,
        authority_threshold: 0.5,
        max_nodes: null,
        filter_mode: 'and'
      };

      console.log('[Sending filter_update]', JSON.stringify(filterMsg));
      ws.send(JSON.stringify(filterMsg));
      return { success: true, message: filterMsg };
    });
    console.log(`   Send result:`, JSON.stringify(sendResult));

    // Wait for server to process and send back filtered graph
    console.log('\n5. Waiting 10s for server to send filtered graph...');
    await page.waitForTimeout(10000);

    // Check responses
    const responses = await page.evaluate(() => {
      return (window as any).__wsResponses || [];
    });

    const filterSuccess = responses.find((r: any) => r.type === 'filter_update_success');
    const graphLoads = responses.filter((r: any) => r.type === 'initialGraphLoad');
    const lastGraphLoad = graphLoads[graphLoads.length - 1];

    console.log(`\n6. WebSocket responses:`);
    console.log(`   - filter_update_success: ${filterSuccess ? 'YES' : 'NO'}`);
    console.log(`   - initialGraphLoad messages: ${graphLoads.length}`);
    if (lastGraphLoad) {
      console.log(`   - Last graph load node count: ${lastGraphLoad.nodes?.length}`);
    }

    // Get final node count from UI
    const finalCount = await page.evaluate(() => {
      const match = document.body.innerText.match(/Nodes[:\s]+([\d,]+)/i);
      return match ? parseInt(match[1].replace(/,/g, ''), 10) : null;
    });

    // Take screenshot
    await page.screenshot({ path: '/tmp/filter-release-test.png', fullPage: true });

    console.log('\n========================================');
    console.log('=== RESULTS ===');
    console.log('========================================');
    console.log(`Initial UI count: ${initialCount}`);
    console.log(`Final UI count:   ${finalCount}`);
    console.log(`Last WS graph:    ${lastGraphLoad?.nodes?.length || 'N/A'}`);

    if (lastGraphLoad?.nodes?.length && lastGraphLoad.nodes.length < 500) {
      console.log(`\n SUCCESS: Server sent filtered graph with ${lastGraphLoad.nodes.length} nodes!`);
      console.log(`   (Expected ~190 nodes with quality >= 0.8)`);
    } else if (finalCount && initialCount && finalCount < initialCount) {
      console.log(`\n PARTIAL: UI shows reduction from ${initialCount} to ${finalCount}`);
    } else {
      console.log(`\n ISSUE: No node reduction observed`);

      // Debug: print last few responses
      console.log('\n   Recent WebSocket messages:');
      const recent = responses.slice(-10);
      for (const r of recent) {
        console.log(`   - ${r.type}: ${JSON.stringify(r).substring(0, 100)}`);
      }
    }

    console.log('\n   Screenshot: /tmp/filter-release-test.png');

  } catch (error) {
    console.error('Test error:', error);
    await page.screenshot({ path: '/tmp/filter-release-error.png' });
  } finally {
    await browser.close();
  }

  console.log('\nTest completed.');
}

testFilterRelease().catch(console.error);
