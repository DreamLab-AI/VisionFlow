import { chromium } from 'playwright';

/**
 * Test filter at 0.5 threshold to verify filtering is working
 * Should show 503 nodes if quality_score exists, 0 if not
 */
async function testFilterThreshold() {
  console.log('=== Filter Threshold Test ===\n');

  const browser = await chromium.launch({
    headless: false,
    args: ['--no-sandbox', '--disable-setuid-sandbox']
  });

  const page = await browser.newPage();
  page.setDefaultTimeout(60000);

  // Track WebSocket
  let wsSocket: any = null;
  let wsResponses: any[] = [];

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
            if (parsed.type === 'initialGraphLoad') {
              console.log('[WS] initialGraphLoad:', parsed.nodes?.length, 'nodes');
            } else if (parsed.type === 'filter_update_success') {
              console.log('[WS] filter_update_success');
            }
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
    if (text.includes('[WS]') || text.includes('initialGraphLoad') || text.includes('nodes')) {
      console.log(`[Browser] ${text}`);
    }
  });

  try {
    console.log('1. Navigating to VisionFlow...');
    await page.goto('http://192.168.0.51:3001', { waitUntil: 'domcontentloaded' });

    console.log('2. Waiting for initial load (15s)...');
    await page.waitForTimeout(15000);

    // Dismiss overlays
    for (let i = 0; i < 3; i++) {
      await page.keyboard.press('Escape');
      await page.waitForTimeout(100);
    }

    // Test multiple thresholds
    const thresholds = [0.0, 0.3, 0.5, 0.8];

    for (const threshold of thresholds) {
      console.log(`\n3. Testing threshold ${threshold}...`);

      // Clear responses
      await page.evaluate(() => {
        (window as any).__wsResponses = [];
      });

      // Send filter_update
      const result = await page.evaluate((th) => {
        const ws = (window as any).__wsSocket;
        if (!ws || ws.readyState !== 1) {
          return { error: 'No WebSocket' };
        }

        ws.send(JSON.stringify({
          type: 'filter_update',
          enabled: true,
          filter_by_quality: true,
          filter_by_authority: false,
          quality_threshold: th,
          authority_threshold: 0.0,
          max_nodes: null,
          filter_mode: 'and'
        }));
        return { success: true };
      }, threshold);

      if (result.error) {
        console.log(`   Error: ${result.error}`);
        continue;
      }

      // Wait for response
      await page.waitForTimeout(5000);

      // Check result
      const responses = await page.evaluate(() => (window as any).__wsResponses || []);
      const graphLoads = responses.filter((r: any) => r.type === 'initialGraphLoad');
      const lastLoad = graphLoads[graphLoads.length - 1];

      console.log(`   Result: ${lastLoad?.nodes?.length ?? 'N/A'} nodes`);
    }

    console.log('\n========================================');
    console.log('=== SUMMARY ===');
    console.log('========================================');
    console.log('If all thresholds return 0 nodes, quality_score is not in node.metadata');
    console.log('If threshold 0.0 returns all nodes, quality_score defaults to 0.5');

  } catch (error) {
    console.error('Test error:', error);
  } finally {
    await browser.close();
  }

  console.log('\nTest completed.');
}

testFilterThreshold().catch(console.error);
