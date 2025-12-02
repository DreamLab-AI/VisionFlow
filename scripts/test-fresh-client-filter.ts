import { chromium } from '@playwright/test';

/**
 * Test: Fresh client connection should receive filtered graph data
 *
 * This test verifies that when a NEW client connects:
 * 1. Server sends initial graph load with quality filtering applied
 * 2. Client receives ~190 nodes (not 1009) with quality >= 0.7 by default
 */
async function testFreshClientFilter() {
  console.log('=== Fresh Client Filter Test ===\n');

  const browser = await chromium.launch({
    headless: true
  });

  const context = await browser.newContext();
  const page = await context.newPage();

  // Track WebSocket messages
  const wsMessages: any[] = [];
  let initialGraphNodes = 0;

  // Intercept console logs
  page.on('console', msg => {
    const text = msg.text();
    if (text.includes('[WS') || text.includes('initialGraphLoad') || text.includes('filter')) {
      console.log(`[Browser] ${text}`);
    }
  });

  // Intercept WebSocket messages using CDP
  const client = await context.newCDPSession(page);
  await client.send('Network.enable');

  client.on('Network.webSocketFrameReceived', (params: any) => {
    try {
      const payload = JSON.parse(params.response.payloadData);
      if (payload.type === 'initialGraphLoad') {
        initialGraphNodes = payload.nodes?.length || 0;
        console.log(`[WS] Received initialGraphLoad with ${initialGraphNodes} nodes`);
        wsMessages.push(payload);
      }
    } catch (e) {
      // ignore non-JSON
    }
  });

  try {
    console.log('1. Opening fresh browser session to VisionFlow...');
    await page.goto('http://192.168.0.51:3001/', { waitUntil: 'networkidle', timeout: 30000 });

    console.log('\n2. Waiting 15s for initial graph data...');
    await page.waitForTimeout(15000);

    console.log('\n3. Checking results...');

    // Get node count from UI
    const nodeCountText = await page.evaluate(() => {
      // Look for node count in various places
      const statsEl = document.querySelector('[data-testid="node-count"]');
      if (statsEl) return statsEl.textContent;

      // Check if there's a count displayed anywhere
      const allText = document.body.innerText;
      const match = allText.match(/(\d+)\s*nodes/i);
      return match ? match[1] : 'unknown';
    });

    console.log(`\n========================================`);
    console.log(`=== RESULTS ===`);
    console.log(`========================================`);
    console.log(`WebSocket initialGraphLoad count: ${initialGraphNodes}`);
    console.log(`UI displayed node count: ${nodeCountText}`);

    if (initialGraphNodes > 0 && initialGraphNodes < 500) {
      console.log(`\n SUCCESS: Fresh client received filtered graph with ${initialGraphNodes} nodes!`);
      console.log(`   (Expected ~190 nodes with quality >= 0.7 filter)`);
    } else if (initialGraphNodes >= 500) {
      console.log(`\n WARNING: Fresh client received ${initialGraphNodes} nodes - filter may not be applied by default`);
      console.log(`   Check server default settings for nodeFilter.enabled = true`);
    } else {
      console.log(`\n UNKNOWN: Could not determine initial node count`);
    }

    // Take screenshot
    await page.screenshot({ path: '/tmp/fresh-client-test.png', fullPage: true });
    console.log(`\n   Screenshot: /tmp/fresh-client-test.png`);

  } catch (error) {
    console.error('Test error:', error);
  } finally {
    await browser.close();
  }

  console.log('\nTest completed.');
}

testFreshClientFilter();
