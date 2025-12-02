import { chromium } from 'playwright';

/**
 * Test filter by directly updating the Zustand store
 * This triggers the proper WebSocket message flow through WebSocketService
 */
async function testFilterViaStore() {
  console.log('=== Store-Based Filter Test v2 ===\n');

  const browser = await chromium.launch({
    headless: false,
    args: ['--no-sandbox', '--disable-setuid-sandbox']
  });

  const page = await browser.newPage();
  page.setDefaultTimeout(30000);

  // Track WebSocket messages
  await page.addInitScript(() => {
    const OriginalWebSocket = window.WebSocket;
    (window as any).__wsMessages = [];
    (window as any).WebSocket = function(url: string, protocols?: string | string[]) {
      console.log('[WS Interceptor] New connection:', url);
      const ws = new OriginalWebSocket(url, protocols);

      // Intercept messages
      ws.addEventListener('message', (event) => {
        if (typeof event.data === 'string') {
          try {
            const parsed = JSON.parse(event.data);
            if (parsed.type) {
              console.log('[WS Response]', parsed.type, JSON.stringify(parsed).substring(0, 150));
              (window as any).__wsMessages.push(parsed);
            }
          } catch {}
        }
      });

      return ws;
    };
    (window as any).WebSocket.prototype = OriginalWebSocket.prototype;
    (window as any).WebSocket.CONNECTING = OriginalWebSocket.CONNECTING;
    (window as any).WebSocket.OPEN = OriginalWebSocket.OPEN;
    (window as any).WebSocket.CLOSING = OriginalWebSocket.CLOSING;
    (window as any).WebSocket.CLOSED = OriginalWebSocket.CLOSED;
  });

  // Log relevant console messages
  page.on('console', msg => {
    const text = msg.text();
    if (text.includes('1009') || text.includes('filter') || text.includes('Filter') ||
        text.includes('WS ') || text.includes('WebSocket') || text.includes('nodeFilter') ||
        text.includes('190') || text.includes('503')) {
      console.log(`[Browser] ${text}`);
    }
  });

  try {
    console.log('1. Navigating to VisionFlow...');
    await page.goto('http://192.168.0.51:3001', { waitUntil: 'domcontentloaded', timeout: 30000 });

    // Wait longer for full initialization including WebSocket registration
    console.log('2. Waiting for full initialization (15s)...');
    await page.waitForTimeout(15000);

    // Dismiss overlays
    for (let i = 0; i < 5; i++) {
      await page.keyboard.press('Escape');
      await page.waitForTimeout(100);
    }

    // Get initial node count
    const initialCount = await page.evaluate(() => {
      const match = document.body.innerText.match(/Nodes[:\s]+(\d[\d,]*)/i);
      return match ? parseInt(match[1].replace(/,/g, ''), 10) : null;
    });
    console.log(`\n3. Initial node count: ${initialCount}`);

    // Check if store is exposed
    const storeExposed = await page.evaluate(() => {
      return typeof (window as any).useSettingsStore !== 'undefined';
    });
    console.log(`4. Store exposed: ${storeExposed}`);

    if (!storeExposed) {
      console.log('   ERROR: useSettingsStore not exposed on window');
      console.log('   Make sure client/src/app/main.tsx exposes the store in dev mode');
      return;
    }

    // Get current filter state
    const currentFilter = await page.evaluate(() => {
      const store = (window as any).useSettingsStore;
      if (!store) return null;
      const state = store.getState();
      return state.settings?.nodeFilter || null;
    });
    console.log(`5. Current filter state: ${JSON.stringify(currentFilter, null, 2)}`);

    // Update filter settings via store
    console.log('\n6. Updating filter via store...');
    const updateResult = await page.evaluate(() => {
      const store = (window as any).useSettingsStore;
      if (!store) return { error: 'Store not found' };

      try {
        // Get the store state and use updateSettingByPath
        const state = store.getState();

        // Update the filter settings
        state.updateSettingByPath('nodeFilter.enabled', true, { path: 'nodeFilter.enabled' });
        state.updateSettingByPath('nodeFilter.filterByQuality', true, { path: 'nodeFilter.filterByQuality' });
        state.updateSettingByPath('nodeFilter.qualityThreshold', 0.8, { path: 'nodeFilter.qualityThreshold' });

        // Get updated state
        const newState = store.getState();
        return {
          success: true,
          newFilter: newState.settings?.nodeFilter
        };
      } catch (e) {
        return { error: String(e) };
      }
    });
    console.log(`   Update result: ${JSON.stringify(updateResult, null, 2)}`);

    // Wait for WebSocket to send the filter update and server to respond
    console.log('\n7. Waiting 10s for server to process filter...');
    await page.waitForTimeout(10000);

    // Check if we received filter_update_success
    const wsMessages = await page.evaluate(() => {
      return (window as any).__wsMessages || [];
    });

    const filterSuccess = wsMessages.find((m: any) => m.type === 'filter_update_success');
    const filterErrors = wsMessages.filter((m: any) =>
      m.type === 'error' && m.message?.includes('filter')
    );

    console.log(`8. Filter success response: ${filterSuccess ? 'YES' : 'NO'}`);
    if (filterSuccess) {
      console.log(`   Response: ${JSON.stringify(filterSuccess)}`);
    }
    if (filterErrors.length > 0) {
      console.log(`   Filter errors: ${JSON.stringify(filterErrors)}`);
    }

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

      // Debug: List all WS messages
      console.log('\n9. All WebSocket messages received:');
      for (const msg of wsMessages.slice(-20)) {
        console.log(`   - ${msg.type}: ${JSON.stringify(msg).substring(0, 100)}`);
      }
    }

  } catch (error) {
    console.error('Test error:', error);
  } finally {
    await browser.close();
  }

  console.log('\nTest completed.');
}

testFilterViaStore().catch(console.error);
