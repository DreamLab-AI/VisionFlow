import { chromium } from 'playwright';

/**
 * Filter test that directly updates Zustand store
 * This bypasses UI interactions and directly triggers the WebSocket filter update
 */
async function testFilterStore() {
  console.log('=== Store-Based Filter Test ===\n');

  const browser = await chromium.launch({
    headless: false,
    args: ['--no-sandbox', '--disable-setuid-sandbox']
  });

  const page = await browser.newPage();
  page.setDefaultTimeout(15000);

  // Capture filter-related console messages
  page.on('console', msg => {
    const text = msg.text();
    if (text.includes('filter') || text.includes('Filter') ||
        text.includes('1009') || text.includes('SendInitialGraphLoad') ||
        text.includes('nodeFilter') || text.includes('subscription') ||
        text.includes('WebSocket') || text.includes('sendMessage')) {
      console.log(`[Browser] ${text}`);
    }
  });

  try {
    console.log('1. Navigating to VisionFlow...');
    await page.goto('http://192.168.0.51:3001', { waitUntil: 'domcontentloaded', timeout: 30000 });
    await page.waitForTimeout(12000); // Wait for graph to fully load

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

    // Check if we can access Zustand store
    console.log('\n3. Checking store access...');
    const storeAccess = await page.evaluate(() => {
      // Try to get the Zustand store - it's typically exposed via React DevTools or window
      // @ts-ignore
      const zustand = window.__ZUSTAND_DEVTOOLS__;
      // @ts-ignore
      const settingsStore = window.useSettingsStore;

      // Try to find it in React Fiber
      const rootElement = document.getElementById('root');
      // @ts-ignore
      const fiberKey = rootElement && Object.keys(rootElement).find(k => k.startsWith('__reactFiber'));

      return {
        hasZustandDevtools: !!zustand,
        hasSettingsStore: !!settingsStore,
        hasFiberRoot: !!fiberKey,
        // Check if there's a global reference
        // @ts-ignore
        hasWebSocketService: !!(window.webSocketService || window.__WS_SERVICE__)
      };
    });
    console.log(`   Store access: ${JSON.stringify(storeAccess)}`);

    // Try to directly update the store by importing it dynamically
    console.log('\n4. Attempting direct store update via window injection...');

    // Inject a script that will update the store
    const updateResult = await page.evaluate(async () => {
      // Find the React root and traverse to find settingsStore
      const rootElement = document.getElementById('root');
      if (!rootElement) return { error: 'No root element' };

      // Try multiple approaches to find and update the store

      // Approach 1: Look for exposed store on window
      // @ts-ignore
      if (window.useSettingsStore) {
        // @ts-ignore
        const state = window.useSettingsStore.getState();
        if (state && state.updateSettings) {
          await state.updateSettings((draft: any) => {
            if (!draft.nodeFilter) draft.nodeFilter = {};
            draft.nodeFilter.enabled = true;
            draft.nodeFilter.filterByQuality = true;
            draft.nodeFilter.qualityThreshold = 0.8;
            draft.nodeFilter.filterByAuthority = false;
            draft.nodeFilter.authorityThreshold = 0.5;
            draft.nodeFilter.filterMode = 'or';
          });
          return {
            success: true,
            method: 'window.useSettingsStore',
            newFilter: state.settings?.nodeFilter
          };
        }
      }

      // Approach 2: Try to find WebSocket service and call sendFilterUpdate directly
      // @ts-ignore
      const wsService = window.webSocketService || window.__WS_SERVICE__;
      if (wsService && typeof wsService.sendFilterUpdate === 'function') {
        wsService.sendFilterUpdate({
          enabled: true,
          filterByQuality: true,
          qualityThreshold: 0.8,
          filterByAuthority: false,
          authorityThreshold: 0.5,
          filterMode: 'or'
        });
        return { success: true, method: 'wsService.sendFilterUpdate' };
      }

      // Approach 3: Try sending raw WebSocket message
      // @ts-ignore
      if (wsService && wsService.sendMessage) {
        wsService.sendMessage('filter_update', {
          enabled: true,
          quality_threshold: 0.8,
          authority_threshold: 0.5,
          filter_by_quality: true,
          filter_by_authority: false,
          filter_mode: 'or'
        });
        return { success: true, method: 'wsService.sendMessage' };
      }

      // Approach 4: Look for any WebSocket and send directly
      // @ts-ignore
      const ws = window.__WS__ || wsService?.socket;
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({
          type: 'filter_update',
          payload: {
            enabled: true,
            quality_threshold: 0.8,
            authority_threshold: 0.5,
            filter_by_quality: true,
            filter_by_authority: false,
            filter_mode: 'or'
          }
        }));
        return { success: true, method: 'raw WebSocket send' };
      }

      return {
        error: 'Could not find store or websocket',
        // @ts-ignore
        windowKeys: Object.keys(window).filter(k =>
          k.includes('store') || k.includes('Store') ||
          k.includes('socket') || k.includes('Socket') ||
          k.includes('WS') || k.includes('zustand')
        ).join(', ')
      };
    });
    console.log(`   Update result: ${JSON.stringify(updateResult, null, 2)}`);

    if (!updateResult.success) {
      // Approach 5: Find the actual toggle button and use Playwright's click
      console.log('\n5. Falling back to Playwright native click...');

      // Switch to Analytics tab first
      const analyticsTab = await page.$('[id*="trigger-analytics"]');
      if (analyticsTab) {
        await analyticsTab.click({ force: true });
        await page.waitForTimeout(1000);
        console.log('   Clicked Analytics tab');
      }

      // Find the filter toggle by its label
      const filterToggle = await page.evaluate(() => {
        const labels = document.querySelectorAll('label');
        for (const label of labels) {
          if (label.textContent?.includes('Filter by Quality')) {
            const container = label.closest('div');
            const button = container?.querySelector('button');
            if (button) {
              const rect = button.getBoundingClientRect();
              return { x: rect.x + rect.width/2, y: rect.y + rect.height/2, found: true };
            }
          }
        }
        return { found: false };
      });

      if (filterToggle.found) {
        console.log(`   Found toggle at (${filterToggle.x}, ${filterToggle.y})`);
        await page.mouse.click(filterToggle.x!, filterToggle.y!);
        await page.waitForTimeout(500);
        console.log('   Clicked toggle with Playwright mouse');

        // Also try to find and set the slider
        const sliderSet = await page.evaluate(() => {
          const sliders = document.querySelectorAll('input[type="range"]');
          for (const slider of sliders) {
            const container = slider.closest('div')?.parentElement;
            if (container?.textContent?.includes('Quality Threshold')) {
              const el = slider as HTMLInputElement;
              // Trigger the React onChange by using the native setter
              const nativeInputValueSetter = Object.getOwnPropertyDescriptor(
                window.HTMLInputElement.prototype, 'value'
              )?.set;
              if (nativeInputValueSetter) {
                nativeInputValueSetter.call(el, '0.8');
                el.dispatchEvent(new Event('input', { bubbles: true }));
                el.dispatchEvent(new Event('change', { bubbles: true }));
                return { set: true, newValue: el.value };
              }
            }
          }
          return { set: false };
        });
        console.log(`   Slider result: ${JSON.stringify(sliderSet)}`);
      }
    }

    console.log('\n6. Waiting 15s for server to apply filter...');
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

      // Debug: Check the current store state
      const currentState = await page.evaluate(() => {
        // @ts-ignore
        const state = window.useSettingsStore?.getState?.();
        return state?.settings?.nodeFilter || 'Store not accessible';
      });
      console.log(`\nCurrent nodeFilter state: ${JSON.stringify(currentState)}`);
    }

  } catch (error) {
    console.error('Test error:', error);
  } finally {
    await browser.close();
  }

  console.log('\nTest completed.');
}

testFilterStore().catch(console.error);
