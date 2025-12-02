import { chromium } from 'playwright';

/**
 * Debug filter test - checks if settings store is being updated
 */
async function testFilterDebug() {
  console.log('=== Filter Debug Test ===\n');

  const browser = await chromium.launch({
    headless: false,
    args: ['--no-sandbox', '--disable-setuid-sandbox']
  });

  const page = await browser.newPage();
  page.setDefaultTimeout(10000);

  // Capture ALL console messages for debugging
  page.on('console', msg => {
    const text = msg.text();
    console.log(`[Browser] ${text}`);
  });

  try {
    // Navigate
    console.log('1. Navigating...');
    await page.goto('http://192.168.0.51:3001', { waitUntil: 'domcontentloaded', timeout: 30000 });
    await page.waitForTimeout(10000);

    // Dismiss overlays
    for (let i = 0; i < 5; i++) {
      await page.keyboard.press('Escape');
      await page.waitForTimeout(100);
    }

    // Remove overlay elements
    await page.evaluate(() => {
      document.querySelectorAll('.fixed').forEach(el => {
        if (el.textContent?.includes('SpaceMouse') || el.textContent?.includes('Secure')) {
          (el as HTMLElement).style.display = 'none';
        }
      });
    });

    // Get initial state
    const initialState = await page.evaluate(() => {
      // @ts-ignore
      const store = window.__ZUSTAND_STORE__ || window.useSettingsStore?.getState?.();
      if (store) {
        return {
          hasStore: true,
          nodeFilter: store.settings?.nodeFilter
        };
      }
      return { hasStore: false };
    });
    console.log(`\n2. Initial store state: ${JSON.stringify(initialState)}`);

    // Switch to Analytics tab
    console.log('\n3. Switching to Analytics tab...');
    await page.evaluate(() => {
      const tab = document.querySelector('[id*="trigger-analytics"]') as HTMLElement;
      if (tab) tab.click();
    });
    await page.waitForTimeout(1000);

    // Check for filter controls
    console.log('\n4. Checking for filter controls...');
    const hasFilterLabel = await page.evaluate(() => {
      const text = document.body.innerText;
      return text.includes('Filter by Quality');
    });
    console.log(`   Filter by Quality visible: ${hasFilterLabel}`);

    if (hasFilterLabel) {
      // Check the current state of the toggle
      const beforeToggle = await page.evaluate(() => {
        const labels = document.querySelectorAll('label');
        for (const label of labels) {
          if (label.textContent?.includes('Filter by Quality')) {
            const container = label.closest('div');
            const button = container?.querySelector('button');
            if (button) {
              const bg = getComputedStyle(button).background;
              const isOn = bg.includes('16, 185, 129') || bg.includes('10b981'); // green
              return {
                found: true,
                buttonStyle: bg.substring(0, 100),
                isOn
              };
            }
          }
        }
        return { found: false };
      });
      console.log(`\n5. Toggle state before click: ${JSON.stringify(beforeToggle)}`);

      // Click the toggle using Playwright's click (not evaluate)
      console.log('\n6. Clicking toggle with Playwright...');

      // First, find the exact toggle button position
      const toggleHandle = await page.evaluateHandle(() => {
        const labels = document.querySelectorAll('label');
        for (const label of labels) {
          if (label.textContent?.includes('Filter by Quality')) {
            const container = label.closest('div');
            return container?.querySelector('button');
          }
        }
        return null;
      });

      if (toggleHandle) {
        const box = await toggleHandle.asElement()?.boundingBox();
        console.log(`   Toggle bounding box: ${JSON.stringify(box)}`);

        if (box) {
          // Click at the center of the toggle
          await page.mouse.click(box.x + box.width / 2, box.y + box.height / 2);
          console.log('   Clicked toggle');
        }
      }
      await page.waitForTimeout(500);

      // Check state after toggle click
      const afterToggle = await page.evaluate(() => {
        // @ts-ignore
        const store = window.__ZUSTAND_STORE__ || window.useSettingsStore?.getState?.();
        const labels = document.querySelectorAll('label');
        let buttonState = {};
        for (const label of labels) {
          if (label.textContent?.includes('Filter by Quality')) {
            const container = label.closest('div');
            const button = container?.querySelector('button');
            if (button) {
              const bg = getComputedStyle(button).background;
              buttonState = {
                buttonBg: bg.substring(0, 100),
                isGreen: bg.includes('16, 185, 129') || bg.includes('10b981')
              };
            }
          }
        }
        return {
          storeFilter: store?.settings?.nodeFilter,
          buttonState
        };
      });
      console.log(`\n7. State after toggle: ${JSON.stringify(afterToggle, null, 2)}`);

      // Now set the threshold slider
      console.log('\n8. Setting quality threshold...');

      // Find and click/drag the slider
      const sliderResult = await page.evaluate(() => {
        const sliders = document.querySelectorAll('input[type="range"]');
        for (const slider of sliders) {
          const container = slider.closest('div')?.parentElement;
          if (container?.textContent?.includes('Quality Threshold')) {
            // Set value programmatically
            (slider as HTMLInputElement).value = '0.8';

            // Fire all possible events
            slider.dispatchEvent(new Event('input', { bubbles: true }));
            slider.dispatchEvent(new Event('change', { bubbles: true }));

            // Also try with InputEvent
            slider.dispatchEvent(new InputEvent('input', { bubbles: true }));

            return { set: true, newValue: (slider as HTMLInputElement).value };
          }
        }
        return { set: false };
      });
      console.log(`   Slider result: ${JSON.stringify(sliderResult)}`);

      // Check store state after slider change
      await page.waitForTimeout(500);
      const afterSlider = await page.evaluate(() => {
        // @ts-ignore
        const store = window.__ZUSTAND_STORE__ || window.useSettingsStore?.getState?.();
        return store?.settings?.nodeFilter;
      });
      console.log(`\n9. Store nodeFilter after slider: ${JSON.stringify(afterSlider)}`);

      // Check if WebSocket is connected and send message manually
      console.log('\n10. Checking WebSocket and sending filter manually...');
      const wsResult = await page.evaluate(() => {
        // @ts-ignore
        const wsService = window.webSocketService || window.__WS_SERVICE__;
        if (wsService && wsService.isConnected) {
          // @ts-ignore
          wsService.sendFilterUpdate({
            enabled: true,
            filterByQuality: true,
            qualityThreshold: 0.8,
            filterByAuthority: false,
            authorityThreshold: 0.5,
            filterMode: 'or'
          });
          return { sent: true };
        }
        return {
          wsFound: !!wsService,
          isConnected: wsService?.isConnected
        };
      });
      console.log(`   WebSocket result: ${JSON.stringify(wsResult)}`);

      // Wait for server response
      console.log('\n11. Waiting 10s for server...');
      await page.waitForTimeout(10000);

      // Check final node count
      const finalCount = await page.evaluate(() => {
        const text = document.body.innerText;
        const match = text.match(/Nodes[:\s]+(\d[\d,]*)/i);
        return match ? match[1] : null;
      });
      console.log(`\n12. Final node count: ${finalCount}`);
    }

  } catch (error) {
    console.error('Test error:', error);
  } finally {
    await browser.close();
  }

  console.log('\nTest completed.');
}

testFilterDebug().catch(console.error);
