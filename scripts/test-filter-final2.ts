import { chromium } from 'playwright';

/**
 * Final filter test - minimal, robust approach
 */
async function testFilterFinal2() {
  console.log('=== Final Filter Test ===\n');

  const browser = await chromium.launch({
    headless: false,
    args: ['--no-sandbox', '--disable-setuid-sandbox']
  });

  const page = await browser.newPage();
  // Shorter default timeout
  page.setDefaultTimeout(10000);

  page.on('console', msg => {
    const text = msg.text();
    if (text.includes('filter') || text.includes('Filter') ||
        text.includes('1009') || text.includes('filter_update') ||
        text.includes('SendInitialGraphLoad')) {
      console.log(`[Browser] ${text}`);
    }
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

    // Remove overlays via JS
    await page.evaluate(() => {
      document.querySelectorAll('.fixed').forEach(el => {
        const text = el.textContent || '';
        if (text.includes('SpaceMouse') || text.includes('Secure')) {
          (el as HTMLElement).style.display = 'none';
        }
      });
    });

    // Get initial node count
    const initialText = await page.evaluate(() => {
      const match = document.body.innerText.match(/Nodes[:\s]+(\d[\d,]*)/i);
      return match ? match[1] : null;
    });
    const initialCount = initialText ? parseInt(initialText.replace(/,/g, ''), 10) : null;
    console.log(`\n2. Initial node count: ${initialCount}`);

    // Screenshot with short timeout
    await page.screenshot({ path: '/tmp/filter-f2-1.png', timeout: 5000 }).catch(() => console.log('   Screenshot 1 skipped'));

    // Click Analytics tab using evaluate + dispatchEvent
    console.log('\n3. Switching to Analytics tab...');
    const tabSwitched = await page.evaluate(() => {
      const tab = document.querySelector('[id*="trigger-analytics"]') as HTMLElement;
      if (tab) {
        // Try multiple event types
        tab.click();

        // Also try setting focus and triggering keyboard event
        tab.focus();

        return { clicked: true, id: tab.id };
      }
      return { clicked: false };
    });
    console.log(`   Tab click result: ${JSON.stringify(tabSwitched)}`);
    await page.waitForTimeout(1000);

    // Check active tab state
    const activeState = await page.evaluate(() => {
      const analyticsTab = document.querySelector('[id*="trigger-analytics"]');
      const graphTab = document.querySelector('[id*="trigger-graph"]');
      return {
        analyticsState: analyticsTab?.getAttribute('data-state'),
        graphState: graphTab?.getAttribute('data-state'),
        analyticsSelected: analyticsTab?.getAttribute('aria-selected')
      };
    });
    console.log(`   Tab states: ${JSON.stringify(activeState)}`);

    // If still not switched, try direct manipulation of the React state
    if (activeState.analyticsState !== 'active') {
      console.log('\n4. Tab not switching - trying keyboard arrow navigation...');

      // Focus any tab first
      await page.focus('[role="tab"]');
      await page.waitForTimeout(100);

      // Use arrow keys to navigate
      await page.keyboard.press('ArrowRight');
      await page.waitForTimeout(100);
      await page.keyboard.press('ArrowRight');
      await page.waitForTimeout(100);
      await page.keyboard.press('ArrowRight');  // Should reach Analytics (4th tab)
      await page.waitForTimeout(100);

      // Press Enter to activate
      await page.keyboard.press('Enter');
      await page.waitForTimeout(500);

      const newState = await page.evaluate(() => {
        const analyticsTab = document.querySelector('[id*="trigger-analytics"]');
        return analyticsTab?.getAttribute('data-state');
      });
      console.log(`   Analytics tab state after arrow nav: ${newState}`);
    }

    // Check visible content
    const contentVisible = await page.evaluate(() => {
      const text = document.body.innerText;
      return {
        graphViz: text.includes('Graph Visualization'),
        analyticsFiltering: text.includes('Analytics & Filtering'),
        filterByQuality: text.includes('Filter by Quality'),
        qualityThreshold: text.includes('Quality Threshold')
      };
    });
    console.log(`\n5. Visible content: ${JSON.stringify(contentVisible)}`);

    await page.screenshot({ path: '/tmp/filter-f2-2.png', timeout: 5000 }).catch(() => console.log('   Screenshot 2 skipped'));

    if (contentVisible.filterByQuality) {
      console.log('\n6. Filter controls visible! Enabling filter...');

      // Enable the filter toggle
      await page.evaluate(() => {
        const labels = document.querySelectorAll('label');
        for (const label of labels) {
          if (label.textContent?.includes('Filter by Quality')) {
            const btn = label.closest('div')?.querySelector('button');
            if (btn) btn.click();
          }
        }
      });
      await page.waitForTimeout(500);

      // Set threshold
      await page.evaluate(() => {
        const sliders = document.querySelectorAll('input[type="range"]');
        sliders.forEach((s, i) => {
          const container = s.closest('div');
          if (container?.textContent?.includes('Quality')) {
            (s as HTMLInputElement).value = '0.8';
            s.dispatchEvent(new Event('input', { bubbles: true }));
            s.dispatchEvent(new Event('change', { bubbles: true }));
          }
        });
      });

      console.log('\n7. Waiting 15s for filter to apply...');
      await page.waitForTimeout(15000);

      // Get final count
      const finalText = await page.evaluate(() => {
        const match = document.body.innerText.match(/Nodes[:\s]+(\d[\d,]*)/i);
        return match ? match[1] : null;
      });
      const finalCount = finalText ? parseInt(finalText.replace(/,/g, ''), 10) : null;

      console.log('\n========================================');
      console.log('=== RESULTS ===');
      console.log('========================================');
      console.log(`Initial: ${initialCount}`);
      console.log(`Final:   ${finalCount}`);

      if (initialCount && finalCount && finalCount < initialCount) {
        console.log(`\n✅ SUCCESS: Reduced by ${initialCount - finalCount} nodes`);
      } else {
        console.log('\n⚠️ No reduction observed');
      }
    } else {
      console.log('\n❌ Analytics tab content not visible');
      console.log('   The Radix Tabs state is not changing properly.');
      console.log('   This appears to be a Radix UI interaction issue with Playwright.');

      // Debug: what IS visible
      const headers = await page.evaluate(() => {
        return Array.from(document.querySelectorAll('h3')).map(h => h.textContent?.trim());
      });
      console.log(`   Visible headers: ${JSON.stringify(headers)}`);
    }

    await page.screenshot({ path: '/tmp/filter-f2-3.png', timeout: 5000 }).catch(() => console.log('   Screenshot 3 skipped'));

  } catch (error) {
    console.error('Test error:', error);
  } finally {
    await browser.close();
  }

  console.log('\nTest completed.');
}

testFilterFinal2().catch(console.error);
