import { chromium } from 'playwright';

/**
 * End-to-end test for quality-based node filtering.
 *
 * This test verifies that:
 * 1. Server-side filtering works (node count decreases)
 * 2. UI controls are accessible via Analytics > Filter tab
 * 3. Quality threshold slider affects node count
 */
async function testFilterE2E() {
  console.log('=== E2E Filter Verification Test ===\n');

  const browser = await chromium.launch({
    headless: false,
    args: ['--no-sandbox', '--disable-setuid-sandbox']
  });

  const page = await browser.newPage();

  // Collect console logs
  const consoleLogs: string[] = [];
  page.on('console', msg => {
    const text = msg.text();
    consoleLogs.push(text);
    if (text.includes('filter') || text.includes('Filter') ||
        text.includes('nodes') || text.includes('WebSocket') ||
        text.includes('filter_update') || text.includes('SendInitialGraphLoad')) {
      console.log(`[Browser] ${text}`);
    }
  });

  try {
    // Step 1: Navigate and wait for graph load
    console.log('1. Navigating to VisionFlow...');
    await page.goto('http://192.168.0.51:3001', { waitUntil: 'domcontentloaded', timeout: 30000 });
    await page.waitForTimeout(8000); // Wait for graph to load

    // Dismiss overlays
    console.log('2. Dismissing overlays...');
    for (let i = 0; i < 3; i++) {
      await page.keyboard.press('Escape');
      await page.waitForTimeout(200);
    }

    // Remove any warning banners by CSS selector
    await page.evaluate(() => {
      // Remove SpaceMouse warning
      document.querySelectorAll('.fixed.top-0').forEach(el => {
        if (el.textContent?.includes('SpaceMouse') || el.textContent?.includes('Connection')) {
          (el as HTMLElement).style.display = 'none';
        }
      });
      // Remove any orange/red gradient banners
      document.querySelectorAll('[class*="from-orange"], [class*="from-red"]').forEach(el => {
        (el as HTMLElement).style.display = 'none';
      });
    });
    await page.waitForTimeout(500);

    // Helper to get node count from the page
    const getNodeCount = async (): Promise<number | null> => {
      return await page.evaluate(() => {
        // Look for node count display in the UI
        const text = document.body.innerText;
        // Pattern: "Nodes: 123" or "123 nodes" or similar
        const patterns = [
          /Nodes[:\s]+([0-9,]+)/i,
          /([0-9,]+)\s*nodes/i,
          /node count[:\s]+([0-9,]+)/i
        ];
        for (const pattern of patterns) {
          const match = text.match(pattern);
          if (match) {
            return parseInt(match[1].replace(/,/g, ''), 10);
          }
        }
        return null;
      });
    };

    // Step 3: Get initial node count
    const initialCount = await getNodeCount();
    console.log(`\n3. Initial node count: ${initialCount}`);
    await page.screenshot({ path: '/tmp/filter-e2e-1-initial.png' });

    // Step 4: Click Analytics tab in Control Center using Playwright locator
    console.log('\n4. Clicking Analytics tab...');

    // Use Playwright's native click - more reliable than evaluate()
    // Look for the tab trigger with "Analytics" text
    try {
      // Wait for Control Center to be visible
      await page.waitForSelector('.control-center', { timeout: 10000 });

      // Click the Analytics tab using multiple selector strategies
      const analyticsTab = page.locator('[role="tab"]').filter({ hasText: 'Analytics' }).first();
      if (await analyticsTab.count() > 0) {
        await analyticsTab.click();
        console.log('   Result: Clicked Analytics tab via Playwright locator');
      } else {
        // Try by value attribute
        const byValue = page.locator('button[value="analytics"]');
        if (await byValue.count() > 0) {
          await byValue.click();
          console.log('   Result: Clicked button[value="analytics"]');
        } else {
          // Debug: list all tabs
          const allTabs = await page.evaluate(() => {
            const tabs = document.querySelectorAll('[role="tab"]');
            return Array.from(tabs).map(t => ({
              text: t.textContent?.trim(),
              value: t.getAttribute('value'),
              dataState: t.getAttribute('data-state')
            }));
          });
          console.log('   Available tabs:', JSON.stringify(allTabs, null, 2));
        }
      }
    } catch (e) {
      console.log(`   Error clicking Analytics: ${e}`);
    }
    await page.waitForTimeout(2000);
    await page.screenshot({ path: '/tmp/filter-e2e-2-analytics.png' });

    // Step 5: The SemanticAnalysisPanel should now be visible with "filter" tab already active
    // Check if filter controls are visible
    console.log('\n5. Checking for filter controls...');
    const filterControlsStatus = await page.evaluate(() => {
      const text = document.body.innerText;
      return {
        hasFilterByQuality: text.includes('Filter by Quality'),
        hasQualityThreshold: text.includes('Quality Threshold'),
        hasAuthorityThreshold: text.includes('Authority Threshold'),
        hasSemanticAnalysis: text.includes('Semantic Analysis'),
        hasFilterTab: !!document.querySelector('[value="filter"]')
      };
    });
    console.log(`   Controls visible: ${JSON.stringify(filterControlsStatus, null, 2)}`);

    // If filter tab is not active, click it
    if (filterControlsStatus.hasFilterTab && !filterControlsStatus.hasFilterByQuality) {
      console.log('   Clicking filter sub-tab...');
      await page.click('[value="filter"]');
      await page.waitForTimeout(500);
    }

    await page.screenshot({ path: '/tmp/filter-e2e-3-filter-tab.png' });

    // Step 6: Enable "Filter by Quality" if not already enabled
    console.log('\n6. Enabling Filter by Quality...');
    const switchResult = await page.evaluate(() => {
      // Find the switch next to "Filter by Quality" label
      const labels = document.querySelectorAll('label');
      for (const label of labels) {
        if (label.textContent?.includes('Filter by Quality')) {
          // Find the nearest switch (sibling or in same parent)
          const parent = label.closest('.flex, div');
          const switchEl = parent?.querySelector('[role="switch"]');
          if (switchEl) {
            const isChecked = switchEl.getAttribute('data-state') === 'checked' ||
                              switchEl.getAttribute('aria-checked') === 'true';
            if (!isChecked) {
              (switchEl as HTMLElement).click();
              return 'Enabled Filter by Quality';
            }
            return 'Filter by Quality already enabled';
          }
        }
      }
      // Fallback: find any switch that is not checked
      const switches = document.querySelectorAll('[role="switch"]');
      for (const sw of switches) {
        const isChecked = sw.getAttribute('data-state') === 'checked';
        if (!isChecked) {
          (sw as HTMLElement).click();
          return 'Clicked first unchecked switch';
        }
      }
      return 'No switch found';
    });
    console.log(`   Result: ${switchResult}`);
    await page.waitForTimeout(1000);

    // Step 7: Set quality threshold to a high value (0.8) to filter more nodes
    console.log('\n7. Setting quality threshold to 0.85...');
    const sliderResult = await page.evaluate(() => {
      // Find the slider for quality threshold
      const sliders = document.querySelectorAll('[role="slider"], input[type="range"]');
      const labels = document.querySelectorAll('label');

      // Look for the quality threshold slider
      for (const label of labels) {
        if (label.textContent?.includes('Quality Threshold')) {
          const parent = label.closest('.space-y-2, .space-y-3, div');
          const slider = parent?.querySelector('[role="slider"], input[type="range"]');
          if (slider) {
            // For Radix slider, we need to use keyboard or click-drag
            // For now, click at 85% position
            const rect = (slider as HTMLElement).getBoundingClientRect();
            if (rect.width > 0) {
              // Calculate 85% position
              const clickX = rect.left + (rect.width * 0.85);
              const clickY = rect.top + (rect.height / 2);

              // Dispatch mousedown, mousemove, mouseup sequence
              const mousedown = new MouseEvent('mousedown', { clientX: clickX, clientY: clickY, bubbles: true });
              const mouseup = new MouseEvent('mouseup', { clientX: clickX, clientY: clickY, bubbles: true });
              slider.dispatchEvent(mousedown);
              slider.dispatchEvent(mouseup);

              return 'Clicked quality threshold slider at 85%';
            }
          }
        }
      }
      return 'Quality threshold slider not found';
    });
    console.log(`   Result: ${sliderResult}`);
    await page.waitForTimeout(1000);
    await page.screenshot({ path: '/tmp/filter-e2e-4-filter-enabled.png' });

    // Step 8: Wait for server to apply filter (WebSocket round-trip)
    console.log('\n8. Waiting for server to apply filter (12 seconds)...');
    await page.waitForTimeout(12000);

    // Step 9: Get new node count
    const filteredCount = await getNodeCount();
    console.log(`\n9. Filtered node count: ${filteredCount}`);
    await page.screenshot({ path: '/tmp/filter-e2e-5-filtered.png' });

    // Results summary
    console.log('\n========================================');
    console.log('=== RESULTS ===');
    console.log('========================================');
    console.log(`Initial nodes:  ${initialCount}`);
    console.log(`Filtered nodes: ${filteredCount}`);

    if (initialCount !== null && filteredCount !== null) {
      const reduction = initialCount - filteredCount;
      const percentage = ((reduction / initialCount) * 100).toFixed(1);
      console.log(`Reduction:      ${reduction} nodes (${percentage}%)`);

      if (filteredCount < initialCount) {
        console.log('\n✅ SUCCESS: Node count DECREASED!');
        console.log('   Server-side per-client filtering is working correctly.');
      } else if (filteredCount === initialCount) {
        console.log('\n⚠️  WARNING: Node count unchanged');
        console.log('   Possible causes:');
        console.log('   1. Filter controls not found/clicked correctly');
        console.log('   2. Server not receiving filter_update messages');
        console.log('   3. All nodes have quality >= threshold');
      }
    } else {
      console.log('\n❌ ERROR: Could not read node counts');
    }

    // Step 10: Debug info
    console.log('\n10. Debug: Page state...');
    const debugInfo = await page.evaluate(() => {
      const text = document.body.innerText;
      return {
        hasNodesLabel: text.includes('Nodes'),
        hasFilterByQuality: text.includes('Filter by Quality'),
        hasQualityThreshold: text.includes('Quality Threshold'),
        connectionFailed: text.includes('Connection to Backend Failed'),
        visibleSwitches: document.querySelectorAll('[role="switch"]').length,
        visibleSliders: document.querySelectorAll('[role="slider"]').length,
        // Find any node count displays
        nodeTexts: Array.from(text.matchAll(/\d+\s*nodes?|\bNodes?[:\s]+\d+/gi)).map(m => m[0])
      };
    });
    console.log(`   ${JSON.stringify(debugInfo, null, 2)}`);

    // Check for filter-related console logs
    const filterLogs = consoleLogs.filter(l =>
      l.toLowerCase().includes('filter') ||
      l.includes('SendInitialGraphLoad') ||
      l.includes('filter_update')
    );
    if (filterLogs.length > 0) {
      console.log('\n11. Filter-related console logs:');
      filterLogs.slice(-10).forEach(log => console.log(`   ${log}`));
    }

    await page.screenshot({ path: '/tmp/filter-e2e-6-final.png', fullPage: true });

  } catch (error) {
    console.error('Test error:', error);
    await page.screenshot({ path: '/tmp/filter-e2e-error.png' });
  } finally {
    await browser.close();
  }

  console.log('\nTest completed. Check /tmp/filter-e2e-*.png');
}

testFilterE2E().catch(console.error);
