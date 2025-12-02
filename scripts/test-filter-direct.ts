import { chromium } from 'playwright';

/**
 * Direct filter test - focuses on clicking the Analytics tab correctly
 */
async function testFilterDirect() {
  console.log('=== Direct Filter Test ===\n');

  const browser = await chromium.launch({
    headless: false,
    args: ['--no-sandbox', '--disable-setuid-sandbox']
  });

  const page = await browser.newPage();

  page.on('console', msg => {
    const text = msg.text();
    if (text.includes('filter') || text.includes('Filter') ||
        text.includes('nodes') || text.includes('1009') ||
        text.includes('filter_update')) {
      console.log(`[Browser] ${text}`);
    }
  });

  try {
    // Navigate
    console.log('1. Navigating to VisionFlow...');
    await page.goto('http://192.168.0.51:3001', { waitUntil: 'domcontentloaded', timeout: 30000 });
    await page.waitForTimeout(10000);  // Wait for graph to fully load

    // Dismiss overlays
    for (let i = 0; i < 5; i++) {
      await page.keyboard.press('Escape');
      await page.waitForTimeout(100);
    }

    // Hide warning banners via JS
    await page.evaluate(() => {
      document.querySelectorAll('[class*="from-orange"], [class*="from-red"], [class*="fixed"]').forEach(el => {
        if (el.textContent?.includes('SpaceMouse') || el.textContent?.includes('Secure Context')) {
          (el as HTMLElement).style.display = 'none';
        }
      });
    });
    await page.waitForTimeout(500);

    await page.screenshot({ path: '/tmp/filter-direct-1-loaded.png' });

    // Get initial node count
    let initialCount: number | null = null;
    const nodeCountText = await page.evaluate(() => {
      // Look for the node count in System Status
      const text = document.body.innerText;
      const match = text.match(/Nodes[:\s]+(\d[\d,]*)/i);
      return match ? match[1] : null;
    });
    if (nodeCountText) {
      initialCount = parseInt(nodeCountText.replace(/,/g, ''), 10);
    }
    console.log(`2. Initial node count: ${initialCount}`);

    // Find and click the Analytics tab
    console.log('\n3. Finding Analytics tab...');

    // Debug: list all tab buttons
    const tabInfo = await page.evaluate(() => {
      const results: any[] = [];
      // Look for buttons that look like tabs
      const buttons = document.querySelectorAll('button');
      buttons.forEach((btn, i) => {
        const text = btn.textContent?.trim();
        const value = btn.getAttribute('value');
        const role = btn.getAttribute('role');
        const dataState = btn.getAttribute('data-state');
        if (role === 'tab' || value || text?.match(/^(Graph|Physics|Effects|Analytics|Quality|System)$/)) {
          results.push({ index: i, text, value, role, dataState });
        }
      });
      return results;
    });
    console.log('   Tab buttons found:', JSON.stringify(tabInfo, null, 2));

    // Click Analytics tab using the BarChart3 icon container or text
    console.log('\n4. Clicking Analytics tab...');
    const clickResult = await page.evaluate(() => {
      // Strategy 1: Find by button value="analytics"
      const byValue = document.querySelector('button[value="analytics"]') as HTMLElement;
      if (byValue) {
        byValue.click();
        return 'clicked button[value="analytics"]';
      }

      // Strategy 2: Find by role="tab" with Analytics text
      const tabs = document.querySelectorAll('[role="tab"]');
      for (const tab of tabs) {
        if (tab.textContent?.includes('Analytics')) {
          (tab as HTMLElement).click();
          return 'clicked [role="tab"] with Analytics text';
        }
      }

      // Strategy 3: Find by buttonKey "4" (from config)
      const buttons = document.querySelectorAll('button');
      for (const btn of buttons) {
        if (btn.textContent?.trim() === 'Analytics' ||
            btn.textContent?.includes('Analytics')) {
          (btn as HTMLElement).click();
          return 'clicked button with Analytics text';
        }
      }

      return 'Analytics tab not found';
    });
    console.log(`   Result: ${clickResult}`);
    await page.waitForTimeout(1000);

    await page.screenshot({ path: '/tmp/filter-direct-2-after-click.png' });

    // Verify what content is now visible
    console.log('\n5. Checking visible content...');
    const contentCheck = await page.evaluate(() => {
      const text = document.body.innerText;
      return {
        hasGraphVisualization: text.includes('Graph Visualization'),
        hasAnalyticsFiltering: text.includes('Analytics & Filtering'),
        hasFilterByQuality: text.includes('Filter by Quality'),
        hasQualityThreshold: text.includes('Quality Threshold'),
        activeTab: document.querySelector('[role="tab"][data-state="active"]')?.textContent?.trim(),
        // Look for toggle switches
        toggleCount: document.querySelectorAll('button[style*="border-radius: 9"]').length,
        sliderCount: document.querySelectorAll('input[type="range"]').length
      };
    });
    console.log('   Content check:', JSON.stringify(contentCheck, null, 2));

    // If we see Filter by Quality, try to enable it
    if (contentCheck.hasFilterByQuality) {
      console.log('\n6. Found filter controls! Enabling Filter by Quality...');

      const toggleResult = await page.evaluate(() => {
        // Find the toggle for "Filter by Quality"
        const labels = document.querySelectorAll('label');
        for (const label of labels) {
          if (label.textContent?.includes('Filter by Quality')) {
            const container = label.closest('div');
            const toggle = container?.querySelector('button');
            if (toggle) {
              toggle.click();
              return 'toggled Filter by Quality';
            }
          }
        }
        return 'toggle not found';
      });
      console.log(`   Toggle result: ${toggleResult}`);
      await page.waitForTimeout(500);

      // Set quality threshold to 0.8
      console.log('\n7. Setting Quality Threshold to 0.8...');
      const sliderResult = await page.evaluate(() => {
        const labels = document.querySelectorAll('label');
        for (const label of labels) {
          if (label.textContent?.includes('Quality Threshold')) {
            const container = label.closest('div')?.parentElement;
            const slider = container?.querySelector('input[type="range"]');
            if (slider) {
              (slider as HTMLInputElement).value = '0.8';
              slider.dispatchEvent(new Event('input', { bubbles: true }));
              slider.dispatchEvent(new Event('change', { bubbles: true }));
              return 'set Quality Threshold to 0.8';
            }
          }
        }
        return 'slider not found';
      });
      console.log(`   Slider result: ${sliderResult}`);

      await page.screenshot({ path: '/tmp/filter-direct-3-filter-enabled.png' });

      // Wait for server to process
      console.log('\n8. Waiting for filter to apply (15s)...');
      await page.waitForTimeout(15000);

      // Get new node count
      const finalCountText = await page.evaluate(() => {
        const text = document.body.innerText;
        const match = text.match(/Nodes[:\s]+(\d[\d,]*)/i);
        return match ? match[1] : null;
      });
      const finalCount = finalCountText ? parseInt(finalCountText.replace(/,/g, ''), 10) : null;
      console.log(`\n9. Final node count: ${finalCount}`);

      await page.screenshot({ path: '/tmp/filter-direct-4-final.png' });

      // Results
      console.log('\n========================================');
      console.log('=== RESULTS ===');
      console.log('========================================');
      console.log(`Initial nodes:  ${initialCount}`);
      console.log(`Final nodes:    ${finalCount}`);

      if (initialCount && finalCount && finalCount < initialCount) {
        const reduction = initialCount - finalCount;
        const pct = ((reduction / initialCount) * 100).toFixed(1);
        console.log(`\n✅ SUCCESS: Nodes reduced by ${reduction} (${pct}%)`);
      } else if (initialCount === finalCount) {
        console.log('\n⚠️  Node count unchanged');
      } else {
        console.log('\n❌ Could not verify node count change');
      }
    } else {
      console.log('\n❌ Filter controls not visible after clicking Analytics');
      console.log('   The tab content did not switch properly');

      // Try clicking again with force
      await page.click('button[value="analytics"]', { force: true }).catch(() => {});
      await page.waitForTimeout(1000);
      await page.screenshot({ path: '/tmp/filter-direct-5-retry.png' });
    }

  } catch (error) {
    console.error('Test error:', error);
    await page.screenshot({ path: '/tmp/filter-direct-error.png' });
  } finally {
    await browser.close();
  }

  console.log('\nTest completed. Check /tmp/filter-direct-*.png');
}

testFilterDirect().catch(console.error);
