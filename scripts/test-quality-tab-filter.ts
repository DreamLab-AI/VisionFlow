import { chromium } from 'playwright';

async function testQualityTabFilter() {
  console.log('=== Quality Tab Filter Verification Test ===\n');
  console.log('This test clicks the Quality tab and adjusts filter settings.\n');

  const browser = await chromium.launch({
    headless: false,
    args: ['--no-sandbox', '--disable-setuid-sandbox']
  });

  const page = await browser.newPage();

  // Track console logs
  const logs: string[] = [];
  page.on('console', msg => {
    const text = msg.text();
    logs.push(`[${msg.type()}] ${text}`);
    if (text.includes('Filter') || text.includes('filter') ||
        text.includes('nodes') || text.includes('Nodes') ||
        text.includes('sent to server') || text.includes('InitialGraphLoad')) {
      console.log(`>>> ${text}`);
    }
  });

  try {
    console.log('1. Navigating to VisionFlow...');
    await page.goto('http://192.168.0.51:3001', { waitUntil: 'domcontentloaded', timeout: 30000 });
    await page.waitForTimeout(6000);

    // Dismiss any dialogs
    await page.keyboard.press('Escape');
    await page.waitForTimeout(500);

    await page.screenshot({ path: '/tmp/quality-tab-1-initial.png' });
    console.log('   Screenshot 1: Initial state');

    // Read initial node count from the "Nodes: X,XXX" display
    const initialNodeCount = await page.evaluate(() => {
      const text = document.body.textContent || '';
      const match = text.match(/Nodes:\s*([\d,]+)/i);
      if (match) {
        return parseInt(match[1].replace(/,/g, ''), 10);
      }
      return null;
    });
    console.log(`\n   Initial node count: ${initialNodeCount}`);

    // Click on the Quality tab
    console.log('\n2. Clicking Quality tab...');
    const qualityTab = await page.locator('button:has-text("Quality"), [role="tab"]:has-text("Quality")').first();
    if (await qualityTab.isVisible().catch(() => false)) {
      await qualityTab.click();
      await page.waitForTimeout(1000);
      console.log('   Clicked Quality tab');
    } else {
      // Try clicking by text content
      await page.click('text=Quality');
      await page.waitForTimeout(1000);
      console.log('   Clicked Quality by text');
    }

    await page.screenshot({ path: '/tmp/quality-tab-2-quality-panel.png' });
    console.log('   Screenshot 2: Quality panel');

    // Now look for filter controls in the Quality tab
    console.log('\n3. Looking for Quality filter controls...');

    // Find checkboxes for "Filter by Quality" or "Enable Filter"
    const filterCheckboxes = await page.locator('input[type="checkbox"]').all();
    console.log(`   Found ${filterCheckboxes.length} checkboxes`);

    // Find sliders that might be quality/authority thresholds
    const sliders = await page.locator('input[type="range"]').all();
    console.log(`   Found ${sliders.length} sliders`);

    // Try to enable filter if there's a checkbox
    if (filterCheckboxes.length > 0) {
      console.log('\n4. Enabling filter checkbox...');
      for (let i = 0; i < Math.min(filterCheckboxes.length, 3); i++) {
        const checkbox = filterCheckboxes[i];
        const isChecked = await checkbox.isChecked().catch(() => false);
        const isVisible = await checkbox.isVisible().catch(() => false);
        if (isVisible && !isChecked) {
          await checkbox.click();
          console.log(`   Enabled checkbox ${i + 1}`);
          await page.waitForTimeout(500);
        }
      }
    }

    await page.screenshot({ path: '/tmp/quality-tab-3-filter-enabled.png' });
    console.log('   Screenshot 3: Filter enabled');

    // Adjust quality threshold slider if found
    if (sliders.length > 0) {
      console.log('\n5. Adjusting quality threshold slider...');
      const slider = sliders[0];
      const isVisible = await slider.isVisible().catch(() => false);
      if (isVisible) {
        // Set to high value to filter out more nodes
        await slider.evaluate((el: HTMLInputElement) => {
          el.value = '0.9';
          el.dispatchEvent(new Event('input', { bubbles: true }));
          el.dispatchEvent(new Event('change', { bubbles: true }));
        });
        console.log('   Set slider to 0.9');
        await page.waitForTimeout(3000);
      }
    }

    await page.screenshot({ path: '/tmp/quality-tab-4-threshold-set.png' });
    console.log('   Screenshot 4: Threshold adjusted');

    // Wait for server response
    console.log('\n6. Waiting for server to process filter...');
    await page.waitForTimeout(5000);

    // Read new node count
    const newNodeCount = await page.evaluate(() => {
      const text = document.body.textContent || '';
      const match = text.match(/Nodes:\s*([\d,]+)/i);
      if (match) {
        return parseInt(match[1].replace(/,/g, ''), 10);
      }
      return null;
    });
    console.log(`   New node count: ${newNodeCount}`);

    await page.screenshot({ path: '/tmp/quality-tab-5-final.png', fullPage: true });
    console.log('   Screenshot 5: Final state\n');

    // Summary
    console.log('=== RESULTS ===');
    if (initialNodeCount !== null && newNodeCount !== null) {
      console.log(`Initial nodes:  ${initialNodeCount}`);
      console.log(`Filtered nodes: ${newNodeCount}`);
      const reduction = initialNodeCount - newNodeCount;
      const percentage = ((reduction / initialNodeCount) * 100).toFixed(1);
      console.log(`Reduction:      ${reduction} nodes (${percentage}%)`);

      if (newNodeCount < initialNodeCount) {
        console.log('\n*** SUCCESS: Node count DECREASED after applying quality filter! ***');
      } else if (newNodeCount === initialNodeCount) {
        console.log('\n*** NOTICE: Node count unchanged. Filter may not be working. ***');
      }
    } else {
      console.log('Could not verify node count change');
    }

    // Print relevant logs
    console.log('\n=== Filter-Related Logs ===');
    const filterLogs = logs.filter(l =>
      l.toLowerCase().includes('filter') ||
      l.toLowerCase().includes('sent to server') ||
      l.includes('InitialGraphLoad')
    );
    filterLogs.slice(-20).forEach(log => console.log(log));

  } catch (error) {
    console.error('Test error:', error);
    await page.screenshot({ path: '/tmp/quality-tab-error.png' });
  } finally {
    await browser.close();
  }

  console.log('\nTest completed. Check /tmp/quality-tab-*.png for screenshots.');
}

testQualityTabFilter().catch(console.error);
