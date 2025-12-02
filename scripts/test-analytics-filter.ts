import { chromium } from 'playwright';

async function testAnalyticsFilter() {
  console.log('=== Analytics Tab Node Filter Test ===\n');
  console.log('Filter settings are in Analytics tab (not Quality tab)\n');

  const browser = await chromium.launch({
    headless: false,
    args: ['--no-sandbox', '--disable-setuid-sandbox']
  });

  const page = await browser.newPage();

  // Track logs
  const logs: string[] = [];
  page.on('console', msg => {
    const text = msg.text();
    logs.push(`[${msg.type()}] ${text}`);
    if (text.includes('Filter') || text.includes('filter') ||
        text.includes('nodes') || text.includes('sent to server') ||
        text.includes('InitialGraphLoad')) {
      console.log(`>>> ${text}`);
    }
  });

  try {
    console.log('1. Navigating to VisionFlow...');
    await page.goto('http://192.168.0.51:3001', { waitUntil: 'domcontentloaded', timeout: 30000 });
    await page.waitForTimeout(6000);

    await page.keyboard.press('Escape');
    await page.waitForTimeout(500);

    await page.screenshot({ path: '/tmp/analytics-1-initial.png' });
    console.log('   Screenshot 1: Initial state');

    // Read initial node count
    const initialCount = await page.evaluate(() => {
      const text = document.body.textContent || '';
      const match = text.match(/Nodes:\s*([\d,]+)/i);
      return match ? parseInt(match[1].replace(/,/g, ''), 10) : null;
    });
    console.log(`\n   Initial node count: ${initialCount}`);

    // Click on the Analytics tab
    console.log('\n2. Clicking Analytics tab...');
    await page.click('text=Analytics');
    await page.waitForTimeout(1500);

    await page.screenshot({ path: '/tmp/analytics-2-analytics-tab.png' });
    console.log('   Screenshot 2: Analytics tab opened');

    // Look for filter controls in Analytics tab
    console.log('\n3. Looking for "Enable Filtering" toggle...');

    // Find all toggles and checkboxes
    const toggles = await page.locator('input[type="checkbox"], [role="switch"]').all();
    console.log(`   Found ${toggles.length} toggles/checkboxes`);

    // Find the "Enable Filtering" toggle
    const filterToggle = await page.locator('label:has-text("Enable Filtering"), label:has-text("Filter")').first();
    if (await filterToggle.isVisible().catch(() => false)) {
      console.log('   Found "Enable Filtering" label');
      await filterToggle.click();
      await page.waitForTimeout(1000);
    } else {
      // Try clicking any toggle that might be for filtering
      console.log('   Looking for filter toggle by other means...');
      const inputsInAnalytics = await page.locator('input[type="checkbox"]').all();
      if (inputsInAnalytics.length > 0) {
        // First checkbox might be the filter enable
        await inputsInAnalytics[0].click();
        console.log('   Clicked first checkbox');
        await page.waitForTimeout(1000);
      }
    }

    await page.screenshot({ path: '/tmp/analytics-3-filter-clicked.png' });
    console.log('   Screenshot 3: After clicking filter toggle');

    // Find and adjust Quality Threshold slider
    console.log('\n4. Looking for Quality Threshold slider...');
    const qualitySliderLabel = await page.locator('label:has-text("Quality Threshold")').first();
    if (await qualitySliderLabel.isVisible().catch(() => false)) {
      console.log('   Found Quality Threshold label');
      // Find the slider near this label
      const nearbySlider = await page.locator('input[type="range"]').first();
      if (await nearbySlider.isVisible().catch(() => false)) {
        // Set to high threshold to filter more nodes
        await nearbySlider.evaluate((el: HTMLInputElement) => {
          el.value = '0.9';
          el.dispatchEvent(new Event('input', { bubbles: true }));
          el.dispatchEvent(new Event('change', { bubbles: true }));
        });
        console.log('   Set Quality Threshold to 0.9');
      }
    } else {
      console.log('   Quality Threshold label not found');
    }

    await page.waitForTimeout(3000);
    await page.screenshot({ path: '/tmp/analytics-4-threshold-set.png' });
    console.log('   Screenshot 4: After setting threshold');

    // Find and enable "Filter by Quality" toggle
    console.log('\n5. Looking for "Filter by Quality" toggle...');
    const filterByQualityLabel = await page.locator('label:has-text("Filter by Quality")').first();
    if (await filterByQualityLabel.isVisible().catch(() => false)) {
      console.log('   Found "Filter by Quality" label');
      await filterByQualityLabel.click();
      await page.waitForTimeout(1000);
    }

    // Wait for server to respond
    console.log('\n6. Waiting for server to process filter...');
    await page.waitForTimeout(5000);

    await page.screenshot({ path: '/tmp/analytics-5-filters-enabled.png' });
    console.log('   Screenshot 5: After enabling filters');

    // Read new node count
    const newCount = await page.evaluate(() => {
      const text = document.body.textContent || '';
      const match = text.match(/Nodes:\s*([\d,]+)/i);
      return match ? parseInt(match[1].replace(/,/g, ''), 10) : null;
    });
    console.log(`   New node count: ${newCount}`);

    await page.screenshot({ path: '/tmp/analytics-6-final.png', fullPage: true });
    console.log('   Screenshot 6: Final state\n');

    // Summary
    console.log('=== RESULTS ===');
    if (initialCount !== null && newCount !== null) {
      console.log(`Initial nodes:  ${initialCount}`);
      console.log(`Filtered nodes: ${newCount}`);
      const reduction = initialCount - newCount;
      const percentage = ((reduction / initialCount) * 100).toFixed(1);
      console.log(`Reduction:      ${reduction} nodes (${percentage}%)`);

      if (newCount < initialCount) {
        console.log('\n*** SUCCESS: Node count DECREASED after applying quality filter! ***');
      } else if (newCount === initialCount) {
        console.log('\n*** NOTICE: Node count unchanged. Filter may not have been triggered. ***');
      }
    } else {
      console.log('Could not verify node count change');
      console.log(`Initial: ${initialCount}, New: ${newCount}`);
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
    await page.screenshot({ path: '/tmp/analytics-error.png' });
  } finally {
    await browser.close();
  }

  console.log('\nTest completed. Check /tmp/analytics-*.png for screenshots.');
}

testAnalyticsFilter().catch(console.error);
