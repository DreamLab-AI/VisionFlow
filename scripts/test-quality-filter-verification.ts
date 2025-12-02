import { chromium } from 'playwright';

async function testQualityFilterVerification() {
  console.log('=== Quality Filter Verification Test ===\n');
  console.log('This test verifies that node count drops when quality filters are enabled.\n');

  const browser = await chromium.launch({
    headless: false,
    args: ['--no-sandbox', '--disable-setuid-sandbox']
  });

  const page = await browser.newPage();

  // Track node counts from console logs
  let initialNodeCount: number | null = null;
  let filteredNodeCount: number | null = null;
  const filterLogs: string[] = [];

  page.on('console', msg => {
    const text = msg.text();
    // Capture node count from various log formats
    if (text.includes('nodes') || text.includes('Nodes')) {
      filterLogs.push(`[${msg.type()}] ${text}`);
      console.log(`>>> ${text}`);
    }
    if (text.includes('Filter') || text.includes('filter_update') || text.includes('sent to server')) {
      filterLogs.push(`[${msg.type()}] ${text}`);
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

    await page.screenshot({ path: '/tmp/quality-filter-1-initial.png' });
    console.log('   Screenshot 1: Initial state\n');

    // Read initial node count from the Control Center display
    console.log('2. Reading initial node count...');
    const nodeCountText = await page.evaluate(() => {
      // Look for node count in various elements
      const elements = document.querySelectorAll('*');
      for (const el of elements) {
        const text = el.textContent || '';
        // Match patterns like "1009 nodes" or "Nodes: 1009"
        const match = text.match(/(\d{2,5})\s*nodes?/i);
        if (match) {
          return match[1];
        }
      }
      // Also check for node count in status bar or stats panel
      const stats = document.querySelector('[data-testid="node-count"]');
      if (stats) return stats.textContent;
      return null;
    });

    if (nodeCountText) {
      initialNodeCount = parseInt(nodeCountText, 10);
      console.log(`   Initial node count: ${initialNodeCount}`);
    } else {
      // Try to get from text content that shows node stats
      const pageContent = await page.content();
      const nodeMatch = pageContent.match(/(\d{3,5})\s*nodes?/i);
      if (nodeMatch) {
        initialNodeCount = parseInt(nodeMatch[1], 10);
        console.log(`   Initial node count (from content): ${initialNodeCount}`);
      } else {
        console.log('   Could not find initial node count directly');
      }
    }

    // Find and enable the quality filter
    console.log('\n3. Finding quality filter controls...');

    // Look for the quality threshold slider or enable toggle
    const qualitySlider = await page.locator('input[type="range"]').first();
    if (await qualitySlider.isVisible().catch(() => false)) {
      console.log('   Found quality slider');

      // Get current value
      const currentValue = await qualitySlider.inputValue();
      console.log(`   Current slider value: ${currentValue}`);

      // Change to a higher threshold (0.7 or above to filter more nodes)
      console.log('\n4. Setting quality threshold to 0.8...');
      await qualitySlider.evaluate((el: HTMLInputElement) => {
        el.value = '0.8';
        el.dispatchEvent(new Event('input', { bubbles: true }));
        el.dispatchEvent(new Event('change', { bubbles: true }));
      });

      await page.waitForTimeout(3000);
      await page.screenshot({ path: '/tmp/quality-filter-2-threshold-set.png' });
      console.log('   Screenshot 2: Quality threshold set to 0.8');

      // Enable the filter checkbox if present
      const enableCheckbox = await page.locator('input[type="checkbox"]').first();
      if (await enableCheckbox.isVisible().catch(() => false)) {
        const isChecked = await enableCheckbox.isChecked();
        if (!isChecked) {
          console.log('\n5. Enabling filter...');
          await enableCheckbox.click();
          await page.waitForTimeout(3000);
        } else {
          console.log('   Filter already enabled');
        }
      }

      await page.screenshot({ path: '/tmp/quality-filter-3-after-filter.png' });
      console.log('   Screenshot 3: After filter applied');

      // Wait for server to process and send filtered data
      console.log('\n6. Waiting for filtered data from server...');
      await page.waitForTimeout(5000);

      // Read new node count
      const newNodeCountText = await page.evaluate(() => {
        const elements = document.querySelectorAll('*');
        for (const el of elements) {
          const text = el.textContent || '';
          const match = text.match(/(\d{2,5})\s*nodes?/i);
          if (match) {
            return match[1];
          }
        }
        return null;
      });

      if (newNodeCountText) {
        filteredNodeCount = parseInt(newNodeCountText, 10);
        console.log(`   Filtered node count: ${filteredNodeCount}`);
      }

      await page.screenshot({ path: '/tmp/quality-filter-4-final.png', fullPage: true });
      console.log('   Screenshot 4: Final state');
    }

    // Summary
    console.log('\n=== RESULTS ===');
    if (initialNodeCount !== null && filteredNodeCount !== null) {
      console.log(`Initial nodes:  ${initialNodeCount}`);
      console.log(`Filtered nodes: ${filteredNodeCount}`);
      const reduction = initialNodeCount - filteredNodeCount;
      const percentage = ((reduction / initialNodeCount) * 100).toFixed(1);
      console.log(`Reduction:      ${reduction} nodes (${percentage}%)`);

      if (filteredNodeCount < initialNodeCount) {
        console.log('\n*** SUCCESS: Node count DECREASED after applying quality filter! ***');
      } else {
        console.log('\n*** NOTICE: Node count did not decrease. Check server logs. ***');
      }
    } else {
      console.log('Could not verify node count change');
    }

    console.log('\n=== Filter-Related Logs ===');
    filterLogs.forEach(log => console.log(log));

    // Check server logs
    console.log('\n7. Checking server logs for filter processing...');

  } catch (error) {
    console.error('Test error:', error);
    await page.screenshot({ path: '/tmp/quality-filter-error.png' });
  } finally {
    await browser.close();
  }

  console.log('\nTest completed. Check /tmp/quality-filter-*.png for screenshots.');
}

testQualityFilterVerification().catch(console.error);
