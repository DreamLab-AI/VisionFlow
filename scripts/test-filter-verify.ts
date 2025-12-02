import { chromium } from 'playwright';

async function testFilterVerify() {
  console.log('=== Filter Verification Test ===\n');

  const browser = await chromium.launch({
    headless: false,
    args: ['--no-sandbox', '--disable-setuid-sandbox']
  });

  const page = await browser.newPage();

  // Track WebSocket messages
  page.on('console', msg => {
    const text = msg.text();
    if (text.includes('filter') || text.includes('Filter') ||
        text.includes('nodes') || text.includes('Nodes') ||
        text.includes('sent to server')) {
      console.log(`>>> ${text}`);
    }
  });

  try {
    console.log('1. Navigating to VisionFlow...');
    await page.goto('http://192.168.0.51:3001', { waitUntil: 'domcontentloaded', timeout: 30000 });
    await page.waitForTimeout(6000);
    await page.keyboard.press('Escape');
    await page.waitForTimeout(500);

    // Get initial node count
    const getNodeCount = async () => {
      return await page.evaluate(() => {
        const text = document.body.textContent || '';
        const match = text.match(/Nodes:\s*([\d,]+)/i);
        return match ? parseInt(match[1].replace(/,/g, ''), 10) : null;
      });
    };

    const initialCount = await getNodeCount();
    console.log(`\n   Initial node count: ${initialCount}`);
    await page.screenshot({ path: '/tmp/filter-verify-1-initial.png' });

    // Click Analytics tab - use JavaScript to bypass any overlays
    console.log('\n2. Opening Analytics tab...');

    // Use JavaScript to click the Analytics tab directly
    await page.evaluate(() => {
      const analyticsTab = document.querySelector('[data-tab="analytics"]') ||
                           Array.from(document.querySelectorAll('div')).find(el => el.textContent === 'Analytics');
      if (analyticsTab) {
        (analyticsTab as HTMLElement).click();
      }
    });
    await page.waitForTimeout(1000);
    await page.screenshot({ path: '/tmp/filter-verify-2-analytics.png' });

    // Find and toggle the filter controls - these are styled toggle switches, not checkboxes
    console.log('\n3. Toggling filter controls...');

    // Click the toggle switch area for "Enable Filtering"
    // The toggle appears to be a label + input combo, clicking the label or toggle area should work
    try {
      // First, try to find the toggle by looking for the switch/toggle element near the label
      const enableFilterToggle = await page.locator('label:has-text("Enable Filtering")').first();
      if (await enableFilterToggle.isVisible()) {
        // Click the toggle - the label may have a sibling toggle element
        await enableFilterToggle.click();
        console.log('   Clicked Enable Filtering toggle');
      }
    } catch (e) {
      // Try direct input click
      await page.click('input#filterEnabled');
      console.log('   Clicked Enable Filtering input');
    }
    await page.waitForTimeout(1000);

    // Set quality threshold high using the slider
    console.log('\n4. Setting quality threshold to 0.85...');
    const qualitySlider = await page.$('input[type="range"]');
    if (qualitySlider) {
      await qualitySlider.evaluate((el: HTMLInputElement) => {
        el.value = '0.85';
        el.dispatchEvent(new Event('input', { bubbles: true }));
        el.dispatchEvent(new Event('change', { bubbles: true }));
      });
      console.log('   Set quality threshold slider to 0.85');
    }
    await page.waitForTimeout(500);

    // Enable "Filter by Quality" toggle
    console.log('\n5. Enabling Filter by Quality...');
    try {
      const filterByQualityToggle = await page.locator('label:has-text("Filter by Quality")').first();
      if (await filterByQualityToggle.isVisible()) {
        await filterByQualityToggle.click();
        console.log('   Clicked Filter by Quality toggle');
      }
    } catch (e) {
      console.log('   Could not find Filter by Quality toggle');
    }

    await page.screenshot({ path: '/tmp/filter-verify-3-settings.png' });

    // Wait for server to process the filter
    console.log('\n6. Waiting for server to apply filter...');
    await page.waitForTimeout(5000);

    const afterFilterCount = await getNodeCount();
    console.log(`   Node count after filter: ${afterFilterCount}`);
    await page.screenshot({ path: '/tmp/filter-verify-4-filtered.png' });

    // Summary
    console.log('\n=== RESULTS ===');
    console.log(`Initial nodes:  ${initialCount}`);
    console.log(`Filtered nodes: ${afterFilterCount}`);

    if (initialCount !== null && afterFilterCount !== null) {
      const reduction = initialCount - afterFilterCount;
      const percentage = ((reduction / initialCount) * 100).toFixed(1);
      console.log(`Reduction:      ${reduction} nodes (${percentage}%)`);

      if (afterFilterCount < initialCount) {
        console.log('\n*** SUCCESS: Node count DECREASED after applying quality filter! ***');
      } else {
        console.log('\n*** NOTICE: Node count unchanged - filter may not be working ***');
      }
    }

    await page.screenshot({ path: '/tmp/filter-verify-5-final.png', fullPage: true });

  } catch (error) {
    console.error('Test error:', error);
    await page.screenshot({ path: '/tmp/filter-verify-error.png' });
  } finally {
    await browser.close();
  }

  console.log('\nTest completed. Check /tmp/filter-verify-*.png');
}

testFilterVerify().catch(console.error);
