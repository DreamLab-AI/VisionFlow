import { chromium } from 'playwright';

/**
 * Playwright-native filter test
 * Uses Playwright locators exclusively for proper event handling
 */
async function testFilterPlaywright() {
  console.log('=== Playwright Native Filter Test ===\n');

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
    await page.waitForTimeout(10000);

    // Dismiss overlays
    for (let i = 0; i < 5; i++) {
      await page.keyboard.press('Escape');
      await page.waitForTimeout(100);
    }
    await page.waitForTimeout(500);

    await page.screenshot({ path: '/tmp/filter-pw-1-loaded.png' });

    // Get initial node count from the visible UI
    const initialNodeText = await page.locator('text=/Nodes.*\\d+/').first().textContent().catch(() => null);
    console.log(`2. Node text found: "${initialNodeText}"`);

    // Extract number from the node count
    const initialMatch = initialNodeText?.match(/(\d[\d,]*)/);
    const initialCount = initialMatch ? parseInt(initialMatch[1].replace(/,/g, ''), 10) : null;
    console.log(`   Initial node count: ${initialCount}`);

    // Wait for the Analytics tab to be visible and click it using Playwright locator
    console.log('\n3. Looking for Analytics tab...');

    // The tab has text "4Analytics" according to previous test output
    // Use Playwright's text matching with regex
    const analyticsTab = page.locator('button[role="tab"]').filter({ hasText: /Analytics/i }).first();
    const tabCount = await analyticsTab.count();
    console.log(`   Found ${tabCount} matching tab(s)`);

    if (tabCount > 0) {
      // Get tab info before click
      const tabText = await analyticsTab.textContent();
      console.log(`   Tab text: "${tabText}"`);

      // Click using Playwright's native click (proper event handling)
      console.log('\n4. Clicking Analytics tab with Playwright...');
      await analyticsTab.click();
      await page.waitForTimeout(1000);

      // Verify data-state changed
      const dataState = await analyticsTab.getAttribute('data-state');
      console.log(`   Tab data-state after click: ${dataState}`);
    } else {
      console.log('   No Analytics tab found!');
    }

    await page.screenshot({ path: '/tmp/filter-pw-2-after-click.png' });

    // Check what content is visible now
    console.log('\n5. Checking for filter controls...');

    // Look for "Filter by Quality" text
    const filterLabel = page.locator('label').filter({ hasText: /Filter by Quality/i }).first();
    const filterLabelVisible = await filterLabel.isVisible().catch(() => false);
    console.log(`   "Filter by Quality" label visible: ${filterLabelVisible}`);

    // Look for "Quality Threshold" text
    const thresholdLabel = page.locator('label').filter({ hasText: /Quality Threshold/i }).first();
    const thresholdLabelVisible = await thresholdLabel.isVisible().catch(() => false);
    console.log(`   "Quality Threshold" label visible: ${thresholdLabelVisible}`);

    // Look for "Analytics & Filtering" title
    const analyticsTitle = page.locator('h3').filter({ hasText: /Analytics/i }).first();
    const analyticsTitleVisible = await analyticsTitle.isVisible().catch(() => false);
    console.log(`   "Analytics..." title visible: ${analyticsTitleVisible}`);

    if (filterLabelVisible) {
      console.log('\n6. Filter controls found! Enabling filter...');

      // Find and click the toggle button near the "Filter by Quality" label
      // The toggle is a button sibling in the same flex container
      const toggleButton = page.locator('label').filter({ hasText: /Filter by Quality/i })
        .locator('..').locator('button').first();

      const toggleVisible = await toggleButton.isVisible().catch(() => false);
      console.log(`   Toggle button visible: ${toggleVisible}`);

      if (toggleVisible) {
        await toggleButton.click();
        console.log('   Clicked toggle');
        await page.waitForTimeout(500);
      }

      // Find and adjust the quality threshold slider
      if (thresholdLabelVisible) {
        console.log('\n7. Setting Quality Threshold to 0.8...');

        // Find the slider input near the threshold label
        const slider = page.locator('label').filter({ hasText: /Quality Threshold/i })
          .locator('..').locator('..').locator('input[type="range"]').first();

        const sliderVisible = await slider.isVisible().catch(() => false);
        console.log(`   Slider visible: ${sliderVisible}`);

        if (sliderVisible) {
          // Fill the slider with value
          await slider.fill('0.8');
          console.log('   Set slider to 0.8');
          await page.waitForTimeout(500);
        }
      }

      await page.screenshot({ path: '/tmp/filter-pw-3-filter-enabled.png' });

      // Wait for server to process
      console.log('\n8. Waiting for filter to apply (15s)...');
      await page.waitForTimeout(15000);

      // Get final node count
      const finalNodeText = await page.locator('text=/Nodes.*\\d+/').first().textContent().catch(() => null);
      const finalMatch = finalNodeText?.match(/(\d[\d,]*)/);
      const finalCount = finalMatch ? parseInt(finalMatch[1].replace(/,/g, ''), 10) : null;
      console.log(`\n9. Final node count: ${finalCount}`);

      await page.screenshot({ path: '/tmp/filter-pw-4-final.png' });

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
      }
    } else {
      console.log('\n❌ Filter controls not visible');
      console.log('   The Analytics tab content did not render');

      // Debug: Take full page screenshot
      await page.screenshot({ path: '/tmp/filter-pw-debug.png', fullPage: true });

      // Debug: Print visible text
      const bodyText = await page.evaluate(() => document.body.innerText.substring(0, 2000));
      console.log('\n   Page text preview:');
      console.log(bodyText.split('\n').slice(0, 30).join('\n'));
    }

  } catch (error) {
    console.error('Test error:', error);
    await page.screenshot({ path: '/tmp/filter-pw-error.png' });
  } finally {
    await browser.close();
  }

  console.log('\nTest completed. Check /tmp/filter-pw-*.png');
}

testFilterPlaywright().catch(console.error);
