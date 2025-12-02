import { chromium } from 'playwright';

/**
 * E2E test for quality-based node filtering - Version 2
 * Uses Playwright native locators exclusively
 */
async function testFilterE2EV2() {
  console.log('=== E2E Filter Verification Test V2 ===\n');

  const browser = await chromium.launch({
    headless: false,
    args: ['--no-sandbox', '--disable-setuid-sandbox']
  });

  const page = await browser.newPage();

  // Collect console logs
  page.on('console', msg => {
    const text = msg.text();
    if (text.includes('filter') || text.includes('Filter') ||
        text.includes('nodes') || text.includes('WebSocket') ||
        text.includes('filter_update') || text.includes('SendInitialGraphLoad')) {
      console.log(`[Browser] ${text}`);
    }
  });

  try {
    // Step 1: Navigate
    console.log('1. Navigating to VisionFlow...');
    await page.goto('http://192.168.0.51:3001', { waitUntil: 'domcontentloaded', timeout: 30000 });
    await page.waitForTimeout(8000);

    // Close any dialogs
    for (let i = 0; i < 3; i++) {
      await page.keyboard.press('Escape');
      await page.waitForTimeout(200);
    }
    await page.waitForTimeout(500);

    // Get initial node count
    let initialCount = await page.evaluate(() => {
      const text = document.body.innerText;
      const match = text.match(/Nodes[:\s]+(\d[\d,]*)/i) || text.match(/(\d[\d,]*)\s*nodes/i);
      return match ? parseInt(match[1].replace(/,/g, ''), 10) : null;
    });
    console.log(`2. Initial node count: ${initialCount}`);
    await page.screenshot({ path: '/tmp/filter-v2-1-initial.png' });

    // Step 3: Find and click Analytics tab using data-value attribute
    console.log('\n3. Clicking Analytics tab...');

    // Wait for Control Center
    await page.waitForSelector('.control-center', { timeout: 10000 });

    // Debug: Log all available tabs first
    const tabs = await page.$$eval('[role="tab"]', tabs =>
      tabs.map(t => ({
        text: t.textContent?.trim(),
        value: t.getAttribute('value'),
        dataState: t.getAttribute('data-state')
      }))
    );
    console.log('   Available tabs:', JSON.stringify(tabs));

    // Click the Analytics tab specifically by value
    const analyticsClicked = await page.evaluate(() => {
      // Find button with value="analytics"
      const tab = document.querySelector('button[value="analytics"]') as HTMLElement;
      if (tab) {
        tab.click();
        return 'clicked via button[value=analytics]';
      }
      // Fallback: find by text
      const buttons = document.querySelectorAll('[role="tab"]');
      for (const btn of buttons) {
        if (btn.textContent?.toLowerCase().includes('analytics')) {
          (btn as HTMLElement).click();
          return `clicked by text: ${btn.textContent?.trim()}`;
        }
      }
      return 'no analytics tab found';
    });
    console.log(`   Result: ${analyticsClicked}`);
    await page.waitForTimeout(1500);

    // Verify tab switch
    const activeTabNow = await page.evaluate(() => {
      const activeTab = document.querySelector('[role="tab"][data-state="active"]');
      return activeTab?.textContent?.trim() || 'unknown';
    });
    console.log(`   Active tab now: ${activeTabNow}`);

    await page.screenshot({ path: '/tmp/filter-v2-2-analytics.png' });

    // Step 4: Check if we see "Analytics & Filtering" or "Filter by Quality" in the content
    console.log('\n4. Checking for filter controls...');
    const bodyText = await page.evaluate(() => document.body.innerText);
    console.log(`   Contains "Filter by Quality": ${bodyText.includes('Filter by Quality')}`);
    console.log(`   Contains "Quality Threshold": ${bodyText.includes('Quality Threshold')}`);
    console.log(`   Contains "Analytics & Filtering": ${bodyText.includes('Analytics & Filtering')}`);

    // Step 5: Find and toggle the "Filter by Quality" switch
    console.log('\n5. Looking for Filter by Quality toggle...');

    // Use label association
    const toggleResult = await page.evaluate(() => {
      // Method 1: Find label with Filter by Quality, then find nearby toggle
      const labels = document.querySelectorAll('label');
      for (const label of labels) {
        if (label.textContent?.includes('Filter by Quality')) {
          // The toggle should be in the same flex container
          const container = label.closest('div');
          if (container) {
            // Find button in same container (custom toggle button)
            const button = container.querySelector('button');
            if (button) {
              button.click();
              return { success: true, method: 'label-sibling-button' };
            }
          }
        }
      }

      // Method 2: Find any toggle by ID containing "filter" or "quality"
      const toggles = document.querySelectorAll('button[id*="filter"], button[id*="quality"], [role="switch"]');
      if (toggles.length > 0) {
        (toggles[0] as HTMLElement).click();
        return { success: true, method: 'id-pattern', count: toggles.length };
      }

      // Method 3: Just list what toggles exist
      const allButtons = document.querySelectorAll('button');
      const toggleLike = Array.from(allButtons)
        .filter(b => b.style.borderRadius?.includes('9') || b.getAttribute('role') === 'switch')
        .map(b => b.id || b.textContent?.slice(0, 20));

      return { success: false, toggleLikeElements: toggleLike };
    });
    console.log(`   Toggle result: ${JSON.stringify(toggleResult)}`);
    await page.waitForTimeout(500);

    await page.screenshot({ path: '/tmp/filter-v2-3-toggle.png' });

    // Step 6: Try to interact with Quality Threshold slider
    console.log('\n6. Looking for Quality Threshold slider...');
    const sliderResult = await page.evaluate(() => {
      // Find slider input for quality threshold
      const inputs = document.querySelectorAll('input[type="range"]');
      for (const input of inputs) {
        const parent = input.closest('div');
        const label = parent?.querySelector('label');
        if (label?.textContent?.includes('Quality')) {
          // Found it! Set to 0.8
          (input as HTMLInputElement).value = '0.8';
          input.dispatchEvent(new Event('input', { bubbles: true }));
          input.dispatchEvent(new Event('change', { bubbles: true }));
          return { success: true, label: label.textContent };
        }
      }
      return { success: false, sliderCount: inputs.length };
    });
    console.log(`   Slider result: ${JSON.stringify(sliderResult)}`);
    await page.waitForTimeout(500);

    await page.screenshot({ path: '/tmp/filter-v2-4-slider.png' });

    // Step 7: Wait for server round-trip
    console.log('\n7. Waiting for filter to apply (10s)...');
    await page.waitForTimeout(10000);

    // Step 8: Get final node count
    const finalCount = await page.evaluate(() => {
      const text = document.body.innerText;
      const match = text.match(/Nodes[:\s]+(\d[\d,]*)/i) || text.match(/(\d[\d,]*)\s*nodes/i);
      return match ? parseInt(match[1].replace(/,/g, ''), 10) : null;
    });
    console.log(`\n8. Final node count: ${finalCount}`);

    await page.screenshot({ path: '/tmp/filter-v2-5-final.png' });

    // Results
    console.log('\n========================================');
    console.log('=== RESULTS ===');
    console.log('========================================');
    console.log(`Initial nodes:  ${initialCount}`);
    console.log(`Final nodes:    ${finalCount}`);

    if (initialCount !== null && finalCount !== null) {
      if (finalCount < initialCount) {
        const reduction = initialCount - finalCount;
        const pct = ((reduction / initialCount) * 100).toFixed(1);
        console.log(`\n✅ SUCCESS: Nodes reduced by ${reduction} (${pct}%)`);
      } else {
        console.log('\n⚠️  Node count unchanged or increased');
        console.log('   Check screenshots and debug info below');
      }
    } else {
      console.log('\n❌ Could not read node counts');
    }

    // Debug: Capture final page state
    const debugInfo = await page.evaluate(() => ({
      activeTab: document.querySelector('[role="tab"][data-state="active"]')?.textContent,
      visibleToggles: document.querySelectorAll('button').length,
      visibleSliders: document.querySelectorAll('input[type="range"]').length,
      hasAnalyticsPanel: document.body.innerText.includes('Analytics & Filtering'),
      connectionError: document.body.innerText.includes('Connection') && document.body.innerText.includes('Failed')
    }));
    console.log('\nDebug info:', JSON.stringify(debugInfo, null, 2));

  } catch (error) {
    console.error('Test error:', error);
    await page.screenshot({ path: '/tmp/filter-v2-error.png' });
  } finally {
    await browser.close();
  }

  console.log('\nTest completed. Check /tmp/filter-v2-*.png');
}

testFilterE2EV2().catch(console.error);
