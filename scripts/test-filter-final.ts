import { chromium } from 'playwright';

async function testFilterFinal() {
  console.log('=== Final Filter Verification Test ===\n');

  const browser = await chromium.launch({
    headless: false,
    args: ['--no-sandbox', '--disable-setuid-sandbox']
  });

  const page = await browser.newPage();

  // Track relevant console logs
  page.on('console', msg => {
    const text = msg.text();
    if (text.includes('filter') || text.includes('Filter') ||
        text.includes('nodes') || text.includes('Nodes') ||
        text.includes('sent to server') || text.includes('WebSocket') ||
        text.includes('filter_update')) {
      console.log(`>>> ${text}`);
    }
  });

  try {
    console.log('1. Navigating to VisionFlow...');
    await page.goto('http://192.168.0.51:3001', { waitUntil: 'domcontentloaded', timeout: 30000 });
    await page.waitForTimeout(6000);

    // Dismiss overlays
    console.log('\n2. Dismissing overlays...');
    for (let i = 0; i < 3; i++) {
      await page.keyboard.press('Escape');
      await page.waitForTimeout(300);
    }

    // Remove SpaceMouse warning banner
    console.log('\n3. Removing warning banners...');
    await page.evaluate(() => {
      document.querySelectorAll('.fixed.top-0, [class*="orange-600"], [class*="red-600"]').forEach(el => {
        if (el.textContent?.includes('SpaceMouse') || el.textContent?.includes('Connection')) {
          el.remove();
        }
      });
    });
    await page.waitForTimeout(500);

    // Get node count
    const getNodeCount = async (): Promise<number | null> => {
      return await page.evaluate(() => {
        const allText = document.body.innerText;
        const match = allText.match(/Nodes[:\s]+([0-9,]+)/i);
        return match ? parseInt(match[1].replace(/,/g, ''), 10) : null;
      });
    };

    const initialCount = await getNodeCount();
    console.log(`\n   Initial node count: ${initialCount}`);
    await page.screenshot({ path: '/tmp/filter-final-1-initial.png' });

    // Step 4: Click Analytics tab in Control Center (main level)
    console.log('\n4. Clicking Analytics tab...');
    const analyticsClicked = await page.evaluate(() => {
      // Find the Analytics tab by looking for tab triggers with "Analytics" text
      const tabs = document.querySelectorAll('[role="tab"], [data-state], button');
      for (const tab of tabs) {
        const text = tab.textContent?.trim();
        if (text === 'Analytics' || text?.includes('Analytics')) {
          (tab as HTMLElement).click();
          return 'main tab clicked';
        }
      }
      // Fallback: click by value attribute
      const byValue = document.querySelector('[value="analytics"]');
      if (byValue) {
        (byValue as HTMLElement).click();
        return 'value=analytics clicked';
      }
      return 'not found';
    });
    console.log(`   Analytics: ${analyticsClicked}`);
    await page.waitForTimeout(1000);
    await page.screenshot({ path: '/tmp/filter-final-2-analytics.png' });

    // Step 5: Click the "filter" sub-tab inside SemanticAnalysisPanel
    console.log('\n5. Clicking Filter sub-tab...');
    const filterTabClicked = await page.evaluate(() => {
      // The SemanticAnalysisPanel has nested tabs: filter, communities, centrality, paths, constraints
      const tabs = document.querySelectorAll('[role="tab"], [data-state], button');
      for (const tab of tabs) {
        const text = tab.textContent?.trim().toLowerCase();
        if (text === 'filter' || text === 'node filter') {
          (tab as HTMLElement).click();
          return 'filter sub-tab clicked';
        }
      }
      // Try by value attribute
      const byValue = document.querySelector('[value="filter"]');
      if (byValue) {
        (byValue as HTMLElement).click();
        return 'value=filter clicked';
      }
      return 'not found';
    });
    console.log(`   Filter sub-tab: ${filterTabClicked}`);
    await page.waitForTimeout(1000);
    await page.screenshot({ path: '/tmp/filter-final-3-filter-tab.png' });

    // Step 6: Toggle "Enable Filtering" switch
    console.log('\n6. Enabling filters...');
    const enableResult = await page.evaluate(() => {
      // Find switch elements next to labels
      const switches = document.querySelectorAll('[role="switch"], button[class*="switch"], button');
      const results: string[] = [];

      // Look for "Enable" switch (first one in filter section)
      for (const sw of switches) {
        const parent = sw.closest('.flex, .space-y-3');
        const labelText = parent?.textContent || '';

        if (labelText.includes('Enable') && !labelText.includes('Filtering')) {
          // This might be the main enable switch
          const checked = sw.getAttribute('data-state') === 'checked' || sw.getAttribute('aria-checked') === 'true';
          if (!checked) {
            (sw as HTMLElement).click();
            results.push('enabled main switch');
          } else {
            results.push('main already enabled');
          }
        }
      }

      // Find and enable "Filter by Quality" switch
      for (const sw of switches) {
        const parent = sw.closest('.flex, div');
        const labelText = parent?.textContent || '';

        if (labelText.includes('Filter by Quality')) {
          const checked = sw.getAttribute('data-state') === 'checked' || sw.getAttribute('aria-checked') === 'true';
          if (!checked) {
            (sw as HTMLElement).click();
            results.push('enabled Filter by Quality');
          } else {
            results.push('Filter by Quality already enabled');
          }
        }

        if (labelText.includes('Filter by Authority')) {
          const checked = sw.getAttribute('data-state') === 'checked' || sw.getAttribute('aria-checked') === 'true';
          if (!checked) {
            (sw as HTMLElement).click();
            results.push('enabled Filter by Authority');
          } else {
            results.push('Filter by Authority already enabled');
          }
        }
      }

      return results.length > 0 ? results.join(', ') : 'no switches found';
    });
    console.log(`   Switches: ${enableResult}`);
    await page.waitForTimeout(1000);

    // Step 7: Set quality threshold slider to high value (0.8)
    console.log('\n7. Setting quality threshold to 0.8...');
    const thresholdResult = await page.evaluate(() => {
      const sliders = document.querySelectorAll('input[type="range"], [role="slider"]');
      for (const slider of sliders) {
        const parent = slider.closest('.space-y-2, .space-y-3, div');
        const labelText = parent?.textContent || '';

        if (labelText.toLowerCase().includes('quality') && labelText.toLowerCase().includes('threshold')) {
          if (slider instanceof HTMLInputElement) {
            slider.value = '0.8';
            slider.dispatchEvent(new Event('input', { bubbles: true }));
            slider.dispatchEvent(new Event('change', { bubbles: true }));
            return 'set to 0.8 via input';
          } else {
            // For custom slider components, try clicking at 80% position
            const rect = slider.getBoundingClientRect();
            const clickX = rect.left + (rect.width * 0.8);
            const clickY = rect.top + (rect.height / 2);
            const clickEvent = new MouseEvent('click', {
              clientX: clickX,
              clientY: clickY,
              bubbles: true
            });
            slider.dispatchEvent(clickEvent);
            return 'clicked at 80% position';
          }
        }
      }
      return 'no quality threshold slider found';
    });
    console.log(`   Threshold: ${thresholdResult}`);

    await page.screenshot({ path: '/tmp/filter-final-4-settings.png' });

    // Wait for server to process
    console.log('\n8. Waiting for server (10 seconds)...');
    await page.waitForTimeout(10000);

    // Get final count
    const filteredCount = await getNodeCount();
    console.log(`\n   Filtered node count: ${filteredCount}`);
    await page.screenshot({ path: '/tmp/filter-final-5-result.png' });

    // Summary
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
        console.log('   Server-side filtering is working correctly.');
      } else {
        console.log('\n⚠️  Node count unchanged or not detected');
      }
    }

    // Debug info
    console.log('\n9. Debug: Checking visible elements...');
    const debug = await page.evaluate(() => {
      return {
        hasFilterByQuality: document.body.innerText.includes('Filter by Quality'),
        hasQualityThreshold: document.body.innerText.includes('Quality Threshold'),
        hasAuthorityThreshold: document.body.innerText.includes('Authority Threshold'),
        activeTabState: document.querySelector('[data-state="active"]')?.textContent || 'unknown'
      };
    });
    console.log(`   ${JSON.stringify(debug, null, 2)}`);

    await page.screenshot({ path: '/tmp/filter-final-6-final.png', fullPage: true });

  } catch (error) {
    console.error('Test error:', error);
    await page.screenshot({ path: '/tmp/filter-final-error.png' });
  } finally {
    await browser.close();
  }

  console.log('\nTest completed. Check /tmp/filter-final-*.png');
}

testFilterFinal().catch(console.error);
