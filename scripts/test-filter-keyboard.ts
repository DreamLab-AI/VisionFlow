import { chromium } from 'playwright';

/**
 * Keyboard-based filter test
 * Uses keyboard navigation to avoid click blocking by overlays
 */
async function testFilterKeyboard() {
  console.log('=== Keyboard Filter Test ===\n');

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

    // Dismiss overlays more aggressively
    console.log('\n2. Dismissing overlays...');
    for (let i = 0; i < 10; i++) {
      await page.keyboard.press('Escape');
      await page.waitForTimeout(100);
    }

    // Hide all fixed-position overlays
    await page.evaluate(() => {
      // Remove all fixed-position elements that could block clicks
      document.querySelectorAll('.fixed, [style*="position: fixed"]').forEach(el => {
        const htmlEl = el as HTMLElement;
        if (htmlEl.textContent?.includes('SpaceMouse') ||
            htmlEl.textContent?.includes('Secure Context') ||
            htmlEl.textContent?.includes('HTTPS')) {
          htmlEl.remove();
        }
      });

      // Also hide any elements with high z-index that aren't the control center
      document.querySelectorAll('[style*="z-index"]').forEach(el => {
        const htmlEl = el as HTMLElement;
        const zIndex = parseInt(getComputedStyle(htmlEl).zIndex || '0');
        if (zIndex > 100 && !htmlEl.classList.contains('control-center')) {
          htmlEl.style.display = 'none';
        }
      });
    });
    await page.waitForTimeout(500);

    await page.screenshot({ path: '/tmp/filter-kb-1-loaded.png' });

    // Get initial node count
    const initialNodeText = await page.locator('text=/Nodes.*\\d+/').first().textContent().catch(() => null);
    const initialMatch = initialNodeText?.match(/(\d[\d,]*)/);
    const initialCount = initialMatch ? parseInt(initialMatch[1].replace(/,/g, ''), 10) : null;
    console.log(`\n3. Initial node count: ${initialCount}`);

    // Use keyboard shortcuts instead of clicking
    // According to the config, Analytics tab has buttonKey: '4'
    console.log('\n4. Pressing keyboard shortcut "4" for Analytics tab...');

    // Focus on the page first
    await page.click('body', { force: true, timeout: 5000 }).catch(() => {});

    // Try keyboard shortcut
    await page.keyboard.press('4');
    await page.waitForTimeout(1000);

    // Check if tab switched
    let activeTabText = await page.evaluate(() => {
      return document.querySelector('[role="tab"][data-state="active"]')?.textContent?.trim();
    });
    console.log(`   Active tab after pressing '4': ${activeTabText}`);

    // If that didn't work, try focusing the Analytics tab directly and pressing Enter/Space
    if (!activeTabText?.includes('Analytics')) {
      console.log('   Keyboard shortcut didn\'t work. Trying focus + Enter...');

      // Find the Analytics tab and focus it
      const analyticsTab = page.locator('#radix-\\:r3\\:-trigger-analytics');
      const exists = await analyticsTab.count();
      console.log(`   Analytics tab by ID exists: ${exists > 0}`);

      if (exists > 0) {
        // Focus with force
        await analyticsTab.focus();
        await page.waitForTimeout(200);
        await page.keyboard.press('Enter');
        await page.waitForTimeout(500);

        activeTabText = await page.evaluate(() => {
          return document.querySelector('[role="tab"][data-state="active"]')?.textContent?.trim();
        });
        console.log(`   Active tab after Enter: ${activeTabText}`);
      }
    }

    // Try one more approach: dispatchEvent
    if (!activeTabText?.includes('Analytics')) {
      console.log('   Trying dispatchEvent approach...');

      await page.evaluate(() => {
        const tab = document.querySelector('[id*="trigger-analytics"]') as HTMLElement;
        if (tab) {
          // Simulate a real click event
          const clickEvent = new MouseEvent('click', {
            bubbles: true,
            cancelable: true,
            view: window,
            clientX: tab.getBoundingClientRect().left + 10,
            clientY: tab.getBoundingClientRect().top + 10
          });
          tab.dispatchEvent(clickEvent);

          // Also trigger pointerdown/pointerup
          tab.dispatchEvent(new PointerEvent('pointerdown', { bubbles: true }));
          tab.dispatchEvent(new PointerEvent('pointerup', { bubbles: true }));
        }
      });
      await page.waitForTimeout(1000);

      activeTabText = await page.evaluate(() => {
        return document.querySelector('[role="tab"][data-state="active"]')?.textContent?.trim();
      });
      console.log(`   Active tab after dispatchEvent: ${activeTabText}`);
    }

    await page.screenshot({ path: '/tmp/filter-kb-2-analytics.png' });

    // Check for filter controls
    console.log('\n5. Checking for filter controls...');
    const hasFilterLabel = await page.locator('label').filter({ hasText: /Filter by Quality/i }).count();
    console.log(`   "Filter by Quality" labels: ${hasFilterLabel}`);

    if (hasFilterLabel > 0) {
      console.log('\n6. Found filter controls! Enabling...');

      // Enable Filter by Quality toggle using keyboard
      await page.locator('label').filter({ hasText: /Filter by Quality/i }).first().click({ force: true, timeout: 5000 }).catch(() => {});
      await page.waitForTimeout(500);

      // Find toggle button next to label and click
      const toggleClicked = await page.evaluate(() => {
        const labels = document.querySelectorAll('label');
        for (const label of labels) {
          if (label.textContent?.includes('Filter by Quality')) {
            const button = label.closest('div')?.querySelector('button');
            if (button) {
              button.click();
              return true;
            }
          }
        }
        return false;
      });
      console.log(`   Toggle clicked: ${toggleClicked}`);

      await page.screenshot({ path: '/tmp/filter-kb-3-toggle.png' });

      // Set threshold
      console.log('\n7. Setting quality threshold...');
      await page.evaluate(() => {
        const sliders = document.querySelectorAll('input[type="range"]');
        sliders.forEach(slider => {
          const parent = slider.closest('div');
          if (parent?.textContent?.includes('Quality Threshold')) {
            (slider as HTMLInputElement).value = '0.8';
            slider.dispatchEvent(new Event('input', { bubbles: true }));
            slider.dispatchEvent(new Event('change', { bubbles: true }));
          }
        });
      });

      await page.waitForTimeout(15000);

      // Get final node count
      const finalNodeText = await page.locator('text=/Nodes.*\\d+/').first().textContent().catch(() => null);
      const finalMatch = finalNodeText?.match(/(\d[\d,]*)/);
      const finalCount = finalMatch ? parseInt(finalMatch[1].replace(/,/g, ''), 10) : null;
      console.log(`\n8. Final node count: ${finalCount}`);

      console.log('\n========================================');
      console.log('=== RESULTS ===');
      console.log('========================================');
      console.log(`Initial: ${initialCount}, Final: ${finalCount}`);

      if (initialCount && finalCount && finalCount < initialCount) {
        console.log(`\n✅ SUCCESS: Reduced by ${initialCount - finalCount} nodes`);
      }
    } else {
      console.log('\n❌ Filter controls not visible');

      // Debug info
      const pageContent = await page.evaluate(() => {
        const h3s = document.querySelectorAll('h3');
        return Array.from(h3s).map(h => h.textContent?.trim()).filter(Boolean);
      });
      console.log('   Visible section headers:', pageContent);
    }

    await page.screenshot({ path: '/tmp/filter-kb-4-final.png' });

  } catch (error) {
    console.error('Test error:', error);
    await page.screenshot({ path: '/tmp/filter-kb-error.png' });
  } finally {
    await browser.close();
  }

  console.log('\nTest completed. Check /tmp/filter-kb-*.png');
}

testFilterKeyboard().catch(console.error);
