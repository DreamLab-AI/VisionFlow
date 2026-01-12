import { test, expect } from '@playwright/test';

test.describe('Settings Panel Debug', () => {
  test('investigate settings panel structure', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');

    // Take initial screenshot
    await page.screenshot({ path: '/tmp/01-initial-page.png', fullPage: true });

    // Look for any settings-related elements
    const settingsButtons = await page.locator('[data-testid*="settings"], button:has-text("Settings"), .settings-toggle, [class*="settings"], [class*="Settings"]').all();
    console.log(`Found ${settingsButtons.length} settings-related elements`);

    // Find and click settings toggle if exists
    const settingsToggle = page.locator('[data-testid="settings-toggle"], button:has-text("Settings"), .settings-toggle, [aria-label*="settings" i]').first();
    if (await settingsToggle.isVisible()) {
      console.log('Clicking settings toggle...');
      await settingsToggle.click();
      // Wait for settings panel to appear
      await page.waitForSelector('[role="tabpanel"], [class*="panel" i], [class*="settings" i]', {
        state: 'visible',
        timeout: 5000
      }).catch(() => { /* Panel may already be visible */ });
      await page.screenshot({ path: '/tmp/02-after-settings-click.png', fullPage: true });
    }

    // Look for tabs or sections
    const tabs = await page.locator('[role="tab"], .tab, [class*="Tab"], button[class*="tab" i]').all();
    console.log(`Found ${tabs.length} tab elements`);
    for (const tab of tabs) {
      const text = await tab.textContent();
      console.log(`  Tab: ${text}`);
    }

    // Look for Analytics section
    const analyticsElements = await page.locator('text=Analytics, text=analytics, [class*="analytics" i]').all();
    console.log(`Found ${analyticsElements.length} analytics-related elements`);

    // Look for Node Filter or Node Confidence
    const nodeFilterElements = await page.locator('text=Node Filter, text=Node Confidence, text=Quality Threshold, text=nodeFilter, [class*="nodeFilter" i]').all();
    console.log(`Found ${nodeFilterElements.length} node filter-related elements`);

    // Check all visible text on the page for debugging
    const bodyText = await page.locator('body').textContent();
    console.log('Page contains "Analytics":', bodyText?.includes('Analytics'));
    console.log('Page contains "Node":', bodyText?.includes('Node'));
    console.log('Page contains "Filter":', bodyText?.includes('Filter'));
    console.log('Page contains "Quality":', bodyText?.includes('Quality'));
    console.log('Page contains "Threshold":', bodyText?.includes('Threshold'));

    // Try clicking on various tabs if they exist
    const allTabs = await page.locator('[role="tab"], .tab, button[class*="tab" i]').all();
    for (let i = 0; i < allTabs.length && i < 10; i++) {
      const tabText = await allTabs[i].textContent();
      console.log(`Clicking tab ${i}: ${tabText}`);
      await allTabs[i].click();
      // Wait for tab panel content to update
      await page.waitForSelector('[role="tabpanel"]', { state: 'visible', timeout: 3000 })
        .catch(() => { /* Tab panel may not exist for this tab */ });
      await page.screenshot({ path: `/tmp/03-tab-${i}.png`, fullPage: true });
    }

    // Final screenshot
    await page.screenshot({ path: '/tmp/04-final-state.png', fullPage: true });
  });

  test('check settings panel DOM structure', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');

    // Get all the panel/sidebar elements
    const panels = await page.locator('[class*="panel" i], [class*="sidebar" i], [class*="drawer" i], aside').all();
    console.log(`Found ${panels.length} panel/sidebar elements`);

    // Check for accordion or expandable sections
    const accordions = await page.locator('[class*="accordion" i], [class*="collapse" i], [class*="expand" i], details').all();
    console.log(`Found ${accordions.length} accordion/expandable elements`);

    // Look for form controls that might be settings
    const formControls = await page.locator('input[type="checkbox"], input[type="range"], select, [role="slider"]').all();
    console.log(`Found ${formControls.length} form controls`);

    // Print class names of main containers
    const mainContainers = await page.locator('div[class]').all();
    const uniqueClasses = new Set<string>();
    for (const container of mainContainers.slice(0, 50)) {
      const className = await container.getAttribute('class');
      if (className) {
        className.split(' ').forEach(c => {
          if (c.toLowerCase().includes('setting') ||
              c.toLowerCase().includes('panel') ||
              c.toLowerCase().includes('filter') ||
              c.toLowerCase().includes('analytics')) {
            uniqueClasses.add(c);
          }
        });
      }
    }
    console.log('Relevant class names found:', Array.from(uniqueClasses));
  });
});
