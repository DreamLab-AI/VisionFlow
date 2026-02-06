import { test, expect } from '@playwright/test';

test('debug Analytics tab click behavior', async ({ page }) => {
  // Go to the app
  await page.goto('/', { waitUntil: 'networkidle' });
  await page.waitForTimeout(3000);

  // Take initial screenshot
  await page.screenshot({ path: '/tmp/debug-1-initial.png', fullPage: false });

  // Check initial tab state
  const tabs = await page.locator('[role="tab"]').all();
  console.log(`Found ${tabs.length} tabs`);

  // Find the Analytics tab specifically
  const analyticsTab = page.locator('[role="tab"]').filter({ hasText: 'Analytics' });
  const analyticsTabCount = await analyticsTab.count();
  console.log(`Found ${analyticsTabCount} Analytics tabs`);

  if (analyticsTabCount > 0) {
    // Get tab attributes
    const tabId = await analyticsTab.getAttribute('data-value') || await analyticsTab.getAttribute('value');
    const ariaSelected = await analyticsTab.getAttribute('aria-selected');
    console.log(`Analytics tab - value: ${tabId}, aria-selected: ${ariaSelected}`);

    // Click the Analytics tab
    await analyticsTab.click();
    await page.waitForTimeout(500);

    // Check aria-selected after click
    const ariaSelectedAfter = await analyticsTab.getAttribute('aria-selected');
    console.log(`After click aria-selected: ${ariaSelectedAfter}`);

    // Take screenshot after click
    await page.screenshot({ path: '/tmp/debug-2-after-analytics-click.png', fullPage: false });

    // Check the tabpanel content
    const tabPanels = await page.locator('[role="tabpanel"]').all();
    console.log(`Found ${tabPanels.length} tabpanels`);

    // Look for SemanticAnalysisPanel content
    const hasSemanticPanel = await page.locator('text=Semantic Analysis').count();
    const hasQualityThreshold = await page.locator('text=Quality Threshold').count();
    const hasAuthorityThreshold = await page.locator('text=Authority Threshold').count();
    const hasFilterPanel = await page.locator('text=Node Filter Settings').count();
    const hasNoSettings = await page.locator('text=No settings available').count();

    console.log(`SemanticAnalysisPanel markers:
      - "Semantic Analysis": ${hasSemanticPanel}
      - "Quality Threshold": ${hasQualityThreshold}
      - "Authority Threshold": ${hasAuthorityThreshold}
      - "Node Filter Settings": ${hasFilterPanel}
      - "No settings available": ${hasNoSettings}
    `);

    // Get all visible text in the tabpanel
    const tabpanel = page.locator('[role="tabpanel"]');
    const panelText = await tabpanel.textContent();
    console.log(`Tabpanel content: "${panelText?.substring(0, 500)}"`);

    // Check if the panel has any sliders
    const sliders = await page.locator('input[type="range"]').all();
    console.log(`Found ${sliders.length} sliders`);

    // Check for specific slider labels
    const qualitySlider = await page.locator('label:has-text("Quality Threshold")').count();
    const authoritySlider = await page.locator('label:has-text("Authority Threshold")').count();
    console.log(`Quality slider label: ${qualitySlider}, Authority slider label: ${authoritySlider}`);
  }

  // Final screenshot
  await page.screenshot({ path: '/tmp/debug-3-final.png', fullPage: false });
});
