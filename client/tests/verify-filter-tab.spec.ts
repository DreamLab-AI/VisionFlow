import { test, expect } from '@playwright/test';

test('verify Filter tab with Node Filter Settings', async ({ page }) => {
  await page.goto('/');
  await page.waitForLoadState('domcontentloaded');
  await page.waitForTimeout(20000); // Wait for app to fully load

  // Take initial screenshot
  await page.screenshot({ path: '/tmp/verify-1-initial.png' });

  // Find and click Analytics tab
  console.log('Step 1: Looking for Analytics tab...');

  const allTabs = await page.evaluate(() => {
    return Array.from(document.querySelectorAll('[role="tab"]'))
      .map(t => ({ text: t.textContent?.trim(), selected: t.getAttribute('aria-selected') }));
  });
  console.log('Available tabs:', JSON.stringify(allTabs));

  // Click Analytics tab
  const analyticsClicked = await page.evaluate(() => {
    const tabs = document.querySelectorAll('[role="tab"]');
    for (const tab of tabs) {
      if (tab.textContent?.toLowerCase().includes('analytics')) {
        (tab as HTMLElement).click();
        return true;
      }
    }
    return false;
  });
  console.log('Analytics tab clicked:', analyticsClicked);
  await page.waitForTimeout(2000);
  await page.screenshot({ path: '/tmp/verify-2-after-analytics.png' });

  // Now look for tab panel content
  const tabContent = await page.evaluate(() => {
    // Look for SemanticAnalysisPanel or Filter-related content
    const body = document.body.textContent || '';
    return {
      hasSemanticAnalysis: body.includes('Semantic Analysis'),
      hasFilter: body.includes('Filter'),
      hasQualityThreshold: body.includes('Quality Threshold'),
      hasAuthorityThreshold: body.includes('Authority Threshold'),
      hasNodeConfidence: body.includes('Node Confidence'),
      hasCommunities: body.includes('Communities'),
      hasCentrality: body.includes('Centrality'),
      hasMetrics: body.includes('Metrics'),
    };
  });
  console.log('Tab content after Analytics click:', JSON.stringify(tabContent, null, 2));

  // Look for inner tabs (Filter, Communities, Centrality, Metrics)
  const innerTabs = await page.evaluate(() => {
    // SemanticAnalysisPanel uses TabsTrigger buttons
    const buttons = Array.from(document.querySelectorAll('button'));
    const tabLikeButtons = buttons.filter(btn => {
      const text = btn.textContent?.toLowerCase() || '';
      return ['filter', 'communities', 'centrality', 'metrics'].some(t => text.includes(t));
    });
    return tabLikeButtons.map(b => b.textContent?.trim());
  });
  console.log('Inner tabs found:', innerTabs);

  // If Filter tab exists, click it
  if (innerTabs.some(t => t?.toLowerCase().includes('filter'))) {
    console.log('Step 2: Clicking Filter tab...');
    await page.evaluate(() => {
      const buttons = Array.from(document.querySelectorAll('button'));
      const filterBtn = buttons.find(btn => btn.textContent?.toLowerCase().includes('filter'));
      if (filterBtn) (filterBtn as HTMLElement).click();
    });
    await page.waitForTimeout(1000);
    await page.screenshot({ path: '/tmp/verify-3-after-filter.png' });
  }

  // Final content check
  const finalContent = await page.evaluate(() => {
    const body = document.body.textContent || '';
    // Look for slider-related elements
    const sliders = document.querySelectorAll('input[type="range"]');
    const sliderLabels = Array.from(document.querySelectorAll('label')).map(l => l.textContent?.trim());

    return {
      hasQualityThreshold: body.includes('Quality Threshold'),
      hasAuthorityThreshold: body.includes('Authority Threshold'),
      hasNodeConfidenceFilter: body.includes('Node Confidence Filter'),
      sliderCount: sliders.length,
      labels: sliderLabels.filter(l => l && l.length < 50).slice(0, 20)
    };
  });
  console.log('Final content check:', JSON.stringify(finalContent, null, 2));

  await page.screenshot({ path: '/tmp/verify-4-final.png' });

  // Assert that Filter settings are visible
  const hasFilterSettings = finalContent.hasQualityThreshold ||
                           finalContent.hasAuthorityThreshold ||
                           finalContent.hasNodeConfidenceFilter;

  console.log('\n=== VERIFICATION RESULT ===');
  console.log('Filter settings visible:', hasFilterSettings);

  expect(hasFilterSettings).toBe(true);
});
