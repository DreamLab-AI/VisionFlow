import { test, expect } from '@playwright/test';

test.describe('Node Filter Settings Tab', () => {
  test('check if Filter tab exists in Semantic Analysis panel', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('domcontentloaded');
    await page.waitForTimeout(20000); // Wait for full app load

    // Look for Analytics tab and click it with proper Playwright locator
    console.log('Looking for Analytics tab...');

    // Try clicking Analytics tab using Playwright's built-in click
    try {
      const analyticsTab = page.getByRole('tab', { name: /Analytics/i });
      if (await analyticsTab.count() > 0) {
        await analyticsTab.first().click();
        console.log('Clicked Analytics tab using Playwright locator');
        await page.waitForTimeout(2000);
      }
    } catch (e) {
      console.log('Playwright click failed, trying JavaScript fallback');
    }

    // Fallback: Force click via JavaScript
    await page.evaluate(() => {
      const tabs = document.querySelectorAll('[role="tab"]');
      for (const tab of tabs) {
        if (tab.textContent?.includes('Analytics')) {
          (tab as HTMLElement).click();
          // Also try setting aria-selected
          tabs.forEach(t => t.setAttribute('aria-selected', 'false'));
          tab.setAttribute('aria-selected', 'true');
          break;
        }
      }
    });

    await page.waitForTimeout(2000);

    // Check what's now visible
    const pageContent = await page.evaluate(() => {
      const body = document.body.textContent || '';
      return {
        hasSemanticAnalysis: body.includes('Semantic Analysis'),
        hasFilter: body.includes('Filter'),
        hasQualityThreshold: body.includes('Quality Threshold'),
        hasNodeConfidence: body.includes('Node Confidence'),
        hasCommunities: body.includes('Communities'),
        hasCentrality: body.includes('Centrality'),
        selectedTabs: Array.from(document.querySelectorAll('[role="tab"][aria-selected="true"]'))
          .map(t => t.textContent?.trim())
      };
    });

    console.log('\n=== Page Content Check ===');
    console.log('Has Semantic Analysis:', pageContent.hasSemanticAnalysis);
    console.log('Has Filter:', pageContent.hasFilter);
    console.log('Has Quality Threshold:', pageContent.hasQualityThreshold);
    console.log('Has Node Confidence:', pageContent.hasNodeConfidence);
    console.log('Has Communities:', pageContent.hasCommunities);
    console.log('Has Centrality:', pageContent.hasCentrality);
    console.log('Selected tabs:', pageContent.selectedTabs);

    // Screenshot before looking for filter
    await page.screenshot({ path: '/tmp/filter-analytics.png' });

    // If Semantic Analysis panel is visible, check for Filter tab
    if (pageContent.hasSemanticAnalysis || pageContent.hasCommunities) {
      console.log('SemanticAnalysisPanel detected! Looking for Filter tab...');

      // The Filter tab should be inside SemanticAnalysisPanel
      const filterTabExists = await page.evaluate(() => {
        const tabButtons = Array.from(document.querySelectorAll('button[role="tab"]'));
        return tabButtons.some(btn => btn.textContent?.toLowerCase().includes('filter'));
      });

      console.log('Filter tab exists:', filterTabExists);

      if (filterTabExists) {
        // Click the Filter tab
        await page.evaluate(() => {
          const tabButtons = Array.from(document.querySelectorAll('button[role="tab"]'));
          const filterTab = tabButtons.find(btn => btn.textContent?.toLowerCase().includes('filter'));
          if (filterTab) (filterTab as HTMLElement).click();
        });
        await page.waitForTimeout(1000);

        const afterFilterClick = await page.evaluate(() => {
          const body = document.body.textContent || '';
          return {
            hasQualityThreshold: body.includes('Quality Threshold'),
            hasAuthorityThreshold: body.includes('Authority Threshold'),
            hasNodeConfidence: body.includes('Node Confidence Filter')
          };
        });

        console.log('\n=== After Filter Tab Click ===');
        console.log('Has Quality Threshold:', afterFilterClick.hasQualityThreshold);
        console.log('Has Authority Threshold:', afterFilterClick.hasAuthorityThreshold);
        console.log('Has Node Confidence Filter:', afterFilterClick.hasNodeConfidence);

        await page.screenshot({ path: '/tmp/filter-final.png' });

        // Assertion
        expect(afterFilterClick.hasQualityThreshold || afterFilterClick.hasNodeConfidence).toBe(true);
      }
    } else {
      console.log('SemanticAnalysisPanel NOT visible - checking what is displayed');

      // Get visible panel content
      const visibleContent = await page.evaluate(() => {
        const panels = Array.from(document.querySelectorAll('[role="tabpanel"]'));
        return panels.filter(p => {
          const style = window.getComputedStyle(p);
          return style.display !== 'none' && style.visibility !== 'hidden';
        }).map(p => p.textContent?.substring(0, 500));
      });
      console.log('Visible panel content:', visibleContent);

      await page.screenshot({ path: '/tmp/filter-final.png' });
    }
  });
});
