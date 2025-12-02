import { chromium } from 'playwright';

async function testNodeFilter() {
  console.log('Starting node filter test...');

  const browser = await chromium.launch({
    headless: false,
    args: ['--no-sandbox', '--disable-setuid-sandbox']
  });

  const page = await browser.newPage();

  // Collect console logs
  const logs: string[] = [];
  page.on('console', msg => {
    const text = msg.text();
    logs.push(`[${msg.type()}] ${text}`);
    if (text.includes('NodeFilter') || text.includes('node filter') || text.includes('filtering')) {
      console.log(`FILTER LOG: ${text}`);
    }
  });

  try {
    console.log('Navigating to VisionFlow...');
    await page.goto('http://localhost:5173', { waitUntil: 'networkidle', timeout: 30000 });

    // Wait for app to load
    await page.waitForTimeout(5000);
    console.log('App loaded, waiting for graph...');

    // Wait for graph nodes to appear
    await page.waitForTimeout(3000);

    // Take initial screenshot
    await page.screenshot({ path: '/tmp/filter-test-1-initial.png' });
    console.log('Screenshot 1: Initial state');

    // Print relevant console logs
    console.log('\n=== Console logs (filtered) ===');
    logs.filter(l =>
      l.includes('NodeFilter') ||
      l.includes('settings') ||
      l.includes('filtering') ||
      l.includes('visible')
    ).forEach(l => console.log(l));

    // Try to find and interact with filter controls
    // Look for filter toggle or slider in Control Center
    const filterToggle = await page.locator('text=Enable Filtering').first();
    if (await filterToggle.isVisible()) {
      console.log('\nFound Enable Filtering toggle');

      // Check if there's a nearby checkbox/switch
      const toggle = await page.locator('[data-testid*="filter"], [class*="filter"] input[type="checkbox"], [class*="toggle"]').first();
      if (await toggle.isVisible()) {
        console.log('Clicking filter toggle...');
        await toggle.click();
        await page.waitForTimeout(2000);
      }
    }

    // Look for quality threshold slider
    const qualitySlider = await page.locator('text=Quality Threshold').first();
    if (await qualitySlider.isVisible()) {
      console.log('\nFound Quality Threshold control');
    }

    // Take screenshot after attempting filter interaction
    await page.screenshot({ path: '/tmp/filter-test-2-after-toggle.png' });
    console.log('Screenshot 2: After toggle attempt');

    // Print final logs
    console.log('\n=== Final console logs ===');
    logs.slice(-30).forEach(l => console.log(l));

  } catch (error) {
    console.error('Test error:', error);
    await page.screenshot({ path: '/tmp/filter-test-error.png' });
  } finally {
    await browser.close();
  }

  console.log('\nTest completed. Check screenshots in /tmp/');
}

testNodeFilter().catch(console.error);
