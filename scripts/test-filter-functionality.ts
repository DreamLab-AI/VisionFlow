import { chromium } from 'playwright';

async function testFilterFunctionality() {
  console.log('Starting filter functionality test...');

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
    if (text.includes('filter') || text.includes('Filter') || text.includes('node') || text.includes('Node')) {
      console.log(`RELEVANT: ${text}`);
    }
  });

  try {
    console.log('Navigating to VisionFlow...');
    await page.goto('http://192.168.0.51:3001', { waitUntil: 'domcontentloaded', timeout: 30000 });
    await page.waitForTimeout(5000);

    // Dismiss SpaceMouse dialog if present
    const dismissButton = await page.locator('button:has-text("Dismiss")').first();
    if (await dismissButton.isVisible().catch(() => false)) {
      await dismissButton.click();
      console.log('Dismissed SpaceMouse dialog');
      await page.waitForTimeout(500);
    }

    // Take initial screenshot
    await page.screenshot({ path: '/tmp/filter-func-1-initial.png' });
    console.log('Screenshot 1: Initial state');

    // Look for node count in System Status
    const nodeCountRegex = /(\d+)\s*nodes/i;
    const pageContent = await page.content();
    const nodeMatch = pageContent.match(nodeCountRegex);
    if (nodeMatch) {
      console.log(`Initial node count from page: ${nodeMatch[1]}`);
    }

    // Find the Enable Filtering toggle
    console.log('\n=== Looking for filter controls ===');

    // Look for "Enable Filtering" text/checkbox
    const filterToggle = await page.locator('text=Enable Filtering').first();
    if (await filterToggle.isVisible().catch(() => false)) {
      console.log('Found "Enable Filtering" label');

      // Find the associated checkbox/toggle - it might be a sibling or parent element
      const checkbox = await page.locator('input[type="checkbox"]').first();
      if (await checkbox.isVisible().catch(() => false)) {
        const isChecked = await checkbox.isChecked();
        console.log(`Checkbox found, checked: ${isChecked}`);

        if (!isChecked) {
          console.log('Enabling filtering...');
          await checkbox.click();
          await page.waitForTimeout(2000);
          await page.screenshot({ path: '/tmp/filter-func-2-enabled.png' });
          console.log('Screenshot 2: Filtering enabled');
        }
      }
    }

    // Look for Quality Threshold slider
    const qualitySlider = await page.locator('input[type="range"]').first();
    if (await qualitySlider.isVisible().catch(() => false)) {
      console.log('Found quality threshold slider');

      // Get current value
      const currentValue = await qualitySlider.inputValue();
      console.log(`Current slider value: ${currentValue}`);

      // Set to high value (0.8) to filter aggressively
      console.log('Setting quality threshold to 0.8 (aggressive filtering)...');
      await qualitySlider.fill('0.8');
      await page.waitForTimeout(2000);

      await page.screenshot({ path: '/tmp/filter-func-3-high-threshold.png' });
      console.log('Screenshot 3: High threshold set');

      // Check new node count
      const newPageContent = await page.content();
      const newNodeMatch = newPageContent.match(nodeCountRegex);
      if (newNodeMatch) {
        console.log(`Node count after filtering: ${newNodeMatch[1]}`);
      }
    }

    // Look for specific UI elements that might show filter status
    console.log('\n=== Checking for filter status indicators ===');

    // Look for "Filtered" text anywhere
    const filteredText = await page.locator('text=/filtered/i').first();
    if (await filteredText.isVisible().catch(() => false)) {
      const text = await filteredText.textContent();
      console.log(`Found filtered indicator: ${text}`);
    }

    // Look for node count display that might update
    const nodeDisplay = await page.locator('[class*="node"], [class*="count"], [class*="status"]').all();
    for (const el of nodeDisplay.slice(0, 5)) {
      const text = await el.textContent().catch(() => '');
      if (text && text.includes('node')) {
        console.log(`Node display element: ${text.trim().substring(0, 100)}`);
      }
    }

    // Print WebSocket activity from logs
    console.log('\n=== WebSocket Activity ===');
    const wsLogs = logs.filter(l => l.toLowerCase().includes('websocket') || l.toLowerCase().includes('ws') || l.toLowerCase().includes('socket'));
    wsLogs.slice(-10).forEach(l => console.log(l));

    // Print filter-related logs
    console.log('\n=== Filter-Related Logs ===');
    const filterLogs = logs.filter(l => l.toLowerCase().includes('filter'));
    filterLogs.slice(-10).forEach(l => console.log(l));

    // Final screenshot
    await page.screenshot({ path: '/tmp/filter-func-4-final.png', fullPage: true });
    console.log('Screenshot 4: Final state');

  } catch (error) {
    console.error('Test error:', error);
    await page.screenshot({ path: '/tmp/filter-func-error.png' });
  } finally {
    await browser.close();
  }

  console.log('\nTest completed. Check screenshots in /tmp/filter-func-*.png');
}

testFilterFunctionality().catch(console.error);
