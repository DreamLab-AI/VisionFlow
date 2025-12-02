import { chromium } from 'playwright';

async function testFilterSync() {
  console.log('Starting filter sync test...\n');

  const browser = await chromium.launch({
    headless: false,
    args: ['--no-sandbox', '--disable-setuid-sandbox']
  });

  const page = await browser.newPage();

  // Collect console logs for filter-related activity
  const logs: string[] = [];
  page.on('console', msg => {
    const text = msg.text();
    logs.push(`[${msg.type()}] ${text}`);
    if (text.includes('Filter') || text.includes('sent to server') ||
        text.includes('filter_update') || text.includes('Filter update')) {
      console.log(`>>> ${text}`);
    }
  });

  try {
    console.log('1. Navigating to VisionFlow...');
    await page.goto('http://192.168.0.51:3001', { waitUntil: 'domcontentloaded', timeout: 30000 });
    await page.waitForTimeout(5000);

    // Dismiss any dialogs or banners by clicking away or pressing Escape
    await page.keyboard.press('Escape');
    await page.waitForTimeout(500);

    // Dismiss SpaceMouse dialog if present
    const dismissButton = await page.locator('button:has-text("Dismiss")').first();
    if (await dismissButton.isVisible().catch(() => false)) {
      await dismissButton.click();
      console.log('   Dismissed SpaceMouse dialog');
      await page.waitForTimeout(500);
    }

    // Close any warning banners
    const closeButton = await page.locator('.fixed.top-0 button, .z-50 button').first();
    if (await closeButton.isVisible().catch(() => false)) {
      try {
        await closeButton.click({ timeout: 2000 });
        console.log('   Closed warning banner');
        await page.waitForTimeout(500);
      } catch (e) {
        // Ignore
      }
    }

    await page.screenshot({ path: '/tmp/filter-sync-1-initial.png' });
    console.log('   Screenshot 1: Initial state\n');

    // Find sliders directly
    console.log('2. Looking for sliders...');
    const sliders = await page.locator('input[type="range"]').all();
    console.log(`   Found ${sliders.length} sliders`);

    if (sliders.length > 0) {
      const qualitySlider = sliders[0];
      const initialValue = await qualitySlider.inputValue();
      console.log(`   Initial slider value: ${initialValue}`);

      // Change the slider value
      console.log('\n3. Changing slider value to 0.5...');
      await qualitySlider.evaluate((el: HTMLInputElement) => {
        el.value = '0.5';
        el.dispatchEvent(new Event('input', { bubbles: true }));
        el.dispatchEvent(new Event('change', { bubbles: true }));
      });
      await page.waitForTimeout(3000);

      await page.screenshot({ path: '/tmp/filter-sync-2-changed.png' });
      console.log('   Screenshot 2: After slider change');

      // Check for filter update logs
      const newLogs = logs.filter(l => l.includes('Filter') || l.includes('filter'));
      if (newLogs.length > 0) {
        console.log('\n   Recent filter logs:');
        newLogs.slice(-5).forEach(l => console.log('   ' + l));
      }

      // Change slider again
      console.log('\n4. Changing slider to 0.3...');
      await qualitySlider.evaluate((el: HTMLInputElement) => {
        el.value = '0.3';
        el.dispatchEvent(new Event('input', { bubbles: true }));
        el.dispatchEvent(new Event('change', { bubbles: true }));
      });
      await page.waitForTimeout(3000);

      await page.screenshot({ path: '/tmp/filter-sync-3-final.png' });
      console.log('   Screenshot 3: After second change');
    }

    // Print all filter-related logs
    console.log('\n=== All Filter-Related Logs ===');
    const filterLogs = logs.filter(l =>
      l.toLowerCase().includes('filter') ||
      l.toLowerCase().includes('sent to server')
    );
    filterLogs.forEach(l => console.log(l));

    // Check server logs
    console.log('\n5. Checking server logs for filter_update...');

  } catch (error) {
    console.error('Test error:', error);
    await page.screenshot({ path: '/tmp/filter-sync-error.png' });
  } finally {
    await browser.close();
  }

  console.log('\nTest completed.');
}

testFilterSync().catch(console.error);
