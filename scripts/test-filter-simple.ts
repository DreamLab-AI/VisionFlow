import { chromium } from 'playwright';

async function testFilterSimple() {
  console.log('Starting simple filter test...\n');

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
    if (text.includes('Filter') || text.includes('filter') ||
        text.includes('sent to server') || text.includes('subscription')) {
      console.log(`>>> ${text}`);
    }
  });

  try {
    console.log('1. Navigating to VisionFlow...');
    await page.goto('http://192.168.0.51:3001', { waitUntil: 'domcontentloaded', timeout: 30000 });
    await page.waitForTimeout(5000);

    // Hide the SpaceMouse warning banner via JavaScript
    console.log('2. Hiding any blocking banners...');
    await page.evaluate(() => {
      // Find and hide the orange/red warning banner
      const banner = document.querySelector('.fixed.top-0.z-50');
      if (banner) {
        (banner as HTMLElement).style.display = 'none';
        console.log('Hidden SpaceMouse banner');
      }
      // Also try by background color
      document.querySelectorAll('div').forEach(el => {
        const style = window.getComputedStyle(el);
        if (style.position === 'fixed' && style.top === '0px' && style.zIndex === '50') {
          (el as HTMLElement).style.display = 'none';
        }
      });
    });
    await page.waitForTimeout(500);

    await page.screenshot({ path: '/tmp/filter-simple-1-initial.png' });
    console.log('   Screenshot 1: Initial\n');

    // Find and interact with the first slider using force click
    console.log('3. Finding and clicking slider...');
    const slider = await page.locator('input[type="range"]').first();

    if (await slider.isVisible().catch(() => false)) {
      // Use force click to bypass any overlays
      await slider.click({ force: true });
      await page.waitForTimeout(500);

      // Change value using keyboard
      console.log('4. Changing slider value via keyboard...');
      await page.keyboard.press('Home');
      await page.waitForTimeout(200);

      for (let i = 0; i < 5; i++) {
        await page.keyboard.press('ArrowRight');
        await page.waitForTimeout(200);
      }

      console.log('   Slider changed');
      await page.waitForTimeout(2000);

      await page.screenshot({ path: '/tmp/filter-simple-2-changed.png' });
    } else {
      console.log('   Slider not visible');
    }

    // Check logs
    console.log('\n=== Filter-Related Console Logs ===');
    const filterLogs = logs.filter(l =>
      l.toLowerCase().includes('filter') ||
      l.toLowerCase().includes('sent to server')
    );
    if (filterLogs.length === 0) {
      console.log('No filter-related logs found');
    } else {
      filterLogs.forEach(l => console.log(l));
    }

    // Final screenshot
    await page.screenshot({ path: '/tmp/filter-simple-3-final.png', fullPage: true });
    console.log('\nScreenshot 3: Final');

  } catch (error) {
    console.error('Test error:', error);
    await page.screenshot({ path: '/tmp/filter-simple-error.png' });
  } finally {
    await browser.close();
  }

  console.log('\nTest completed.');
}

testFilterSimple().catch(console.error);
