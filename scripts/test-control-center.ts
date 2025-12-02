import { chromium } from 'playwright';

async function testControlCenter() {
  console.log('Starting Control Center test...');

  const browser = await chromium.launch({
    headless: false,
    args: ['--no-sandbox'],
  });

  const page = await browser.newPage();

  // Log all console messages
  page.on('console', msg => console.log('Browser:', msg.type(), msg.text()));
  page.on('pageerror', err => console.log('Page Error:', err.message));

  try {
    console.log('Navigating to VisionFlow...');
    await page.goto('http://172.18.0.11:3001', { timeout: 60000, waitUntil: 'domcontentloaded' });
    console.log('Page loaded');

    await page.waitForTimeout(8000);

    // Take screenshot
    await page.screenshot({ path: '/tmp/cc-unified-test.png', timeout: 10000 });
    console.log('Screenshot saved to /tmp/cc-unified-test.png');

    // Keep browser open for inspection
    console.log('Keeping browser open for 90 seconds...');
    await page.waitForTimeout(90000);

  } catch (error) {
    console.error('Error:', error);
    await page.screenshot({ path: '/tmp/cc-error.png', timeout: 5000 }).catch(() => {});
  } finally {
    await browser.close();
  }
}

testControlCenter().catch(console.error);
