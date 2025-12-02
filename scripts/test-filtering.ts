import { chromium } from 'playwright';

async function testFiltering() {
  console.log('Starting filtering test...');

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
    if (text.includes('filter') || text.includes('Filter') || text.includes('auth') || text.includes('nostr')) {
      console.log(`RELEVANT LOG: ${text}`);
    }
  });

  try {
    console.log('Navigating to VisionFlow at http://192.168.0.51:3001 ...');
    await page.goto('http://192.168.0.51:3001', { waitUntil: 'domcontentloaded', timeout: 30000 });

    // Wait for app to load
    await page.waitForTimeout(8000);
    console.log('App loaded');

    // Take screenshot of initial state
    await page.screenshot({ path: '/tmp/filter-test-1-initial.png', fullPage: true });
    console.log('Screenshot 1: Initial state saved');

    // Check if we see a login screen or the main app
    const loginButton = await page.locator('text=Login with Nostr').first();
    const hasLoginScreen = await loginButton.isVisible().catch(() => false);

    if (hasLoginScreen) {
      console.log('Login screen detected - Nostr auth is working!');
      await page.screenshot({ path: '/tmp/filter-test-2-login-screen.png' });
      console.log('Screenshot 2: Login screen');
    } else {
      console.log('No login screen - checking for Control Center...');

      // Look for Control Center
      const controlCenter = await page.locator('text=Control Center').first();
      const hasControlCenter = await controlCenter.isVisible().catch(() => false);

      if (hasControlCenter) {
        console.log('Control Center found');

        // Look for Graph Visualization section
        const graphViz = await page.locator('text=Graph Visualization').first();
        if (await graphViz.isVisible().catch(() => false)) {
          console.log('Graph Visualization section found');
        }

        // Look for filter controls
        const filterToggle = await page.locator('text=Enable Filtering').first();
        if (await filterToggle.isVisible().catch(() => false)) {
          console.log('Enable Filtering toggle found!');
        }

        // Look for quality threshold
        const qualityThreshold = await page.locator('text=Quality Threshold').first();
        if (await qualityThreshold.isVisible().catch(() => false)) {
          console.log('Quality Threshold slider found!');
        }

        // Look for system status
        const systemStatus = await page.locator('text=System Status').first();
        if (await systemStatus.isVisible().catch(() => false)) {
          console.log('System Status section found!');
        }

        // Look for node count display
        const nodeCount = await page.locator('text=/\\d+ nodes/').first();
        if (await nodeCount.isVisible().catch(() => false)) {
          const text = await nodeCount.textContent();
          console.log(`Node count display: ${text}`);
        }

        await page.screenshot({ path: '/tmp/filter-test-3-control-center.png' });
        console.log('Screenshot 3: Control Center');
      }
    }

    // Print recent logs
    console.log('\n=== Recent Console Logs ===');
    logs.slice(-20).forEach(l => console.log(l));

  } catch (error) {
    console.error('Test error:', error);
    await page.screenshot({ path: '/tmp/filter-test-error.png' });
  } finally {
    await browser.close();
  }

  console.log('\nTest completed. Check screenshots in /tmp/');
}

testFiltering().catch(console.error);
