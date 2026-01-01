#!/usr/bin/env node
/**
 * Playwright Screenshot Script
 * Takes screenshots of the VisionFlow website via HTTPS bridge
 * Uses system Chromium instead of Playwright's bundled browser
 */

const { chromium } = require('playwright');
const path = require('path');

const SCREENSHOT_DIR = process.env.SCREENSHOT_DIR || '/home/devuser/workspace/project/screenshots';
const URL = process.env.TARGET_URL || 'https://localhost:3001';

async function takeScreenshots() {
  console.log('Launching system Chromium browser...');

  const browser = await chromium.launch({
    executablePath: '/usr/bin/chromium',  // Use system chromium
    headless: false,  // Use headed mode for VNC display
    args: [
      '--no-sandbox',
      '--disable-setuid-sandbox',
      '--disable-dev-shm-usage',
      '--ignore-certificate-errors',  // Accept self-signed cert
      '--disable-web-security',
      '--disable-gpu'  // For Xvfb compatibility
    ]
  });

  const context = await browser.newContext({
    ignoreHTTPSErrors: true,
    viewport: { width: 1920, height: 1080 }
  });

  const page = await context.newPage();

  console.log(`Navigating to ${URL}...`);

  try {
    await page.goto(URL, {
      waitUntil: 'networkidle',
      timeout: 30000
    });

    console.log('Page loaded successfully');

    // Take full page screenshot
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const fullPagePath = path.join(SCREENSHOT_DIR, `visionflow-fullpage-${timestamp}.png`);
    await page.screenshot({
      path: fullPagePath,
      fullPage: true
    });
    console.log(`Full page screenshot saved: ${fullPagePath}`);

    // Take viewport screenshot
    const viewportPath = path.join(SCREENSHOT_DIR, `visionflow-viewport-${timestamp}.png`);
    await page.screenshot({
      path: viewportPath,
      fullPage: false
    });
    console.log(`Viewport screenshot saved: ${viewportPath}`);

    // Get page title
    const title = await page.title();
    console.log(`Page title: ${title}`);

    // Wait a bit to see it on VNC
    await page.waitForTimeout(3000);

  } catch (error) {
    console.error('Error:', error.message);

    // Take error screenshot anyway
    const errorPath = path.join(SCREENSHOT_DIR, `error-screenshot-${Date.now()}.png`);
    try {
      await page.screenshot({ path: errorPath });
      console.log(`Error screenshot saved: ${errorPath}`);
    } catch (e) {
      console.error('Could not take error screenshot');
    }
  }

  await browser.close();
  console.log('Browser closed');
}

takeScreenshots().catch(console.error);
