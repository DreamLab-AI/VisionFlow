#!/usr/bin/env node

/**
 * Playwright MCP Demo Script
 * Demonstrates the capabilities of Playwright MCP Server
 */

const { chromium } = require('playwright');

async function runDemo() {
  console.log('🎭 Playwright MCP Demo');
  console.log('======================\n');

  console.log('📋 Available MCP Tools:');
  console.log('  • playwright_navigate - Navigate to a URL');
  console.log('  • playwright_screenshot - Capture screenshots');
  console.log('  • playwright_click - Click elements');
  console.log('  • playwright_fill - Fill form fields');
  console.log('  • playwright_evaluate - Execute JavaScript');
  console.log('  • playwright_wait_for_selector - Wait for elements');
  console.log('  • playwright_get_content - Get page content\n');

  console.log('🚀 Starting browser demo...\n');

  try {
    // Launch browser
    const browser = await chromium.launch({
      headless: true,
      args: ['--no-sandbox', '--disable-setuid-sandbox']
    });
    
    const context = await browser.newContext({
      viewport: { width: 1280, height: 720 }
    });
    
    const page = await context.newPage();
    
    // Demo 1: Navigation
    console.log('1️⃣ Navigating to Playwright.dev...');
    await page.goto('https://playwright.dev/');
    console.log('   ✅ Page loaded successfully');
    
    // Demo 2: Screenshot
    console.log('\n2️⃣ Taking screenshot...');
    await page.screenshot({ path: '/workspace/playwright-demo-homepage.png' });
    console.log('   ✅ Screenshot saved to: /workspace/playwright-demo-homepage.png');
    
    // Demo 3: Interaction
    console.log('\n3️⃣ Searching documentation...');
    await page.click('button[aria-label="Search"]');
    await page.fill('input[type="search"]', 'MCP Server');
    console.log('   ✅ Search performed');
    
    // Demo 4: Evaluation
    console.log('\n4️⃣ Evaluating JavaScript in page context...');
    const title = await page.evaluate(() => document.title);
    console.log(`   ✅ Page title: "${title}"`);
    
    // Demo 5: Element inspection
    console.log('\n5️⃣ Inspecting page elements...');
    const navLinks = await page.$$eval('nav a', links => 
      links.slice(0, 5).map(link => ({
        text: link.textContent,
        href: link.href
      }))
    );
    console.log('   ✅ Navigation links found:');
    navLinks.forEach(link => {
      console.log(`      - ${link.text}`);
    });
    
    // Demo 6: Mobile emulation
    console.log('\n6️⃣ Testing mobile viewport...');
    await page.setViewportSize({ width: 375, height: 667 });
    await page.screenshot({ path: '/workspace/playwright-demo-mobile.png' });
    console.log('   ✅ Mobile screenshot saved');
    
    // Demo 7: Network monitoring
    console.log('\n7️⃣ Monitoring network activity...');
    const requests = [];
    page.on('request', request => requests.push(request.url()));
    await page.reload();
    console.log(`   ✅ Captured ${requests.length} network requests`);
    
    await browser.close();
    
    console.log('\n✨ Demo completed successfully!');
    console.log('\n💡 Tips:');
    console.log('  • Use `playwright-server` to start the MCP server');
    console.log('  • Use `playwright-codegen` to record browser actions');
    console.log('  • Check /workspace for generated screenshots');
    console.log('  • Run `playwright-test` to execute test suites\n');

  } catch (error) {
    console.error('❌ Demo error:', error.message);
    console.log('\n💡 Troubleshooting:');
    console.log('  • Ensure Playwright browsers are installed: npx playwright install');
    console.log('  • Check if running in Docker with proper display settings');
    console.log('  • Try running with DISPLAY=:99 if using Xvfb\n');
  }
}

// MCP Server integration example
function showMCPExample() {
  console.log('\n📝 MCP Server Integration Example:');
  console.log('```javascript');
  console.log(`// Using Playwright through MCP
const mcp = require('@executeautomation/playwright-mcp-server');

// Navigate to a page
await mcp.call('playwright_navigate', {
  url: 'https://example.com'
});

// Take a screenshot
await mcp.call('playwright_screenshot', {
  path: 'screenshot.png',
  fullPage: true
});

// Fill a form
await mcp.call('playwright_fill', {
  selector: '#username',
  value: 'testuser'
});

// Click a button
await mcp.call('playwright_click', {
  selector: 'button[type="submit"]'
});
`);
  console.log('```\n');
}

// Run the demo
if (require.main === module) {
  runDemo()
    .then(() => showMCPExample())
    .catch(console.error);
}

module.exports = { runDemo };