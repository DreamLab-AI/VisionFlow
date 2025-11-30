import { test, expect } from '@playwright/test';

test('debug console errors', async ({ page }) => {
  // Collect console messages
  const consoleLogs: string[] = [];
  page.on('console', msg => {
    consoleLogs.push(`[${msg.type()}] ${msg.text()}`);
  });

  // Collect page errors
  const pageErrors: string[] = [];
  page.on('pageerror', error => {
    pageErrors.push(error.message);
  });

  await page.goto('/');
  await page.waitForLoadState('domcontentloaded');
  await page.waitForTimeout(15000);

  // Take screenshot
  await page.screenshot({ path: '/tmp/debug-current.png', fullPage: true });

  // Get page text content
  const pageText = await page.evaluate(() => document.body.textContent?.substring(0, 2000));

  // Get error boundary text
  const errorText = await page.evaluate(() => {
    const errorElements = document.querySelectorAll('[class*="error"], [class*="Error"]');
    return Array.from(errorElements).map(el => el.textContent?.substring(0, 500)).join('\n');
  });

  console.log('\n=== Console Logs ===');
  consoleLogs.forEach(log => console.log(log));

  console.log('\n=== Page Errors ===');
  pageErrors.forEach(err => console.log(err));

  console.log('\n=== Error Elements ===');
  console.log(errorText || 'No error elements found');

  console.log('\n=== Page Content (first 500 chars) ===');
  console.log(pageText?.substring(0, 500));

  // Get visible HTML structure
  const htmlStructure = await page.evaluate(() => {
    const body = document.body;
    function getStructure(el: Element, depth: number): string {
      if (depth > 3) return '';
      const children = Array.from(el.children);
      const tag = el.tagName.toLowerCase();
      const id = el.id ? `#${el.id}` : '';
      const cls = el.className ? `.${el.className.toString().split(' ').slice(0, 2).join('.')}` : '';
      let result = `${'  '.repeat(depth)}${tag}${id}${cls}\n`;
      children.slice(0, 5).forEach(child => {
        result += getStructure(child, depth + 1);
      });
      return result;
    }
    return getStructure(body, 0);
  });

  console.log('\n=== HTML Structure ===');
  console.log(htmlStructure);
});
