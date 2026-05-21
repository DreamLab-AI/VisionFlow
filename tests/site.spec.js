import AxeBuilder from '@axe-core/playwright';
import { test, expect } from './fixtures/cdp.js';

const sections = [
  'hero',
  'problem',
  'evolution',
  'substrates',
  'broker',
  'economic',
  'cases',
  'competitive',
  'scaling',
  'repos'
];

test('renders the complete VisionFlow landing page', async ({ page }) => {
  const consoleErrors = [];
  page.on('console', (message) => {
    if (message.type() === 'error') consoleErrors.push(message.text());
  });

  await page.goto('/');
  await expect(page).toHaveTitle(/VisionFlow/);
  await expect(page.getByRole('heading', { name: /Federated Human/i })).toBeVisible();

  for (const section of sections) {
    await expect(page.locator(`#${section}`), `${section} section`).toHaveCount(1);
  }

  await expect(page.locator('#mesh-canvas')).toBeVisible();
  await expect(page.locator('[id^="particle-canvas-"]')).toHaveCount(4);
  expect(consoleErrors).toEqual([]);
});

test('navigation exposes all primary page sections', async ({ page }) => {
  await page.goto('/');

  const expectedLinks = [
    ['Problem', '#problem'],
    ['Evolution', '#evolution'],
    ['Substrates', '#substrates'],
    ['Governance', '#broker'],
    ['Economics', '#economic'],
    ['Cases', '#cases'],
    ['Landscape', '#competitive'],
    ['Scaling', '#scaling'],
    ['Repos', '#repos']
  ];

  for (const [name, href] of expectedLinks) {
    const link = page.locator(`nav a[href="${href}"]`);
    await expect(link, `${name} nav link`).toHaveCount(1);
    await expect(link).toContainText(name);
  }
});

test('mobile menu opens and links remain reachable', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto('/');

  await page.getByRole('button', { name: /toggle navigation/i }).click();
  await expect(page.locator('#nav-links')).toHaveClass(/open/);
  await page.locator('nav a[href="#repos"]').click();
  await expect(page).toHaveURL(/#repos$/);
});

test('@a11y has no automatically detectable accessibility violations', async ({ page }) => {
  await page.goto('/');
  const results = await new AxeBuilder({ page }).analyze();
  expect(results.violations).toEqual([]);
});

test('@perf initial static payload stays within the PRD budget', async ({ page }) => {
  await page.goto('/', { waitUntil: 'networkidle' });

  const metrics = await page.evaluate(() => {
    const entries = performance.getEntriesByType('navigation').concat(performance.getEntriesByType('resource'));
    const resources = entries.map((entry) => ({
      name: entry.name,
      transferSize: entry.transferSize || entry.encodedBodySize || 0
    }));

    return {
      resourceCount: resources.length,
      totalBytes: resources.reduce((total, resource) => total + resource.transferSize, 0)
    };
  });

  expect(metrics.resourceCount).toBeGreaterThan(0);
  expect(metrics.totalBytes).toBeLessThanOrEqual(800 * 1024);
});
