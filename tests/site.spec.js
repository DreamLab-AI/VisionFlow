import AxeBuilder from '@axe-core/playwright';
import { test, expect } from './fixtures/cdp.js';

// Every top-level <section> on the page, in document order.
const sections = [
  'hero',
  'questions',
  'problem',
  'evolution',
  'substrates',
  'dreaming',
  'guarantees',
  'immersive',
  'broker',
  'economic',
  'sidechains',
  'loom',
  'cases',
  'competitive',
  'scaling',
  'doors',
  'status',
  'estate',
  'repos',
  'contact'
];

test('renders the complete VisionFlow landing page', async ({ page }) => {
  const consoleErrors = [];
  page.on('console', (message) => {
    if (message.type() === 'error') consoleErrors.push(message.text());
  });

  await page.goto('/');
  await expect(page).toHaveTitle(/VisionFlow/);
  await expect(page.getByRole('heading', { level: 1, name: /Federated Human/i })).toBeVisible();

  for (const section of sections) {
    await expect(page.locator(`section#${section}`), `${section} section`).toHaveCount(1);
  }

  // The page has one decorative WebGL mesh backdrop (static/js/mesh-webgl.js),
  // hidden from assistive technology. The earlier #mesh-canvas and four
  // particle-canvas-* elements were removed in c5ac1da (WASM particles replaced
  // by the WebGL mesh), so none may reappear.
  const mesh = page.locator('canvas#mesh-gl');
  await expect(mesh).toHaveCount(1);
  await expect(mesh).toHaveAttribute('aria-hidden', 'true');
  await expect(page.locator('canvas')).toHaveCount(1);
  await expect(mesh).toHaveClass(/ready/);
  expect(consoleErrors).toEqual([]);
});

test('navigation exposes all primary page sections', async ({ page }) => {
  await page.goto('/');

  // The desktop bar (#nav-links). The mobile index (#m-index) lists the same
  // sections again, so the selector is scoped to the bar to keep each count at one.
  const expectedLinks = [
    ['Problem', '#problem'],
    ['Substrates', '#substrates'],
    ['Dreaming', '#dreaming'],
    ['Guarantees', '#guarantees'],
    ['Immersive', '#immersive'],
    ['Governance', '#broker'],
    ['Sidechains', '#sidechains'],
    ['Contact', '#contact'],
    ['Grounding', '#loom'],
    ['Landscape', '#competitive'],
    ['Status', '#status'],
    ['Repos', '#repos']
  ];

  for (const [name, href] of expectedLinks) {
    const link = page.locator(`#nav-links a[href="${href}"]`);
    await expect(link, `${name} nav link`).toHaveCount(1);
    await expect(link).toContainText(name);
    await expect(page.locator(href), `${name} target`).toHaveCount(1);
  }
});

test('mobile index opens, lists every section and navigates', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto('/');

  const toggle = page.getByRole('button', { name: /toggle navigation/i });
  const overlay = page.locator('#m-index');
  await expect(overlay).not.toHaveClass(/open/);

  await toggle.click();
  await expect(overlay).toHaveClass(/open/);
  await expect(toggle).toHaveAttribute('aria-expanded', 'true');
  await expect(overlay.locator('a.m-index-row')).toHaveCount(17);

  await overlay.locator('a[href="#repos"]').click();
  await expect(overlay).not.toHaveClass(/open/);
  await expect(toggle).toHaveAttribute('aria-expanded', 'false');
  await expect(page).toHaveURL(/#repos$/);
});

test('reading switch swaps technical copy for plain English and back', async ({ page }) => {
  await page.goto('/');

  const problem = page.locator('#problem');
  const technical = problem.locator('.rl-face-tech');
  const plain = problem.locator('.rl-face-plain').first();
  await expect(problem.getByRole('heading', { name: 'Hard problems share a common shape' })).toBeVisible();
  await expect(technical.first()).toHaveAttribute('aria-hidden', 'false');
  await expect(plain).toHaveAttribute('aria-hidden', 'true');

  // The floating pill is CSS-hidden below 768px, where the nav knob replaces it.
  const narrow = (page.viewportSize()?.width ?? 1280) < 768;
  const toPlain = narrow ? page.locator('#m-reading') : page.getByRole('button', { name: 'Plain English' });
  const toTechnical = narrow ? page.locator('#m-reading') : page.getByRole('button', { name: 'Technical' });

  await toPlain.click();
  await expect(page.locator('body')).toHaveClass(/reading-plain/);
  await expect(plain).toHaveAttribute('aria-hidden', 'false');
  await expect(technical.first()).toHaveAttribute('aria-hidden', 'true');
  await expect(problem.getByRole('heading', { name: 'More AI means more coordination, not less' })).toBeVisible();
  await expect(problem.getByText('Teams are wiring AI tools together faster')).toBeVisible();
  expect(await page.evaluate(() => localStorage.getItem('vf-reading'))).toBe('plain');

  await toTechnical.click();
  await expect(page.locator('body')).not.toHaveClass(/reading-plain/);
  await expect(plain).toHaveAttribute('aria-hidden', 'true');
  await expect(problem.getByRole('heading', { name: 'Hard problems share a common shape' })).toBeVisible();
});

test('@a11y has no automatically detectable accessibility violations', async ({ page }) => {
  await page.goto('/');
  const results = await new AxeBuilder({ page }).analyze();
  expect(results.violations).toEqual([]);
});

test('@perf initial static payload stays within the PRD budget', async ({ page }) => {
  // Not `networkidle`: the page carries two <video preload="metadata"> elements
  // (a scroll-activated backdrop and the immersive demo). Chrome leaves their
  // open-ended `Range: bytes=0-` requests pending, which Playwright counts as
  // in-flight, so the network never goes idle however small the payload is.
  // 'load' plus settled fonts covers every stylesheet, script, image and font
  // the first screen needs.
  // The estate-health data is fetched from script and nothing makes 'load' wait
  // for it, so wait for it by name: it is part of the payload being budgeted.
  const estateHealth = page.waitForResponse((response) => /estate-health\.json(\?|$)/.test(response.url()));
  await page.goto('/', { waitUntil: 'load' });
  await estateHealth;
  await page.evaluate(() => document.fonts.ready);

  // Video is left out of the sum below, which is only sound while no video is
  // fetched eagerly. Hold that here; the bytes a metadata preload does fetch
  // (the container index) are not measured by this test.
  const preloads = await page.locator('video').evaluateAll((videos) => videos.map((video) => video.getAttribute('preload')));
  expect(preloads.length).toBeGreaterThan(0);
  for (const preload of preloads) expect(['metadata', 'none']).toContain(preload);
  expect(await page.locator('video[autoplay]').count()).toBe(0);

  const metrics = await page.evaluate(() => {
    const entries = performance.getEntriesByType('navigation').concat(performance.getEntriesByType('resource'));
    // Streamed media is bounded by preload="metadata" and Range requests, not by
    // the static-payload budget. It is excluded here and not measured.
    const resources = entries
      .filter((entry) => !/\.(mp4|webm)(\?|$)/.test(entry.name))
      .map((entry) => ({
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
