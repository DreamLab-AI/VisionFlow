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
  // Not `networkidle`: the page carries two <video> elements
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
    // Streamed media is not part of the static-payload budget. It is excluded
    // here and capped separately by the media-bytes test below.
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

// The static budget above leaves video out, so the bytes the browser fetches
// for media while the page loads are bounded here instead. They are counted at
// the protocol level (Network.dataReceived) because a media request is an
// open-ended Range request that never "finishes", so Playwright's own response
// and resource-timing entries do not report what was actually read.
//
// Measured on 2026-11-21 with the Chrome sidecar, page loaded and left idle for
// 2 s: with `preload="metadata"` on both videos the browser read about 1.0 MiB
// from each (2.1 MiB in total on desktop, 1.3 MiB on the mobile project, against
// a 280 KB page), because Chrome buffers
// well past the container index. With `preload="none"` (the featured film shows
// a poster frame instead) it reads nothing. The cap leaves 64 KiB of headroom
// over that zero, so a return to a metadata preload, or an eager video of any
// size, fails this test. Playback after a click or after scrolling past the hero
// is not part of the load window.
const MEDIA_BYTES_CAP = 64 * 1024;

test('@perf media bytes fetched during load stay within the media cap', async ({ page }) => {
  const cdp = await page.context().newCDPSession(page);
  await cdp.send('Network.enable');
  const mediaRequests = new Map();
  const received = new Map();
  cdp.on('Network.requestWillBeSent', (event) => {
    if (/\.(mp4|webm)(\?|$)/.test(event.request.url)) mediaRequests.set(event.requestId, event.request.url);
  });
  cdp.on('Network.dataReceived', (event) => {
    if (mediaRequests.has(event.requestId)) {
      received.set(event.requestId, (received.get(event.requestId) ?? 0) + event.dataLength);
    }
  });

  await page.goto('/', { waitUntil: 'load' });
  await page.evaluate(() => document.fonts.ready);
  // Chrome starts a metadata preload shortly after load; give it time to show.
  await page.waitForTimeout(2000);

  // The measurement is only meaningful while the videos are on the page.
  expect(await page.locator('video').count()).toBeGreaterThan(0);
  const mediaBytes = [...received.values()].reduce((total, bytes) => total + bytes, 0);
  expect(mediaBytes, `media bytes read during load: ${mediaBytes}`).toBeLessThanOrEqual(MEDIA_BYTES_CAP);
});
