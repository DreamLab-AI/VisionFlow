import { test, expect } from '@playwright/test';

test.describe('VisionFlow Visualization', () => {
  test('loads the main page', async ({ page }) => {
    await page.goto('/');
    await expect(page).toHaveTitle(/VisionFlow/i);
  });

  test('API returns graph data', async ({ request }) => {
    const response = await request.get('/api/graph/data');
    expect(response.ok()).toBeTruthy();

    const json = await response.json();
    expect(json).toHaveProperty('data');
    expect(json.data).toHaveProperty('nodes');
    expect(json.data).toHaveProperty('edges');
    expect(json.data.nodes.length).toBeGreaterThan(0);
    expect(json.data.edges.length).toBeGreaterThan(0);

    console.log(`Graph loaded: ${json.data.nodes.length} nodes, ${json.data.edges.length} edges`);
  });

  test('API returns node filter settings', async ({ request }) => {
    const response = await request.get('/api/settings/node-filter');
    expect(response.ok()).toBeTruthy();

    const settings = await response.json();
    expect(settings).toHaveProperty('enabled');
    expect(settings).toHaveProperty('qualityThreshold');
    expect(settings).toHaveProperty('authorityThreshold');
    expect(settings).toHaveProperty('filterMode');

    console.log(`Node filter settings: enabled=${settings.enabled}, qualityThreshold=${settings.qualityThreshold}`);
  });

  test('canvas element renders', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');

    // Wait for Three.js canvas to appear
    const canvas = page.locator('canvas');
    await expect(canvas).toBeVisible({ timeout: 10000 });
  });

  test('settings panel can be toggled', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');

    // Look for settings button or panel
    const settingsButton = page.locator('[data-testid="settings-toggle"], button:has-text("Settings"), .settings-toggle').first();
    if (await settingsButton.isVisible()) {
      await settingsButton.click();
      // Check if settings panel appears
      await page.waitForTimeout(500);
    }
  });

  test('node filter settings can be updated', async ({ request }) => {
    // Get current settings
    const getResponse = await request.get('/api/settings/node-filter');
    const currentSettings = await getResponse.json();

    // Update with new threshold
    const newThreshold = currentSettings.qualityThreshold === 0.7 ? 0.8 : 0.7;
    const updateResponse = await request.put('/api/settings/node-filter', {
      data: {
        ...currentSettings,
        qualityThreshold: newThreshold,
      },
    });
    expect(updateResponse.ok()).toBeTruthy();

    // Verify update
    const verifyResponse = await request.get('/api/settings/node-filter');
    const updatedSettings = await verifyResponse.json();
    expect(updatedSettings.qualityThreshold).toBe(newThreshold);

    // Restore original setting
    await request.put('/api/settings/node-filter', {
      data: currentSettings,
    });
  });
});
