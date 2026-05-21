import { chromium, expect, test as base } from '@playwright/test';
import { existsSync, readFileSync } from 'node:fs';
import { lookup } from 'node:dns/promises';
import { isIP } from 'node:net';

function isDockerRuntime() {
  if (existsSync('/.dockerenv')) return true;
  try {
    return readFileSync('/proc/1/cgroup', 'utf8').includes('docker');
  } catch {
    return false;
  }
}

const DEFAULT_CDP_HOST = process.env.BROWSER_CDP_HOST || (isDockerRuntime() ? 'browsercontainer' : 'localhost');
const DEFAULT_CDP_PORT = process.env.BROWSER_CDP_PORT || (isDockerRuntime() ? '9223' : '9222');

function cdpEndpoint() {
  return process.env.CDP_ENDPOINT || process.env.CHROME_CDP_URL || `http://${DEFAULT_CDP_HOST}:${DEFAULT_CDP_PORT}`;
}

async function resolvedCdpWebSocket() {
  const endpoint = new URL(cdpEndpoint());

  if (endpoint.hostname !== 'localhost' && !isIP(endpoint.hostname)) {
    const address = await lookup(endpoint.hostname);
    endpoint.hostname = address.address;
  }

  if (endpoint.protocol === 'ws:' || endpoint.protocol === 'wss:') {
    return endpoint.toString();
  }

  endpoint.pathname = '/json/version';
  endpoint.search = '';
  endpoint.hash = '';

  const response = await fetch(endpoint);
  if (!response.ok) {
    throw new Error(`CDP sidecar returned HTTP ${response.status} from ${endpoint}`);
  }

  const version = await response.json();
  const browserWs = new URL(version.webSocketDebuggerUrl);
  browserWs.protocol = endpoint.protocol === 'https:' ? 'wss:' : 'ws:';
  browserWs.host = endpoint.host;
  return browserWs.toString();
}

function contextOptions(projectUse) {
  const allowed = [
    'baseURL',
    'viewport',
    'userAgent',
    'deviceScaleFactor',
    'isMobile',
    'hasTouch',
    'colorScheme',
    'locale',
    'timezoneId',
    'permissions',
    'extraHTTPHeaders',
    'geolocation',
    'javaScriptEnabled'
  ];

  return Object.fromEntries(
    allowed
      .filter((key) => projectUse[key] !== undefined)
      .map((key) => [key, projectUse[key]])
  );
}

export const test = base.extend({
  page: async ({}, use, testInfo) => {
    const browser = await chromium.connectOverCDP(await resolvedCdpWebSocket(), { timeout: 10_000 });
    const context = await browser.newContext(contextOptions(testInfo.project.use));
    const page = await context.newPage();

    try {
      await use(page);
    } finally {
      await context.close();
      if (typeof browser.disconnect === 'function') {
        await browser.disconnect();
      } else {
        await browser.close();
      }
    }
  }
});

export { expect };
