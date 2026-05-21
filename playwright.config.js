import { defineConfig, devices } from '@playwright/test';
import { existsSync, readFileSync } from 'node:fs';
import { networkInterfaces } from 'node:os';

function isDockerRuntime() {
  if (existsSync('/.dockerenv')) return true;
  try {
    return readFileSync('/proc/1/cgroup', 'utf8').includes('docker');
  } catch {
    return false;
  }
}

function dockerReachableHost() {
  const addresses = Object.values(networkInterfaces())
    .flat()
    .filter((network) => network && network.family === 'IPv4' && !network.internal)
    .map((network) => network.address);

  return addresses.find((address) => address.startsWith('172.20.')) || addresses[0] || '127.0.0.1';
}

const siteBaseURL = process.env.SITE_BASE_URL || `http://${isDockerRuntime() ? dockerReachableHost() : '127.0.0.1'}:4173`;

export default defineConfig({
  testDir: './tests',
  timeout: 30_000,
  expect: {
    timeout: 5_000
  },
  use: {
    baseURL: siteBaseURL,
    trace: 'on-first-retry'
  },
  webServer: {
    command: 'python3 -m http.server 4173 --bind 0.0.0.0 --directory website/dist',
    url: 'http://127.0.0.1:4173',
    reuseExistingServer: !process.env.CI,
    timeout: 10_000
  },
  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] }
    },
    {
      name: 'mobile-chrome',
      use: { ...devices['Pixel 5'] }
    }
  ]
});
