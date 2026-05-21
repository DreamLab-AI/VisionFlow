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

const DEFAULT_HOST = process.env.BROWSER_CDP_HOST || (isDockerRuntime() ? 'browsercontainer' : 'localhost');
const DEFAULT_PORT = process.env.BROWSER_CDP_PORT || (isDockerRuntime() ? '9223' : '9222');
const endpoint = process.env.CDP_ENDPOINT || process.env.CHROME_CDP_URL || `http://${DEFAULT_HOST}:${DEFAULT_PORT}`;

async function resolvedUrl(value) {
  const url = new URL(value);
  if (url.hostname !== 'localhost' && !isIP(url.hostname)) {
    const address = await lookup(url.hostname);
    url.hostname = address.address;
  }
  return url;
}

async function versionUrlFor(value) {
  const url = await resolvedUrl(value);
  if (url.protocol === 'ws:') url.protocol = 'http:';
  if (url.protocol === 'wss:') url.protocol = 'https:';
  url.pathname = '/json/version';
  url.search = '';
  url.hash = '';
  return url.toString();
}

function normalizedWebSocketUrl(version, versionUrl) {
  const browserWs = new URL(version.webSocketDebuggerUrl);
  const resolvedVersion = new URL(versionUrl);
  browserWs.protocol = resolvedVersion.protocol === 'https:' ? 'wss:' : 'ws:';
  browserWs.host = resolvedVersion.host;
  return browserWs.toString();
}

const versionUrl = await versionUrlFor(endpoint);
const controller = new AbortController();
const timeout = setTimeout(() => controller.abort(), 5_000);

try {
  const response = await fetch(versionUrl, { signal: controller.signal });
  if (!response.ok) {
    throw new Error(`HTTP ${response.status} from ${versionUrl}`);
  }

  const version = await response.json();
  console.log(`CDP sidecar reachable: ${version.Browser || 'unknown browser'}`);
  console.log(`Protocol: ${version['Protocol-Version'] || 'unknown'}`);
  if (version.webSocketDebuggerUrl) {
    console.log(`WebSocket: ${normalizedWebSocketUrl(version, versionUrl)}`);
  }
} catch (error) {
  console.error(`CDP sidecar not reachable at ${versionUrl}`);
  console.error(error.message);
  process.exit(1);
} finally {
  clearTimeout(timeout);
}
