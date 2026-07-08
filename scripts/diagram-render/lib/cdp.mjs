// Minimal raw Chrome DevTools Protocol client for the diagram render gate.
//
// Uses node's built-in global WebSocket (node >= 22) so the gate carries no npm
// dependency. Two ways to obtain a browser:
//   * connect  — attach to an already-running Chrome (the browsercontainer GPU
//                sidecar in this environment; any CDP endpoint in CI via env).
//   * launch   — spawn a local Chrome/Chromium for CI runners that ship one.
// Chrome's DevTools HTTP endpoint rejects a hostname Host header, so we resolve
// the sidecar hostname to an IP before querying /json/version (mirrors the logic
// in scripts/check-cdp-sidecar.mjs).

import { spawn } from 'node:child_process';
import { lookup } from 'node:dns/promises';
import { isIP } from 'node:net';
import { existsSync, readFileSync, rmSync } from 'node:fs';
import { mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

async function resolvedUrl(value) {
  const url = new URL(value);
  if (url.hostname !== 'localhost' && !isIP(url.hostname)) {
    const address = await lookup(url.hostname);
    url.hostname = address.address;
  }
  return url;
}

/** Query /json/version and return a browser WebSocket endpoint we can reach. */
export async function browserEndpointFrom(httpEndpoint) {
  const base = await resolvedUrl(httpEndpoint);
  if (base.protocol === 'ws:') base.protocol = 'http:';
  if (base.protocol === 'wss:') base.protocol = 'https:';
  const versionUrl = new URL('/json/version', base);

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 8_000);
  try {
    const res = await fetch(versionUrl, { signal: controller.signal });
    if (!res.ok) throw new Error(`HTTP ${res.status} from ${versionUrl}`);
    const version = await res.json();
    const ws = new URL(version.webSocketDebuggerUrl);
    ws.protocol = base.protocol === 'https:' ? 'wss:' : 'ws:';
    ws.host = base.host; // rewrite localhost/original host to the reachable one
    return ws.toString();
  } finally {
    clearTimeout(timer);
  }
}

/** Spawn a local headless Chrome and return its browser WS endpoint + cleanup. */
export async function launchLocalChrome(chromeBin) {
  const userDataDir = mkdtempSync(join(tmpdir(), 'diagram-render-'));
  const child = spawn(chromeBin, [
    '--headless=new',
    '--no-sandbox',
    '--disable-gpu',
    '--hide-scrollbars',
    '--remote-debugging-port=0',
    `--user-data-dir=${userDataDir}`,
    'about:blank',
  ], { stdio: ['ignore', 'ignore', 'pipe'] });

  const portFile = join(userDataDir, 'DevToolsActivePort');
  const deadline = Date.now() + 20_000;
  while (Date.now() < deadline) {
    if (existsSync(portFile)) {
      const port = readFileSync(portFile, 'utf8').split('\n')[0].trim();
      if (port) {
        const endpoint = await browserEndpointFrom(`http://127.0.0.1:${port}`);
        return {
          endpoint,
          kill() {
            try { child.kill('SIGKILL'); } catch { /* already gone */ }
            try { rmSync(userDataDir, { recursive: true, force: true }); } catch { /* best effort */ }
          },
        };
      }
    }
    await new Promise((r) => setTimeout(r, 200));
  }
  child.kill('SIGKILL');
  throw new Error(`Chrome at ${chromeBin} did not expose a DevTools port within 20s`);
}

/** A CDP connection with flat-session multiplexing over one WebSocket. */
export class CdpConnection {
  #sock;
  #nextId = 0;
  #pending = new Map();

  static async open(wsEndpoint) {
    const conn = new CdpConnection();
    conn.#sock = new WebSocket(wsEndpoint);
    await new Promise((resolve, reject) => {
      conn.#sock.onopen = resolve;
      conn.#sock.onerror = () => reject(new Error(`WebSocket failed to open: ${wsEndpoint}`));
    });
    conn.#sock.onmessage = (ev) => {
      const msg = JSON.parse(ev.data);
      if (msg.id && conn.#pending.has(msg.id)) {
        const { resolve, reject } = conn.#pending.get(msg.id);
        conn.#pending.delete(msg.id);
        if (msg.error) reject(new Error(`${msg.error.message} (${msg.method ?? 'cdp'})`));
        else resolve(msg.result);
      }
    };
    return conn;
  }

  send(method, params = {}, sessionId) {
    const id = ++this.#nextId;
    const frame = { id, method, params };
    if (sessionId) frame.sessionId = sessionId;
    return new Promise((resolve, reject) => {
      this.#pending.set(id, { resolve, reject });
      this.#sock.send(JSON.stringify(frame));
    });
  }

  /** Open a fresh page target and return its attached sessionId + targetId. */
  async newPage() {
    const { targetId } = await this.send('Target.createTarget', { url: 'about:blank' });
    const { sessionId } = await this.send('Target.attachToTarget', { targetId, flatten: true });
    await this.send('Page.enable', {}, sessionId);
    await this.send('Runtime.enable', {}, sessionId);
    return { sessionId, targetId };
  }

  async closePage(targetId) {
    try { await this.send('Target.closeTarget', { targetId }); } catch { /* best effort */ }
  }

  close() {
    try { this.#sock.close(); } catch { /* already closed */ }
  }
}
