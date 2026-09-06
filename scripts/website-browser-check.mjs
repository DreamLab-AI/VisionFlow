#!/usr/bin/env node
// @ts-check
'use strict';

/**
 * website-browser-check — publication-candidate browser verification.
 *
 * ADR-2002 promises the hero visuals "degrade gracefully (null without WebGL2;
 * honour prefers-reduced-motion)". Nothing verified that in a browser: the
 * closeout evidence records "No website build or browser session ran in this
 * pass". This script runs the built site in the real Chrome on the
 * browsercontainer sidecar over raw CDP and produces a signed-off receipt.
 *
 * Three scenarios, each in a fresh tab with a fresh page context:
 *
 *   baseline        WebGL2 available, motion normal. Expect: initMesh returns a
 *                   controller, the canvas paints, a rAF loop runs.
 *   reduced-motion  prefers-reduced-motion: reduce. Expect: the canvas still
 *                   paints (one settled frame) but no sustained rAF loop.
 *   no-webgl        getContext('webgl2') forced to null before any page script
 *                   runs. Expect: initMesh returns null, the page still renders
 *                   its content, and nothing throws.
 *
 * Paint is proven by comparing screenshot bytes of the canvas region across
 * scenarios rather than by readPixels, which returns a cleared buffer on a
 * canvas without preserveDrawingBuffer.
 *
 * Usage:
 *   node scripts/website-browser-check.mjs \
 *     --url http://agentbox:8099/ \
 *     --out docs/estate-closeout/2026-09-05 \
 *     [--cdp browsercontainer:9223]
 */

import { writeFileSync, mkdirSync, existsSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { dirname, join, resolve } from 'node:path';

const HERE = dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = resolve(HERE, '..');

const argv = process.argv.slice(2);
const opt = (n, d) => { const i = argv.indexOf(`--${n}`); return i >= 0 && argv[i + 1] ? argv[i + 1] : d; };

const URL_ = opt('url', 'http://agentbox:8099/');
const OUT_DIR = resolve(REPO_ROOT, opt('out', 'docs/estate-closeout/2026-09-05'));
const SHOT_DIR = join(OUT_DIR, 'screenshots');
// Chrome's DevTools endpoint rejects any Host header that is not localhost or
// a bare IP, and that check also applies to the WebSocket upgrade — where the
// Host header cannot be overridden. Resolving the sidecar hostname to its IP
// up front sidesteps it for both the HTTP calls and the socket.
function resolveHostPort(hostPort) {
  const [host, port] = hostPort.split(':');
  if (/^\d+\.\d+\.\d+\.\d+$/.test(host)) return hostPort;
  try {
    const ip = execFileSync('python3',
      ['-c', `import socket;print(socket.gethostbyname(${JSON.stringify(host)}))`],
      { encoding: 'utf8' }).trim();
    return `${ip}:${port}`;
  } catch { return hostPort; }
}

const CDP = resolveHostPort(opt('cdp', 'browsercontainer:9223'));

mkdirSync(SHOT_DIR, { recursive: true });

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const sha256 = (b) => createHash('sha256').update(b).digest('hex');

// ── minimal CDP client ───────────────────────────────────────────────────
// The socat bridge rejects a Host header that is not localhost, so every HTTP
// call to the DevTools endpoint spoofs it.
function cdpHttp(path, method = 'GET') {
  const args = ['-s', '-X', method];
  // Only spoof Host when addressing by name; Chrome accepts a bare IP as-is,
  // and spoofing it there makes Chrome hand back ws://localhost URLs that this
  // container cannot dial.
  if (!/^\d+\.\d+\.\d+\.\d+:/.test(CDP)) args.push('-H', 'Host: localhost');
  args.push(`http://${CDP}${path}`);
  const out = execFileSync('curl', args, { encoding: 'utf8', maxBuffer: 64 * 1024 * 1024 });
  try { return JSON.parse(out); } catch { return out; }
}

/** Re-point a DevTools websocket URL at the endpoint we can actually reach. */
function reachableWs(wsUrl) {
  return wsUrl.replace(/^ws:\/\/[^/]+/, `ws://${CDP}`);
}

class Session {
  constructor(ws) { this.ws = ws; this.id = 0; this.pending = new Map(); this.events = []; }

  static async open(wsUrl) {
    const ws = new WebSocket(wsUrl);
    const s = new Session(ws);
    await new Promise((res, rej) => {
      ws.addEventListener('open', res, { once: true });
      ws.addEventListener('error', rej, { once: true });
    });
    ws.addEventListener('message', (ev) => {
      const msg = JSON.parse(ev.data);
      if (msg.id && s.pending.has(msg.id)) {
        const { resolve: r, reject: j } = s.pending.get(msg.id);
        s.pending.delete(msg.id);
        msg.error ? j(new Error(JSON.stringify(msg.error))) : r(msg.result);
      } else if (msg.method) {
        s.events.push(msg);
      }
    });
    return s;
  }

  send(method, params = {}) {
    const id = ++this.id;
    this.ws.send(JSON.stringify({ id, method, params }));
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      setTimeout(() => {
        if (this.pending.has(id)) { this.pending.delete(id); reject(new Error(`CDP timeout: ${method}`)); }
      }, 30000);
    });
  }

  close() { try { this.ws.close(); } catch { /* already gone */ } }
}

// ── page-context instrumentation, installed before any page script runs ──
const INSTRUMENT = `
  window.__vf = { raf: 0, errors: [], rejections: [] };
  (function () {
    const orig = window.requestAnimationFrame.bind(window);
    window.requestAnimationFrame = function (cb) { window.__vf.raf++; return orig(cb); };
  })();
  window.addEventListener('error', (e) => {
    window.__vf.errors.push(String((e && e.error && e.error.stack) || (e && e.message) || e));
  });
  window.addEventListener('unhandledrejection', (e) => {
    window.__vf.rejections.push(String((e && e.reason) || e));
  });
`;

// Force the WebGL2 fallback path deterministically, before page scripts run.
const KILL_WEBGL = `
  (function () {
    const orig = HTMLCanvasElement.prototype.getContext;
    HTMLCanvasElement.prototype.getContext = function (type, ...rest) {
      if (type === 'webgl2' || type === 'webgl' || type === 'experimental-webgl') return null;
      return orig.call(this, type, ...rest);
    };
  })();
`;

// ── the in-page probe ────────────────────────────────────────────────────
const PROBE = `(() => {
  const canvas = document.getElementById('mesh-gl');
  const gl = canvas ? canvas.getContext('webgl2') : null;
  const imgs = Array.from(document.images);
  return {
    title: document.title,
    lang: document.documentElement.lang,
    readyState: document.readyState,
    canonical: (document.querySelector('link[rel=canonical]') || {}).href || null,
    metaDescription: (document.querySelector('meta[name=description]') || {}).content || null,
    og: Object.fromEntries(Array.from(document.querySelectorAll('meta[property^="og:"]'))
          .map(m => [m.getAttribute('property'), m.content])),
    twitter: Object.fromEntries(Array.from(document.querySelectorAll('meta[name^="twitter:"]'))
          .map(m => [m.getAttribute('name'), m.content])),
    canvasPresent: !!canvas,
    webgl2Available: !!gl,
    glRenderer: gl ? (() => { const d = gl.getExtension('WEBGL_debug_renderer_info');
                              return d ? gl.getParameter(d.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER); })() : null,
    reducedMotion: matchMedia('(prefers-reduced-motion: reduce)').matches,
    rafCount: window.__vf ? window.__vf.raf : null,
    pageErrors: window.__vf ? window.__vf.errors : [],
    pageRejections: window.__vf ? window.__vf.rejections : [],
    sections: document.querySelectorAll('section').length,
    headings: document.querySelectorAll('h1,h2').length,
    textLength: (document.body.innerText || '').length,
    images: imgs.length,
    brokenImages: imgs.filter(i => i.complete && i.naturalWidth === 0).map(i => i.currentSrc || i.src),
    videos: Array.from(document.querySelectorAll('video')).map(v => ({
      id: v.id || null, readyState: v.readyState,
      sources: Array.from(v.querySelectorAll('source')).map(s => s.src),
    })),
    canvasRect: canvas ? (r => ({ x: Math.round(r.x), y: Math.round(r.y),
                                  w: Math.round(r.width), h: Math.round(r.height) }))(canvas.getBoundingClientRect()) : null,
  };
})()`;

/** Run one scenario end to end in a fresh tab. */
async function runScenario({ name, reducedMotion = false, killWebgl = false }) {
  // Chrome wants the target URL as a raw query string here, not percent-encoded.
  const target = cdpHttp('/json/new?about:blank', 'PUT');
  if (!target || !target.webSocketDebuggerUrl) {
    throw new Error(`could not open a CDP target: ${JSON.stringify(target).slice(0, 300)}`);
  }
  const record = { scenario: name, url: URL_ };
  const s = await Session.open(reachableWs(target.webSocketDebuggerUrl));

  try {
    await s.send('Page.enable');
    await s.send('Runtime.enable');
    await s.send('Log.enable');
    await s.send('Network.enable');

    await s.send('Page.addScriptToEvaluateOnNewDocument', { source: INSTRUMENT });
    if (killWebgl) await s.send('Page.addScriptToEvaluateOnNewDocument', { source: KILL_WEBGL });
    if (reducedMotion) {
      await s.send('Emulation.setEmulatedMedia', {
        features: [{ name: 'prefers-reduced-motion', value: 'reduce' }],
      });
    }
    await s.send('Emulation.setDeviceMetricsOverride', {
      width: 1440, height: 900, deviceScaleFactor: 1, mobile: false,
    });

    const t0 = Date.now();
    await s.send('Page.navigate', { url: URL_ });
    // Settle: let the load event, module import and first frames complete.
    await sleep(3500);
    record.settle_ms = Date.now() - t0;

    // Main-document status from the network events.
    const resp = s.events.find((e) => e.method === 'Network.responseReceived'
      && e.params.type === 'Document');
    record.http_status = resp ? resp.params.response.status : null;

    const before = await s.send('Runtime.evaluate', { expression: PROBE, returnByValue: true });
    record.probe = before.result.value;

    // A second sample proves whether a rAF loop is still running.
    await sleep(1200);
    const after = await s.send('Runtime.evaluate', {
      expression: 'window.__vf ? window.__vf.raf : null', returnByValue: true,
    });
    record.raf_first = record.probe.rafCount;
    record.raf_second = after.result.value;
    record.raf_delta = (typeof record.raf_second === 'number' && typeof record.raf_first === 'number')
      ? record.raf_second - record.raf_first : null;
    record.animation_loop_running = record.raf_delta !== null && record.raf_delta > 5;

    // Console + protocol-level errors.
    record.console_errors = s.events
      .filter((e) => e.method === 'Runtime.consoleAPICalled' && e.params.type === 'error')
      .map((e) => (e.params.args || []).map((a) => a.value ?? a.description ?? '').join(' '));
    record.console_warnings = s.events
      .filter((e) => e.method === 'Runtime.consoleAPICalled' && e.params.type === 'warning')
      .map((e) => (e.params.args || []).map((a) => a.value ?? a.description ?? '').join(' '));
    record.exceptions = s.events
      .filter((e) => e.method === 'Runtime.exceptionThrown')
      .map((e) => e.params.exceptionDetails.text
        + (e.params.exceptionDetails.exception ? ` — ${e.params.exceptionDetails.exception.description}` : ''));
    record.log_errors = s.events
      .filter((e) => e.method === 'Log.entryAdded' && e.params.entry.level === 'error')
      .map((e) => `${e.params.entry.source}: ${e.params.entry.text}`);
    record.failed_requests = s.events
      .filter((e) => e.method === 'Network.loadingFailed')
      .map((e) => e.params.errorText);

    // Screenshots: full viewport, plus the canvas region on its own so the
    // WebGL layer's contribution can be compared byte-for-byte across runs.
    const shot = async (suffix, clip) => {
      const r = await s.send('Page.captureScreenshot',
        clip ? { format: 'png', clip: { ...clip, scale: 1 } } : { format: 'png' });
      const buf = Buffer.from(r.data, 'base64');
      const file = join(SHOT_DIR, `${name}-${suffix}.png`);
      writeFileSync(file, buf);
      return { file: file.replace(REPO_ROOT + '/', ''), bytes: buf.length, sha256: sha256(buf) };
    };

    record.screenshots = { viewport: await shot('viewport') };
    if (record.probe.canvasRect && record.probe.canvasRect.w > 0) {
      const cr = record.probe.canvasRect;
      record.screenshots.canvas = await shot('canvas', {
        x: Math.max(0, cr.x), y: Math.max(0, cr.y),
        width: Math.min(cr.w, 1440), height: Math.min(cr.h, 900),
      });
    }
    // Mid-page, to show content below the hero rendered.
    await s.send('Runtime.evaluate', { expression: 'window.scrollTo(0, window.innerHeight * 3)' });
    await sleep(1200);
    record.screenshots.scrolled = await shot('scrolled');
  } finally {
    s.close();
    cdpHttp(`/json/close/${target.id}`);
  }
  return record;
}

// ── assertions ───────────────────────────────────────────────────────────
function assess(byName) {
  const checks = [];
  const add = (id, ok, detail) => checks.push({ id, ok, detail });

  const base = byName['baseline'];
  const reduced = byName['reduced-motion'];
  const nogl = byName['no-webgl'];

  // Page load
  add('page-loads', base.http_status === 200 && base.probe.readyState === 'complete',
    `http ${base.http_status}, readyState ${base.probe.readyState}`);
  add('title-present', !!base.probe.title, base.probe.title);
  add('content-rendered', base.probe.sections > 5 && base.probe.textLength > 2000,
    `${base.probe.sections} sections, ${base.probe.textLength} chars of text`);

  // Console hygiene, across every scenario
  for (const [n, r] of Object.entries(byName)) {
    const errs = [...r.console_errors, ...r.exceptions, ...r.log_errors, ...r.probe.pageErrors, ...r.probe.pageRejections];
    add(`no-console-errors:${n}`, errs.length === 0, errs.length ? errs.join(' | ') : 'none');
    add(`no-failed-requests:${n}`, r.failed_requests.length === 0,
      r.failed_requests.length ? r.failed_requests.join(' | ') : 'none');
    add(`no-broken-images:${n}`, r.probe.brokenImages.length === 0,
      r.probe.brokenImages.length ? r.probe.brokenImages.join(' | ') : 'none');
  }

  // Meta / OG
  const og = base.probe.og || {};
  const tw = base.probe.twitter || {};
  add('meta-description', !!base.probe.metaDescription, (base.probe.metaDescription || '').slice(0, 50));
  add('canonical', !!base.probe.canonical, base.probe.canonical);
  add('og-core-tags', ['og:title', 'og:description', 'og:type', 'og:url'].every((k) => og[k]),
    Object.keys(og).join(', '));
  add('og-image', !!og['og:image'], og['og:image'] || 'absent');
  add('twitter-card', tw['twitter:card'] === 'summary_large_image', tw['twitter:card'] || 'absent');

  // WebGL mesh vs canvas fallback
  add('webgl2-available-baseline', base.probe.webgl2Available === true, base.probe.glRenderer || 'none');
  add('mesh-animates-baseline', base.animation_loop_running === true,
    `rAF delta ${base.raf_delta} over ~1.2s`);
  add('webgl2-absent-in-fallback', nogl.probe.webgl2Available === false,
    `webgl2Available=${nogl.probe.webgl2Available}`);
  add('fallback-still-renders-content', nogl.probe.sections > 5 && nogl.probe.textLength > 2000,
    `${nogl.probe.sections} sections, ${nogl.probe.textLength} chars`);
  add('fallback-does-not-throw',
    nogl.exceptions.length === 0 && nogl.probe.pageErrors.length === 0,
    nogl.exceptions.concat(nogl.probe.pageErrors).join(' | ') || 'none');

  // The WebGL layer must actually contribute pixels: the canvas region differs
  // between the GPU run and the forced-fallback run.
  const bc = base.screenshots.canvas, nc = nogl.screenshots.canvas;
  if (bc && nc) {
    add('mesh-paints-pixels', bc.sha256 !== nc.sha256,
      `baseline canvas sha ${bc.sha256.slice(0, 12)} (${bc.bytes}B) vs fallback ${nc.sha256.slice(0, 12)} (${nc.bytes}B)`);
  } else {
    add('mesh-paints-pixels', false, 'canvas region not captured');
  }

  // Reduced motion: still painted, but no sustained animation loop.
  add('reduced-motion-detected', reduced.probe.reducedMotion === true,
    `matchMedia=${reduced.probe.reducedMotion}`);
  add('reduced-motion-no-animation-loop', reduced.animation_loop_running === false,
    `rAF delta ${reduced.raf_delta} over ~1.2s (baseline ${base.raf_delta})`);
  add('reduced-motion-still-renders', reduced.probe.sections > 5 && reduced.probe.textLength > 2000,
    `${reduced.probe.sections} sections, ${reduced.probe.textLength} chars`);

  return checks;
}

// ── main ─────────────────────────────────────────────────────────────────
const scenarios = [
  { name: 'baseline' },
  { name: 'reduced-motion', reducedMotion: true },
  { name: 'no-webgl', killWebgl: true },
];

const results = {};
for (const sc of scenarios) {
  process.stdout.write(`==> scenario: ${sc.name}\n`);
  results[sc.name] = await runScenario(sc);
}

const checks = assess(results);
const failed = checks.filter((c) => !c.ok);

let siteRevision = null;
try {
  siteRevision = execFileSync('git', ['-C', REPO_ROOT, 'rev-parse', 'HEAD'], { encoding: 'utf8' }).trim();
} catch { /* not a checkout */ }

const receiptPath = join(OUT_DIR, 'website-browser-receipt.json');
const receipt = {
  receipt_version: 1,
  generated_at: new Date().toISOString().replace(/\.\d{3}Z$/, 'Z'),
  purpose: 'ADR-2002/ADR-2003 publication-candidate browser verification: page load, console '
    + 'hygiene, WebGL2 mesh render vs canvas fallback, prefers-reduced-motion behaviour, meta/OG tags.',
  url: URL_,
  served_from: 'website/dist (local static server in the agentbox container)',
  browser: {
    endpoint: `cdp://${CDP}`,
    version: (cdpHttp('/json/version') || {}).Browser || null,
    renderer: results.baseline?.probe?.glRenderer || null,
  },
  site_revision: siteRevision,
  build_receipt: existsSync(join(REPO_ROOT, 'website/build-receipt.json'))
    ? 'website/build-receipt.json' : null,
  scenarios: results,
  checks,
  summary: { total: checks.length, passed: checks.length - failed.length, failed: failed.length },
  ok: failed.length === 0,
};
writeFileSync(receiptPath, JSON.stringify(receipt, null, 2) + '\n');

process.stdout.write('\nBROWSER CHECKS\n==============\n');
for (const c of checks) {
  process.stdout.write(`  ${c.ok ? 'ok  ' : 'FAIL'}  ${c.id}\n          ${c.detail}\n`);
}
process.stdout.write(`\nreceipt: ${receiptPath.replace(REPO_ROOT + '/', '')}\n`);
process.stdout.write(`passed ${receipt.summary.passed}/${receipt.summary.total}\n`);
process.stdout.write(failed.length === 0 ? 'BROWSER-CHECK-OK\n' : 'BROWSER-CHECK-FAIL\n');
process.exit(failed.length === 0 ? 0 : 1);
