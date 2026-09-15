#!/usr/bin/env node
// Live edge-forum probe for the augmentation-conditions rollout (PRD-augmentation-conditions M4).
// Read-only. Records what the LIVE relay and auth API actually serve, so that the
// pre-deploy baseline and the post-deploy state are both evidence, never inference.
// Usage: node scripts/live-forum-probe.mjs [out.json]
import { writeFileSync } from 'node:fs';

const RELAY_HTTP = process.env.FORUM_RELAY_HTTP || 'https://dreamlab-nostr-relay.solitary-paper-764d.workers.dev';
const RELAY_WS = process.env.FORUM_RELAY_WS || 'wss://dreamlab-nostr-relay.solitary-paper-764d.workers.dev';
const AUTH_API = process.env.FORUM_AUTH_API || 'https://dreamlab-auth-api.solitary-paper-764d.workers.dev';
const OUT = process.argv[2] || 'docs/estate-closeout/2026-09-14/live-forum-probe.json';
const TIMEOUT_MS = 12000;
const TP_TAGS = ['tp-verifiability', 'tp-reversibility', 'tp-stakes', 'calibration-sample-rate', 'max-pending-hours', 'probe-agent', 'probe'];

const receipt = { probed_at: new Date().toISOString(), relay: RELAY_WS, auth_api: AUTH_API, probes: [] };
const record = (name, result) => { receipt.probes.push({ name, ...result }); console.log(`${result.ok ? 'PASS' : 'INFO'} ${name}: ${result.summary}`); };

async function fetchJson(url, init = {}) {
  const ctl = new AbortController(); const t = setTimeout(() => ctl.abort(), TIMEOUT_MS);
  try {
    const r = await fetch(url, { ...init, signal: ctl.signal });
    const text = await r.text(); let body = null; try { body = JSON.parse(text); } catch { body = text.slice(0, 300); }
    return { status: r.status, body };
  } finally { clearTimeout(t); }
}

// Probe 1: NIP-11 — is the escalation default advertised, and (post-deploy) is enforcement declared?
try {
  const { status, body } = await fetchJson(RELAY_HTTP, { headers: { Accept: 'application/nostr+json' } });
  const esc = body && typeof body === 'object' ? Object.fromEntries(Object.entries(body).filter(([k]) => /escalation|governance|augmentation/i.test(k))) : {};
  record('nip11-escalation-default', { ok: status === 200 && Object.keys(esc).length > 0, status, escalation: esc,
    summary: `HTTP ${status}; escalation keys: ${JSON.stringify(esc)}` });
} catch (e) { record('nip11-escalation-default', { ok: false, error: String(e), summary: `error ${e}` }); }

// Probe 2: subscribe governance kinds and inspect tags for the new schema.
await new Promise((resolve) => {
  const ws = new WebSocket(RELAY_WS);
  const events = []; const done = (why) => { if (ws._done) return; ws._done = true; try { ws.close(); } catch {} ;
    const tagHits = {}; for (const ev of events) for (const t of ev.tags || []) if (TP_TAGS.includes(t[0])) tagHits[t[0]] = (tagHits[t[0]] || 0) + 1;
    const byKind = {}; for (const ev of events) byKind[ev.kind] = (byKind[ev.kind] || 0) + 1;
    const undecidedProbeVisible = events.some(ev => ev.kind === 31402 && (ev.tags || []).some(t => t[0] === 'probe'));
    record('relay-governance-kinds', { ok: why === 'EOSE', why, count: events.length, by_kind: byKind, new_schema_tags_seen: tagHits,
      undecided_probe_tag_visible: undecidedProbeVisible,
      summary: `${why}; ${events.length} events ${JSON.stringify(byKind)}; new tags ${JSON.stringify(tagHits)}` });
    resolve(); };
  const timer = setTimeout(() => done('TIMEOUT'), TIMEOUT_MS);
  ws.onopen = () => ws.send(JSON.stringify(['REQ', 'ac-probe', { kinds: [31400, 31401, 31402, 31403, 31404, 31405], limit: 200 }]));
  ws.onmessage = (m) => { try { const f = JSON.parse(m.data); if (f[0] === 'EVENT') events.push(f[2]); if (f[0] === 'EOSE') { clearTimeout(timer); done('EOSE'); } if (f[0] === 'NOTICE' || f[0] === 'CLOSED') { clearTimeout(timer); done(`${f[0]}:${f[2] ?? f[1]}`); } } catch {} };
  ws.onerror = (e) => { clearTimeout(timer); done(`ERROR:${e.message || 'ws'}`); };
  ws.onclose = () => { clearTimeout(timer); if (!ws._done) { ws._done = true; done('CLOSED'); } };
});

// Probe 3/4: the new auth-worker endpoints. Unauthenticated: 401/403 means deployed and gated; 404 means not yet deployed.
for (const [name, url, init] of [
  ['auth-reviewers-endpoint', `${AUTH_API}/api/governance/reviewers`, {}],
  ['auth-application-receipt-endpoint', `${AUTH_API}/api/governance/receipts/0000000000000000000000000000000000000000000000000000000000000000/application`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ stage: 'consumer-received' }) }],
  ['auth-existing-receipts-endpoint', `${AUTH_API}/api/governance/receipts`, {}],
]) {
  try {
    const { status, body } = await fetchJson(url, init);
    // 404 = route absent; 2xx/403/405/409 = route present; 401 is AMBIGUOUS — the auth worker
    // gates every /api/governance path before routing, so an unauthenticated 401 proves nothing.
    const verdict = status === 404 ? 'not-deployed' : status === 401 ? 'ambiguous-401-pre-auth' : 'present';
    record(name, { ok: verdict === 'present', status, verdict, body_excerpt: typeof body === 'string' ? body : JSON.stringify(body).slice(0, 200),
      summary: `HTTP ${status} → ${verdict}${verdict === 'ambiguous-401-pre-auth' ? ' (needs NIP-98 admin credential to resolve)' : ''}` });
  } catch (e) { record(name, { ok: false, error: String(e), summary: `error ${e}` }); }
}

writeFileSync(OUT, JSON.stringify(receipt, null, 2) + '\n');
console.log(`receipt → ${OUT}`);
process.exit(0);
