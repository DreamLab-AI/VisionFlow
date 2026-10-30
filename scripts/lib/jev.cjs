'use strict';

/**
 * jev.cjs — the one fail-open client for the opt-in "Jev option" (System One
 * typed decisions: Choice / Score / Noul over POST /v1/systemone).
 *
 * Every Jev integration in this repository goes through `decide()`. It returns
 * an OUTCOME and never throws:
 *
 *   { outcome: 'accepted', answers, model, version }
 *   { outcome: 'fallback', reason,  version }
 *
 * `reason` is one of: flag-off, no-endpoint, local-only-refused, timeout,
 * transport, http-<code>, malformed, low-confidence, error. On every fallback
 * the caller must behave exactly as it did before Jev existed.
 *
 * Authority stays in code: this module only asks and validates. It writes
 * nothing, approves nothing and marks nothing successful.
 *
 * Logging: ONE stderr line per call (`jev: <accepted|fallback> ...`) carrying
 * the reason code and the question-set version. Never the state, a question,
 * a response body or a key. A disabled flag logs nothing (nothing happened).
 *
 * Endpoint resolution (first match wins):
 *   1. VISIONFLOW_JEV_ENDPOINT (+ VISIONFLOW_JEV_MODEL, default jev-latest;
 *      VISIONFLOW_JEV_KEY_ENV names the env var holding a Bearer key, if any).
 *   2. AGENTBOX_SYSTEM_ONE_ENDPOINT + AGENTBOX_SYSTEM_ONE_MODEL when
 *      AGENTBOX_SYSTEM_ONE_PROXY=1 (agentbox loopback proxy, holds the vendor
 *      key: NO Authorization header is sent).
 *   3. https://api.typesafe.ai/v1/systemone, model jev-latest, Bearer
 *      $TYPESAFE_API_KEY, only if that key is set.
 *   4. Otherwise: unavailable (`no-endpoint`).
 * A `local-only` consumer refuses steps 2 and 3 unless the endpoint is declared
 * local (AGENTBOX_SYSTEM_ONE_BACKEND_LOCAL=1).
 */

const DEFAULT_API = 'https://api.typesafe.ai/v1/systemone';
const DEFAULT_MODEL = 'jev-latest';
/** Batch CLIs, not hooks: a human is waiting on purpose, but never indefinitely. */
const DEFAULT_DEADLINE_MS = 15000;
/** UNFITTED. Thresholds must be fitted on labelled data before anything relies on them. */
const DEFAULT_MIN_CONFIDENCE = 0.5;
const ITEMS = ['diagram', 'claim', 'health', 'triage'];

/**
 * Is the Jev option on for `item`? `cli` is the value of `--jev` (true / a
 * comma list) or undefined; it wins over VISIONFLOW_JEV. Off by default.
 * @param {string} item one of ITEMS
 * @param {{env?: object, cli?: (boolean|string)}} [src]
 */
function enabled(item, { env = process.env, cli } = {}) {
  const spec = cli !== undefined && cli !== false ? cli : env.VISIONFLOW_JEV;
  if (spec === undefined || spec === null || spec === false) return false;
  const s = String(spec).trim().toLowerCase();
  if (!s || s === '0' || s === 'false' || s === 'off' || s === 'no') return false;
  if (s === '1' || s === 'true' || s === 'on' || s === 'all') return true;
  return s.split(',').map((x) => x.trim()).includes(item);
}

/** Extract `--jev` / `--jev=a,b` from an argv array. Returns undefined when absent. */
function cliFlag(argv) {
  for (const a of argv) {
    if (a === '--jev') return true;
    if (a.startsWith('--jev=')) return a.slice('--jev='.length);
  }
  return undefined;
}

/**
 * Resolve the endpoint per the contract. Pure.
 * @returns {{url:string, model:string, headers:object}|{reason:string}}
 */
function resolveEndpoint(env, dataClass = 'public') {
  const base = { 'Content-Type': 'application/json' };
  const local = env.AGENTBOX_SYSTEM_ONE_BACKEND_LOCAL === '1';
  const proxyUrl = env.AGENTBOX_SYSTEM_ONE_ENDPOINT || env.AGENTBOX_SKILL_ROUTE_API;
  const proxyModel = env.AGENTBOX_SYSTEM_ONE_MODEL || env.AGENTBOX_SKILL_ROUTE_MODEL;
  const proxy = env.AGENTBOX_SYSTEM_ONE_PROXY === '1' && proxyUrl && proxyModel;
  // Agentbox projects its private local route as the explicit endpoint, marked
  // VISIONFLOW_JEV_LOCAL=1. That route is for local-only data; a public question
  // goes to the proxy, which serves the selected backend. An endpoint named by
  // hand (no LOCAL mark) always wins.
  const projectedLocal = env.VISIONFLOW_JEV_LOCAL === '1';
  if (env.VISIONFLOW_JEV_ENDPOINT && !(projectedLocal && dataClass === 'public' && proxy)) {
    const headers = { ...base };
    const keyEnv = env.VISIONFLOW_JEV_KEY_ENV;
    if (keyEnv && env[keyEnv]) headers.Authorization = `Bearer ${env[keyEnv]}`;
    return { url: env.VISIONFLOW_JEV_ENDPOINT, model: env.VISIONFLOW_JEV_MODEL || DEFAULT_MODEL, headers };
  }
  if (proxy) {
    if (dataClass === 'local-only' && !local) return { reason: 'local-only-refused' };
    return { url: proxyUrl, model: proxyModel, headers: base };
  }
  if (env.TYPESAFE_API_KEY) {
    if (dataClass === 'local-only' && !local) return { reason: 'local-only-refused' };
    return { url: DEFAULT_API, model: DEFAULT_MODEL, headers: { ...base, Authorization: `Bearer ${env.TYPESAFE_API_KEY}` } };
  }
  return { reason: 'no-endpoint' };
}

/** Confidence of one validated answer in [0,1]. Noul has none: distance from 0.5, scaled. */
function confidenceOf(a, type) {
  if (type === 'noul') return Math.abs(a.noul - 0.5) * 2;
  return a.confidence;
}

/**
 * Validate a parsed response against the questions asked. Returns a reason
 * string (`malformed` / `low-confidence`) or null when every answer is usable.
 */
function validate(body, questions, minConfidence) {
  if (!body || typeof body !== 'object' || !body.answers || typeof body.answers !== 'object') return 'malformed';
  for (const [id, q] of Object.entries(questions)) {
    const a = body.answers[id];
    // System One need not repeat the question's type; an explicit conflicting one is still malformed.
    if (!a || typeof a !== 'object' || (a.type !== undefined && a.type !== q.type)) return 'malformed';
    if (q.type === 'choice') {
      if (typeof a.choice !== 'string' || !Object.prototype.hasOwnProperty.call(q.criteria, a.choice)) return 'malformed';
      if (typeof a.confidence !== 'number' || !Number.isFinite(a.confidence)) return 'malformed';
    } else if (q.type === 'score') {
      if (typeof a.score !== 'number' || !Number.isFinite(a.score) || a.score < 0 || a.score > q.criteria.length - 1) return 'malformed';
      if (typeof a.confidence !== 'number' || !Number.isFinite(a.confidence)) return 'malformed';
    } else if (q.type === 'noul') {
      if (typeof a.noul !== 'number' || !Number.isFinite(a.noul) || a.noul < 0 || a.noul > 1) return 'malformed';
    } else return 'malformed';
  }
  for (const [id, q] of Object.entries(questions)) {
    if (confidenceOf(body.answers[id], q.type) < minConfidence) return 'low-confidence';
  }
  return null;
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

/**
 * Ask Jev. Never throws, never blocks beyond `deadlineMs`.
 * @param {object} o
 * @param {string} o.version stable question-set version string (logged)
 * @param {object} o.state   structured state (never logged)
 * @param {object} o.questions map id -> {type, instructions, criteria?}
 * @param {boolean} [o.enabled=true] false => flag-off, no request, no log
 * @param {'public'|'local-only'} [o.dataClass='public']
 * @param {object} [o.env=process.env]
 * @param {Function} [o.fetchImpl=globalThis.fetch]
 * @param {number} [o.deadlineMs] overall hard deadline (also VISIONFLOW_JEV_TIMEOUT_MS)
 * @param {number} [o.retries=1] bounded 429/529 retries inside the deadline
 * @param {number} [o.minConfidence] (also VISIONFLOW_JEV_MIN_CONFIDENCE)
 * @param {Function} [o.log] receives the single log line (default: stderr)
 */
async function decide(o) {
  const version = o && o.version ? String(o.version) : 'unversioned';
  const log = (o && o.log) || ((line) => process.stderr.write(line + '\n'));
  const fallback = (reason, quiet) => {
    if (!quiet) { try { log(`jev: fallback reason=${reason} set=${version}`); } catch { /* logging must not throw */ } }
    return { outcome: 'fallback', reason, version };
  };
  try {
    if (o.enabled === false) return fallback('flag-off', true);
    const env = o.env || process.env;
    const ep = resolveEndpoint(env, o.dataClass || 'public');
    if (ep.reason) return fallback(ep.reason);
    const fetchImpl = o.fetchImpl || globalThis.fetch;
    if (typeof fetchImpl !== 'function') return fallback('no-endpoint');
    const num = (v, d) => { const n = Number(v); return Number.isFinite(n) && n > 0 ? n : d; };
    const deadlineMs = num(o.deadlineMs !== undefined ? o.deadlineMs : env.VISIONFLOW_JEV_TIMEOUT_MS, DEFAULT_DEADLINE_MS);
    const minConfidence = o.minConfidence !== undefined ? o.minConfidence
      : (env.VISIONFLOW_JEV_MIN_CONFIDENCE !== undefined ? Number(env.VISIONFLOW_JEV_MIN_CONFIDENCE) : DEFAULT_MIN_CONFIDENCE);
    const retries = o.retries === undefined ? 1 : o.retries;
    const payload = JSON.stringify({ model: ep.model, state: o.state, questions: o.questions });

    const t0 = Date.now();
    const ctl = new AbortController();
    const timer = setTimeout(() => ctl.abort(), deadlineMs);
    try {
      for (let attempt = 0; ; attempt++) {
        let res;
        try {
          res = await fetchImpl(ep.url, { method: 'POST', headers: ep.headers, body: payload, signal: ctl.signal });
        } catch (e) {
          return fallback(ctl.signal.aborted || (e && (e.name === 'AbortError' || e.name === 'TimeoutError')) ? 'timeout' : 'transport');
        }
        const status = Number(res && res.status);
        if ((status === 429 || status === 529) && attempt < retries) {
          const ra = Number(res.headers && typeof res.headers.get === 'function' ? res.headers.get('retry-after') : NaN);
          const wait = Number.isFinite(ra) && ra >= 0 ? ra * 1000 : 250 * 2 ** attempt;
          if (Date.now() - t0 + wait >= deadlineMs) return fallback(`http-${status}`);
          await sleep(wait);
          continue;
        }
        if (!(status >= 200 && status < 300)) return fallback(`http-${status || 0}`);
        let body;
        try { body = JSON.parse(await res.text()); } catch (e) {
          return fallback(ctl.signal.aborted ? 'timeout' : 'malformed');
        }
        const bad = validate(body, o.questions, minConfidence);
        if (bad) return fallback(bad);
        try { log(`jev: accepted set=${version} model=${String(body.model || ep.model)}`); } catch { /* ignore */ }
        return { outcome: 'accepted', answers: body.answers, model: body.model || ep.model, version };
      }
    } finally {
      clearTimeout(timer);
    }
  } catch {
    return fallback('error');
  }
}

module.exports = {
  decide, enabled, cliFlag, resolveEndpoint, validate, confidenceOf,
  DEFAULT_API, DEFAULT_MODEL, DEFAULT_DEADLINE_MS, DEFAULT_MIN_CONFIDENCE, ITEMS,
};
