#!/usr/bin/env node
// @ts-check
'use strict';

/**
 * estate-health — nightly whole-estate health snapshot, collected by CI and
 * read by the dream cycle (ADR-2008).
 *
 * The defect this closes: the estate's health lived in a hand-maintained
 * closeout table. A table written by hand is a claim about the past — it went
 * stale the moment a workflow went red, and nothing anywhere noticed. There was
 * no machine-readable answer to "is the estate green tonight?", so the site
 * could publish a confident status section while three repositories were
 * failing CI and a published surface was returning 404.
 *
 * This collector replaces the claim with a measurement:
 *
 *   collect  Reads scripts/estate-health/roster.json (the declared estate) and
 *            interrogates the GitHub API, the public surfaces and the package
 *            registries, writing website/static/data/estate-health.json. Runs
 *            in CI at 02:30 UTC where the tokens are, and NOWHERE else.
 *   check    OFFLINE. Reads the committed snapshot and prints a verdict. This is
 *            the dream-cycle evaluator entrypoint: the 03:00 UTC dream night
 *            runs in an annexe that holds no tokens and must stay offline-safe,
 *            so a dream night READS the snapshot and never re-collects it.
 *
 * Design rules that matter here:
 *
 *   DEGRADE, NEVER THROW. `collect` is a nightly unattended job whose output is
 *   committed. A single 403, a slow surface or a repository that has been made
 *   private must degrade THAT FIELD (null + a recorded note) and leave the rest
 *   of the snapshot intact. A collector that exits non-zero on a transient
 *   network failure publishes nothing, which is strictly worse than publishing
 *   a snapshot that says which parts could not be read. The only non-zero exit
 *   from `collect` is a usage error or an unwritable output path.
 *
 *   THE SNAPSHOT SAYS WHAT IT DOES NOT KNOW. Every unreadable repository is
 *   `readable:false` with `ci.state:"unknown"` and null counts — never an
 *   optimistic zero. `unknown` and `green` are different answers and the page
 *   renders them differently.
 *
 *   DETERMINISTIC ORDERING. Repositories, surfaces and registries appear in
 *   roster order; CI runs are sorted by workflow name. Two collections of an
 *   unchanged estate differ only in `generated_at` and the latency figures, so
 *   the nightly commit is a real diff rather than reordering noise.
 *
 * Zero dependencies: Node 22 native fetch, AbortSignal.timeout, node:fs.
 *
 * Usage
 *   node scripts/estate-health.mjs collect [--out PATH] [--roster PATH]
 *   node scripts/estate-health.mjs check   [--file PATH] [--max-age-hours 36]
 *
 * Environment (collect only)
 *   GITHUB_TOKEN       required. `Authorization: Bearer` for the GitHub API.
 *   ESTATE_READ_TOKEN  optional. Retried for a repository the primary token
 *                      cannot see (404/403) — private, cross-owner repositories
 *                      such as jjohare/visionGraph. Absent => readable:false.
 *                      A CI job's GITHUB_TOKEN is scoped to its own repository
 *                      and WILL 404 on a repository under another owner, so in
 *                      CI this is the only way that row is readable: a
 *                      fine-grained PAT with Contents:read, Actions:read and
 *                      Metadata:read on jjohare/visionGraph, added as a secret
 *                      on DreamLab-AI/VisionFlow. A local run with a full-scope
 *                      user PAT reads it without the fallback, so local success
 *                      does NOT evidence the CI path.
 *   Tokens are read, never printed: no log line in this file interpolates one.
 *
 * Contract additions beyond the published schema, all additive (a consumer that
 * ignores them reads the documented schema unchanged):
 *
 *   repos[].notes         string[], why a field on this repository degraded
 *   registries[].note     string|null, likewise for a registry lookup
 *   repos[].pages.build_type  "workflow"|"legacy"|null, which explains why a
 *                             LIVE Pages site can report status:null
 *
 * They exist so that a null in the snapshot is explicable from the snapshot,
 * without re-running the collector to find out what went wrong.
 */

import { readFileSync, writeFileSync, mkdirSync, existsSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { dirname, join, resolve } from 'node:path';

const HERE = dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = resolve(HERE, '..');

// ── constants ────────────────────────────────────────────────────────────
const SCHEMA = 'visionflow.estate-health/1';
const COLLECTOR = 'scripts/estate-health.mjs';

/** Identifies this collector to every host it touches; crates.io REQUIRES it. */
const USER_AGENT = 'visionflow-estate-health (https://www.visionflow.info)';

const GITHUB_API = 'https://api.github.com';
const GITHUB_HEADERS = {
  'Accept': 'application/vnd.github+json',
  'X-GitHub-Api-Version': '2022-11-28',
};

const GITHUB_TIMEOUT_MS = 15_000;   // API calls
const SURFACE_TIMEOUT_MS = 10_000;  // public surfaces
const CONCURRENCY = 4;              // ≤ 4 in flight, estate-wide

/**
 * GitHub run conclusions, partitioned. `startup_failure` sits with the failures
 * because a workflow that could not start did not pass; anything GitHub adds
 * later that we do not recognise is treated as amber rather than silently
 * counted green — an unknown conclusion is a thing to look at, not a pass.
 */
const CONCLUSION_RED = new Set(['failure', 'timed_out', 'startup_failure']);
const CONCLUSION_GREEN = new Set(['success', 'skipped', 'neutral']);
const CONCLUSION_AMBER = new Set(['cancelled', 'action_required', 'stale']);

/** Only runs GitHub attributes to the branch itself count; PR runs do not. */
const CI_EVENTS = new Set(['push', 'schedule', 'workflow_dispatch']);

// ── argument parsing ─────────────────────────────────────────────────────
const argv = process.argv.slice(2);
const CMD = argv[0];

/** Read `--flag value`, or a default. */
function opt(name, fallback) {
  const i = argv.indexOf(`--${name}`);
  return i >= 0 && argv[i + 1] ? argv[i + 1] : fallback;
}

if (!['collect', 'check'].includes(CMD)) {
  process.stderr.write(
    'usage: estate-health.mjs collect [--out PATH] [--roster PATH]\n'
    + '       estate-health.mjs check  [--file PATH] [--max-age-hours 36]\n');
  process.exit(2);
}

const OUT_PATH = resolve(REPO_ROOT, opt('out', 'website/static/data/estate-health.json'));
const ROSTER_PATH = resolve(REPO_ROOT, opt('roster', 'scripts/estate-health/roster.json'));
const CHECK_PATH = resolve(REPO_ROOT, opt('file', 'website/static/data/estate-health.json'));
const MAX_AGE_HOURS = Number(opt('max-age-hours', '36'));

// ── small helpers ────────────────────────────────────────────────────────
const readJsonFile = (p) => JSON.parse(readFileSync(p, 'utf8'));

/** First line of a commit message, trimmed. */
const firstLine = (s) => String(s ?? '').split('\n')[0].trim();

/** An error's message, never its cause chain (which can carry request detail). */
const reason = (err) => (err && err.name === 'TimeoutError'
  ? 'timeout'
  : String((err && err.message) || err || 'unknown error'));

/**
 * Run `fn` over `items` with at most `limit` in flight, returning results in
 * INPUT order. Deliberately tiny: a worker pool drawing from a shared cursor.
 */
async function mapPool(items, limit, fn) {
  const out = new Array(items.length);
  let cursor = 0;
  const worker = async () => {
    while (cursor < items.length) {
      const i = cursor++;
      out[i] = await fn(items[i], i);
    }
  };
  await Promise.all(Array.from({ length: Math.min(limit, items.length) }, worker));
  return out;
}

/**
 * One HTTP request. Never throws: a transport failure, a timeout or an abort
 * comes back as `{ ok:false, status:null, error }` so every caller can degrade
 * a field instead of unwinding the run.
 *
 * @returns {Promise<{ok: boolean, status: number|null, headers: Headers|null,
 *                     text: string|null, latency_ms: number, error: string|null}>}
 */
async function request(url, { method = 'GET', headers = {}, timeoutMs = GITHUB_TIMEOUT_MS, wantBody = true } = {}) {
  const started = Date.now();
  try {
    const res = await fetch(url, {
      method,
      redirect: 'follow',
      headers: { 'User-Agent': USER_AGENT, ...headers },
      signal: AbortSignal.timeout(timeoutMs),
    });
    const text = wantBody && method !== 'HEAD' ? await res.text() : null;
    return {
      ok: res.ok,
      status: res.status,
      headers: res.headers,
      text,
      latency_ms: Date.now() - started,
      error: null,
    };
  } catch (err) {
    return {
      ok: false,
      status: null,
      headers: null,
      text: null,
      latency_ms: Date.now() - started,
      error: reason(err),
    };
  }
}

/**
 * A GitHub API call, parsed. `data` is null unless the response was 2xx AND the
 * body parsed as JSON; a 404 is a normal, expected answer here (no release, no
 * Pages site) and is reported as a status, not an error.
 *
 * @returns {Promise<{status: number|null, data: any, headers: Headers|null, error: string|null}>}
 */
async function githubJson(path, token) {
  const res = await request(`${GITHUB_API}${path}`, {
    headers: { ...GITHUB_HEADERS, ...(token ? { Authorization: `Bearer ${token}` } : {}) },
  });
  if (!res.ok) {
    return { status: res.status, data: null, headers: res.headers, error: res.error };
  }
  try {
    return { status: res.status, data: JSON.parse(res.text ?? 'null'), headers: res.headers, error: null };
  } catch {
    return { status: res.status, data: null, headers: res.headers, error: 'unparseable JSON body' };
  }
}

// ── GitHub: repository access ────────────────────────────────────────────
/**
 * Resolve which token can see a repository. The primary token is tried first;
 * a 404 or 403 (GitHub answers 404, not 403, for a repository a token cannot
 * see) is retried with ESTATE_READ_TOKEN when one is configured and differs.
 * The winning token is then reused for that repository's remaining calls, so
 * the fallback costs at most one extra request per unreadable repository.
 *
 * @returns {Promise<{repo: any, token: string|undefined, status: number|null,
 *                    error: string|null, used_fallback: boolean}>}
 */
async function resolveRepoAccess(fullName, primary, fallback) {
  const first = await githubJson(`/repos/${fullName}`, primary);
  if (first.data) {
    return { repo: first.data, token: primary, status: first.status, error: null, used_fallback: false };
  }
  const retryable = first.status === 404 || first.status === 403;
  if (retryable && fallback && fallback !== primary) {
    const second = await githubJson(`/repos/${fullName}`, fallback);
    if (second.data) {
      return { repo: second.data, token: fallback, status: second.status, error: null, used_fallback: true };
    }
    return {
      repo: null,
      token: undefined,
      status: second.status,
      error: second.error ?? `HTTP ${second.status} with both tokens`,
      used_fallback: true,
    };
  }
  return {
    repo: null,
    token: undefined,
    status: first.status,
    error: first.error ?? `HTTP ${first.status}`,
    used_fallback: false,
  };
}

// ── GitHub: CI state ─────────────────────────────────────────────────────
/**
 * Most recent qualifying run per workflow, from one `/actions/runs` page.
 *
 * "Qualifying" is branch-attributed events only (push, schedule,
 * workflow_dispatch): a pull_request run reports the state of a proposal, not
 * of the default branch, and counting it would make a repository red because
 * somebody opened a Dependabot PR.
 *
 * @returns {Map<string, any>} workflow_id (as string) -> run
 */
function latestRunPerWorkflow(runs) {
  const latest = new Map();
  for (const run of runs) {
    if (!CI_EVENTS.has(run.event)) continue;
    const key = String(run.workflow_id ?? run.name ?? 'unknown');
    const prev = latest.get(key);
    if (!prev || isNewerRun(run, prev)) latest.set(key, run);
  }
  return latest;
}

/** Newest by creation time, tie-broken by run_number (monotonic per workflow). */
function isNewerRun(a, b) {
  const ta = Date.parse(a.created_at ?? a.run_started_at ?? 0) || 0;
  const tb = Date.parse(b.created_at ?? b.run_started_at ?? 0) || 0;
  if (ta !== tb) return ta > tb;
  return (a.run_number ?? 0) > (b.run_number ?? 0);
}

/**
 * Fold the latest runs into one estate-legible state.
 *
 *   red     any latest run concluded failure / timed_out / startup_failure
 *   amber   none red, and some run is cancelled, action_required, stale,
 *           still queued or in progress, or carries a conclusion we do not know
 *   green   every latest run concluded success, skipped or neutral
 *   none    no qualifying runs at all
 *
 * Red dominates: one failing workflow makes the repository red however many
 * others are green, because the failing one is the defect.
 */
function ciStateFrom(runRecords) {
  if (runRecords.length === 0) return 'none';
  let amber = false;
  for (const r of runRecords) {
    if (r.status !== 'completed') { amber = true; continue; }
    if (CONCLUSION_RED.has(r.conclusion)) return 'red';
    if (CONCLUSION_GREEN.has(r.conclusion)) continue;
    amber = true; // cancelled, action_required, stale, or unrecognised
  }
  return amber ? 'amber' : 'green';
}

/** The snapshot's per-run record: what the page shows and check greps. */
function runRecord(run) {
  return {
    workflow: run.name ?? 'unknown',
    conclusion: run.conclusion ?? null,
    status: run.status ?? null,
    url: run.html_url ?? null,
    updated_at: run.updated_at ?? run.created_at ?? null,
  };
}

// ── GitHub: open pull requests ───────────────────────────────────────────
/**
 * Open PR count from `/pulls?state=open&per_page=1`: with one item per page the
 * `rel="last"` page number IS the count. Preferred over /search/issues because
 * search is separately rate-limited (30/min), lags its index, and behaves
 * differently on private repositories.
 *
 * @returns {Promise<{count: number|null, note: string|null}>}
 */
async function openPrCount(fullName, token) {
  const res = await githubJson(`/repos/${fullName}/pulls?state=open&per_page=1`, token);
  if (!res.data || !Array.isArray(res.data)) {
    return { count: null, note: `open PR count unavailable (${res.error ?? `HTTP ${res.status}`})` };
  }
  const last = lastPageFromLink(res.headers?.get('link'));
  if (last !== null) return { count: last, note: null };
  return { count: res.data.length, note: null };
}

/** `<...&page=7>; rel="last"` -> 7; no rel="last" -> null. */
function lastPageFromLink(link) {
  if (!link) return null;
  for (const part of link.split(',')) {
    if (!/rel="last"/.test(part)) continue;
    const m = part.match(/[?&]page=(\d+)/);
    if (m) return Number(m[1]);
  }
  return null;
}

// ── GitHub: one repository ───────────────────────────────────────────────
/**
 * Everything the snapshot records about one repository. Six API calls at most,
 * each of which may degrade its own field independently.
 */
async function collectRepo(entry, primary, fallback) {
  const url = `https://github.com/${entry.full_name}`;
  /** @type {string[]} */
  const notes = [];

  const base = {
    name: entry.name,
    full_name: entry.full_name,
    url,
    provenance: entry.provenance,
    role: entry.role,
  };

  const access = await resolveRepoAccess(entry.full_name, primary, fallback);
  if (!access.repo) {
    notes.push(`repository not readable: ${access.error}`
      + (fallback ? ' (ESTATE_READ_TOKEN also tried)' : ' (no ESTATE_READ_TOKEN configured)'));
    return {
      ...base,
      visibility: null,
      archived: null,
      readable: false,
      default_branch: null,
      head: null,
      ci: { state: 'unknown', runs: [] },
      open_prs: null,
      open_issues: null,
      release: null,
      pages: null,
      notes,
    };
  }

  const repo = access.repo;
  const token = access.token;
  const branch = repo.default_branch ?? 'main';
  if (access.used_fallback) notes.push('read with ESTATE_READ_TOKEN');

  const [head, ci, prs, release, pages] = await Promise.all([
    collectHead(entry.full_name, branch, token, notes),
    collectCi(entry.full_name, branch, token, notes),
    openPrCount(entry.full_name, token),
    collectRelease(entry.full_name, token, notes),
    collectPages(entry.full_name, token, notes),
  ]);

  if (prs.note) notes.push(prs.note);

  // `open_issues_count` counts issues AND pull requests; the difference is the
  // issue count. Clamped because the two calls are not a single transaction.
  const openIssues = typeof repo.open_issues_count === 'number' && prs.count !== null
    ? Math.max(0, repo.open_issues_count - prs.count)
    : null;
  if (openIssues === null) notes.push('open issue count unavailable (needs the PR count to separate issues from PRs)');

  return {
    ...base,
    visibility: repo.private ? 'private' : 'public',
    archived: Boolean(repo.archived),
    readable: true,
    default_branch: branch,
    head,
    ci,
    open_prs: prs.count,
    open_issues: openIssues,
    release,
    pages,
    notes,
  };
}

/** Tip commit of the default branch. */
async function collectHead(fullName, branch, token, notes) {
  const res = await githubJson(
    `/repos/${fullName}/commits?sha=${encodeURIComponent(branch)}&per_page=1`, token);
  const commit = Array.isArray(res.data) ? res.data[0] : null;
  if (!commit) {
    notes.push(`head commit unavailable (${res.error ?? `HTTP ${res.status}`})`);
    return null;
  }
  const sha = String(commit.sha ?? '');
  return {
    sha,
    short: sha.slice(0, 7),
    date: commit.commit?.committer?.date ?? commit.commit?.author?.date ?? null,
    message: firstLine(commit.commit?.message),
    url: commit.html_url ?? null,
  };
}

/** CI state on the default branch, from a single runs page. */
async function collectCi(fullName, branch, token, notes) {
  const res = await githubJson(
    `/repos/${fullName}/actions/runs?branch=${encodeURIComponent(branch)}&per_page=100`, token);
  if (!res.data || !Array.isArray(res.data.workflow_runs)) {
    notes.push(`CI runs unavailable (${res.error ?? `HTTP ${res.status}`})`);
    return { state: 'unknown', runs: [] };
  }
  const all = res.data.workflow_runs;
  const latest = latestRunPerWorkflow(all);
  const runs = [...latest.values()]
    .map(runRecord)
    .sort((a, b) => a.workflow.localeCompare(b.workflow));
  if (all.length > 0 && runs.length === 0) {
    notes.push('runs exist on the default branch but none from push/schedule/workflow_dispatch');
  }
  return { state: ciStateFrom(runs), runs };
}

/** Latest published release; 404 simply means there is none. */
async function collectRelease(fullName, token, notes) {
  const res = await githubJson(`/repos/${fullName}/releases/latest`, token);
  if (!res.data) {
    if (res.status !== 404) notes.push(`release unavailable (${res.error ?? `HTTP ${res.status}`})`);
    return null;
  }
  return {
    tag: res.data.tag_name ?? null,
    date: res.data.published_at ?? res.data.created_at ?? null,
    url: res.data.html_url ?? null,
  };
}

/**
 * GitHub Pages configuration; 404/403 means no Pages site or no visibility.
 *
 * `status` is recorded exactly as GitHub reports it, INCLUDING null, and
 * `build_type` is recorded beside it to explain the null. A Pages site built by
 * Actions (`build_type: "workflow"`, which is how this very site publishes)
 * leaves the legacy `status` field null while serving perfectly; a legacy
 * branch-built site (`build_type: "legacy"`) populates it with "built". So null
 * here means "not reported", never "errored", and the pair of fields lets a
 * consumer tell those apart without guessing.
 *
 * Reachability is deliberately NOT inferred from either field — that is the
 * surfaces probe's job, because it actually fetches the URL. Of the status
 * values only `errored` is a defect signal.
 */
async function collectPages(fullName, token, notes) {
  const res = await githubJson(`/repos/${fullName}/pages`, token);
  if (!res.data) {
    if (res.status !== 404 && res.status !== 403) {
      notes.push(`Pages status unavailable (${res.error ?? `HTTP ${res.status}`})`);
    }
    return null;
  }
  return {
    url: res.data.html_url ?? null,
    status: res.data.status ?? null,
    build_type: res.data.build_type ?? null,
  };
}

// ── surfaces ─────────────────────────────────────────────────────────────
/**
 * Reachability of one public surface.
 *
 * HEAD first — a reachability probe should not pull an ontology down the wire —
 * falling back to GET when a host rejects the method, and using GET outright
 * where the roster asks for the body to be parsed.
 *
 * `ok` is STATUS ONLY. A content-type that is not what the roster expects is
 * recorded in the note but does not mark the surface down: a served-but-
 * mislabelled document is a different, lesser defect than an unreachable one,
 * and conflating them would make the page cry wolf.
 */
async function probeSurface(surface) {
  const wantsBody = surface.parse === 'json';
  /** @type {string[]} */
  const notes = [];

  let res = await request(surface.url, {
    method: wantsBody ? 'GET' : 'HEAD',
    timeoutMs: SURFACE_TIMEOUT_MS,
    wantBody: wantsBody,
  });

  // Some hosts and CDNs refuse HEAD outright; that is a method restriction,
  // not an outage, so confirm with GET before calling the surface down.
  if (!wantsBody && (res.status === 405 || res.status === 501 || res.status === 403)) {
    notes.push(`HEAD refused (HTTP ${res.status}); retried with GET`);
    res = await request(surface.url, { method: 'GET', timeoutMs: SURFACE_TIMEOUT_MS, wantBody: true });
  }

  const contentType = res.headers?.get('content-type') ?? null;
  if (res.error) notes.push(res.error);
  if (surface.expect_content_type && contentType
      && !contentType.toLowerCase().includes(surface.expect_content_type.toLowerCase())) {
    notes.push(`expected content-type ${surface.expect_content_type}, got ${contentType}`);
  }
  if (wantsBody && res.ok) notes.push(...summariseJsonBody(res.text));

  return {
    name: surface.name,
    url: surface.url,
    status: res.status,
    ok: Boolean(res.status && res.status >= 200 && res.status < 300),
    content_type: contentType,
    latency_ms: res.latency_ms,
    note: notes.length ? notes.join('; ') : null,
  };
}

/**
 * Corpus-size signal from a stats document, best effort. Any shape we do not
 * recognise yields nothing rather than a guess.
 */
function summariseJsonBody(text) {
  if (!text) return ['body empty'];
  let data;
  try {
    data = JSON.parse(text);
  } catch {
    return ['body did not parse as JSON'];
  }
  const bits = [];
  for (const key of ['classes', 'pages', 'nodes', 'edges']) {
    const v = data?.[key] ?? data?.totals?.[key] ?? data?.stats?.[key];
    if (typeof v === 'number') bits.push(`${key}=${v}`);
  }
  return bits.length ? [bits.join(' ')] : [];
}

// ── registries ───────────────────────────────────────────────────────────
/** Dispatch on the roster's registry name. */
function collectRegistry(entry) {
  if (entry.registry === 'crates.io') return collectCrate(entry.name);
  if (entry.registry === 'npm') return collectNpmPackage(entry.name);
  return Promise.resolve({
    registry: entry.registry, name: entry.name, version: null,
    published_at: null, url: null, note: `unsupported registry "${entry.registry}"`,
  });
}

/**
 * crates.io. `max_version` (not `max_stable_version`) is read, and never fallen
 * back to: `max_stable_version` is NULL for a crate that has only ever shipped
 * pre-releases (nostr-bbs-core, at 1.0.0-beta.10), which would report a live
 * published crate as unpublished. A pre-release IS what the estate published.
 *
 * `published_at` is the creation time of the version actually being reported,
 * matched by number rather than assuming `versions[0]` is the maximum, falling
 * back to `versions[0]` when no entry matches. Deliberately NOT
 * `crate.updated_at`, which moves on any crate metadata change and would
 * misdate the release.
 */
async function collectCrate(name) {
  const url = `https://crates.io/crates/${name}`;
  const res = await request(`https://crates.io/api/v1/crates/${encodeURIComponent(name)}`, {
    headers: { Accept: 'application/json' },
  });
  const base = { registry: 'crates.io', name, version: null, published_at: null, url };
  if (!res.ok) {
    return { ...base, note: `crates.io unavailable (${res.error ?? `HTTP ${res.status}`})` };
  }
  let body;
  try {
    body = JSON.parse(res.text ?? 'null');
  } catch {
    return { ...base, note: 'crates.io response did not parse as JSON' };
  }
  const version = body?.crate?.max_version ?? null;
  const versions = Array.isArray(body?.versions) ? body.versions : [];
  const match = versions.find((v) => v?.num === version) ?? versions[0];
  return {
    ...base,
    version,
    published_at: match?.created_at ?? null,
    note: version ? null : 'crates.io returned no max_version',
  };
}

/** npm registry: dist-tags.latest, timestamped from the `time` map. */
async function collectNpmPackage(name) {
  const url = `https://www.npmjs.com/package/${name}`;
  const res = await request(`https://registry.npmjs.org/${name.replace('/', '%2F')}`, {
    headers: { Accept: 'application/json' },
  });
  const base = { registry: 'npm', name, version: null, published_at: null, url };
  if (!res.ok) {
    return { ...base, note: `npm unavailable (${res.error ?? `HTTP ${res.status}`})` };
  }
  let body;
  try {
    body = JSON.parse(res.text ?? 'null');
  } catch {
    return { ...base, note: 'npm response did not parse as JSON' };
  }
  const version = body?.['dist-tags']?.latest ?? null;
  return {
    ...base,
    version,
    published_at: version ? (body?.time?.[version] ?? null) : null,
    note: version ? null : 'npm returned no dist-tags.latest',
  };
}

// ── provenance of the snapshot itself ────────────────────────────────────
/** Which checkout produced this snapshot, and which run, when run by CI. */
function generatorBlock() {
  return {
    revision: process.env.GITHUB_SHA || gitRevision() || 'local',
    workflow_run: workflowRunUrl(),
    collector: COLLECTOR,
  };
}

function gitRevision() {
  try {
    return execFileSync('git', ['rev-parse', 'HEAD'], {
      cwd: REPO_ROOT, encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'],
    }).trim() || null;
  } catch {
    return null;
  }
}

function workflowRunUrl() {
  const { GITHUB_SERVER_URL, GITHUB_REPOSITORY, GITHUB_RUN_ID } = process.env;
  if (!GITHUB_REPOSITORY || !GITHUB_RUN_ID) return null;
  const server = GITHUB_SERVER_URL || 'https://github.com';
  return `${server}/${GITHUB_REPOSITORY}/actions/runs/${GITHUB_RUN_ID}`;
}

// ── summary ──────────────────────────────────────────────────────────────
/** Counts the page shows without walking the arrays itself. */
function summarise(repos, surfaces) {
  const state = (s) => repos.filter((r) => r.ci.state === s).length;
  return {
    repos: repos.length,
    green: state('green'),
    red: state('red'),
    amber: state('amber'),
    none: state('none'),
    unreadable: repos.filter((r) => !r.readable).length,
    open_prs: repos.reduce((n, r) => n + (r.open_prs ?? 0), 0),
    surfaces_ok: surfaces.filter((s) => s.ok).length,
    surfaces_total: surfaces.length,
  };
}

// ── collect ──────────────────────────────────────────────────────────────
async function collect() {
  if (!existsSync(ROSTER_PATH)) {
    process.stderr.write(`ERROR: roster not found: ${ROSTER_PATH}\n`);
    return 1;
  }
  const roster = readJsonFile(ROSTER_PATH);
  const primary = process.env.GITHUB_TOKEN || undefined;
  const fallback = process.env.ESTATE_READ_TOKEN || undefined;

  if (!primary) {
    // Unauthenticated GitHub is 60 requests/hour: the run would degrade into a
    // snapshot of nothing but rate-limit notes. Say so, then collect anyway —
    // the surfaces and registries need no token and are still worth recording.
    process.stderr.write(
      'WARNING: GITHUB_TOKEN is not set. GitHub fields will degrade to unreadable.\n');
  }

  const [repos, surfaces, registries] = await Promise.all([
    mapPool(roster.repos, CONCURRENCY, (entry) => collectRepo(entry, primary, fallback)),
    mapPool(roster.surfaces, CONCURRENCY, probeSurface),
    mapPool(roster.registries, CONCURRENCY, collectRegistry),
  ]);

  const snapshot = {
    schema: SCHEMA,
    generated_at: new Date().toISOString(),
    generator: generatorBlock(),
    summary: summarise(repos, surfaces),
    repos,
    surfaces,
    registries,
  };

  mkdirSync(dirname(OUT_PATH), { recursive: true });
  writeFileSync(OUT_PATH, `${JSON.stringify(snapshot, null, 2)}\n`);

  const s = snapshot.summary;
  process.stdout.write(
    `wrote ${OUT_PATH}\n`
    + `repos: ${s.repos} (green ${s.green}, red ${s.red}, amber ${s.amber}, `
    + `none ${s.none}, unreadable ${s.unreadable})\n`
    + `open PRs: ${s.open_prs}  surfaces: ${s.surfaces_ok}/${s.surfaces_total} ok  `
    + `registries: ${registries.filter((r) => r.version).length}/${registries.length} resolved\n`
    + 'ESTATE-HEALTH-COLLECTED\n');
  return 0;
}

// ── check (offline) ──────────────────────────────────────────────────────
/**
 * The dream-cycle evaluator. Reads the committed snapshot, prints one line per
 * red or amber repository and per unreachable surface, then exactly one verdict
 * line. Touches no network and no token.
 *
 * Verdict precedence: RED beats STALE. A stale snapshot showing a failure still
 * shows a failure, and reporting staleness would bury the defect; STALE is
 * therefore only reached when the estate is otherwise clean.
 */
function check() {
  if (!existsSync(CHECK_PATH)) {
    process.stdout.write(`DOWN snapshot missing ${CHECK_PATH}\nESTATE-HEALTH-STALE\n`);
    return 1;
  }

  let snap;
  try {
    snap = readJsonFile(CHECK_PATH);
  } catch (err) {
    process.stdout.write(`DOWN snapshot unparseable ${reason(err)}\nESTATE-HEALTH-STALE\n`);
    return 1;
  }

  if (snap?.schema !== SCHEMA) {
    process.stdout.write(
      `DOWN snapshot schema ${snap?.schema ?? 'absent'} (expected ${SCHEMA})\n`
      + 'ESTATE-HEALTH-STALE\n');
    return 1;
  }

  let red = false;

  for (const repo of snap.repos ?? []) {
    if (repo.ci?.state === 'red' || repo.ci?.state === 'amber') {
      const label = repo.ci.state === 'red' ? 'RED' : 'AMBER';
      if (repo.ci.state === 'red') red = true;
      for (const run of repo.ci.runs ?? []) {
        const bad = run.status !== 'completed' || !CONCLUSION_GREEN.has(run.conclusion);
        if (!bad) continue;
        const outcome = run.status === 'completed' ? (run.conclusion ?? 'null') : run.status;
        process.stdout.write(`${label} ${repo.name} ${run.workflow}=${outcome} ${run.url ?? repo.url}\n`);
      }
    } else if (repo.readable === false) {
      process.stdout.write(`AMBER ${repo.name} repository=unreadable ${repo.url}\n`);
    }
  }

  for (const surface of snap.surfaces ?? []) {
    if (surface.ok) continue;
    red = true;
    process.stdout.write(`DOWN ${surface.name} ${surface.status ?? surface.note ?? 'unreachable'}\n`);
  }

  if (red) {
    process.stdout.write('ESTATE-HEALTH-RED\n');
    return 1;
  }

  const ageHours = (Date.now() - Date.parse(snap.generated_at)) / 3_600_000;
  if (!Number.isFinite(ageHours) || ageHours > MAX_AGE_HOURS) {
    const age = Number.isFinite(ageHours) ? `${ageHours.toFixed(1)} h` : 'unknown age';
    process.stdout.write(`DOWN snapshot stale ${age} (limit ${MAX_AGE_HOURS} h)\nESTATE-HEALTH-STALE\n`);
    return 1;
  }

  process.stdout.write('ESTATE-HEALTH-OK\n');
  return 0;
}

// ── entrypoint ───────────────────────────────────────────────────────────
process.exitCode = CMD === 'collect' ? await collect() : check();
