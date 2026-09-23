#!/usr/bin/env node
'use strict';

/*
 * diagram-index-gen.cjs — walk the diagrams-as-code tree (docs/diagrams), parse
 * per-file YAML frontmatter, validate every fenced ```mermaid block, and
 * (re)generate the machine-readable coverage indexes.
 *
 * Usage:  node scripts/diagram-index-gen.cjs <dir> [--check] [--render] [--jobs N] [--only <substr>]
 *
 *   <dir>       diagrams root (docs/diagrams). README.md / COVERAGE.md and the
 *               hero/ + archive/ subtrees are skipped.
 *   --check     validate frontmatter, heading ids and block structure only; do
 *               not write the indexes (still exits 1 on error). A `sources:` or
 *               `governing:` path missing from the working tree is accepted when
 *               it exists at the topic's DECLARED revision (a sibling checkout
 *               that is behind is not evidence the file is gone); the count is
 *               printed per topic, and a path present at neither still fails.
 *   --cite-check resolve every `path:line` citation inside a diagram against the
 *               file's `sources:` list (a citation whose file is NOT listed is a
 *               warning — it was never checked), and assert the file is long
 *               enough; warns when the cited line is blank or a lone closing brace,
 *               and when a bare basename matches more than one sources: entry
 *               (ambiguous — qualify the path). Advisory unless --strict-citations is set.
 *   --strict-citations  run citation checks and fail on any diagnostic; requires source access.
 *               Citations listed in <dir>/citation-allowlist.json are exempted from the
 *               revision requirement, counted and printed as "allowlisted (unverifiable)"
 *               rather than as verified; an allowlist entry that matches nothing is stale
 *               and fails the run.
 *   --worktree-citations  read current source bytes instead of declared revisions.
 *   --check-verification  no source access needed: assert VERIFICATION.md was regenerated
 *               against the revisions the topics currently declare, and that the citation
 *               allowlist is well formed and names real topics. This is how a hosted
 *               runner, which cannot resolve cross-repository revisions at all, gates the
 *               freshness of the local --strict-citations run. See docs/diagrams/README.md.
 *   --render    additionally render every mermaid block through `mmdc` (the
 *               Mermaid CLI) into <dir>/rendered/<file>/<id>.svg; any parse
 *               error fails the run and is reported as file:block-id:line, and
 *               any render wider than 4500px fails as illegible.
 *   --jobs N    render concurrency (default 6).
 *   --only S    restrict to files whose relative path contains S.
 *   --no-source-paths  skip the sources:/governing: existence checks (hosted CI has no
 *               sibling checkouts); --cite-check is meaningless with it.
 *
 * Exit codes: 0 ok, 1 validation/render error, 2 usage / IO error.
 *
 * Diagram-file contract (one file = one topic, many diagrams):
 *
 *   ---
 *   id: VC-03                      # area prefix + 2-digit number, unique
 *   title: REST request lifecycle
 *   area: visionclaw               # visionclaw | agentbox | estate
 *   governing:                     # governing docs (repo-relative paths, optional #anchor)
 *     - docs/IDENTITY-authority-chain.md
 *   adrs: [ADR-2009, ADR-2011]     # ledger records this file evidences
 *   sources:                       # repo-relative code paths the diagrams were verified against
 *     - src/main.rs
 *     - src/middleware/rbac_gate.rs
 *   verified_commit: b00c28a0d
 *   ---
 *   ## VC-03.1 GET /api/graph/data
 *   ```mermaid
 *   sequenceDiagram
 *   ...
 *   ```
 *
 * Every mermaid block must sit under an H2 whose first token is `<file-id>.<n>`;
 * that token is the diagram id and must be unique across the tree.
 * The generated README.md diagram table and COVERAGE.md are build artefacts.
 */

const fs = require('fs');
const path = require('path');
const os = require('os');
const { spawn } = require('child_process');

const AREAS = new Set(['visionflow', 'visionclaw', 'agentbox', 'solid-pod-rs', 'nostr-rust-forum', 'dreamlab-ai-website', 'vowl-wasm', 'knowledgegraph', 'visiongraph', 'sidestr-rs', 'estate']);
const SKIP_DIRS = new Set(['hero', 'archive', 'rendered', 'src', 'upgraded', 'regen-2026-06-14', 'triptych-src', '.claude-flow', 'node_modules']);
const SKIP_FILES = new Set(['README.md', 'COVERAGE.md']);
const MAX_WIDTH = 4500; // px — wider renders are illegible at any zoom
const REQUIRED = ['id', 'title', 'area', 'governing', 'adrs', 'sources', 'verified_commit'];

function usage(msg) {
  if (msg) console.error(msg);
  console.error('Usage: node scripts/diagram-index-gen.cjs <dir> [--check] [--render] [--cite-check] [--strict-citations] [--worktree-citations] [--check-verification] [--no-source-paths] [--jobs N] [--only S]');
  process.exit(2);
}

const argv = process.argv.slice(2);
if (argv.length < 1) usage();
const root = path.resolve(argv[0]);
const flags = { check: false, render: false, cite: false, jobs: 6, only: null };
for (let i = 1; i < argv.length; i++) {
  const a = argv[i];
  if (a === '--check') flags.check = true;
  else if (a === '--render') flags.render = true;
  else if (a === '--cite-check') flags.cite = true;
  else if (a === '--strict-citations') { flags.cite = true; flags.strictCitations = true; }
  else if (a === '--worktree-citations') { flags.cite = true; flags.worktreeCitations = true; }
  else if (a === '--jobs') flags.jobs = parseInt(argv[++i], 10) || 6;
  else if (a === '--only') flags.only = argv[++i];
  else if (a === '--no-source-paths') flags.noSourcePaths = true; // CI: sibling checkouts absent
  else if (a === '--check-verification') flags.checkVerification = true;
  else usage(`unknown flag ${a}`);
}
if (flags.strictCitations && flags.noSourcePaths) usage('--strict-citations requires source paths; do not combine with --no-source-paths');
if (flags.checkVerification && flags.only) usage('--check-verification judges the whole tree; do not combine with --only');
if (!fs.existsSync(root) || !fs.statSync(root).isDirectory()) usage(`not a directory: ${root}`);
const repoRoot = path.resolve(root, '..', '..');

// ---------------------------------------------------------------- walk
function walk(dir, out) {
  for (const ent of fs.readdirSync(dir, { withFileTypes: true })) {
    if (ent.isDirectory()) {
      if (SKIP_DIRS.has(ent.name) || ent.name.startsWith('.')) continue;
      walk(path.join(dir, ent.name), out);
    } else if (ent.isFile() && ent.name.endsWith('.md')) {
      if (dir === root && SKIP_FILES.has(ent.name)) continue;
      if (dir === root) continue; // topic files live in area subdirs only
      out.push(path.join(dir, ent.name));
    }
  }
  return out;
}

// ---------------------------------------------------------------- yaml (minimal)
function parseScalar(s) {
  s = s.trim();
  if ((s.startsWith('"') && s.endsWith('"')) || (s.startsWith("'") && s.endsWith("'"))) return s.slice(1, -1);
  return s;
}
function parseInline(s) {
  s = s.trim();
  if (s.startsWith('[') && s.endsWith(']')) {
    const inner = s.slice(1, -1).trim();
    if (!inner) return [];
    return inner.split(',').map(parseScalar).filter(Boolean);
  }
  return parseScalar(s);
}
function parseFrontmatter(text, file, errors) {
  if (!text.startsWith('---\n')) { errors.push(`${file}: missing frontmatter`); return null; }
  const end = text.indexOf('\n---', 4);
  if (end < 0) { errors.push(`${file}: unterminated frontmatter`); return null; }
  const block = text.slice(4, end).split('\n');
  const fm = {};
  let key = null;
  for (const raw of block) {
    if (!raw.trim() || raw.trim().startsWith('#')) continue;
    const m = raw.match(/^([A-Za-z_][A-Za-z0-9_]*):\s*(.*)$/);
    if (m) {
      key = m[1];
      fm[key] = m[2].trim() === '' ? [] : parseInline(m[2]);
    } else if (/^\s*-\s+/.test(raw) && key) {
      if (!Array.isArray(fm[key])) fm[key] = [];
      fm[key].push(parseScalar(raw.replace(/^\s*-\s+/, '')));
    } else {
      errors.push(`${file}: unparseable frontmatter line: ${raw}`);
    }
  }
  return { fm, body: text.slice(end + 4) };
}

// ---------------------------------------------------------------- source existence
// Collected per topic and printed once, so a run always states how much of its
// evidence came from a revision rather than from the checkout on disk.
const SOURCE_AT_REVISION = [];
const existsCache = new Map();
// 'worktree' | 'revision' | null. `repoOf` and `shaFor` are the citation
// checker's own resolvers, so a path is judged against exactly the repository
// and sha its citations are judged against.
function existsForTopic(fm, p) {
  if (fs.existsSync(path.join(repoRoot, p))) return 'worktree';
  const r = repoOf(p);
  if (!r) return null;
  const sha = shaFor({ fm }, r.key);
  if (!sha) return null;
  const key = `${r.abs}:${sha}:${r.rel}`;
  if (existsCache.has(key)) return existsCache.get(key);
  let out = null;
  try {
    require('child_process').execFileSync('git', ['-C', r.abs, 'cat-file', '-e', `${sha}:${r.rel}`], { stdio: 'ignore' });
    out = 'revision';
  } catch { out = null; }
  existsCache.set(key, out);
  return out;
}

// ---------------------------------------------------------------- parse topic files
function parseTopic(file, errors) {
  const rel = path.relative(root, file);
  const text = fs.readFileSync(file, 'utf8');
  const parsed = parseFrontmatter(text, rel, errors);
  if (!parsed) return null;
  const { fm, body } = parsed;
  for (const k of REQUIRED) if (!(k in fm)) errors.push(`${rel}: missing frontmatter field '${k}'`);
  for (const k of ['governing', 'adrs', 'sources']) if (k in fm && !Array.isArray(fm[k])) fm[k] = [fm[k]];
  if (fm.area && !AREAS.has(fm.area)) errors.push(`${rel}: area '${fm.area}' not in ${[...AREAS].join('|')}`);
  const areaDir = rel.split(path.sep)[0];
  if (fm.area && areaDir !== fm.area) errors.push(`${rel}: area '${fm.area}' does not match directory '${areaDir}'`);
  if (fm.id && !/^[A-Z]{2,3}-\d{2,3}$/.test(fm.id)) errors.push(`${rel}: id '${fm.id}' must match /^[A-Z]{2,3}-\\d{2,3}$/`);
  {
    // verified_commit must be a git sha (7-40 hex) or a {repo: sha} map of them. A
    // label such as `worktree-2026-09-07` is not a revision and cannot be checked.
    const v = fm.verified_commit;
    const SHA = /^[0-9a-f]{7,40}$/;
    let ok = false;
    if (typeof v === 'string' && v.trim().startsWith('{')) {
      const pairs = v.trim().slice(1, -1).split(',').map((x) => x.split(':').map((y) => y.trim()));
      ok = pairs.length > 0 && pairs.every(([k, sha]) => k && SHA.test(sha || ''));
    } else if (v && typeof v === 'object') ok = Object.values(v).every((sha) => SHA.test(String(sha)));
    else ok = SHA.test(String(v || ''));
    if (!ok) errors.push(`${rel}: verified_commit '${v}' is not a git sha (7-40 hex) or a {repo: sha} map`);
  }
  {
    // A declared path that is absent from the working tree is not yet evidence
    // that the file does not exist: a sibling checkout is often simply BEHIND
    // the revision its topic declares, and a corpus whose whole claim is "this
    // was read at this commit" must not call that a missing source. So the
    // existence check follows the same rule the citation check does — ask the
    // owning repository whether the blob exists at the topic's sha for it, and
    // accept it when it does. A path that exists at NEITHER is still an error,
    // and the revision-only fallback is counted and printed per topic so it can
    // never pass silently. `--no-source-paths` is unchanged: it skips all of it.
    let atRevision = 0;
    const judge = (p, what) => {
      if (flags.noSourcePaths) return;
      const where = existsForTopic(fm, p);
      if (!where) errors.push(`${rel}: ${what} does not exist: ${p}`);
      else if (where === 'revision') atRevision++;
    };
    for (const s of fm.sources || []) judge(s.split(':')[0], 'source path');
    for (const g of fm.governing || []) judge(g.split('#')[0], 'governing doc');
    if (atRevision) SOURCE_AT_REVISION.push(`${rel}: ${atRevision} source(s) exist at the declared revision but not in the working tree`);
  }

  // headings + mermaid blocks
  const lines = body.split('\n');
  const diagrams = [];
  let currentH2 = null;
  let inFence = false, fenceLang = null, fenceStart = 0, buf = [];
  let proseLines = 0;
  for (let i = 0; i < lines.length; i++) {
    const ln = lines[i];
    if (!inFence) {
      const f = ln.match(/^```(\w*)/);
      if (f) { inFence = true; fenceLang = f[1]; fenceStart = i; buf = []; continue; }
      const h = ln.match(/^##\s+(\S+)\s*(.*)$/);
      if (h) { currentH2 = { id: h[1], title: h[2].trim(), line: i }; continue; }
      if (ln.startsWith('#')) continue;
      if (ln.trim() && !ln.trim().startsWith('<!--')) proseLines++;
    } else {
      if (ln.startsWith('```')) {
        inFence = false;
        if (fenceLang === 'mermaid') {
          if (!currentH2) errors.push(`${rel}: mermaid block at line ${fenceStart + 1} has no H2 heading`);
          else {
            const expect = new RegExp(`^${(fm.id || '').replace('-', '\\-')}\\.\\d+$`);
            if (!expect.test(currentH2.id)) errors.push(`${rel}: H2 id '${currentH2.id}' must be '${fm.id}.<n>'`);
            const src = buf.join('\n');
            for (const m of src.matchAll(/rect\s+rgba?\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)/g)) {
              const lum = 0.299 * +m[1] + 0.587 * +m[2] + 0.114 * +m[3];
              if (lum < 140) errors.push(`${rel}:${currentH2.id} — dark rect fill rgb(${m[1]},${m[2]},${m[3]}) makes message text unreadable on the light render; use a pastel (luminance >= 140)`);
            }
            if (/^\s*(mindmap|pie|quadrantChart|timeline|journey)\b/m.test(src)) errors.push(`${rel}:${currentH2.id} — forbidden diagram kind (no information density)`);
            diagrams.push({ id: currentH2.id, title: currentH2.title, src, line: fenceStart + 1, kind: (buf[0] || '').trim().split(/\s/)[0] });
          }
        }
        continue;
      }
      buf.push(ln);
    }
  }
  if (inFence) errors.push(`${rel}: unterminated code fence`);
  if (diagrams.length === 0) errors.push(`${rel}: no mermaid diagrams`);
  if (proseLines > diagrams.length * 3) errors.push(`${rel}: ${proseLines} prose lines for ${diagrams.length} diagrams — this tree is diagrams-only (max 3 lines per diagram)`);
  return { file, rel, fm, diagrams };
}

// ---------------------------------------------------------------- citation check
// Every fact in this tree is anchored as a `path:line` (or `path:a-b`) inside a
// participant, message or Note. The path is usually written short (a basename or
// a trailing fragment), so resolve it against the file's own `sources:` list.
const CITE_RE = /([A-Za-z0-9_./-]*[A-Za-z0-9_-]\.[A-Za-z0-9]{1,12}):(\d+)(?:\s*-\s*(\d+))?/g;

// ── Revision-pinned reads ────────────────────────────────────────────────────
// Citations are verified against the topic's DECLARED revision, not the working
// tree: another session's uncommitted edits must not flag a correct anchor, and a
// stamp must mean "true at that sha". Repo of a source path = longest matching
// prefix below; sha = the topic's verified_commit (string → the area's own repo
// only; {repo: sha} map → by key, case-insensitive). Unknown sha or non-git path
// falls back to the working tree.
const REPO_PREFIXES = [
  { key: 'agentbox', prefix: '../project/agentbox/' },
  { key: 'unmute', prefix: '../project/voice-stack/unmute/' },   // nested repo, not tracked by visionclaw
  { key: 'visionclaw', prefix: '../project/' },
  { key: 'solid-pod-rs', prefix: '../solid-pod-rs/' },
  { key: 'nostr-rust-forum', prefix: '../nostr-rust-forum/' },
  { key: 'dreamlab-ai-website', prefix: '../dreamlab-ai-website/' },
  { key: 'vowl-wasm', prefix: '../vowl-wasm/' },
  { key: 'knowledgegraph', prefix: '../knowledgeGraph/' },
  { key: 'visiongraph', prefix: '../visionGraph/' },
  { key: 'ruview', prefix: '../RuView/' },
  { key: 'wasmvowl', prefix: '../WasmVOWL/' },
  { key: 'loom', prefix: '../loom/' },
  { key: 'sidestr-rs', prefix: '../sidestr-rs/' },
  { key: 'dream-machine', prefix: '../dream-machine/' },
  { key: 'prose-sanitiser', prefix: '../prose-sanitiser/' },
  { key: 'diagram-ir', prefix: '../diagram-ir/' },
  { key: 'visionflow', prefix: '' },
];
const AREA_REPO = { visionflow: 'visionflow', visionclaw: 'visionclaw', agentbox: 'agentbox', 'solid-pod-rs': 'solid-pod-rs', 'nostr-rust-forum': 'nostr-rust-forum', 'dreamlab-ai-website': 'dreamlab-ai-website', 'vowl-wasm': 'vowl-wasm', knowledgegraph: 'knowledgegraph', visiongraph: 'visiongraph', 'sidestr-rs': 'sidestr-rs' };
function repoOf(p) {
  const clean = p.replace(/^\.\//, '');
  for (const r of REPO_PREFIXES) if (r.prefix && clean.startsWith(r.prefix)) return { key: r.key, rel: clean.slice(r.prefix.length), abs: path.join(repoRoot, r.prefix) };
  if (!clean.startsWith('../')) return { key: 'visionflow', rel: clean, abs: repoRoot };
  return null;
}
function shaFor(t, repoKey) {
  const v = t.fm.verified_commit;
  if (!v) return null;
  if (typeof v === 'string' && v.trim().startsWith('{')) {
    for (const pair of v.trim().slice(1, -1).split(',')) {
      const [k, sha] = pair.split(':').map((x) => x.trim());
      if (k && sha && k.toLowerCase() === repoKey) return sha;
    }
    return null;
  }
  if (typeof v === 'object') { for (const [k, sha] of Object.entries(v)) if (k.toLowerCase() === repoKey) return sha; return null; }
  return AREA_REPO[t.fm.area] === repoKey && /^[0-9a-f]{7,40}$/.test(String(v)) ? String(v) : null;
}
const revCache = new Map();
let CURRENT_TOPIC = null;
// Every path that does NOT end in 'revision' means the bytes judged were the
// working tree's, not the declared revision's. That fallback is why a corpus
// can report zero warnings while a slice of it was never verified against any
// commit, so each reason is named and counted rather than silently absorbed.
const UNVERIFIED_REASON = {
  'no-repo-mapping': 'its repository has no entry in REPO_PREFIXES, so no declared revision can apply',
  'no-sha-for-repo': 'its repository is absent from this topic\'s verified_commit map',
  'revision-read-failed': 'the declared revision does not contain that file (bad sha, or the file did not exist yet)',
};
function revisionLines(t, p) {
  const r = repoOf(p);
  let resolution = null, sha = null;
  if (flags.worktreeCitations) resolution = 'worktree-flag';
  else if (!r) resolution = 'no-repo-mapping';
  else {
    sha = shaFor(t, r.key);
    if (!sha) resolution = 'no-sha-for-repo';
  }
  const key = `${sha || resolution || 'WT'}:${p}`;
  if (revCache.has(key)) return revCache.get(key);
  let lines = null;
  if (sha) {
    try {
      const out = require('child_process').execFileSync('git', ['-C', r.abs, 'show', `${sha}:${r.rel}`], { encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'], maxBuffer: 64 * 1024 * 1024 });
      lines = out.split('\n');
      resolution = 'revision';
    } catch { lines = null; resolution = 'revision-read-failed'; }
  }
  if (!lines) { try { lines = fs.readFileSync(path.join(repoRoot, p), 'utf8').split('\n'); } catch { lines = null; } }
  const res = { lines, resolution, repoKey: r ? r.key : null, sha };
  revCache.set(key, res);
  return res;
}
// ── The citation allowlist ───────────────────────────────────────────────────
// Some citations can never be verified at a declared revision because the file
// they name exists in no commit — a gitignored build artefact is the honest
// case. Refusing them forever keeps the strict gate red; ignoring them silently
// is the failure mode this whole checker exists to prevent. So they are named,
// one per citation, in <dir>/citation-allowlist.json with a reason, and then
// counted and printed SEPARATELY from what was actually verified.
//
// A sidecar JSON rather than a frontmatter field: the frontmatter parser above
// is a flat scalar/list reader with no nested maps, so a {path, line, reason}
// record cannot be expressed in a topic's frontmatter without growing a YAML
// implementation; and a debt register wants one reviewable surface.
const ALLOWLIST_FILE = path.join(root, 'citation-allowlist.json');
function loadAllowlist(errors) {
  if (!fs.existsSync(ALLOWLIST_FILE)) return [];
  let doc;
  try { doc = JSON.parse(fs.readFileSync(ALLOWLIST_FILE, 'utf8')); }
  catch (e) { errors.push(`citation-allowlist.json: not valid JSON (${e.message})`); return []; }
  if (!doc || !Array.isArray(doc.allow)) { errors.push('citation-allowlist.json: missing an `allow` array'); return []; }
  const out = [];
  doc.allow.forEach((e, i) => {
    const at = `citation-allowlist.json: allow[${i}]`;
    if (!e || typeof e !== 'object') { errors.push(`${at} is not an object`); return; }
    for (const k of ['topic', 'path', 'reason']) if (typeof e[k] !== 'string' || !e[k].trim()) { errors.push(`${at} is missing a non-empty '${k}'`); return; }
    if ('diagram' in e && typeof e.diagram !== 'string') { errors.push(`${at} has a non-string 'diagram'`); return; }
    if ('line' in e && !Number.isInteger(e.line)) { errors.push(`${at} has a non-integer 'line'`); return; }
    if (e.diagram && !e.diagram.startsWith(e.topic + '.')) { errors.push(`${at} names diagram '${e.diagram}', which is not in topic '${e.topic}'`); return; }
    out.push({ ...e, used: 0, index: i });
  });
  return out;
}
let ALLOWLIST = [];
// `cited` is the path as written in the diagram, `src` the sources: entry it
// resolved to (absent when it resolved to nothing). Either may be named.
function allowMatch(topic, diagramId, cited, src, line) {
  const n = Number(line);
  for (const e of ALLOWLIST) {
    if (e.topic !== topic.fm.id) continue;
    if (e.diagram && e.diagram !== diagramId) continue;
    if ('line' in e && e.line !== n) continue;
    const p = e.path;
    const namesIt = [cited, src].filter(Boolean).some((c) => c === p || c.endsWith('/' + p) || p.endsWith('/' + c));
    if (!namesIt) continue;
    e.used++;
    return e;
  }
  return null;
}

// Reset per citeCheck run; surfaced on the summary line and in COVERAGE.md.
let CITE_STATS = null;
// Allowlisted citations: reported, never failed. Kept apart from `warnings` so
// that --strict-citations, which turns every warning into an error, cannot turn
// a waiver into one.
let CITE_NOTES = [];
function newCiteStats() { return { checked: 0, unverified: 0, unresolvable: 0, allowlisted: 0, byReason: {}, byTopic: {} }; }

// The set of revisions the tree currently declares, as sorted `repo@sha` tokens.
// Derived from frontmatter alone, so it is computable without any source access —
// which is what lets a hosted runner check that VERIFICATION.md is current.
function declaredRevisions(topics) {
  const vcs = new Set();
  for (const t of topics) {
    const v = t.fm.verified_commit;
    if (v && typeof v === 'object') for (const [k, sha] of Object.entries(v)) vcs.add(`${k}@${sha}`);
    else if (typeof v === 'string' && v.trim().startsWith('{')) {
      // inline-map form kept as a string by the frontmatter parser: {repo: sha, repo: sha}
      for (const pair of v.trim().slice(1, -1).split(',')) { const [k, sha] = pair.split(':').map((x) => x.trim()); if (k && sha) vcs.add(`${k}@${sha}`); }
    } else vcs.add(String(v));
  }
  return [...vcs].sort();
}
const REVISIONS_ANCHOR = '<!-- declared-revisions: ';

function writeVerificationReport(st, topics) {
  const rows = Object.entries(st.byTopic).filter(([, r]) => r.unverified || r.unresolvable || r.allowlisted).sort();
  const out = [
    '<!-- GENERATED BY scripts/diagram-index-gen.cjs --cite-check — DO NOT EDIT BY HAND -->',
    '# Citation verification',
    '',
    'What the last local `--cite-check` run actually resolved. A citation counts as',
    'verified only when its bytes were read from the revision its topic declares',
    '(`git show <sha>:<path>`). Anything else was read from the working tree, which',
    'means it was judged against whatever happens to be checked out, not against a',
    'commit. That distinction is invisible in a warning count, which is why it is',
    'stated here: a green run reports what it checked, not merely that nothing objected.',
    '',
    `- **${st.checked}** verified at the declared revision`,
    `- **${st.unverified}** unverified (read from the working tree)`,
    `- **${st.unresolvable}** unresolvable (never checked at all)`,
    `- **${st.allowlisted}** allowlisted (unverifiable), waived by [citation-allowlist.json](citation-allowlist.json)`,
    '',
    'An allowlisted citation names a file that exists in no commit, so no declared',
    'revision can ever contain it. It is waived by name, one entry per citation, with',
    'a reason; it is never counted as verified. The line checks still applied to it.',
    '',
    `${REVISIONS_ANCHOR}${declaredRevisions(topics).join(' ')} -->`,
    '',
    'The comment above records the revisions the tree declared when this file was',
    'written. `--check-verification` recomputes it from the topics and refuses a',
    'mismatch, which is how CI knows a re-stamped topic has had its citations',
    're-resolved locally rather than merely renumbered.',
    '',
  ];
  for (const [reason, n] of Object.entries(st.byReason).sort((a, b) => b[1] - a[1])) {
    out.push(`Unverified, ${n}: ${UNVERIFIED_REASON[reason] || reason}.`);
    out.push('');
  }
  if (rows.length) {
    out.push('| Topic | unverified | unresolvable | allowlisted |', '|---|---|---|---|');
    for (const [rel, r] of rows) out.push(`| [${rel}](${rel}) | ${r.unverified} | ${r.unresolvable} | ${r.allowlisted} |`);
    out.push('');
  } else {
    out.push('Every citation in the tree resolved against its declared revision.', '');
  }
  fs.writeFileSync(path.join(root, 'VERIFICATION.md'), out.join('\n'));
  console.log('wrote VERIFICATION.md');
}
function bumpTopic(t, k) {
  const r = (CITE_STATS.byTopic[t.rel] = CITE_STATS.byTopic[t.rel] || { checked: 0, unverified: 0, unresolvable: 0, allowlisted: 0 });
  r[k]++;
}
function citeCheck(topics) {
  const warnings = [];
  CITE_STATS = newCiteStats();
  CITE_NOTES = [];
  let linesOf = (p) => revisionLines(CURRENT_TOPIC, p); // bound per topic below
  // Bare `:NNN` (no path) is a continuation of the last path cited earlier on
  // the SAME line/label (`proxy.mjs:100<br/>verify :340`); with no path on the
  // line it is unresolvable (or a port) and is reported as such. Previously
  // bare cites were never resolved and never checked (23 % of all citations).
  const BARE_RE = /(^|[^A-Za-z0-9_./:-]):(\d+)(?:\s*-\s*(\d+))?(?![\d.])/g;
  for (const t of topics) {
    CURRENT_TOPIC = t;
    for (const d of t.diagrams) {
      const check = (cited, a, b) => {
        if (/^\d+(\.\d+)+$/.test(cited)) return; // host:port such as 127.0.0.1:8080, not a citation
        // An allowlisted citation is waived, counted apart and printed as such.
        // Returns true when the caller should stop: the diagnostic it was about
        // to raise is the one the allowlist exists to waive.
        const waive = (src) => {
          const e = allowMatch(t, d.id, cited, src, a);
          if (!e) return false;
          CITE_STATS.allowlisted++;
          bumpTopic(t, 'allowlisted');
          CITE_NOTES.push(`${t.rel}:${d.id} — ${src || cited}:${a} allowlisted (unverifiable): ${e.reason}`);
          return true;
        };
        // Exact match wins over suffix match: a repo-root file (README.md) is a
        // suffix of every deeper twin, so `README.md:12` must resolve to it alone.
        const srcPaths = (t.fm.sources || []).map((s) => s.split(':')[0]);
        const exact = srcPaths.filter((sp) => sp === cited || sp === './' + cited);
        const hits = exact.length ? exact : srcPaths.filter((sp) => sp.endsWith('/' + cited));
        if (hits.length === 0) { if (waive(null)) return; CITE_STATS.unresolvable++; bumpTopic(t, 'unresolvable'); warnings.push(`${t.rel}:${d.id} — ${cited}:${a} cites a file that is not in this topic's sources: (unresolvable, never checked)`); return; }
        if (hits.length > 1) { if (waive(null)) return; CITE_STATS.unresolvable++; bumpTopic(t, 'unresolvable'); warnings.push(`${t.rel}:${d.id} — ${cited}:${a} is ambiguous: matches ${hits.length} sources: entries`); return; }
        const src = hits[0];
        const { lines, resolution } = linesOf(src);
        if (!lines) { if (waive(src)) return; CITE_STATS.unresolvable++; bumpTopic(t, 'unresolvable'); warnings.push(`${t.rel}:${d.id} — ${src} could not be read`); return; }
        // Judged against working-tree bytes rather than the declared revision:
        // the line checks below still run, but this citation is NOT verified.
        // An allowlisted citation is waived HERE, not earlier: the line checks
        // below still run against the working-tree bytes, because the waiver is
        // of the revision requirement alone, never of whether the line is right.
        if (resolution !== 'revision' && resolution !== 'worktree-flag') {
          if (!waive(src)) {
            CITE_STATS.unverified++;
            CITE_STATS.byReason[resolution] = (CITE_STATS.byReason[resolution] || 0) + 1;
            bumpTopic(t, 'unverified');
            warnings.push(`${t.rel}:${d.id} — ${src}:${a} was read from the working tree, not the declared revision: ${UNVERIFIED_REASON[resolution] || resolution} (unverified)`);
          }
        } else if (resolution === 'revision') { CITE_STATS.checked++; bumpTopic(t, 'checked'); }
        // Both endpoints must exist; only the anchor line's CONTENT is judged —
        // a range legitimately ends on a closing brace.
        for (const n of [a, b].filter(Boolean).map(Number)) {
          if (n > lines.length) warnings.push(`${t.rel}:${d.id} — ${src}:${n} past EOF (file has ${lines.length} lines)`);
        }
        const n = Number(a);
        if (n <= lines.length) {
          const txt = (lines[n - 1] || '').trim();
          if (!txt) warnings.push(`${t.rel}:${d.id} — ${src}:${n} is blank`);
          else if (/^[)\]}>;,]+$/.test(txt)) warnings.push(`${t.rel}:${d.id} — ${src}:${n} is punctuation only ('${txt}')`);
        }
      };
      // A literal \n inside a classDiagram `note for` runs straight into the path
      // (`…\npath.rs:NN`) and CITE_RE would swallow the `n`; split on it first.
      const text = d.src.replace(/\\n/g, '\n');
      // sequenceDiagram convention: `participant X as Label<br/>path:NN` binds a
      // file to X; a message `X->>Y: … (:NNN)` or `Note over X,Y: … :NNN` with a
      // bare line then means that line in the SENDER's (first named) file.
      const partFile = new Map();   // participant id → bound file, or null when declared without a path
      for (const pm of text.matchAll(/^[ \t]*(?:participant|actor)\s+(\w+)(?:\s+as\s+(.+))?$/gm)) {
        const c = new RegExp(CITE_RE.source).exec(pm[2] || '');
        partFile.set(pm[1], c ? c[1] : null);
      }
      // Returns a file, or the string 'UNBOUND' when the line's participant is
      // declared WITHOUT a path — a bare ref there must not silently inherit the
      // last path in the diagram (it lands on a real line of the wrong file).
      const lineContext = (line) => {
        const msg = /^[ \t]*(\w+)\s*(?:-->>|->>|-->|->|--x|-x|--\)|-\))\s*[+-]?\s*(\w+)\s*:/.exec(line);
        const note = /^[ \t]*Note\s+(?:over|left of|right of)\s+(\w+)(?:\s*,\s*(\w+))?\s*:/.exec(line);
        const ids = msg ? [msg[1], msg[2]] : note ? [note[1], note[2]].filter(Boolean) : [];
        if (!ids.length) return null;
        for (const id of ids) if (partFile.get(id)) return partFile.get(id);
        return ids.some((id) => partFile.has(id)) ? 'UNBOUND' : null;
      };
      // Fallback for flowchart nodes and alt/loop/else lines: a bare :NNN continues
      // the most recent path cited earlier in the diagram (document order).
      let lastPath = null;
      for (const rawLine of text.split('\n')) {
        // `:987,1053,1060` cites three lines; expand so every number is checked.
        const line = rawLine.replace(/:(\d+)((?:,\s*\d+)+)/g, (m0, a, rest) => ':' + a + rest.replace(/,\s*(\d+)/g, ' :$1'));
        const paths = [];
        const stripped = line.replace(CITE_RE, (m0, cited, a, b, off) => {
          paths.push({ cited, off });
          lastPath = cited;
          check(cited, a, b);
          return ' '.repeat(m0.length);
        });
        for (const bm of stripped.matchAll(BARE_RE)) {
          const off = bm.index + bm[1].length;
          const before = paths.filter((p) => p.off < off);
          const lc = before.length ? null : lineContext(line);
          if (lc === 'UNBOUND') { warnings.push(`${t.rel}:${d.id} — bare :${bm[2]} on a message whose participant is declared without a path (bind the participant to a file, or qualify the ref)`); continue; }
          const ctx = before.length ? before[before.length - 1].cited : (lc || lastPath);
          if (!ctx) { warnings.push(`${t.rel}:${d.id} — bare :${bm[2]} has no path anywhere before it in the diagram (qualify it, or reword if it is a port)`); continue; }
          check(ctx, bm[2], bm[3]);
        }
      }
    }
  }
  // An allowlist entry that matches nothing is stale: the citation it waived has
  // been fixed, renumbered or deleted. Left alone it silently broadens over time
  // into a blanket waiver, so it fails the run instead. (--only sees part of the
  // tree, so a miss there means nothing.)
  if (!flags.only) {
    for (const e of ALLOWLIST) {
      if (e.used) continue;
      const at = `${e.topic}${e.diagram ? ' ' + e.diagram : ''} ${e.path}${'line' in e ? ':' + e.line : ''}`;
      warnings.push(`citation-allowlist.json: allow[${e.index}] (${at}) matched no citation — it is stale; remove it`);
    }
  }
  return warnings;
}

// A relocation that preserves a line's text preserves whatever the citation
// meant — including a citation that was already pointing at the wrong line. The
// one cheap check that catches THAT: a participant labelled with a function name
// should cite a line inside (or immediately above) that function's definition.
const PART_RE = /^[ \t]*(?:participant|actor)\s+\w+\s+as\s+(.+)$/gm;
const FN_RE = /\b([a-z_][a-z0-9_]{3,})\b/g;
function symbolCheck(topics) {
  const warnings = [];
  let linesOf = (p) => revisionLines(CURRENT_TOPIC, p); // bound per topic below
  for (const t of topics) {
    CURRENT_TOPIC = t;
    for (const m of t.diagrams.map((d) => d.src).join('\n').matchAll(PART_RE)) {
      const label = m[1];
      const c = new RegExp(CITE_RE.source).exec(label);
      if (!c) continue;
      const [, cited, a, b] = c, ln = +a, end = b ? +b : ln;
      const hits = (t.fm.sources || []).filter((s) => {
        const sp = s.split(':')[0];
        return sp === cited || sp.endsWith('/' + cited);
      });
      if (hits.length !== 1) continue;
      const src = hits[0].split(':')[0], lines = linesOf(src).lines;
      if (!lines || ln > lines.length) continue;
      const near = lines.slice(Math.max(0, ln - 4), ln + 3).join('\n');
      const names = [...new Set([...label.slice(0, c.index).matchAll(FN_RE)].map((x) => x[1]))];
      // A participant deliberately covering several operations carries its own
      // comma-list of citations; only the first is parsed here, so leave it be.
      // ...as does one written `path:a,b` with a line per operation.
      if (names.length > 1 || /^\s*,\s*\d+/.test(label.slice(c.index + c[0].length))) continue;
      for (const name of names) {
        if (near.includes(name)) continue;
        const def = new RegExp(`^\\s*(?:pub\\s+)?(?:async\\s+)?fn\\s+${name}\\b|^\\s*(?:export\\s+)?(?:async\\s+)?function\\s+${name}\\b`);
        const at = lines.reduce((acc, txt, i) => (def.test(txt) ? acc.concat(i + 1) : acc), []);
        if (at.length !== 1) continue;
        // Citing a STEP INSIDE the named function is correct and common, so the
        // test is body containment, not proximity: walk braces from the
        // definition to its close and accept anything from its doc comment to
        // its last line.
        let depth = 0, endOfBody = at[0], seen = false;
        for (let i = at[0] - 1; i < lines.length; i++) {
          for (const ch of lines[i]) { if (ch === '{') { depth++; seen = true; } else if (ch === '}') depth--; }
          if (seen && depth <= 0) { endOfBody = i + 1; break; }
        }
        if ((ln < at[0] - 3 && end < at[0]) || ln > endOfBody)
          warnings.push(`${t.rel} — ${src}:${ln} is labelled '${name}' but that function spans :${at[0]}-${endOfBody}`);
      }
    }
  }
  return warnings;
}

// ---------------------------------------------------------------- render
function renderOne(topic, d, outDir) {
  return new Promise((resolve) => {
    const mmd = path.join(outDir, `${d.id}.mmd`);
    const svg = path.join(outDir, `${d.id}.svg`);
    fs.writeFileSync(mmd, d.src + '\n');
    const child = spawn('mmdc', ['-i', mmd, '-o', svg, '-q'], { stdio: ['ignore', 'pipe', 'pipe'] });
    let err = '';
    child.stderr.on('data', (c) => { err += c.toString(); });
    child.stdout.on('data', (c) => { err += c.toString(); });
    child.on('close', (code) => {
      if (code === 0) {
        try {
          const svgText = fs.readFileSync(svg, 'utf8');
          const vb = svgText.match(/viewBox="[\d.\-]+ [\d.\-]+ ([\d.]+) ([\d.]+)"/);
          const w = vb ? Math.round(+vb[1]) : 0;
          if (w > MAX_WIDTH) return resolve(`${topic.rel}:${d.id} (md line ${d.line}) — rendered ${w}px wide (max ${MAX_WIDTH}); wrap long Notes with <br/>, cap them at ~90 chars, or split the diagram`);
        } catch (e) { /* ignore census failure */ }
        return resolve(null);
      }
      const m = err.match(/Parse error on line (\d+):[\s\S]*?\n([\s\S]*?)(?:\n\s+at |$)/);
      const detail = m ? `mermaid line ${m[1]}: ${m[2].split('\n').slice(0, 3).join(' | ')}` : err.split('\n').filter((l) => l.trim() && !/^\s+at /.test(l)).slice(0, 3).join(' | ');
      resolve(`${topic.rel}:${d.id} (md line ${d.line}) — ${detail}`);
    });
  });
}
async function renderAll(topics) {
  const jobs = [];
  for (const t of topics) {
    CURRENT_TOPIC = t;
    const outDir = path.join(root, 'rendered', t.rel.replace(/\.md$/, ''));
    fs.mkdirSync(outDir, { recursive: true });
    for (const d of t.diagrams) jobs.push(() => renderOne(t, d, outDir));
  }
  const errors = [];
  let next = 0;
  async function worker() {
    while (next < jobs.length) {
      const j = jobs[next++];
      const e = await j();
      if (e) errors.push(e);
    }
  }
  await Promise.all(Array.from({ length: Math.min(flags.jobs, jobs.length) }, worker));
  return { errors, count: jobs.length };
}

// ---------------------------------------------------------------- indexes
function mdLink(rel, anchor) {
  return anchor ? `${rel}#${anchor}` : rel;
}
function slug(id, title) {
  return `${id} ${title}`.toLowerCase().replace(/[^a-z0-9\s-]/g, '').trim().replace(/\s+/g, '-');
}
function writeIndexes(topics) {
  const byArea = Object.fromEntries([...AREAS].map((a) => [a, []]));
  for (const t of topics) byArea[t.fm.area].push(t);
  for (const k of Object.keys(byArea)) byArea[k].sort((a, b) => a.fm.id.localeCompare(b.fm.id, undefined, { numeric: true }));

  // README table (regenerated block between markers)
  const readme = path.join(root, 'README.md');
  let text = fs.existsSync(readme) ? fs.readFileSync(readme, 'utf8') : '';
  const START = '<!-- BEGIN GENERATED DIAGRAM INDEX -->', END = '<!-- END GENERATED DIAGRAM INDEX -->';
  const rows = [];
  for (const area of AREAS) {
    if (!byArea[area] || byArea[area].length === 0) continue;
    rows.push(`\n### ${area}\n`);
    rows.push('| ID | Topic | Diagrams | Kinds | Governing | ADRs |');
    rows.push('|----|-------|----------|-------|-----------|------|');
    for (const t of byArea[area]) {
      const kinds = [...new Set(t.diagrams.map((d) => d.kind))].join(', ');
      const gov = t.fm.governing.map((g) => `[${path.basename(g.split('#')[0])}](../../${g})`).join(', ');
      rows.push(`| ${t.fm.id} | [${t.fm.title}](${t.rel}) | ${t.diagrams.length} | ${kinds} | ${gov} | ${t.fm.adrs.join(', ')} |`);
    }
  }
  const total = topics.reduce((n, t) => n + t.diagrams.length, 0);
  const gen = `${START}\n_${topics.length} topic files, ${total} diagrams. Regenerate with_ \`node scripts/diagram-index-gen.cjs docs/diagrams\`.\n${rows.join('\n')}\n${END}`;
  if (text.includes(START) && text.includes(END)) {
    text = text.slice(0, text.indexOf(START)) + gen + text.slice(text.indexOf(END) + END.length);
  } else {
    text = text.trimEnd() + '\n\n## Diagram index\n\n' + gen + '\n';
  }
  fs.writeFileSync(readme, text);

  // COVERAGE.md — three inverted indexes
  const adrIdx = new Map(), govIdx = new Map(), srcIdx = new Map();
  for (const t of topics) {
    CURRENT_TOPIC = t;
    for (const a of t.fm.adrs) {
      // ADR numbers are repository-local. Estate topics must supply an owner
      // explicitly; do not infer adoption from a coincidentally matching number.
      const key = a.includes(':') ? a : `${t.fm.area === 'estate' ? 'estate-unresolved' : t.fm.area}:${a}`;
      if (!adrIdx.has(key)) adrIdx.set(key, []);
      adrIdx.get(key).push(t);
    }
    for (const g of t.fm.governing) { const k = g.split('#')[0]; if (!govIdx.has(k)) govIdx.set(k, []); govIdx.get(k).push(t); }
    for (const s of t.fm.sources) { const k = s.split(':')[0]; if (!srcIdx.has(k)) srcIdx.set(k, []); srcIdx.get(k).push(t); }
  }
  const sortKeys = (m) => [...m.keys()].sort((a, b) => a.localeCompare(b, undefined, { numeric: true }));
  const out = [];
  out.push('<!-- GENERATED BY scripts/diagram-index-gen.cjs — DO NOT EDIT BY HAND -->');
  out.push('# Diagram coverage index\n');
  // verified_commit is a sha (single-repo topic) or a {repo: sha} map (estate topic
  // whose sources span repos); render both forms as `repo@sha` / `sha`.
  out.push(`${topics.length} topic files · ${total} diagrams · declared source revisions: ${declaredRevisions(topics).join(', ')}\n`);
  out.push('Revision labels are author declarations. This index checks structure and references, not semantic accuracy, clean working trees, deployment or system acceptance. See the [dated estate audit](../estate-review/2026-09-07-estate-audit.md) for evidence and limits.\n');
  out.push('## Diagrams\n');
  out.push('| Diagram | Kind | Topic file |');
  out.push('|---------|------|------------|');
  for (const area of AREAS) for (const t of byArea[area] || []) for (const d of t.diagrams) out.push(`| [${d.id} ${d.title}](${t.rel}#${slug(d.id, d.title)}) | ${d.kind} | ${t.fm.id} |`);
  out.push('\n## By ADR\n');
  out.push('Keys are repository-qualified. `estate-unresolved` preserves an unqualified cross-repository reference pending owner resolution; it must not be read as one shared decision.\n');
  out.push('| ADR | Topic files |');
  out.push('|-----|-------------|');
  for (const k of sortKeys(adrIdx)) out.push(`| ${k} | ${adrIdx.get(k).map((t) => `[${t.fm.id}](${t.rel})`).join(', ')} |`);
  out.push('\n## By governing document\n');
  out.push('| Governing doc | Topic files |');
  out.push('|---------------|-------------|');
  for (const k of sortKeys(govIdx)) out.push(`| [${k}](../../${k}) | ${govIdx.get(k).map((t) => `[${t.fm.id}](${t.rel})`).join(', ')} |`);
  out.push('\n## By source path\n');
  out.push('| Source | Topic files |');
  out.push('|--------|-------------|');
  for (const k of sortKeys(srcIdx)) out.push(`| \`${k}\` | ${srcIdx.get(k).map((t) => `[${t.fm.id}](${t.rel})`).join(', ')} |`);
  fs.writeFileSync(path.join(root, 'COVERAGE.md'), out.join('\n') + '\n');
}

// ---------------------------------------------------------------- main
(async () => {
  const errors = [];
  ALLOWLIST = loadAllowlist(errors);
  let files = walk(root, []);
  if (flags.only) files = files.filter((f) => path.relative(root, f).includes(flags.only));
  const topics = files.map((f) => parseTopic(f, errors)).filter(Boolean);
  const seenTopic = new Map(), seenDiag = new Map();
  for (const t of topics) {
    CURRENT_TOPIC = t;
    if (seenTopic.has(t.fm.id)) errors.push(`${t.rel}: duplicate topic id ${t.fm.id} (also ${seenTopic.get(t.fm.id)})`);
    seenTopic.set(t.fm.id, t.rel);
    for (const d of t.diagrams) {
      if (seenDiag.has(d.id)) errors.push(`${t.rel}: duplicate diagram id ${d.id} (also ${seenDiag.get(d.id)})`);
      seenDiag.set(d.id, t.rel);
    }
  }
  const total = topics.reduce((n, t) => n + t.diagrams.length, 0);
  console.log(`parsed ${topics.length} topic files, ${total} mermaid diagrams`);
  for (const x of SOURCE_AT_REVISION) console.log(`  ~ ${x}`);
  // An allowlist entry for a topic that no longer exists is as stale as one for a
  // citation that no longer exists, and is caught here because it never reaches
  // the citation loop at all.
  if (!flags.only) {
    for (const e of ALLOWLIST) {
      if (!seenTopic.has(e.topic)) errors.push(`citation-allowlist.json: allow[${e.index}] names topic '${e.topic}', which is not in this tree`);
    }
  }
  if (flags.cite) {
    const w = citeCheck(topics).concat(symbolCheck(topics));
    const st = CITE_STATS;
    console.log(`cite-check: ${w.length} warning(s)`);
    // A green run must state what it checked, not merely that nothing objected.
    console.log(`citations: ${st.checked} verified at the declared revision, ${st.unverified} unverified (working-tree fallback), ${st.unresolvable} unresolvable, ${st.allowlisted} allowlisted (unverifiable)`);
    for (const [reason, n] of Object.entries(st.byReason).sort((a, b) => b[1] - a[1])) {
      console.log(`  unverified: ${n} — ${UNVERIFIED_REASON[reason] || reason}`);
    }
    // Written only by a run that actually resolved citations, and deliberately
    // NOT part of COVERAGE.md: the hosted runner has no sibling checkouts, so
    // its counts would differ from a local run and the in-sync diff would fail
    // on a true statement. The gate stays honest by not diffing this file.
    writeVerificationReport(st, topics);
    for (const x of CITE_NOTES) console.log(`  ~ ${x}`);
    for (const x of w) console.warn(`  ! ${x}`);
    if (flags.strictCitations) errors.push(...w.map(x => `citation: ${x}`));
  }
  if (flags.checkVerification) {
    // The runner cannot resolve cross-repository revisions, so it cannot repeat
    // the citation check. What it CAN do is refuse a VERIFICATION.md written
    // against revisions the tree no longer declares — which is exactly the case
    // where a topic was re-stamped without its citations being re-resolved.
    const vfile = path.join(root, 'VERIFICATION.md');
    const text = fs.existsSync(vfile) ? fs.readFileSync(vfile, 'utf8') : null;
    const m = text && text.match(/<!-- declared-revisions:([^>]*?)-->/);
    if (!text) errors.push('VERIFICATION.md is absent — run: node scripts/diagram-index-gen.cjs docs/diagrams --check --strict-citations');
    else if (!m) errors.push('VERIFICATION.md carries no declared-revisions anchor — regenerate it: node scripts/diagram-index-gen.cjs docs/diagrams --check --strict-citations');
    else {
      const recorded = m[1].trim().split(/\s+/).filter(Boolean);
      const current = declaredRevisions(topics);
      const added = current.filter((x) => !recorded.includes(x));
      const gone = recorded.filter((x) => !current.includes(x));
      if (added.length || gone.length) {
        errors.push(`VERIFICATION.md is stale: declared revisions have moved since it was written${added.length ? `; now declared but not recorded: ${added.join(', ')}` : ''}${gone.length ? `; recorded but no longer declared: ${gone.join(', ')}` : ''}. Re-resolve the citations locally and commit the result: node scripts/diagram-index-gen.cjs docs/diagrams --check --strict-citations`);
      } else {
        console.log(`verification: fresh — VERIFICATION.md records all ${current.length} declared revision(s)`);
      }
    }
    console.log(`allowlist: ${ALLOWLIST.length} well-formed entr${ALLOWLIST.length === 1 ? 'y' : 'ies'}`);
  }
  if (flags.render) {
    const { errors: rerr, count } = await renderAll(topics);
    console.log(`rendered ${count - rerr.length}/${count} diagrams via mmdc`);
    errors.push(...rerr);
  }
  if (errors.length) {
    console.error(`\n${errors.length} error(s):`);
    for (const e of errors) console.error(`  - ${e}`);
    process.exit(1);
  }
  if (!flags.check && !flags.only) {
    writeIndexes(topics);
    console.log('wrote README.md index block + COVERAGE.md');
  }
})().catch((e) => { console.error(e); process.exit(2); });
