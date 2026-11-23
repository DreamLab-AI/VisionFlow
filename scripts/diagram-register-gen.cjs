#!/usr/bin/env node
'use strict';

/*
 * diagram-register-gen.cjs — collect the register markers (`**Tension:**`,
 * `**Debt:**`, `**Drift:**`, `**Open:**`, `**Invariant:**`, each optionally
 * qualified as `**Debt (parallel, …):**`) from the topic files of one
 * diagrams-as-code area and write docs/diagrams/REGISTER-<area>.md.
 *
 * Usage:
 *   node scripts/diagram-register-gen.cjs [--area visionclaw]            # write from the working tree
 *   node scripts/diagram-register-gen.cjs [--area visionclaw] --check    # exit 1 if the committed file is stale
 *   node scripts/diagram-register-gen.cjs [--area visionclaw] --stdout   # print, do not write
 *   node scripts/diagram-register-gen.cjs --rev <sha> --stdout           # read the area at a git revision
 *
 *   --root <dir>  repository root (default: the parent of this script's directory).
 *
 * Only the VisionClaw register uses this format. REGISTER-diagram-ir.md and
 * REGISTER-prose-sanitiser.md come from the diagrams-as-code skill's collector
 * and a different layout, so this script refuses any area not in TITLES rather
 * than overwrite them.
 *
 * Exit codes: 0 ok, 1 --check found the committed register stale, 2 usage error.
 */

const fs = require('node:fs');
const path = require('node:path');
const { execFileSync } = require('node:child_process');

const TITLES = { visionclaw: 'VisionClaw' };
const KINDS = ['Tension', 'Debt', 'Drift', 'Open', 'Invariant'];
const MARKER = /^\*\*(Tension|Debt|Drift|Open|Invariant)(?: \(([^)]*(?:\([^)]*\)[^)]*)*)\))?:\*\*\s*(.*)$/;
const TOPIC = /\/\d\d-[^/]*\.md$/;

function usage(msg) {
  process.stderr.write(`diagram-register-gen: ${msg}\n`);
  process.exit(2);
}

function parseArgs(argv) {
  const opts = { area: 'visionclaw', root: path.resolve(__dirname, '..'), rev: null, check: false, stdout: false };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    const value = () => (i + 1 < argv.length ? argv[++i] : usage(`${a} needs a value`));
    if (a === '--area') opts.area = value();
    else if (a === '--root') opts.root = path.resolve(value());
    else if (a === '--rev') opts.rev = value();
    else if (a === '--check') opts.check = true;
    else if (a === '--stdout') opts.stdout = true;
    else usage(`unknown argument ${a}`);
  }
  if (!Object.prototype.hasOwnProperty.call(TITLES, opts.area)) {
    usage(`area "${opts.area}" does not use this register format (supported: ${Object.keys(TITLES).join(', ')})`);
  }
  if (opts.rev && !opts.stdout && !opts.check) usage('--rev reads a historical tree; combine it with --stdout');
  if (opts.rev && opts.check) usage('--check compares the working tree; it does not take --rev');
  return opts;
}

function git(root, args) {
  return execFileSync('git', ['-C', root, ...args], { encoding: 'utf8', maxBuffer: 1 << 28 });
}

/** List the area's topic files (repo-relative paths) and a reader for each. */
function source(root, area, rev) {
  const dir = `docs/diagrams/${area}/`;
  if (rev) {
    const files = git(root, ['ls-tree', '--name-only', rev, dir]).split('\n').filter((f) => TOPIC.test(f)).sort();
    return { files, read: (f) => git(root, ['show', `${rev}:${f}`]) };
  }
  const files = fs.readdirSync(path.join(root, dir), { withFileTypes: true })
    .filter((d) => d.isFile())
    .map((d) => dir + d.name)
    .filter((f) => TOPIC.test(f))
    .sort();
  return { files, read: (f) => fs.readFileSync(path.join(root, f), 'utf8') };
}

/** Build the register text for one area. Output is byte-stable: the committed file is compared to it verbatim. */
function generate({ root, area, rev }) {
  const { files, read } = source(root, area, rev);
  const out = Object.fromEntries(KINDS.map((k) => [k, []]));
  let parallel = 0;
  for (const f of files) {
    const txt = read(f);
    const id = (txt.match(/^id:\s*(\S+)/m) || [])[1];
    txt.split('\n').forEach((line, i) => {
      const m = line.match(MARKER);
      if (!m) return;
      const [, kind, qualifier, rest] = m;
      if (qualifier && /\bparallel\b/.test(qualifier)) parallel++;
      out[kind].push(`- ${id}:${i + 1} ${qualifier ? `(${qualifier}) ` : ''}${rest}`);
    });
  }
  const totals = KINDS.map((k) => `${k} ${out[k].length}`).join(', ');
  let s = `# ${TITLES[area]} register\n\nGenerated from the register markers in docs/diagrams/${area}/*.md.\n\nTotals: ${totals}; ${parallel} tagged parallel.\n`;
  for (const k of KINDS) s += `\n## ${k}\n\n${out[k].join('\n')}\n`;
  return s;
}

function main() {
  const opts = parseArgs(process.argv.slice(2));
  const text = generate(opts);
  if (opts.stdout) {
    process.stdout.write(text);
    return;
  }
  const target = path.join(opts.root, 'docs/diagrams', `REGISTER-${opts.area}.md`);
  if (opts.check) {
    const committed = fs.existsSync(target) ? fs.readFileSync(target, 'utf8') : null;
    if (committed !== text) {
      process.stderr.write(`diagram-register-gen: ${path.relative(opts.root, target)} is stale — run: node scripts/diagram-register-gen.cjs --area ${opts.area}\n`);
      process.exit(1);
    }
    process.stdout.write(`diagram-register-gen: ${path.relative(opts.root, target)} is up to date\n`);
    return;
  }
  fs.writeFileSync(target, text);
  process.stdout.write(`diagram-register-gen: wrote ${path.relative(opts.root, target)}\n`);
}

if (require.main === module) main();

module.exports = { generate, TITLES };
