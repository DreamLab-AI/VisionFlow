#!/usr/bin/env node
// RES-b diagram render gate — text-visibility + label assertion.
//
// Guards against the shipped defect: mermaid diagrams authored for a dark theme
// exported onto the white page of a PDF with near-white text on a dropped-away
// transparent background, so every label was invisible. This checker fails
// non-zero unless every rendered SVG under presentation/report/diagrams/rendered
// (or a directory passed as the first argument) has:
//
//   1. visible <text> — every non-empty text/tspan node resolves to a fill that
//      is NOT none / transparent / white / near-white, and is not made invisible
//      by a near-zero fill-opacity; and each diagram has at least one such node;
//   2. its key label strings — every distinctive label word extracted from the
//      matching .mmd source appears in the diagram's VISIBLE text.
//
// Pure node built-ins; no browser, no npm dependency — runs anywhere, including
// against the committed baseline with zero rendering. A second mode,
//   node scripts/check-diagram-text.js --diff <baselineDir> <candidateDir>
// compares the two directories by each SVG's sorted set of visible words
// (font-metric independent) and fails on any drift — the CI drift guard.

import { readFileSync, readdirSync, existsSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { extractLabelWords } from './diagram-render/lib/preprocess.mjs';

const HERE = dirname(fileURLToPath(import.meta.url));
const REPO = resolve(HERE, '..');
const SRC_DIR = join(REPO, 'presentation', 'report', 'diagrams');

/** Parse `fill:#xxx` default declared in the SVG's <style> block, if any. */
function styleDefaultFill(svg) {
  const style = /<style[^>]*>([\s\S]*?)<\/style>/.exec(svg);
  if (!style) return null;
  const m = /fill:\s*(#[0-9a-fA-F]{3,6}|rgba?\([^)]*\)|[a-zA-Z]+)/.exec(style[1]);
  return m ? m[1] : null;
}

/** Attribute value from an open-tag string, or null. */
function attr(tag, name) {
  const m = new RegExp(`\\b${name}\\s*=\\s*"([^"]*)"`).exec(tag);
  return m ? m[1] : null;
}

/** Is this fill invisible against a white page? */
function isInvisibleFill(fill, fillOpacity) {
  if (fillOpacity != null && fillOpacity !== '' && Number(fillOpacity) < 0.1) return true;
  if (!fill) return false; // absent -> inherits a visible default; handled by caller
  const f = fill.trim().toLowerCase();
  if (f === 'none' || f === 'transparent') return true;
  if (f === 'white' || f === '#fff' || f === '#ffffff') return true;
  const rgb = /rgba?\(([^)]+)\)/.exec(f);
  if (rgb) {
    const parts = rgb[1].split(',').map((n) => parseFloat(n));
    const [r, g, b, a] = parts;
    if (a != null && a < 0.1) return true;
    if (Number.isNaN(r) || Number.isNaN(g) || Number.isNaN(b)) return false; // odd but not white
    return r >= 240 && g >= 240 && b >= 240;
  }
  const hex = /^#([0-9a-f]{3}|[0-9a-f]{6})$/.exec(f);
  if (hex) {
    let h = hex[1];
    if (h.length === 3) h = h.split('').map((c) => c + c).join('');
    const r = parseInt(h.slice(0, 2), 16);
    const g = parseInt(h.slice(2, 4), 16);
    const b = parseInt(h.slice(4, 6), 16);
    return r >= 240 && g >= 240 && b >= 240;
  }
  return false; // named colours other than white are treated as visible
}

/**
 * Walk every text-bearing leaf (a <text> with no tspans, or each <tspan>),
 * resolving fill inheritance (tspan -> parent text -> <style> default -> #000).
 * Returns { visible:[strings], invisible:[{text,fill}] }.
 */
function extractTextNodes(svg) {
  const defaultFill = styleDefaultFill(svg);
  const visible = [];
  const invisible = [];

  for (const block of svg.matchAll(/<text\b([^>]*)>([\s\S]*?)<\/text>/g)) {
    const textTag = block[1];
    const inner = block[2];
    const textFill = attr(`<x${textTag}>`, 'fill');
    const textOpacity = attr(`<x${textTag}>`, 'fill-opacity');

    const tspans = [...inner.matchAll(/<tspan\b([^>]*)>([\s\S]*?)<\/tspan>/g)];
    const leaves = tspans.length
      ? tspans.map((t) => ({
          content: t[2].replace(/<[^>]+>/g, ''),
          fill: attr(`<x${t[1]}>`, 'fill') ?? textFill,
          opacity: attr(`<x${t[1]}>`, 'fill-opacity') ?? textOpacity,
        }))
      : [{ content: inner.replace(/<[^>]+>/g, ''), fill: textFill, opacity: textOpacity }];

    for (const leaf of leaves) {
      const content = leaf.content.replace(/&[a-z]+;/gi, ' ').trim();
      if (!content) continue;
      const effectiveFill = leaf.fill ?? defaultFill ?? '#000000';
      if (isInvisibleFill(effectiveFill, leaf.opacity)) {
        invisible.push({ text: content, fill: effectiveFill });
      } else {
        visible.push(content);
      }
    }
  }
  return { visible, invisible };
}

function listSvgs(dir) {
  return readdirSync(dir).filter((f) => f.endsWith('.svg')).sort();
}

/**
 * --diff mode: compare two render directories by each SVG's set of visible
 * words. Word-level (not phrase-level, not byte-level) so it is invariant to the
 * sub-pixel font metrics that make one Chrome wrap a label across a different
 * number of <tspan> lines than another, and to SVG element ordering — while
 * still catching a label that was added, removed or changed because the
 * committed baseline went stale against an edited .mmd.
 */
function runDiff(baselineDir, candidateDir) {
  const words = (dir, f) => {
    const set = new Set();
    for (const s of extractTextNodes(readFileSync(join(dir, f), 'utf8')).visible) {
      for (const w of s.toLowerCase().split(/[^a-z0-9]+/)) if (w) set.add(w);
    }
    return set;
  };

  const a = new Set(listSvgs(baselineDir));
  const b = new Set(listSvgs(candidateDir));
  let failures = 0;

  for (const f of [...new Set([...a, ...b])].sort()) {
    if (!a.has(f)) { console.error(`  DRIFT  ${f}: present in candidate, absent from baseline`); failures++; continue; }
    if (!b.has(f)) { console.error(`  DRIFT  ${f}: present in baseline, absent from candidate`); failures++; continue; }
    const bs = words(baselineDir, f);
    const cs = words(candidateDir, f);
    const gone = [...bs].filter((x) => !cs.has(x));
    const added = [...cs].filter((x) => !bs.has(x));
    if (gone.length || added.length) {
      console.error(`  DRIFT  ${f}: visible words differ from committed baseline`);
      if (gone.length) console.error(`           removed: ${gone.slice(0, 8).map((x) => JSON.stringify(x)).join(', ')}`);
      if (added.length) console.error(`           added:   ${added.slice(0, 8).map((x) => JSON.stringify(x)).join(', ')}`);
      failures++;
    } else {
      console.log(`  ok     ${f}: visible words match baseline`);
    }
  }
  if (failures) {
    console.error(`\n${failures} diagram(s) drifted from the committed baseline.`);
    process.exit(1);
  }
  console.log(`\nAll rendered diagrams match the committed baseline's visible words.`);
}

/** Default mode: validate every SVG in a render directory. */
function runCheck(renderDir) {
  if (!existsSync(renderDir)) {
    console.error(`render directory not found: ${renderDir}`);
    process.exit(1);
  }
  const svgs = listSvgs(renderDir);
  if (svgs.length === 0) {
    console.error(`no rendered SVGs in ${renderDir}`);
    process.exit(1);
  }

  let failures = 0;
  for (const svgName of svgs) {
    const base = svgName.replace(/\.svg$/, '');
    const svg = readFileSync(join(renderDir, svgName), 'utf8');
    const { visible, invisible } = extractTextNodes(svg);
    const problems = [];

    if (visible.length === 0) problems.push('no visible text nodes');
    if (invisible.length > 0) {
      const sample = invisible.slice(0, 4).map((n) => `${JSON.stringify(n.text.slice(0, 24))}=${n.fill}`);
      problems.push(`${invisible.length} invisible text node(s): ${sample.join(', ')}`);
    }

    const mmdPath = join(SRC_DIR, `${base}.mmd`);
    let labelInfo = 'no .mmd source';
    if (existsSync(mmdPath)) {
      const raw = readFileSync(mmdPath, 'utf8');
      const wanted = extractLabelWords(raw);
      const haystack = visible.join(' ').toLowerCase();
      const missing = wanted.filter((w) => !haystack.includes(w));
      labelInfo = `${wanted.length - missing.length}/${wanted.length} label words`;
      if (missing.length > 0) {
        problems.push(`missing label words in visible text: ${missing.slice(0, 8).join(', ')}${missing.length > 8 ? ' …' : ''}`);
      }
    } else {
      problems.push('no matching .mmd source');
    }

    if (problems.length) {
      console.error(`  FAIL  ${svgName}: ${problems.join('; ')}`);
      failures++;
    } else {
      console.log(`  ok    ${svgName}: ${visible.length} visible text nodes, ${labelInfo}`);
    }
  }

  if (failures) {
    console.error(`\n${failures}/${svgs.length} diagram(s) failed the text-visibility gate.`);
    process.exit(1);
  }
  console.log(`\nAll ${svgs.length} diagram(s) passed: visible text + key labels present.`);
}

function main() {
  const argv = process.argv.slice(2);
  if (argv[0] === '--diff') {
    if (argv.length !== 3) {
      console.error('usage: check-diagram-text.js --diff <baselineDir> <candidateDir>');
      process.exit(2);
    }
    runDiff(resolve(argv[1]), resolve(argv[2]));
    return;
  }
  const renderDir = argv[0] ? resolve(argv[0]) : join(SRC_DIR, 'rendered');
  runCheck(renderDir);
}

main();
