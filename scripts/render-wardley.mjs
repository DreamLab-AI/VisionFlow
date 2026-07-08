#!/usr/bin/env node
// RES-e — clean, print-resolution re-export of the five Wardley maps.
//
// The maps ship as self-contained HTML (presentation/report/wardley/*.html)
// produced by the project's WardleyMapGenerator. Each HTML embeds a clean 1200x800
// <svg> map wrapped in the tool's UI chrome — a page card, a "Wardley Map" heading
// and Export SVG / Export PNG / Toggle Grid buttons. The book figures were baked as
// full-page screenshots that captured that chrome and sat below print resolution
// (wardley-01/04/05 at 1400x1094).
//
// This script re-exports WITHOUT chrome at print resolution, deterministically and
// with no network or sidecar dependency:
//   1. extract the embedded <svg> from each HTML (drops the card, heading, buttons);
//   2. add an explicit white background and an Arial-metric-compatible sans font, so
//      the raster matches the browser-rendered figure on a white page;
//   3. rasterise the clean SVG to a >=2800px-wide PNG via ImageMagick's librsvg
//      delegate (RSVG 2.62.1) at a density that yields 3000x2000 (>=300 DPI for the
//      book's ~9.6in figure width);
//   4. write the clean SVG + PNG to wardley/rendered/ and replace the chrome-baked
//      images/wardley-0N-*.png the .tex includes, keeping the filenames.
//
// Pure node built-ins + the `magick` CLI already present in the toolchain.

import { readFileSync, writeFileSync, readdirSync, mkdirSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { dirname, join, resolve, basename } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const REPO = resolve(HERE, '..');
const WARDLEY = join(REPO, 'presentation', 'report', 'wardley');
const RENDERED = join(WARDLEY, 'rendered');
const IMAGES = join(REPO, 'presentation', 'report', 'images');

// Target raster: 1200px SVG at density 240 => 3000px wide (>=2800 required).
const DENSITY = 240;
const FONT_FAMILY = "'Liberation Sans', 'DejaVu Sans', Arial, Helvetica, sans-serif";

mkdirSync(RENDERED, { recursive: true });

/** Extract the embedded map <svg>…</svg>, strip chrome, add white bg + sans font. */
function cleanSvg(html) {
  const m = /<svg\b[\s\S]*?<\/svg>/.exec(html);
  if (!m) throw new Error('no <svg> found');
  let svg = m[0];
  // Component names carry literal newlines (e.g. "Jagged Frontier\nNavigation").
  // A browser (white-space:normal) collapses each to a single space; librsvg drops
  // it, joining the words. Collapse whitespace inside every <text> node to match the
  // browser figure the book already shipped.
  svg = svg.replace(/(<text\b[^>]*>)([\s\S]*?)(<\/text>)/g,
    (_all, open, body, close) => open + body.replace(/\s+/g, ' ').trim() + close);
  // Rewrite the opening <svg …> tag: add a font-family so librsvg renders the
  // font-less <text> nodes in a sans-serif matching the browser figure.
  svg = svg.replace(/<svg\b([^>]*)>/, (_all, attrs) => {
    const cleaned = attrs.replace(/\s+$/, '');
    return `<svg${cleaned} font-family="${FONT_FAMILY}">` +
      '\n<rect x="0" y="0" width="100%" height="100%" fill="#ffffff"/>';
  });
  return svg;
}

const htmlFiles = readdirSync(WARDLEY)
  .filter((f) => /^0\d-.*\.html$/.test(f))
  .sort();

if (htmlFiles.length !== 5) {
  console.error(`expected 5 numbered map HTML files, found ${htmlFiles.length}: ${htmlFiles.join(', ')}`);
  process.exit(1);
}

for (const file of htmlFiles) {
  const stem = basename(file, '.html');            // e.g. 01-coordination-value-chain
  const html = readFileSync(join(WARDLEY, file), 'utf8');
  const svg = cleanSvg(html);

  const svgOut = join(RENDERED, `${stem}.svg`);
  const pngRendered = join(RENDERED, `${stem}.png`);
  const pngBook = join(IMAGES, `wardley-${stem}.png`);   // filename the .tex expects

  writeFileSync(svgOut, svg, 'utf8');
  execFileSync('magick', ['-density', String(DENSITY), '-background', 'white', svgOut, pngRendered]);
  execFileSync('magick', ['-density', String(DENSITY), '-background', 'white', svgOut, pngBook]);

  const dims = execFileSync('identify', ['-format', '%wx%h', pngBook]).toString();
  console.log(`  ${file} -> ${svgOut}, ${pngBook} (${dims})`);
}

console.log('done: 5 maps re-exported clean at print resolution.');
