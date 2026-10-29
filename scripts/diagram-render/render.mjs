// Render presentation/report/diagrams/*.mmd to light-theme SVG (RES-b gate).
//
// Renders each Mermaid source with Pele (the sha256-pinned build in
// scripts/vendor/pele), in-process with no browser, and writes a static SVG:
// every Pele CSS variable is replaced by its concrete light default, a white
// background is baked in, and every text node carries an explicit fill, so PDF
// embedding never depends on a stylesheet or container CSS — the
// transparent-background trap that once let dark-theme text vanish on a page.
// Output lands in presentation/report/diagrams/rendered/.

import { readFileSync, writeFileSync, readdirSync, mkdirSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createRequire } from 'node:module';
import { preprocessSource } from './lib/preprocess.mjs';

const require = createRequire(import.meta.url);
const peleStatic = require('../lib/pele-static.cjs');

const HERE = dirname(fileURLToPath(import.meta.url));
const REPO = resolve(HERE, '..', '..');
const SRC_DIR = join(REPO, 'presentation', 'report', 'diagrams');
const OUT_DIR = join(SRC_DIR, 'rendered');
const TEXT_FILL = '#1f1f1f'; // Pele's own --pele-fg default: dark, print-safe

/** Give every <text> without a fill the print fill, so a checker needs no CSS. */
function bakeTextFill(svg) {
  return svg.replace(/<text\b(?![^>]*\sfill=)/g, `<text fill="${TEXT_FILL}"`);
}

async function main() {
  const sources = readdirSync(SRC_DIR).filter((f) => f.endsWith('.mmd')).sort();
  if (sources.length === 0) throw new Error(`No .mmd sources under ${SRC_DIR}`);
  mkdirSync(OUT_DIR, { recursive: true });
  const pele = await peleStatic.loadPele();

  let failures = 0;
  for (const file of sources) {
    const name = file.replace(/\.mmd$/, '');
    const definition = preprocessSource(readFileSync(join(SRC_DIR, file), 'utf8'));
    try {
      const { svg, width } = peleStatic.renderStatic(pele, definition, {
        background: '#ffffff',
        options: { idPrefix: `diagram-${name.replace(/[^a-zA-Z0-9]/g, '-')}` },
      });
      const out = bakeTextFill(svg);
      const total = (out.match(/<text\b/g) || []).length;
      writeFileSync(join(OUT_DIR, `${name}.svg`), `<?xml version="1.0" encoding="UTF-8"?>\n${out}\n`);
      console.log(`  rendered ${file} -> rendered/${name}.svg  (width ${width}px, text nodes: ${total})`);
    } catch (error) {
      failures++;
      console.error(`  FAILED  ${file}: ${peleStatic.describeError(error)}`);
    }
  }

  if (failures > 0) {
    console.error(`\n${failures} diagram(s) failed to render.`);
    process.exit(1);
  }
  console.log(`\nRendered ${sources.length} diagram(s) to ${OUT_DIR}`);
}

main().catch((error) => {
  console.error(`render-diagrams: ${error.message}`);
  process.exit(1);
});
