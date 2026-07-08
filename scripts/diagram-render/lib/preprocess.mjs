// Shared source transform for the diagram-as-code render gate (RES-b).
//
// The ten .mmd sources under presentation/report/diagrams/ were authored for a
// DARK mermaid theme: each carries a leading `%%{init: {'theme':'dark', ...}}%%`
// directive and hardcoded `style`/`linkStyle`/`classDef` lines with dark node
// fills and near-white text (`color:#e0e0e0`). Exported onto the white page of a
// PDF, the theme's transparent background dropped away and the light text became
// invisible — the defect this gate exists to stop shipping.
//
// The gate renders every source with mermaid's LIGHT `default` theme so all text
// resolves to a dark, print-safe fill on a white background. To let the light
// theme win, we strip the dark theme directive and the dark palette overrides.
// Node labels are rendered as real <text> elements (htmlLabels:false, set by the
// renderer) so the text-fill assertion in check-diagram-text.js has something to
// measure. HTML emphasis tags (<b>/<i>) are dropped because the non-HTML label
// path would otherwise print them literally; <br/> is kept (mermaid splits it).

/** Strip the dark-theme directive and dark palette so the light theme renders. */
export function preprocessSource(raw) {
  return raw
    // Drop leading init directives — they force `theme: dark` and dark themeVariables.
    .replace(/^\s*%%\{[^\n]*\}%%\s*$/gm, '')
    // Drop hardcoded dark palette overrides (fills + near-white text colours).
    .replace(/^\s*(style|linkStyle|classDef)\s+.*$/gm, '')
    // Drop emphasis tags that the non-HTML label renderer would print verbatim.
    .replace(/<\/?[bi]>/gi, '')
    // Collapse the blank lines the strips leave behind.
    .replace(/\n{3,}/g, '\n\n')
    .trim();
}

/**
 * Distinctive label words expected to appear as rendered text in the SVG.
 * Pulled from quoted and bracketed label content in the ORIGINAL source, with
 * markup removed. Word-level (not phrase-level) so it survives mermaid wrapping
 * a multi-word label across several <tspan> lines.
 */
export function extractLabelWords(raw) {
  const fragments = [];
  for (const m of raw.matchAll(/"([^"]*)"/g)) fragments.push(m[1]);
  for (const m of raw.matchAll(/\[([^\]]*)\]/g)) fragments.push(m[1]);

  const stop = new Set([
    'with', 'that', 'this', 'from', 'into', 'over', 'each', 'they', 'them',
    'have', 'more', 'less', 'than', 'once', 'other', 'their', 'across', 'about',
    'which', 'while', 'when', 'what',
  ]);

  const words = new Set();
  for (const fragment of fragments) {
    const clean = fragment
      .replace(/<[^>]+>/g, ' ')       // strip any markup
      .replace(/&[a-z]+;/gi, ' ')     // strip entities
      .replace(/[^A-Za-z ]/g, ' ');   // keep letters only, word-level
    for (const word of clean.split(/\s+/)) {
      const lower = word.toLowerCase();
      if (word.length >= 4 && !stop.has(lower)) words.add(lower);
    }
  }
  return [...words];
}
