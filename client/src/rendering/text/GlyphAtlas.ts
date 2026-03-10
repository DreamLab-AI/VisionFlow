import * as THREE from 'three';

export interface GlyphMetrics {
  u: number;  // UV origin x (0..1)
  v: number;  // UV origin y (0..1)
  w: number;  // UV width (0..1)
  h: number;  // UV height (0..1)
  advance: number;  // horizontal advance in normalized units
  xOffset: number;  // bearing offset
  yOffset: number;  // baseline offset
}

export interface GlyphAtlasResult {
  texture: THREE.CanvasTexture;
  metrics: Map<string, GlyphMetrics>;
  lineHeight: number;  // normalized line height
}

// Characters to include: printable ASCII + common Unicode symbols used in labels
const ATLAS_CHARS =
  ' !"#$%&\'()*+,-./0123456789:;<=>?@' +
  'ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`' +
  'abcdefghijklmnopqrstuvwxyz{|}~' +
  '\u25CF\u25C7\u25CB\u25C9\u27E8\u27E9\u21B3\u26A1\u2665\u2713\u26A0\u2605\u2606\u00B7\u00BD';

const ATLAS_SIZE = 1024;
const GLYPH_FONT_SIZE = 48;
const GLYPH_PADDING = 2;

let _cached: GlyphAtlasResult | null = null;

export function createGlyphAtlas(fontSize: number = GLYPH_FONT_SIZE): GlyphAtlasResult {
  if (_cached) return _cached;

  const canvas = document.createElement('canvas');
  canvas.width = ATLAS_SIZE;
  canvas.height = ATLAS_SIZE;
  const ctx = canvas.getContext('2d')!;

  ctx.clearRect(0, 0, ATLAS_SIZE, ATLAS_SIZE);

  const font = `${fontSize}px system-ui, sans-serif`;
  ctx.font = font;
  ctx.textBaseline = 'top';

  const metrics = new Map<string, GlyphMetrics>();

  let cursorX = GLYPH_PADDING;
  let cursorY = GLYPH_PADDING;
  let rowHeight = 0;

  for (const char of ATLAS_CHARS) {
    const measured = ctx.measureText(char);
    const charWidth = Math.ceil(measured.width) + GLYPH_PADDING * 2;
    const charHeight = fontSize + GLYPH_PADDING * 2;

    // Wrap to next row if needed
    if (cursorX + charWidth > ATLAS_SIZE) {
      cursorX = GLYPH_PADDING;
      cursorY += rowHeight + GLYPH_PADDING;
      rowHeight = 0;
    }

    // Out of atlas space
    if (cursorY + charHeight > ATLAS_SIZE) break;

    // Draw black outline for readability
    ctx.strokeStyle = '#000000';
    ctx.lineWidth = 3;
    ctx.strokeText(char, cursorX + GLYPH_PADDING, cursorY + GLYPH_PADDING);

    // Draw white fill (allows per-instance color tinting via vColor * texSample.r)
    ctx.fillStyle = '#ffffff';
    ctx.fillText(char, cursorX + GLYPH_PADDING, cursorY + GLYPH_PADDING);

    metrics.set(char, {
      u: cursorX / ATLAS_SIZE,
      v: cursorY / ATLAS_SIZE,
      w: charWidth / ATLAS_SIZE,
      h: charHeight / ATLAS_SIZE,
      advance: measured.width / fontSize,
      xOffset: 0,
      yOffset: 0,
    });

    cursorX += charWidth + GLYPH_PADDING;
    rowHeight = Math.max(rowHeight, charHeight);
  }

  const texture = new THREE.CanvasTexture(canvas);
  texture.flipY = false;  // UVs are computed in canvas-space (y-down); don't invert
  texture.minFilter = THREE.LinearFilter;
  texture.magFilter = THREE.LinearFilter;
  texture.generateMipmaps = false;
  texture.needsUpdate = true;

  const lineHeight = (fontSize + GLYPH_PADDING * 2) / fontSize;

  _cached = { texture, metrics, lineHeight };
  return _cached;
}
