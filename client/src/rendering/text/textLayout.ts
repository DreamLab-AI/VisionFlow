import type { GlyphAtlasResult } from './GlyphAtlas';

export interface GlyphInstance {
  localX: number;   // offset from label center
  localY: number;   // offset from label center (line stacking)
  scaleX: number;   // glyph quad width
  scaleY: number;   // glyph quad height
  u: number;        // atlas UV rect origin x
  v: number;        // atlas UV rect origin y
  uw: number;       // atlas UV rect width
  vh: number;       // atlas UV rect height
}

// Fallback metrics for characters not in the atlas
const FALLBACK_ADVANCE = 0.5;

/**
 * Convert multi-line label text into positioned glyph instances.
 * Each glyph gets a local offset from the label's anchor point (center-top).
 * The caller combines these offsets with the node's world position.
 */
export function layoutText(
  lines: ReadonlyArray<{ text: string; fontSize: number }>,
  atlas: GlyphAtlasResult,
  maxWidth: number,
): GlyphInstance[] {
  const result: GlyphInstance[] = [];
  let lineOffsetY = 0;

  for (const line of lines) {
    const scale = line.fontSize;
    const glyphHeight = atlas.lineHeight * scale;

    // First pass: compute total line width for centering
    let lineWidth = 0;
    for (const char of line.text) {
      const m = atlas.metrics.get(char);
      lineWidth += m ? m.advance * scale : FALLBACK_ADVANCE * scale;
    }

    // Truncate at maxWidth
    const effectiveWidth = Math.min(lineWidth, maxWidth);
    let cursorX = -effectiveWidth * 0.5;
    let accumulated = 0;

    for (const char of line.text) {
      const m = atlas.metrics.get(char);
      if (!m) {
        // Skip characters not in the atlas
        accumulated += FALLBACK_ADVANCE * scale;
        cursorX += FALLBACK_ADVANCE * scale;
        if (accumulated > effectiveWidth) break;
        continue;
      }

      const glyphW = m.w * 1024 / 48 * scale;  // atlas pixel width → world scale
      const glyphH = m.h * 1024 / 48 * scale;

      if (accumulated + m.advance * scale > effectiveWidth + 0.01) break;

      result.push({
        localX: cursorX + m.xOffset * scale,
        localY: -lineOffsetY - m.yOffset * scale,
        scaleX: glyphW,
        scaleY: glyphH,
        u: m.u,
        v: m.v,
        uw: m.w,
        vh: m.h,
      });

      cursorX += m.advance * scale;
      accumulated += m.advance * scale;
    }

    lineOffsetY += glyphHeight * 1.3;
  }

  return result;
}

/**
 * Zero-allocation version: writes glyph data directly into pre-allocated
 * InstancedBufferAttribute arrays. Returns the number of glyphs written.
 * Used by InstancedLabels useFrame to avoid GC pressure from per-node allocations.
 */
export function layoutTextInline(
  lines: ReadonlyArray<{ text: string; fontSize: number; color: string }>,
  atlas: GlyphAtlasResult,
  maxWidth: number,
  // Output arrays (pre-allocated, written starting at `offset`)
  localOffArr: Float32Array,
  scaleArr: Float32Array,
  uvRectArr: Float32Array,
  colorArr: Float32Array,
  opacityArr: Float32Array,
  labelPosArr: Float32Array,
  // Per-label values
  worldX: number,
  worldY: number,
  worldZ: number,
  opacity: number,
  offset: number,
  maxGlyphs: number,
  // Scratch color object (reused by caller to avoid allocation)
  tempColor: { r: number; g: number; b: number; set(c: string): void },
): number {
  let glyphIdx = offset;
  let lineOffsetY = 0;

  for (const line of lines) {
    if (glyphIdx >= maxGlyphs) break;

    const scale = line.fontSize;
    const glyphHeight = atlas.lineHeight * scale;
    tempColor.set(line.color);
    const cr = tempColor.r, cg = tempColor.g, cb = tempColor.b;

    // First pass: compute total line width for centering
    let lineWidth = 0;
    for (const char of line.text) {
      const m = atlas.metrics.get(char);
      lineWidth += m ? m.advance * scale : FALLBACK_ADVANCE * scale;
    }

    const effectiveWidth = Math.min(lineWidth, maxWidth);
    let cursorX = -effectiveWidth * 0.5;
    let accumulated = 0;

    for (const char of line.text) {
      if (glyphIdx >= maxGlyphs) break;

      const m = atlas.metrics.get(char);
      if (!m) {
        accumulated += FALLBACK_ADVANCE * scale;
        cursorX += FALLBACK_ADVANCE * scale;
        if (accumulated > effectiveWidth) break;
        continue;
      }

      const glyphW = m.w * 1024 / 48 * scale;
      const glyphH = m.h * 1024 / 48 * scale;

      if (accumulated + m.advance * scale > effectiveWidth + 0.01) break;

      const i3 = glyphIdx * 3;
      const i2 = glyphIdx * 2;
      const i4 = glyphIdx * 4;

      labelPosArr[i3] = worldX;
      labelPosArr[i3 + 1] = worldY;
      labelPosArr[i3 + 2] = worldZ;

      localOffArr[i2] = cursorX + m.xOffset * scale;
      localOffArr[i2 + 1] = -lineOffsetY - m.yOffset * scale;

      scaleArr[i2] = glyphW;
      scaleArr[i2 + 1] = glyphH;

      uvRectArr[i4] = m.u;
      uvRectArr[i4 + 1] = m.v;
      uvRectArr[i4 + 2] = m.w;
      uvRectArr[i4 + 3] = m.h;

      colorArr[i3] = cr;
      colorArr[i3 + 1] = cg;
      colorArr[i3 + 2] = cb;

      opacityArr[glyphIdx] = opacity;

      cursorX += m.advance * scale;
      accumulated += m.advance * scale;
      glyphIdx++;
    }

    lineOffsetY += glyphHeight * 1.3;
  }

  return glyphIdx - offset;
}
