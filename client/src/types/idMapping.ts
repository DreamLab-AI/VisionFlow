/**
 * Deterministic FNV-1a hash: string -> u32.
 * Stable across reloads and re-indexes, unlike index-based mapping.
 * Used by both graph.worker.ts and graphDataManager.ts to map
 * non-numeric string node IDs to u32 IDs for the binary WebSocket protocol.
 */
export function stringToU32(str: string): number {
  let hash = 0x811c9dc5; // FNV offset basis
  for (let i = 0; i < str.length; i++) {
    hash ^= str.charCodeAt(i);
    hash = Math.imul(hash, 0x01000193); // FNV prime
  }
  return hash >>> 0; // Ensure unsigned 32-bit
}
