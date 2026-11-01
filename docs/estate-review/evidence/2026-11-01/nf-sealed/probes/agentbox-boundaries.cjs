'use strict';
// Audit-only synthetic fixtures; no accounts, storage, or relay connection.
const assert = require('node:assert/strict');
const root = process.argv[2];
const { readOutcome } = require(root + '/management-api/lib/zone-keys.js');
const { finalizeEvent, getPublicKey, nip44 } = require(root + '/management-api/node_modules/nostr-tools');
const key = n => Uint8Array.from({ length: 32 }, () => n);
const author = key(0x11), migrator = key(0x22), zone = key(0x33);
const channel = 'c'.repeat(64);
const inner = finalizeEvent({ kind: 42, created_at: 1600000000, tags: [['e', channel, '', 'root']], content: 'synthetic history' }, author);
const lookup = () => ({ zone: 'zone3', epoch: 1, secret: Buffer.from(zone).toString('hex'), pubkey: getPublicKey(zone) });
function envelope(text) {
  return finalizeEvent({ kind: 42, created_at: inner.created_at,
    tags: [['e', channel, '', 'root'], ['zk', 'zone3', '1', getPublicKey(zone)], ['sealed', inner.id, '1']],
    content: nip44.encrypt(text, nip44.getConversationKey(migrator, getPublicKey(zone))) }, migrator);
}
const good = readOutcome(envelope(JSON.stringify(inner)), lookup);
assert.equal(good.sealed.id, inner.id);
console.log('PASS: valid sealed original restores original identity');
const duplicated = '{"content":"discarded duplicate",' + JSON.stringify(inner).slice(1);
const duplicateOutcome = readOutcome(envelope(duplicated), lookup);
assert.equal(duplicateOutcome.sealed.id, inner.id);
console.log('REPRODUCED: duplicate content property accepted after JSON.parse');
const malformed = finalizeEvent({ kind: 42, created_at: inner.created_at, tags: [['e', channel, '', 'root'], ['sealed', inner.id, '99']], content: 'untrusted envelope content' }, migrator);
assert.equal(readOutcome(malformed, lookup).type, 'plain');
console.log('REPRODUCED: sealed marker without zk falls through to plain');
const badVersion = envelope(JSON.stringify(inner));
badVersion.tags[2][2] = '99';
assert.equal(readOutcome(badVersion, lookup).type, 'failed');
console.log('PASS: unsupported sealed version with zk fails');
