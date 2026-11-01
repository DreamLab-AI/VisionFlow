// Provision the poker table's house seat (forum ADR-2020, website ADR-2009)
// on the live relay: publish its kind-0 profile so the forum shows a name
// instead of hex. The relay row (whitelist, cohorts dreamlab+agent) is the
// generic step:
//
//   node scripts/seed/whitelist-admin-recipient.mjs --pubkey=<citizen hex> --cohorts=dreamlab,agent
//
// then this script:
//
//   POKER_CITIZEN_KEY_FILE=~/workspace/sidestr/agents/poker-citizen.key \
//     node scripts/seed/provision-poker-citizen.mjs
//
// The key file holds 64 hex characters or an nsec1…; it is read, never
// printed. Idempotent: kind-0 is replaceable.
//
// RE-RUN HAZARD: POST /api/admin/reset-db drops the whitelist row (not this
// profile, which lives in the events table); re-run the whitelist step after
// any relay reseed, or every member's wrap to the house is refused
// "gift-wrap recipient not whitelisted".
import { readFileSync } from 'node:fs';
import { finalizeEvent, getPublicKey } from 'nostr-tools/pure';
import { decode } from 'nostr-tools/nip19';

const RELAY_WS = 'wss://dreamlab-nostr-relay.solitary-paper-764d.workers.dev';
const keyFile = process.env.POKER_CITIZEN_KEY_FILE
  || `${process.env.HOME}/workspace/sidestr/agents/poker-citizen.key`;

function readKey(path) {
  const text = readFileSync(path, 'utf8').trim();
  if (/^nsec1/i.test(text)) {
    const d = decode(text);
    if (d.type !== 'nsec') throw new Error('not an nsec');
    return d.data;
  }
  if (!/^[0-9a-f]{64}$/i.test(text)) throw new Error('key file is not 64 hex or an nsec');
  return Uint8Array.from(Buffer.from(text, 'hex'));
}

const sk = readKey(keyFile);
const pubkey = getPublicKey(sk);
const now = () => Math.floor(Date.now() / 1000);
const kind0 = finalizeEvent({
  kind: 0,
  created_at: now(),
  tags: [],
  content: JSON.stringify({
    name: process.env.POKER_CITIZEN_NAME || 'poker-citizen',
    display_name: process.env.POKER_CITIZEN_DISPLAY_NAME || 'The House (poker)',
    about: process.env.POKER_CITIZEN_ABOUT || 'The forum poker table’s house seat. Deals DREAM hands, plays the house bot, settles every hand on sidestr:dreamlab. Testnet: no value. Do not DM; sit at /community/table.',
    bot: true,
  }),
}, sk);

await new Promise((resolve, reject) => {
  const ws = new WebSocket(RELAY_WS);
  const timer = setTimeout(() => reject(new Error('timeout')), 15000);
  ws.onmessage = (m) => {
    const d = JSON.parse(m.data);
    if (d[0] === 'AUTH') {
      ws.send(JSON.stringify(['AUTH', finalizeEvent({ kind: 22242, created_at: now(), tags: [['relay', RELAY_WS], ['challenge', d[1]]], content: '' }, sk)]));
      setTimeout(() => ws.send(JSON.stringify(['EVENT', kind0])), 600);
    } else if (d[0] === 'OK') {
      console.log('kind-0 poker-citizen:', d[2] ? 'ACCEPTED' : 'REJECTED ' + (d[3] || ''));
      clearTimeout(timer);
      ws.close();
      resolve();
    }
  };
  ws.onerror = (e) => { clearTimeout(timer); reject(e); };
});
console.log('pubkey:', pubkey);
