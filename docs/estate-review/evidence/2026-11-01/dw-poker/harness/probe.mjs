// Play one hand against the live house seat as a member, end to end over
// the production relay: the protocol of forum ADR-2020 (gift-wrapped rumors
// of kind 20779), with the member's side done the way the forum client does
// it, minus the wallet (the settlement transfer is left for the client; this
// probe only reports what the hand owes).
//
//   HERO_KEY_FILE=/path/to/hero.key \
//     node scripts/seed/test-poker-citizen.mjs [--bb=2] [--citizen=<hex>]
//
// The hero key must be whitelisted on the relay and hold at least the buy-in
// in DREAM (the house refuses otherwise, which this probe also reports). The
// key is read from a file, never printed. The hero calls or checks every
// turn, so a hand ends in a few seconds.
import { readFileSync } from 'node:fs';
import { createHash, randomBytes } from 'node:crypto';
import { finalizeEvent, getPublicKey } from 'nostr-tools/pure';
import { nip59 } from 'nostr-tools';

const RELAY_WS = 'wss://dreamlab-nostr-relay.solitary-paper-764d.workers.dev';
const RUMOR_KIND = 20779;
const VERSION = 1;
const arg = (name, dflt) => {
  const a = process.argv.find((x) => x.startsWith(`--${name}=`));
  return a ? a.slice(name.length + 3) : dflt;
};
const CITIZEN = arg('citizen', 'd4bda43addb1419ad2930401322403c774bc30a141a6eb8fdba0314ee4086df9');
const BB = Number(arg('bb', '2'));

const keyText = readFileSync(process.env.HERO_KEY_FILE, 'utf8').trim();
if (!/^[0-9a-f]{64}$/i.test(keyText)) throw new Error('HERO_KEY_FILE must hold 64 hex characters');
const sk = Uint8Array.from(Buffer.from(keyText, 'hex'));
const me = getPublicKey(sk);
const now = () => Math.floor(Date.now() / 1000);
const sha256 = (s) => createHash('sha256').update(s).digest('hex');

const ws = new WebSocket(RELAY_WS);
const send = (msg) => {
  const rumor = { kind: RUMOR_KIND, created_at: now(), tags: [['p', CITIZEN]], content: JSON.stringify(msg), pubkey: me };
  const wrap = nip59.wrapEvent(rumor, sk, CITIZEN);
  ws.send(JSON.stringify(['EVENT', wrap]));
};
let nonce = null;
let commitInPlay = null;
let hands = 0;
const seen = new Set();
const deadline = setTimeout(() => { console.error('FAIL: timed out'); process.exit(2); }, 90_000);

ws.onopen = () => console.log('connected as', me.slice(0, 12) + '…');
ws.onmessage = (m) => {
  const d = JSON.parse(m.data);
  if (d[0] === 'AUTH') {
    ws.send(JSON.stringify(['AUTH', finalizeEvent({ kind: 22242, created_at: now(), tags: [['relay', RELAY_WS], ['challenge', d[1]]], content: '' }, sk)]));
    return;
  }
  if (d[0] === 'OK') {
    if (d[1] && !d[2]) console.error('relay refused', d[1].slice(0, 12), d[3]);
    if (d[2] && d[1].startsWith('auth-')) {}
    return;
  }
  if (d[0] === 'NOTICE') { console.log('notice:', d[1]); return; }
  if (d[0] === 'EOSE') return;
  if (d[0] !== 'EVENT') return;
  const ev = d[2];
  if (seen.has(ev.id)) return;
  seen.add(ev.id);
  let rumor;
  try { rumor = nip59.unwrapEvent(ev, sk); } catch { return; }
  if (rumor.kind !== RUMOR_KIND || rumor.pubkey !== CITIZEN) return;
  if (now() - rumor.created_at > 600) return;
  const msg = JSON.parse(rumor.content);
  handle(msg);
};
// after AUTH, open the inbox and say hello
setTimeout(() => {
  ws.send(JSON.stringify(['REQ', 'inbox', { kinds: [1059], '#p': [me], since: now() - 2 * 86400 - 3600 }]));
  send({ t: 'hello', v: VERSION });
  console.log('→ hello');
}, 1200);

function handle(msg) {
  switch (msg.t) {
    case 'offer': {
      console.log(`← offer from ${msg.name} (${msg.profile}); tables ${msg.tables.map((t) => t.label).join(' ')}; present ${msg.present.length}; owed ${msg.owed.length}; cap left ${msg.daily_cap_left}`);
      if (hands >= 1) { console.log('PASS: one hand played and settled'); clearTimeout(deadline); ws.close(); process.exit(0); }
      nonce = randomBytes(32).toString('hex');
      commitInPlay = msg.commit;
      send({ t: 'sit', v: VERSION, commit: msg.commit, asset: 'dream', bb: BB, nonce });
      console.log('→ sit', BB);
      break;
    }
    case 'state': {
      const v = msg.view;
      const mine = v.seats[msg.seat];
      console.log(`← state ${v.street} pot ${v.pot} me ${mine.stack} hole ${JSON.stringify(v.hole)} toAct ${v.toAct}`);
      if (v.phase === 'act' && v.toAct === msg.seat) {
        const call = Math.min(Math.max(v.currentBet - mine.streetCommit, 0), mine.stack);
        const action = call === 0 ? 'check' : 'call';
        send({ t: 'act', commit: msg.commit, action: { seat: msg.seat, action } });
        console.log('→', action);
      }
      break;
    }
    case 'done': {
      hands += 1;
      const ok = sha256(msg.secret) === msg.commit && sha256(msg.secret + msg.nonces.join('')) === msg.seed && msg.nonces[msg.seat] === nonce;
      console.log(`← done: commit ${ok ? 'verified' : 'MISMATCH'}; root ${msg.root.slice(0, 16)}…; settlement ${JSON.stringify(msg.settlement)}; winners ${JSON.stringify(msg.view.result?.winners)}`);
      if (!ok) { console.error('FAIL: commitment or nonce mismatch'); process.exit(1); }
      if (msg.settlement && msg.settlement.from === 'Hero') {
        console.log(`(the hero owes ${msg.settlement.amount} DREAM for hand:${msg.root}; the forum client pays this from the wallet)`);
      }
      break;
    }
    case 'paid':
      console.log('← paid', msg.txid);
      break;
    case 'error':
      console.error('← error:', msg.message);
      if (/buy-in|owe|protocol|cannot cover|daily limit/.test(msg.message)) { console.error('FAIL'); process.exit(1); }
      break;
    default:
      console.log('←', msg.t);
  }
}
