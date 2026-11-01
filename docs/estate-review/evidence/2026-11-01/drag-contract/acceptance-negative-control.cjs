// Run the unchanged acceptance script against a deliberately wrong local peer.
// Usage: node acceptance-negative-control.cjs <script.mjs> <ws-module-directory>
// No production endpoint is contacted. The peer never releases its held node.
const { spawn } = require('node:child_process');
const { WebSocketServer } = require(process.argv[3]);
const server = new WebSocketServer({ host: '127.0.0.1', port: 0 });
const peers = [];
let released = false;
let framesAfterRelease = 0;
let position = 0;
function frame(peer, nodes) {
  if (released) framesAfterRelease++;
  const b = Buffer.alloc(9 + 52 * nodes.length);
  b[0] = 5;
  b.writeBigUInt64LE(++peer.seq, 1);
  nodes.forEach(([id, x], i) => {
    const p = 9 + 52 * i;
    b.writeUInt32LE(id, p);
    b.writeFloatLE(x, p + 4);
  });
  peer.ws.send(b);
}
server.on('connection', ws => {
  const peer = { ws, seq: 0n, index: peers.length };
  peers.push(peer);
  ws.on('message', (raw, binary) => {
    if (binary) return; // Acks are accepted; this fixture does not model server pacing.
    const m = JSON.parse(raw.toString());
    if (m.type === 'authenticate') ws.send(JSON.stringify({ type: 'authenticate_success' }));
    if (m.type === 'requestInitialData') frame(peer, [[1, 0], [2, 10], [3, 20]]);
    if (m.type === 'nodeDragStart') ws.send(JSON.stringify({ type: 'nodeDragStartAck' }));
    if (m.type === 'nodeDragUpdate') {
      position = m.data.position.x;
      // Dragger receives nearby nodes but has a stale held-node sample.
      frame(peers[0], [[2, 10 + position / 10], [3, 20 + position / 10]]);
      frame(peers[1], [[1, position], [2, 10 + position / 10], [3, 20 + position / 10]]);
    }
    if (m.type === 'nodeDragEnd') {
      released = true;
      ws.send(JSON.stringify({ type: 'nodeDragEndAck' }));
      // Intentionally wrong: retain the pin and emit no post-release positions.
    }
  });
});
server.on('listening', () => {
  const child = spawn(process.execPath, [process.argv[2], '--url',
    `ws://127.0.0.1:${server.address().port}`, '--ws-module', process.argv[3],
    '--node', '1', '--neighbours', '2,3', '--drag-ms', '200', '--rate', '20',
    '--observe-ms', '300', '--quiet-ms', '100'], { stdio: ['ignore', 'pipe', 'pipe'] });
  let output = '';
  child.stdout.on('data', b => { output += b; process.stdout.write(b); });
  child.stderr.on('data', b => process.stderr.write(b));
  const timer = setTimeout(() => child.kill('SIGTERM'), 15000);
  child.on('exit', code => {
    clearTimeout(timer);
    const reproduced = code === 0 && released && framesAfterRelease === 0 &&
      output.includes('PASS  released node rejoined the physics');
    console.log(JSON.stringify({ negativeControl: true, scriptExit: code,
      nodeStillPinned: true, pinnedPosition: position, framesAfterRelease, reproduced }));
    peers.forEach(p => p.ws.terminate());
    server.close(() => { process.exitCode = reproduced ? 0 : 1; });
  });
});
