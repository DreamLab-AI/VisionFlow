'use strict';
// Preload (node --require) that replaces globalThis.fetch with a deterministic
// System One mock. No network. Selected by JEV_MOCK:
//   accept | lowconf | malformed | missing | badchoice | http500 | http429 | error | timeout
// Every request body is appended as a JSON line to JEV_MOCK_LOG, with the
// headers, so a test can assert the wire shape.
const fs = require('node:fs');

globalThis.fetch = async (url, init = {}) => {
  const mode = process.env.JEV_MOCK || 'accept';
  const body = JSON.parse(init.body);
  if (process.env.JEV_MOCK_LOG) fs.appendFileSync(process.env.JEV_MOCK_LOG, JSON.stringify({ url, headers: init.headers, body }) + '\n');
  if (mode === 'error') throw new TypeError('fetch failed');
  if (mode === 'timeout') {
    return new Promise((_, reject) => init.signal.addEventListener('abort', () => reject(Object.assign(new Error('aborted'), { name: 'AbortError' }))));
  }
  const reply = (status, text, headers = {}) => ({ status, ok: status >= 200 && status < 300, headers: { get: (k) => headers[k.toLowerCase()] ?? null }, text: async () => text });
  if (mode === 'http500') return reply(500, '{"error":"boom"}');
  if (mode === 'http429') return reply(429, '{}', { 'retry-after': '0' });
  if (mode === 'malformed') return reply(200, '<html>not json');
  const conf = mode === 'lowconf' ? 0.05 : 0.9;
  const answers = {};
  let n = 0;
  for (const [id, q] of Object.entries(body.questions)) {
    const i = n++;
    if (q.type === 'noul') answers[id] = { type: 'noul', noul: mode === 'lowconf' ? 0.5 : 0.1 };
    else if (q.type === 'score') answers[id] = { type: 'score', score: Math.min(i, q.criteria.length - 1), confidence: conf, legend: {}, probabilities: {} };
    else {
      const keys = Object.keys(q.criteria);
      let choice = keys[0];
      if (id.startsWith('claim_')) choice = i === 0 ? 'overstated' : 'supported';
      if (id === 'verdict') choice = 'contradicted';
      if (id.startsWith('triage_')) choice = 'source-build';
      if (mode === 'badchoice') choice = 'not-offered';
      answers[id] = { type: 'choice', choice, confidence: conf, probabilities: {} };
    }
  }
  if (mode === 'missing') delete answers[Object.keys(answers)[0]];
  return reply(200, JSON.stringify({ model: 'jev-mock', answers, usage: { input_tokens: 1, output_tokens: 1 } }));
};
