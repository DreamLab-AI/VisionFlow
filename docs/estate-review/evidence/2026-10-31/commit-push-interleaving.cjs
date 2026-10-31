// Deterministic audit probe: real Git, temporary local repositories only.
// Usage: node commit-push-interleaving.cjs /absolute/path/to/agentbox
// The injected writer runs after the ahead-count read, before production push.
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { execFileSync } = require('node:child_process');
const assert = require('node:assert/strict');
const source = path.resolve(process.argv[2]);
const { commitPage } = require(path.join(source, 'management-api/lib/ontology-apply.js'));
const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'vf-push-interleaving-'));
const repo = path.join(tmp, 'work');
const remote = path.join(tmp, 'remote.git');
const env = { ...process.env, GIT_CONFIG_NOSYSTEM: '1', GIT_CONFIG_GLOBAL: '/dev/null',
  GIT_AUTHOR_NAME: 'Audit fixture', GIT_AUTHOR_EMAIL: 'audit@example.invalid',
  GIT_COMMITTER_NAME: 'Audit fixture', GIT_COMMITTER_EMAIL: 'audit@example.invalid' };
// Do not inherit shell repository selectors into these isolated fixtures.
for (const key of ['GIT_DIR', 'GIT_WORK_TREE', 'GIT_INDEX_FILE', 'GIT_COMMON_DIR']) delete env[key];
const git = args => execFileSync('git', ['-c', 'core.hooksPath=/dev/null',
  '-c', 'commit.gpgsign=false', ...args], { env, encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'] });
(async () => {
  try {
    git(['init', '--bare', remote]);
    git(['init', '-b', 'main', repo]);
    fs.mkdirSync(path.join(repo, 'knowledge'));
    fs.writeFileSync(path.join(repo, 'knowledge/A.md'), 'original\n');
    git(['-C', repo, 'add', '.']);
    git(['-C', repo, 'commit', '-m', 'base']);
    git(['-C', repo, 'remote', 'add', 'origin', remote]);
    git(['-C', repo, 'push', '-u', 'origin', 'main']);
    fs.writeFileSync(path.join(repo, 'knowledge/A.md'), 'approved change\n');
    let injected = null;
    let observedAhead = null;
    const runGit = async args => {
      const output = git(args);
      if (args.includes('rev-list')) {
        observedAhead = Number(output.trim());
        fs.writeFileSync(path.join(repo, 'unrelated.txt'), 'other writer content\n');
        git(['-C', repo, 'add', 'unrelated.txt']);
        git(['-C', repo, 'commit', '-m', 'other writer after ahead check']);
        injected = git(['-C', repo, 'rev-parse', 'HEAD']).trim();
      }
      return output;
    };
    const result = await commitPage('knowledge/A.md', { outcome: 'promote',
      page: 'A', digest: 'a'.repeat(64), eventId: 'b'.repeat(64), signerNpub: 'fixture' }, { repo, runGit });
    const remoteHead = git(['--git-dir', remote, 'rev-parse', 'refs/heads/main']).trim();
    const unrelated = git(['--git-dir', remote, 'show', 'refs/heads/main:unrelated.txt']);
    assert.equal(observedAhead, 1);
    assert.equal(result.pushed, true);
    assert.equal(remoteHead, injected);
    assert.notEqual(result.commit, remoteHead);
    assert.equal(unrelated, 'other writer content\n');
    console.log(JSON.stringify({ method: 'Production commitPage; real local Git; deterministic writer inserted after ahead count. No network or live corpus.',
      observedAhead, result, injectedCommit: injected, remoteHead,
      unrelatedFileOnRemote: unrelated, reproduced: true }, null, 2));
  } finally { fs.rmSync(tmp, { recursive: true, force: true }); }
})().catch(err => { console.error(err); process.exitCode = 1; });
