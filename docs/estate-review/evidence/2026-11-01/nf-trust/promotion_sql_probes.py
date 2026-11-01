"""Exact promotion SQL on synthetic SQLite; not Worker/D1 execution."""
from pathlib import Path
import argparse
import hashlib
import json
import re
import sqlite3

e = Path(__file__).resolve().parent
source = (e / 'source/crates/nostr-bbs-relay-worker/src/trust.rs').read_text()
body = source[source.index('pub async fn check_promotion'):source.index('// No per-pubkey')]
strings = re.findall(r'"([^"\\]*(?:\\.[^"\\]*)*)"', body, re.S)

def extract(prefix):
    raw = next(s for s in strings if s.startswith(prefix))
    return re.sub(r'\\\n\s*', '', raw), hashlib.sha256(raw.encode()).hexdigest()

update, update_hash = extract('UPDATE whitelist SET trust_level')
audit, audit_hash = extract('INSERT INTO admin_log')

def db():
    c = sqlite3.connect(':memory:', isolation_level=None)
    c.executescript('''
      CREATE TABLE whitelist(pubkey TEXT PRIMARY KEY, trust_level INTEGER,
        trust_level_updated_at REAL, is_admin INTEGER, last_active_at REAL,
        days_active INTEGER, posts_read INTEGER, posts_created INTEGER,
        mod_actions_against INTEGER);
      CREATE TABLE admin_log(actor_pubkey TEXT, action TEXT, target_pubkey TEXT,
        previous_value TEXT, new_value TEXT, reason TEXT, created_at INTEGER);
      INSERT INTO whitelist VALUES ('key',0,NULL,0,100,14,50,10,0);
    ''')
    return c

def apply(c):
    # The handler read a qualifying TL0 row before the selected interleaving.
    # Actual Rust computes TL2 from these counters. Both .run() errors are ignored.
    errors = []
    for name, statement, args in [
        ('update', update, (2, 1000, 'key')),
        ('audit', audit, ('system', 'trust_level_change', 'key', '0', '2', 'auto-promotion', 1000)),
    ]:
        try:
            c.execute(statement, args)
        except sqlite3.DatabaseError as err:
            errors.append({'operation': name, 'error': str(err)})
    level = c.execute('SELECT trust_level FROM whitelist').fetchone()
    return {'stored_level': level[0] if level else None,
            'audit_rows': c.execute('SELECT previous_value,new_value FROM admin_log').fetchall(),
            'errors': errors}

cases = []
for name, setup, level, audit_count, errors in [
    ('serial_success_control', '', 2, 1, 0),
    ('concurrent_tl3_grant_overwritten', 'UPDATE whitelist SET trust_level=3', 2, 1, 0),
    ('concurrent_moderation_count_ignored', 'UPDATE whitelist SET mod_actions_against=1', 2, 1, 0),
    ('update_failure_still_creates_audit', "CREATE TRIGGER reject_update BEFORE UPDATE ON whitelist BEGIN SELECT RAISE(ABORT,'injected update failure'); END;", 0, 1, 1),
    ('audit_failure_leaves_unaudited_promotion', "CREATE TRIGGER reject_audit BEFORE INSERT ON admin_log BEGIN SELECT RAISE(ABORT,'injected audit failure'); END;", 2, 0, 1),
    ('deleted_row_still_gets_audit', 'DELETE FROM whitelist', None, 1, 0),
]:
    c = db()
    if setup:
        c.executescript(setup)
    result = apply(c)
    assert result['stored_level'] == level, (name, result)
    assert len(result['audit_rows']) == audit_count, (name, result)
    assert len(result['errors']) == errors, (name, result)
    cases.append({'case': name, **result})
    c.close()

p = argparse.ArgumentParser()
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
a.out.write_text(json.dumps({'cases': cases, 'queries': {
    'promotion_update': {'sql': update, 'raw_sha256': update_hash},
    'promotion_audit': {'sql': audit, 'raw_sha256': audit_hash}},
    'limits': ['Synthetic SQLite tables and exact extracted statements.',
               'Chosen interleavings after a valid snapshot; no live D1, Worker, admin endpoint or HTTP.',
               'The fixture mirrors ignored statement errors and separate operations; the full async Rust handler is source-traced, not executed.']}, indent=2) + '\n')
print(f'{len(cases)} promotion scenarios passed')
