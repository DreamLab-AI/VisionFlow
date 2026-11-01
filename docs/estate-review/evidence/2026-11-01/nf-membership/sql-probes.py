#!/usr/bin/env python3
"""Run exported Rust SQL on disposable SQLite; no Worker/D1 execution."""
import hashlib
import json
from pathlib import Path
import re
import sqlite3

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "source"
for entry in json.loads((ROOT / "source-manifest.json").read_text()):
    assert hashlib.sha256((SOURCE / entry["path"]).read_bytes()).hexdigest() == entry["sha256"]


def statements(path):
    text = (SOURCE / path).read_text()
    for match in re.finditer(r'"(?:\\.|[^"\\])*"', text, re.S):
        raw = re.sub(r"\\\n\s*", "", match.group())
        try:
            yield json.loads(raw)
        except json.JSONDecodeError:
            pass


used = {}


def sql(name, path, prefix):
    matches = set(s for s in statements(path) if s.startswith(prefix))
    assert len(matches) == 1, (name, matches)
    value = matches.pop()
    used[name] = {"path": path, "sql": value}
    return value


CORE = "crates/nostr-bbs-core/src/"
AUTH = "crates/nostr-bbs-auth-worker/src/"
RELAY = "crates/nostr-bbs-relay-worker/src/"
create = sql("create", CORE + "whitelist_sql.rs", "CREATE TABLE IF NOT EXISTS whitelist (")
grant = sql("grant", CORE + "whitelist_sql.rs", "INSERT INTO whitelist (pubkey, cohorts, added_at, added_by) ")
admit = sql("admission", RELAY + "relay_do/storage.rs", "SELECT 1 as found FROM whitelist WHERE")
read = sql("invite_read", AUTH + "invites.rs", "SELECT cohorts FROM whitelist WHERE")
replace = sql("invite_write", AUTH + "invites.rs", "UPDATE whitelist SET cohorts =")
set_admin = sql("relay_set_admin", RELAY + "whitelist.rs", "UPDATE whitelist SET is_admin =")
member_create = sql("auth_members", AUTH + "schema.rs", "CREATE TABLE IF NOT EXISTS members (")
relay_admin = sql("relay_admin", CORE + "admin_shared.rs", "SELECT is_admin FROM whitelist WHERE")
auth_admin = sql("auth_admin", CORE + "admin_shared.rs", "SELECT is_admin FROM members WHERE")
invite_create = sql("invitations", AUTH + "schema.rs", "CREATE TABLE IF NOT EXISTS invitations (")
invite_alter = sql("invite_zone", AUTH + "schema.rs", "ALTER TABLE invitations ADD COLUMN zone_id")
invite_revoke = sql("invite_revoke_lookup", AUTH + "invites.rs", "SELECT id, code, issued_by, max_uses, uses, expires_at, revoked_at, revoked_by, created_at, zone_id FROM invitations WHERE id =")
delete = sql("delete_member", RELAY + "user_admin.rs", "DELETE FROM whitelist WHERE pubkey =")
PK = "ab" * 32
cases = []


def db():
    conn = sqlite3.connect(":memory:")
    conn.execute(create)
    return conn


def seed(conn, pk=PK):
    conn.execute(grant, (pk, '["members"]', 10, "fixture"))


with db() as conn:
    seed(conn)
    conn.execute("UPDATE whitelist SET expires_at = 50 WHERE pubkey = ?", (PK,))
    conn.execute(grant, (PK, '["zoneA"]', 200, "admin"))
    row = conn.execute("SELECT cohorts, expires_at FROM whitelist").fetchone()
    assert json.loads(row[0]) == ["members", "zoneA"] and row[1] == 50
    assert conn.execute(admit, (PK, 200)).fetchone() is None
    conn.execute("UPDATE whitelist SET expires_at = NULL")
    assert conn.execute(admit, (PK, 200)).fetchone() == (1,)
    cases.append({"name": "approval_keeps_expiry", "expired_admitted": False, "null_expiry_control_admitted": True})

for interleaved in (True, False):
    with db() as conn:
        seed(conn)
        if not interleaved:
            conn.execute(grant, (PK, '["zoneB"]', 20, "admin"))
        old = json.loads(conn.execute(read, (PK,)).fetchone()[0])
        if interleaved:
            conn.execute(grant, (PK, '["zoneB"]', 20, "admin"))
        # Selected inputs reproduce merge_cohorts_json: append one absent string.
        merged = old + ["zoneA"]
        conn.execute(replace, (json.dumps(merged), PK))
        actual = json.loads(conn.execute(read, (PK,)).fetchone()[0])
        assert actual == (["members", "zoneA"] if interleaved else ["members", "zoneB", "zoneA"])
        cases.append({"name": "invite_interleaved" if interleaved else "invite_serial_control", "cohorts": actual})

with db() as conn:
    conn.execute(member_create)
    seed(conn)
    conn.execute("INSERT INTO members (pubkey,is_admin,created_at) VALUES (?,1,10)", (PK,))
    conn.execute(set_admin, (1, PK))
    conn.execute(set_admin, (0, PK))
    relay = conn.execute(relay_admin, (PK,)).fetchone()[0]
    auth = conn.execute(auth_admin, (PK,)).fetchone()[0]
    assert (relay, auth) == (0, 1)
    cases.append({"name": "relay_demotion_leaves_auth_flag", "relay_flag": relay, "auth_flag": auth})

with db() as conn:
    seed(conn, PK.upper())
    assert conn.execute(admit, (PK, 20)).fetchone() is None
    assert conn.execute(admit, (PK.upper(), 20)).fetchone() == (1,)
    cases.append({"name": "case_sensitive_row", "lowercase_match": False, "uppercase_control_match": True})

with db() as conn:
    conn.execute(invite_create)
    conn.execute(invite_alter)
    invite_id, code = "abcdefgh", "abcdefghijklmnop"
    conn.execute("INSERT INTO invitations (id,code,issued_by,expires_at,created_at) VALUES (?,?,?,1000,10)", (invite_id, code, PK))
    assert conn.execute(invite_revoke, (code,)).fetchone() is None
    assert conn.execute(invite_revoke, (invite_id,)).fetchone()[0] == invite_id
    cases.append({"name": "revoke_code_id_mismatch", "code_found": False, "id_control_found": True})

with db() as conn:
    seed(conn)
    conn.execute(delete, (PK,))
    assert conn.execute(admit, (PK, 20)).fetchone() is None
    seed(conn)
    assert conn.execute(admit, (PK, 20)).fetchone() == (1,)
    cases.append({"name": "delete_has_no_tombstone", "regrant_creates_row": True})

print(json.dumps({"sqlite_version": sqlite3.sqlite_version, "scope": "Exact extracted SQL, synthetic inputs and selected schedules; not live D1 or handlers", "cases": cases, "statements": used}, indent=2))
