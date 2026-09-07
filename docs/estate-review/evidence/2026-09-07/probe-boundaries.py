#!/usr/bin/env python3
"""Isolated synthetic probes of source queries and public inference; no live data."""
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import sys
import tempfile

workspace = Path(__file__).resolve().parents[5]
trust = workspace / 'nostr-rust-forum/crates/nostr-bbs-relay-worker/src/trust_sweep.rs'
source = trust.read_text()
def sql(prefix):
    match = re.search(r'"(' + re.escape(prefix) + r'.*?)"', source, re.S)
    if not match:
        raise RuntimeError('Production query changed; re-review probe')
    return re.sub(r'\\\n\s*', '', match[1])
db = sqlite3.connect(':memory:')
db.executescript('''CREATE TABLE whitelist(pubkey TEXT, trust_level INT, trust_level_updated_at INT);
CREATE TABLE admin_log(actor_pubkey TEXT, action TEXT, target_pubkey TEXT, previous_value TEXT, new_value TEXT, reason TEXT, created_at INT);
INSERT INTO whitelist VALUES('synthetic-key', 3, 0);''')
with db:
    changed = db.execute(sql('UPDATE whitelist SET trust_level = ?1'), (0, 123, 'synthetic-key', 1)).rowcount
    db.execute(sql('INSERT INTO admin_log '), ('system', 'trust_level_change', 'synthetic-key', '1', '0', 'auto-demotion (hysteresis)', 123))
trust_result = {'update_changes': changed, 'stored_level': db.execute('SELECT trust_level FROM whitelist').fetchone()[0], 'audit_rows': db.execute('SELECT COUNT(*) FROM admin_log').fetchone()[0]}
sys.path.insert(0, str(workspace / 'visionGraph'))
from pipeline.jsonld_parser import PageData, OntologyEntity, WikilinkRef
from pipeline.reason import compute_closure, emit_inferred_ttl

def page(name, public, parent=None):
    iri = 'urn:ngm:class:' + name
    refs = [WikilinkRef(iri='urn:ngm:class:' + parent, label=parent)] if parent else []
    return PageData(path=Path(name + '.md'), page_iri='urn:ngm:page:' + name, slug=name, title=name, is_public=public, schema_version=2, ontology_class=OntologyEntity(iri=iri, label=name, entity_type='Class', domain='synthetic', definition='Synthetic audit fixture', sub_class_of=refs))
pages = [page('audit-public-child', True, 'audit-private-parent'), page('audit-private-parent', False, 'audit-private-grandparent'), page('audit-private-grandparent', False)]
with tempfile.TemporaryDirectory(prefix='visionflow-public-inference-') as directory:
    output = Path(directory) / 'inferred.ttl'
    triples = emit_inferred_ttl(pages, compute_closure(pages), output)
    inferred_result = {'inferred_triples': triples, 'private_grandparent_present': 'urn:ngm:class:audit-private-grandparent' in output.read_text()}
paths = [trust, workspace / 'visionGraph/pipeline/jsonld_parser.py', workspace / 'visionGraph/pipeline/reason.py']
result = {'date':'2026-09-07','scope':'Exact source SQL in fresh in-memory SQLite and synthetic ontology pages in temporary storage; no live account, private corpus, external database or publication used. Exit zero means probes ran, not that products conform.', 'trust_conflict':trust_result,'public_inference':inferred_result,'sources':{str(p.relative_to(workspace)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
Path(__file__).with_name('boundary-probes.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
