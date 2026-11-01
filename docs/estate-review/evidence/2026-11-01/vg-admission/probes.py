"""Exercise the exported space-corpus executable, never a live vault.

Usage: python3 probes.py BINARY EXPORTED_VISIONGRAPH_ROOT EXECUTABLE_TEMP_ROOT
The fake `vault` records arguments and returns controlled responses. It proves
only the importer's subprocess boundary, not behaviour of the real vault CLI.
"""
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

binary, exported, temp_root = map(Path, sys.argv[1:])
results = []


def candidate(title="A", **overrides):
    value = dict(title=title, domain="space-science-and-systems", branch="test",
                 sources=[{"id": "slso", "resource": "urn:source:A"}], relations={})
    value.update(overrides)
    return value


def catalogue(pages=None, **overrides):
    value = dict(generated_at="2026-11-01T00:00:00Z", public=False,
                 pages=pages if pages is not None else [candidate()], mappings=[])
    value.update(overrides)
    return value


with tempfile.TemporaryDirectory(prefix="admission-probes-", dir=temp_root) as td:
    root = Path(td)

    def fixture(name):
        d = root / name
        (d / "repo/knowledge").mkdir(parents=True)
        (d / "repo/ontology").mkdir()
        shutil.copyfile(exported / "ontology/vocabulary.yaml", d / "repo/ontology/vocabulary.yaml")
        return d

    def run(name, d, args, success, env=None):
        p = subprocess.run([str(binary), *map(str, args)], text=True,
                           capture_output=True, timeout=30, env=env)
        row = dict(case=name, returncode=p.returncode, stdout=p.stdout, stderr=p.stderr)
        results.append(row)
        assert (p.returncode == 0) == success, row
        return row

    def stage(name, c, success=True, prepare=None):
        d = fixture(name)
        if prepare:
            prepare(d)
        (d / "catalogue.json").write_text(json.dumps(c))
        run(name, d, ["stage", d / "catalogue.json", "--repo", d / "repo",
                      "--out", d / "staged"], success)
        results[-1]["staged_files"] = sorted(p.name for p in (d / "staged").glob("*.md"))
        return d

    control = stage("valid-private-header", catalogue())
    private_header = (control / "staged/A.md").read_text()
    assert "public: false" in private_header
    stage("missing-source-list-refused", catalogue([candidate(sources=[])]), False)
    stage("empty-source-object-accepted", catalogue([candidate(sources=[{}])]))
    stage("unregistered-source-accepted", catalogue([candidate(sources=[{"id": "unadmitted", "resource": "urn:unadmitted"}])]))
    stage("unknown-relation-refused", catalogue([candidate(relations={"unknown-predicate": ["A"]})]), False)
    stage("unresolved-target-refused", catalogue([candidate(relations={"is-a": ["Missing"]})]), False)
    stage("self-relation-refused", catalogue([candidate(relations={"is-a": ["A"]})]), False)
    stage("two-page-cycle-accepted", catalogue([candidate(relations={"is-a": ["B"]}), candidate("B", relations={"is-a": ["A"]})]))
    stage("missing-source-mappings-accepted", catalogue(mappings=[]))
    stage("dangling-mapping-target-accepted", catalogue(mappings=[{"source_iri": "urn:absent", "title": "Missing", "disposition": "create"}]))
    stage("unresolved-mapping-refused", catalogue(mappings=[{"disposition": "unresolved"}]), False)
    public = stage("public-catalogue-stages-public", catalogue(public=True))
    assert "public: true" in (public / "staged/A.md").read_text()
    stage("alias-collision-accepted", catalogue([candidate(aliases=["Shared"]), candidate("B", aliases=["Shared"])]))
    stage("slug-collision-refused", catalogue([candidate("Same Name"), candidate("Same-Name")]), False)
    partial = stage("late-source-error-leaves-partial-stage", catalogue([candidate(), candidate("B", sources=[])]), False)
    assert (partial / "staged/A.md").exists()
    stage("empty-generation-time-accepted", catalogue(generated_at=""))

    d = fixture("audit-reused-edge")
    for title in ["A", "B"]:
        (d / f"repo/knowledge/{title}.md").write_text(private_header.replace("title: A", f"title: {title}"))
    c = catalogue([candidate(reuse="A", relations={"is-a": ["B"]})])
    (d / "catalogue.json").write_text(json.dumps(c))
    run("audit-accepts-absent-reused-edge", d, ["audit", d / "catalogue.json", "--repo", d / "repo", "--out", d / "audit.json"], True)
    results[-1]["audit"] = json.loads((d / "audit.json").read_text())

    (d / "cases.json").write_text(json.dumps({"cases": []}))
    run("zero-competency-cases-pass", d, ["check-cases", d / "cases.json", "--repo", d / "repo", "--out", d / "cases-result.json"], True)

    rdf = fixture("inventory")
    ttl = '''@prefix owl: <http://www.w3.org/2002/07/owl#>.
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#>.
<urn:A> a owl:Class; rdfs:subClassOf [ a owl:Restriction; owl:onProperty <urn:p>; owl:someValuesFrom <urn:B> ].
<urn:ontology> owl:imports <https://invalid.example/not-fetched>.
'''
    (rdf / "source.ttl").write_text(ttl)
    run("inventory-retains-reference-not-restriction-closure", rdf, ["inventory", rdf / "source.ttl", "--out", rdf / "out.json"], True)
    data = json.loads((rdf / "out.json").read_text())
    assert data["quad_count"] == 6
    assert len(data["classes"]) == 1
    assert "http://www.w3.org/2002/07/owl#onProperty" not in data["classes"][0]["predicates"]
    results[-1]["inventory"] = data

    # An instrumented subprocess, never the installed vault executable.
    fake = root / "bin"
    fake.mkdir()
    logger = fake / "vault"
    logger.write_text("#!" + sys.executable + "\nimport os,json,sys\nfrom pathlib import Path\np=Path(os.environ['PROBE_CALLS'])\nwith p.open('a') as f: f.write(json.dumps(sys.argv[1:])+'\\n')\nprint(json.dumps({'created':True,'page':'probe'}))\n")
    logger.chmod(0o755)
    d = fixture("apply")
    (d / "staged").mkdir()
    (d / "staged/A.md").write_text("not reviewed Markdown\n")
    env = dict(os.environ, PATH=str(fake), PROBE_CALLS=str(d / "calls.jsonl"))
    run("apply-forwards-unreviewed-file", d, ["apply", d / "staged", "--repo", d / "repo", "--expect", 1], True, env)
    results[-1]["calls"] = [json.loads(x) for x in (d / "calls.jsonl").read_text().splitlines()]
    run("apply-total-mismatch-refused", d, ["apply", d / "staged", "--repo", d / "repo", "--expect", 2], False, env)
    (d / "repo/knowledge/A.md").write_text("changed")
    run("apply-resume-byte-mismatch-refused", d, ["apply", d / "staged", "--repo", d / "repo", "--expect", 1, "--resume"], False, env)
    (d / "repo/knowledge/A.md").write_text(private_header)
    (d / "catalogue.json").write_text(json.dumps(catalogue()))
    run("promote-trusts-zero-exit-without-postread", d, ["promote", d / "catalogue.json", "--repo", d / "repo", "--expect", 1], True, env)
    assert (d / "repo/knowledge/A.md").read_text() == private_header
    results[-1]["page_still_private"] = True

print(json.dumps({"scope": "exact exported importer; current pinned vault-core with isolated regenerated lock; fake vault only for subprocess-boundary cases", "cases": results}, indent=2))
