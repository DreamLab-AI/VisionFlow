"""Run against a git-exported archive, never the authored corpus.

Usage: PYTHONPATH=<export>/archive/logseq-era-2026-09-22 python probes.py
Assertions describe observed defects, not desired acceptance behaviour.
"""
import json
import re
import tempfile
from pathlib import Path

from rdflib import Graph
from pipeline.ontocast_import import ImportConfig, import_graph, write_candidates

PREFIX = '''@prefix ex: <https://example.test/> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
'''
CFG = ImportConfig('probe', 'test', 'synthetic:test', 'urn:probe:class/root', 'Root')


def imported(ttl):
    return import_graph(Graph().parse(data=PREFIX + ttl, format='turtle'), CFG)


def blocks(candidate):
    return [json.loads(x) for x in re.findall(r'```json-ld\n(.*?)\n```', candidate.markdown, re.S)]


def main():
    results = []
    report = imported('ex:A a owl:Class; rdfs:label "A"; ex:evidence [ex:detail "NESTED_SENTINEL"].')
    assert len(report.candidates) == 1
    page, entity, evidence = blocks(report.candidates[0])
    assert page['vc:public'] is False and evidence['vc:reviewStatus'] == 'pending'
    assert 'NESTED_SENTINEL' not in report.candidates[0].markdown
    assert any(s['object']['kind'] == 'blank-node' for s in evidence['vc:statements'])
    results.append({'probe': 'blank-node-evidence', 'result': 'reference retained; nested statement omitted; private pending control passes'})

    report = imported('ex:A a owl:Class; rdfs:label "Same". ex:B a owl:Class; rdfs:label "Same". ex:C a owl:Class; rdfs:label "Child"; rdfs:subClassOf ex:B.')
    assert report.skipped[0]['sourceIri'] == 'https://example.test/B'
    child = next(c for c in report.candidates if c.title == 'Child')
    assert blocks(child)[1]['subClassOf'][0]['@id'] == 'urn:probe:class/same'
    survivor = next(c for c in report.candidates if c.title == 'Same')
    assert survivor.source_iri == 'https://example.test/A'
    results.append({'probe': 'collision-reference', 'result': 'child of skipped B maps to retained A local identifier'})

    report = imported('ex:P a owl:Class, owl:NamedIndividual; rdfs:label "Punned".')
    assert len(report.candidates) == 2
    assert {blocks(c)[1]['@id'] for c in report.candidates} == {'urn:probe:individual/punned'}
    with tempfile.TemporaryDirectory() as d:
        try:
            write_candidates(report, Path(d))
            raise AssertionError('expected second candidate collision')
        except FileExistsError:
            assert len(list(Path(d).glob('*.md'))) == 1
    results.append({'probe': 'dual-type', 'result': 'two candidates share individual identifier; write fails after first file'})

    report = imported('ex:A a owl:Class; rdfs:label "A".')
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        staging = root / 'staging'
        staging.mkdir()
        target = root / 'outside.md'
        (staging / 'A.md').symlink_to(target)
        write_candidates(report, staging)
        assert target.read_text() == report.candidates[0].markdown
        try:
            write_candidates(report, staging)
            raise AssertionError('expected existing-file rejection')
        except FileExistsError:
            pass
    results.append({'probe': 'dangling-symlink', 'result': 'pre-existing dangling symlink followed outside staging; subsequent existing-file refusal passes'})
    print(json.dumps({'scope': 'archived adapter, synthetic fixtures only', 'passed': len(results), 'results': results}, indent=2))


if __name__ == '__main__':
    main()
