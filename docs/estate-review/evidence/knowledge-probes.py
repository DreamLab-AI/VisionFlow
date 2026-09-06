#!/usr/bin/env python3
"""Local pipeline review receipts; all generated build/probe data is temporary.
Uses knowledgeGraph's existing venv. Never publishes or prints private corpus text.
"""
import contextlib
import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[3]
PYTHON = WORKSPACE / 'knowledgeGraph/.venv/bin/python'


def worker(repo):
    root = WORKSPACE / repo
    sys.path.insert(0, str(root))
    from pipeline.jsonld_parser import parse_corpus, parse_page, PageData, OntologyEntity, WikilinkRef
    from pipeline.validate import validate_corpus
    from pipeline.jsonld_to_turtle import build_graph
    from pipeline.jsonld_to_page_api import build_page_api
    from rdflib.namespace import RDF, OWL, RDFS
    corpus = root / {'knowledgeGraph': 'ontology/pages', 'logseq': 'mainKnowledgeGraph/pages', 'visionGraph': 'knowledge/pages'}[repo]
    pages = parse_corpus(corpus)
    result = {'repo': repo, 'census': {
        'markdown_files': len(list(corpus.glob('*.md'))), 'recursive_markdown_files': len(list(corpus.rglob('*.md'))), 'parsed_pages': len(pages),
        'public_pages': sum(bool(p.is_public) for p in pages),
        'non_boolean_public_values': sum(type(p.is_public) is not bool for p in pages),
        'entities': sum(p.ontology_class is not None for p in pages),
        'source_individuals': sum(p.ontology_class is not None and p.ontology_class.entity_type == 'Individual' for p in pages),
        'validation': validate_corpus(pages).summary(),
    }}
    with tempfile.TemporaryDirectory(prefix='estate-knowledge-') as tmp:
        base = Path(tmp)
        bad = base / 'malformed.md'
        bad.write_text('```json-ld\n{broken}\n```\n')
        parsed = parse_corpus(base)
        result['malformed_fence'] = {'input_files': 1, 'parsed_pages': len(parsed), 'validation': validate_corpus(parsed).summary()}
        false = base / 'false-string.md'
        false.write_text('```json-ld\n' + json.dumps({'@type':'Page','@id':'urn:test:page', 'vc:slug':'false-string','vc:public':'false','vc:schemaVersion':2}) + '\n```\nSynthetic probe body')
        p = parse_page(false)
        build_page_api([p], base / 'string-api/pages')
        result['string_false_public'] = {'parsed_type': type(p.is_public).__name__, 'validation':validate_corpus([p]).summary(), 'page_published': (base/'string-api/pages/false-string.json').exists()}
        def page(slug, public, parents=()):
            entity=OntologyEntity(iri='urn:ngm:class:'+slug,label=slug,entity_type='Class',domain='ai',definition='Synthetic fixture',sub_class_of=[WikilinkRef('urn:ngm:class:'+x,x) for x in parents])
            return PageData(path=base/(slug+'.md'),page_iri='urn:test:page:'+slug,slug=slug,title=slug,is_public=public,schema_version=2,ontology_class=entity)
        fixture=[page('public-child',True,['private-parent']),page('private-parent',False,['private-grandparent']),page('private-grandparent',False)]
        if repo in ('logseq', 'visionGraph'):
            from pipeline.reason import compute_closure, emit_inferred_ttl
            from pipeline.scaffold_index import emit_scaffold_index
            closure=compute_closure(fixture)
            build_page_api(fixture,base/'private-api/pages',closure=closure)
            api=json.loads((base/'private-api/pages/public-child.json').read_text())
            scaffold=emit_scaffold_index(fixture,closure,{},base/'scaffold.json')
            emit_inferred_ttl(fixture,closure,base/'inferred.ttl')
            result['private_ancestor_probe']={
                'public_page_inferred_superclasses':api.get('inferredSuperClasses',api.get('inferredSuperclasses')),
                'public_page_keys':list(api),
                'private_grandparent_present_in_public_page': 'private-grandparent' in json.dumps(api),
                'scaffold_inferred_ancestors':scaffold['classes']['public-child']['isup'],
                'private_ancestor_present_in_inferred_turtle':'private-grandparent' in (base/'inferred.ttl').read_text(),
                'private_page_published':(base/'private-api/pages/private-parent.json').exists(),
            }
            from pipeline.jsonld_to_turtle import _iri_to_uriref
            result['iri_representation']={'asserted_turtle':str(_iri_to_uriref('urn:ngm:class:public-child')),'closure_iri':closure.iris['public-child']}
        else:
            from pipeline.build import build
            from rdflib import Graph
            out=base/'full-build'
            log=io.StringIO()
            with contextlib.redirect_stdout(log):
                report=build(corpus,out)
            graph=Graph().parse(out/'data/ontology.ttl',format='turtle')
            result['full_build']={
                'validation':report.summary(),
                'stats':json.loads((out/'data/graph/stats.json').read_text()),
                'rdf_triples':len(graph),'rdf_named_individuals':len(set(graph.subjects(RDF.type,OWL.NamedIndividual))),
                'rdf_named_individual_iris':sorted(str(x) for x in graph.subjects(RDF.type,OWL.NamedIndividual)),
                'rdf_classes':len(set(graph.subjects(RDF.type,OWL.Class))),
                'scaffold_index_emitted':(out/'data/scaffold-index.json').exists(),
                'inferred_turtle_emitted':(out/'data/ontology-inferred.ttl').exists(),
            }
    return result


def run(args,cwd):
    r=subprocess.run(args,cwd=cwd,capture_output=True,text=True,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'},timeout=180)
    return {'command':args,'cwd':str(cwd),'exit_code':r.returncode,'stdout':r.stdout,'stderr':r.stderr}

if len(sys.argv)>1:
    print(json.dumps(worker(sys.argv[1])))
else:
    receipt={'captured_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Local source pipelines, existing tests and synthetic publication-boundary fixtures; no production or credential reads.','repositories':{},'workers':[],'tests':[],'source_sha256':{}}
    for repo in ['knowledgeGraph','logseq']:
        root=WORKSPACE/repo
        receipt['repositories'][repo]={
            'head':run(['git','rev-parse','HEAD'],root)['stdout'].strip(),
            'tracked_changes':run(['git','status','--porcelain','--untracked-files=no'],root)['stdout'].splitlines(),
        }
        r=run([str(PYTHON),'-B',str(Path(__file__).resolve()),repo],root)
        if r['exit_code']==0:
            r['result']=json.loads(r.pop('stdout'))
        receipt['workers'].append(r)
        receipt['tests'].append(run([str(PYTHON),'-B','-m','pytest','pipeline/tests','-q','-p','no:cacheprovider'],root))
        for p in sorted((root/'pipeline').glob('*.py')):
            receipt['source_sha256'][str(p.relative_to(WORKSPACE))]=hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((root/'.github/workflows').glob('*.yml')):
            receipt['source_sha256'][str(p.relative_to(WORKSPACE))]=hashlib.sha256(p.read_bytes()).hexdigest()
    (HERE/'knowledge-snapshot.json').write_text(json.dumps(receipt,indent=2)+'\n')
    for r in receipt['workers']:
        print('Pipeline probe:',r['cwd'],'exit',r['exit_code'])
    for r in receipt['tests']:
        print('Tests:',r['cwd'],'exit',r['exit_code'],r['stdout'][-150:])
