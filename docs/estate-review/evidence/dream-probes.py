#!/usr/bin/env python3
"""Compile the actual verdict module in a temporary local harness; no services."""
import hashlib, json, pathlib, subprocess, tempfile
ROOT = pathlib.Path(__file__).resolve().parents[4]
ENGINE = ROOT / 'project/agentbox/services/dream-engine'
FILES = ['engine.rs','dispatch.rs','verdict.rs','persist.rs','witness.rs','compile.rs','config.rs','main.rs','context.rs','llm.rs','ruvector.rs']
with tempfile.TemporaryDirectory(prefix='estate-dream-') as td:
    td = pathlib.Path(td)
    source = '#[path = ' + json.dumps(str(ENGINE / 'src/verdict.rs')) + '] mod verdict;\nfn main() {\n'
    cases = {
        'explicit_accept_after_failure': 'Evaluator: FAIL\nVERDICT: ACCEPT',
        'all_blocked_with_accept': 'All evaluators BLOCKED\nVERDICT: ACCEPT',
        'negated_fallback': 'No evidence justifies ACCEPT',
        'explicit_inconclusive': 'ACCEPT considered\nVERDICT: INCONCLUSIVE',
        'malformed_explicit_fallback': 'VERDICT: unknown\nExample: ACCEPT',
    }
    for name, report in cases.items():
        source += 'println!("' + name + '={}", verdict::parse_verdict(' + json.dumps(report) + ').as_str());\n'
    source += '}\n'
    (td/'probe.rs').write_text(source)
    compile_result = subprocess.run(['rustc','--edition=2021','-Awarnings',str(td/'probe.rs'),'-o',str(td/'probe')],capture_output=True,text=True)
    compile_result.check_returncode()
    run = subprocess.run([str(td/'probe')],capture_output=True,text=True,check=True)
    print(json.dumps({'scope':'actual verdict module; synthetic reports; no LLM, SSH, push or memory write', 'source_sha256':{f:hashlib.sha256((ENGINE/'src'/f).read_bytes()).hexdigest() for f in FILES},'reports':cases,'result':dict(line.split('=',1) for line in run.stdout.splitlines())},indent=2))
