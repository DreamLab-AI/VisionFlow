#!/usr/bin/env python3
"""Run selected unchanged classes/methods without importing NumPy/SciPy."""
import ast
import dataclasses
import enum
import hashlib
import json
import logging
from pathlib import Path
from typing import List, Optional
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[4] / 'RuView'
BASE = ROOT / 'v1/src/sensing'
namespace = dict(dataclass=dataclasses.dataclass, field=dataclasses.field, Enum=enum.Enum,
                 List=List, Optional=Optional, logger=logging.getLogger('probe'))
def load_nodes(file, names):
    tree = ast.parse((BASE / file).read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in names]
    assert len(nodes) == len(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(BASE / file), 'exec'), namespace)

load_nodes('feature_extractor.py', {'RssiFeatures'})
load_nodes('classifier.py', {'MotionLevel', 'SensingResult', 'PresenceClassifier'})
features = namespace['RssiFeatures']; classifier = namespace['PresenceClassifier']()
empty = classifier.classify(features())
assert empty.presence_detected is False and empty.confidence == 1.0
boundary = classifier.classify(features(variance=0.5, motion_band_power=0.1))
assert boundary.motion_level.value == 'active' and boundary.confidence == 1.0
# Extract actual methods: early-return path and latest-sample-relative trimming only.
tree = ast.parse((BASE / 'feature_extractor.py').read_text())
cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'RssiFeatureExtractor')
methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in {'extract','_trim_to_window'}]
# Postpone annotations so no numerical import is needed for unused later paths.
module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), *methods], type_ignores=[])
exec(compile(ast.fix_missing_locations(module), 'actual-extractor-methods', 'exec'), namespace)
short = namespace['extract'](SimpleNamespace(), [])
assert short.n_samples == 0 and classifier.classify(short).confidence == 1.0
old_samples = [SimpleNamespace(timestamp=x) for x in [1,2,3,4]]
trimmed = namespace['_trim_to_window'](SimpleNamespace(_window_seconds=30), old_samples)
assert trimmed == old_samples
files = ['v1/src/sensing/'+x for x in ['backend.py','classifier.py','feature_extractor.py','rssi_collector.py','ws_server.py']]+['v1/src/cli.py','v1/setup.py','v1/tests/unit/test_sensing.py']
tests = ast.parse((ROOT / 'v1/tests/unit/test_sensing.py').read_text())
result = {'scope':'Four isolated assertions on unchanged AST-selected classes/methods; no NumPy/SciPy execution, live collector, complete suite or accuracy validation',
          'assertions_passed':4, 'empty_result':{'presence':empty.presence_detected,'confidence':empty.confidence},
          'threshold_result':{'motion':boundary.motion_level.value,'confidence':boundary.confidence},
          'old_sample_timestamps_retained':[s.timestamp for s in trimmed],
          'declared_test_methods':sum(isinstance(n,ast.FunctionDef) and n.name.startswith('test_') for n in ast.walk(tests)),
          'commodity_proof_directory_exists':(ROOT/'v1/data/proof/commodity').is_dir(),
          'sources':{f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}}
output = Path(__file__).with_suffix('.json')
output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k != 'sources'}))
