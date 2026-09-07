#!/usr/bin/env python3
"""Render the evidence-bound current board; preserve the historical closure table."""
from collections import Counter
import json
from pathlib import Path

here = Path(__file__).resolve().parent
vf = here.parents[3]
data = json.loads((here / 'master-dispositions.json').read_text())
rows = data['items']
assert len(rows) == 61 and len({r['id'] for r in rows}) == 61
assert all(r['status'] in {'closed', 'blocked', 'in-progress', 'excluded'} for r in rows)
counts = Counter(r['status'] for r in rows)
base = '../../VisionFlow/docs/estate-review/'
lines = ['# Unified TODO — VisionClaw and Agentbox estate', '',
    '**Reconciled:** 2026-09-07. ' + ('Execution recorded.' if data['final'] else 'Execution in progress; pending checks remain explicit.'), '',
    f"The execution-start register contained **61 distinct open IDs**. Current dispositions: **{counts['closed']} closed, {counts['blocked']} blocked, {counts['in-progress']} in progress, {counts['excluded']} excluded**. Paired rows are counted individually. These figures are generated from the [disposition receipt](" + base + 'closeout/execution-2026-09-07/master-dispositions.json).', '',
    '**Governed by:** [PRD-024](archive/prd/PRD-024-final-mile-closeout.md), [ADR-133](archive/adr/ADR-133-final-mile-sprint.md), and the [estate sprint](' + base + 'closeout/2026-09-07-sprint.md). The [execution report](' + base + '2026-09-07-estate-closeout-execution.md) connects repository changes, tests, browser receipts and remaining system boundaries.', '',
    'A source closure fulfils the named implementation task; it does not assert a running deployment or complete system journey. A blocked row names the concrete acceptance problem. Excluded profiles remain disabled or outside the selected scope. Historical resolved items and frozen programmes are not silently reopened or counted as new completions.', '',
    'The [pre-execution board](' + base + 'closeout/execution-2026-09-07/master-before.md) preserves all earlier wording, counts and evidence. ADR IDs are always repository-local; see the [status contract](../../VisionFlow/docs/architecture/adr-status-contract.md).']
for title, statuses in [('Remaining work and concrete problems', {'blocked', 'in-progress'}), ('Removed as resolved in this execution', {'closed'}), ('Explicit scope exclusions', {'excluded'})]:
    lines += ['', '## ' + title, '', '| ID | State | Evidence and remaining boundary |', '|---|---|---|']
    for row in rows:
        if row['status'] not in statuses:
            continue
        report = base + 'closeout/2026-09-07-execution-' + row['lane'] + '.md'
        lines.append(f"| {row['id']} | {row['status']} | {row['result']} [Evidence]({report}). |")
lines += ['', '## Added XR visual scope', '',
    'The user extended this execution to the Godot XR visual experience. [ADR-2107](adr/ADR-2107-compatible-xr-visual-experience.md) records the implemented scene, material, HUD, menu and comfort upgrades; the [visual report](' + base + 'closeout/2026-09-07-xr-visual-upgrade.md) links actual Godot captures and tests. This extension does not inflate the original 61-item count. Headset and Android acceptance remain under L-5/X-6 and the frozen target boundary.', '',
    '## Frozen and historical scope', '',
    'The ADR-073–085 window, ADR-122/123, RVF file-store proposal and XR APK programme retain their recorded frozen/optional boundaries. The standalone WasmVOWL demo remains held: [current merged-source failures](' + base + 'closeout/2026-09-07-execution-canon.md#wasmvowl-remote-integration). Public publisher explorer success does not certify that variant. RuView is explicitly excluded by user instruction.', '',
    '## Earlier resolved records (historical evidence)', '']
snapshot = (here / 'master-before.md').read_text()
history = snapshot.split('## Removed as resolved (with evidence)', 1)[1].rsplit('````', 1)[0].strip()
lines.append(history)
(vf.parent / 'project/docs/TODO-unified.md').write_text('\n'.join(lines) + '\n')
print(dict(counts))
