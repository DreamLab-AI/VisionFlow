"""Execute the pinned upstream interrupt handler with an injected queue hand-off.

No network/GPU/server startup. Compile only post_interrupt's unchanged AST body;
strip its routing decorator. Queue, request, response and interrupt effects are
controlled test doubles. The calling Client.cancel is the actual local source.
Usage: python media-cancel-probe.py CLIENT_PY UPSTREAM_SERVER_PY OUTPUT_JSON
"""
import ast
import asyncio
import hashlib
import importlib.util
import json
import logging
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

client_file, server_file, output = map(Path, sys.argv[1:])
spec = importlib.util.spec_from_file_location("comfy_client_audit", client_file)
client_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(client_module)
source = server_file.read_text()
parsed = ast.parse(source)
handler = next(n for n in ast.walk(parsed) if isinstance(n, ast.AsyncFunctionDef) and n.name == "post_interrupt")
original_ast = ast.dump(handler)
handler.decorator_list = []
code = compile(ast.fix_missing_locations(ast.Module(body=[handler], type_ignores=[])), str(server_file), "exec")

results = []
for scenario in ["own_still_running", "already_finished", "handoff_after_server_snapshot"]:
    running = {"id": "owned-A" if scenario != "already_finished" else "other-B"}
    interrupts = []
    posts = []

    class Queue:
        def get_current_queue(self):
            snapshot = [(1, running["id"], {}, {})]
            if scenario == "handoff_after_server_snapshot":
                running["id"] = "other-B"
            return snapshot, []

    env = {"json": json, "logging": logging,
           "self": SimpleNamespace(prompt_queue=Queue()),
           "nodes": SimpleNamespace(interrupt_processing=lambda: interrupts.append(running["id"])),
           "web": SimpleNamespace(Response=lambda **kw: kw)}
    exec(code, env)

    class Client(client_module.Client):
        def request(self, path, data=None, **kw):
            if path == "/queue":
                # Client snapshot sees its own job; the server may be later.
                return {"queue_running": [[1, "owned-A", {}, {}]], "queue_pending": []}
            assert path == "/interrupt" and data == {"prompt_id": "owned-A"}
            posts.append({"path": path, "body": data})
            class Request:
                async def json(self):
                    return data
            return asyncio.run(env["post_interrupt"](Request()))

    with tempfile.TemporaryDirectory() as td:
        receipt = Path(td) / "receipt.json"
        receipt.write_text(json.dumps({"url": "http://fixture.invalid", "prompt_id": "owned-A"}))
        result = Client("http://fixture.invalid").cancel(receipt)
        journal = json.loads(receipt.read_text())
    expected = {"own_still_running": ["owned-A"], "already_finished": [],
                "handoff_after_server_snapshot": ["other-B"]}[scenario]
    assert interrupts == expected, (scenario, interrupts)
    assert result["cancelled"] is True  # request-level signal, not terminal history
    assert journal["cancel_requested"] == "interrupt"
    results.append({"scenario": scenario, "interrupted": interrupts, "client_result": result, "posts": posts})

output.write_text(json.dumps({"upstream_revision": "faa7e494f3a0c9e45e1c3db8d20ee48fd5fe1b52",
    "server_sha256": hashlib.sha256(source.encode()).hexdigest(),
    "handler_lines": [handler.lineno, handler.end_lineno],
    "handler_ast_sha256": hashlib.sha256(original_ast.encode()).hexdigest(),
    "scope": "Exact handler AST with injected queue hand-off; actual client method; no live sidecar", "cases": results}, indent=2)+"\n")
print("PASS: owned, finished and hand-off scenarios; the hand-off interrupts other-B")
