"""Prepare the isolated Rust harness from a committed VisionClaw export.

Usage: python3 prepare.py EXPORTED_PROJECT_ROOT NEW_HARNESS_DIRECTORY
No network, server startup or live pod is used. Only the upstream puller's test
module is omitted; its production code and the complete generation module are
copied unchanged. Tests inject Fetch and use the actual filesystem store.
"""
from pathlib import Path
import shutil
import sys

source, output = map(Path, sys.argv[1:])
here = Path(__file__).resolve().parent
output.mkdir(parents=True, exist_ok=False)
(output / "src/services").mkdir(parents=True)
(output / "tests").mkdir()
for name in ["Cargo.toml", "Cargo.lock", "tests/identity.rs"]:
    shutil.copyfile(here / name, output / name)
for module in ["ontology_pull", "ontology_generation"]:
    text = (source / f"src/services/{module}.rs").read_text()
    if module == "ontology_pull":
        assert "#[cfg(test)]" in text
        text = text.split("#[cfg(test)]", 1)[0]
    (output / f"src/services/{module}.rs").write_text(text)
(output / "src/lib.rs").write_text(
    "pub mod services { pub mod ontology_generation; pub mod ontology_pull; }\n"
)
