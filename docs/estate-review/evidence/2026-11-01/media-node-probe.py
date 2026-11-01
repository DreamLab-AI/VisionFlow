"""CPU-only probes of current identity receipts and the post-build node gate.
Usage: python media-node-probe.py NODES_DIR VERIFY_SPEDE_SH OUTPUT_JSON
"""
import contextlib
import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

nodes, verify, output = map(Path, sys.argv[1:])
sys.path.insert(0, str(nodes))
from agentbox_video import identity
from agentbox_video.tests.test_identity import FakeEmbedder, frame, unit

observations = []


class Audit(unittest.TestCase):
    def test_drift_receipt_omits_two_required_fields(self):
        r = identity.drift_report([frame(1), frame(1)], [], {"a": [unit(1, 0, 0)]},
                                  FakeEmbedder({1: [(50, unit(1, 0, 0))]}), every=1)
        self.assertNotIn("provisional", r)
        self.assertNotIn("acceptance_backing", r)
        self.assertEqual(r["acceptance"], "editorial")
        self.assertFalse(r["gate"])
        observations.append({"case": "drift_receipt", "receipt": r})

    def test_calibration_claims_remain_refused(self):
        for c in [{"calibrated": True}, {"calibration": None}, {"calibration": {}}]:
            with self.subTest(config=c):
                with self.assertRaisesRegex(identity.CalibrationError, "acceptance is editorial"):
                    identity.ThresholdSet.from_dict(c)
        observations.append({"case": "calibration_claims", "refused": 3})

    def test_postbuild_gate_does_not_require_load_image(self):
        source = verify.read_text()
        # Execute only the existing second Python here-document: registration checks.
        code = source.split("python - <<'PY'\n")[2].split("\nPY\n", 1)[0]
        core = ["SAM3_Detect", "QwenImage21Cache", "TextEncodeQwenImage21", "LoadMoGeModel", "TripoSplatConditioning"]
        video = ["AgentboxIdentityScore", "AgentboxOpeningFrameChooser", "AgentboxRelayPrepare", "AgentboxRetakePrep",
                 "AgentboxTrackMask", "AgentboxLoadLatentVerified", "AgentboxSaveLatentVerified", "AgentboxSaveImageVerified"]
        full = {k: {} for k in core + video + [f"SpEdE_fixture_{n}" for n in range(30)]}
        for absent, expected in [("AgentboxLoadImageVerified", 0), ("AgentboxTrackMask", 1)]:
            info = {k: v for k, v in full.items() if k != absent}
            log = io.StringIO()
            with patch("urllib.request.urlopen", return_value=io.BytesIO(json.dumps(info).encode())), contextlib.redirect_stdout(log):
                with self.assertRaises(SystemExit) as caught:
                    exec(compile(code, str(verify), "exec"), {})
            self.assertEqual(caught.exception.code, expected)
            observations.append({"case": "postbuild_registration", "absent": absent, "exit": expected,
                                 "log": log.getvalue(), "scope": "exact registration snippet, synthetic object_info"})


suite = unittest.defaultTestLoader.loadTestsFromTestCase(Audit)
result = unittest.TextTestRunner(verbosity=2).run(suite)
output.write_text(json.dumps(observations, indent=2)+"\n")
sys.exit(0 if result.wasSuccessful() else 1)
