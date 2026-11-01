# Ontomark reasoning and certificate evidence

Source revision: `0caae5d4b695cb720942b9b9212017c00a80ad41`. See `verification.json` for whole-file source hashes and exported ranges. Full exports preserve source bytes; `.rs.txt` exports prefix selected lines with their original line number. `old-pin-runtime-diff.patch` compares the relevant changed files against OM-04's older declared revision.

Run the probes only in a disposable archive of that commit. Copy `audit_certificate.rs` into `crates/ontomark-compile/tests/` in the archive. Set `OO_CERT_BIN` to an absolute path containing the checksum-verified Linux checker from the URL recorded in `verification.json`. The audit probe requires the checker; the upstream checker tests can silently skip it when the variable is absent.

```bash
cargo test --locked --offline -p rdfs-certificate -p ontomark-reason -p ontomark-compile --lib
cargo test --locked --offline -p ontomark-reason
cargo test --locked --offline -p rdfs-certificate --test oo_cert -- --nocapture
cargo test --locked --offline -p ontomark-compile --test audit_certificate -- --nocapture
```

Offline execution requires the locked dependency cache already present. No deployed corpus or external reasoning service is used by the synthetic tests. The checker binary is a separately downloaded executable, identified by SHA-256 rather than committed into this evidence directory.

`probes-initial.log` records the constructor-name compilation error. `probes-second.log` records the earlier fixture-validation failures. `probes-third.log` is the final nine-test passing run; its assertions deliberately pin defects as well as positive controls. Baseline, reasoner and independent-checker outputs are retained separately. Test success does not mean the recorded defects are fixed.

The six changed panels were inspected after SVG-to-PNG conversion. Render logs and the verification manifest record diagram validation; generated SVGs and disposable build files are scratch artefacts rather than source changes.
