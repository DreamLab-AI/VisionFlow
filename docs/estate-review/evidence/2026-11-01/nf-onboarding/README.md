# Recovery and onboarding evidence

Observed 1 November 2026. Forum source is pinned by `source-manifest.json`; all fourteen exports equal the old NF-05 source revision, twelve equal the website-consumed revision. `source-differences.json` preserves the two k256 argument-conversion differences. These are source identities, not deployment attestations.

## Native reproduction

Create a disposable archive of forum commit `c39f41f104b114d6a3fa29013daaf6fa58b45080`. The preparer refuses a Git checkout and refuses to replace an existing test. From this evidence directory:

```sh
python3 prepare_tests.py --kit /path/to/disposable-forum-archive --out /path/to/generated-fixtures
AUDIT_ONBOARDING_OUTPUT=/path/to/html-output cargo test --manifest-path /path/to/disposable-forum-archive/Cargo.toml -p nostr-bbs-core --test audit_onboarding
```

The observed run reused `/home/devuser/workspace/nostr-rust-forum/target` through `CARGO_TARGET_DIR`; production source was unchanged. `native-tests.log` records 19 passing cases: five unchanged decoder tests, thirteen audit behaviour cases and one fixture-emission case. `native-first-hypothesis.log` retains a failed first run whose uppercase-nsec assumption was disproved. The final test expects rejection in both layers; the first run is not a current implementation failure.

`prepare_tests.py` extracts nine original slices from the retained exports. `extracted-source.json` preserves each complete slice and SHA-256. `audit_onboarding.rs` is the generated harness, with `fixture-prefix.rs` supplying mocks and `audit-cases.rs` supplying probes. The HTML formatter's signature changes to return the formatted string before its browser-download calls. The login slice ends before state mutation and returns the derived public key. Wrapper syntax changes allow callbacks/effects to execute as ordinary functions; source bodies are retained.

Mocked boundaries: window/location/history, replace and print results, reactive signals, parent callback, local-auth completion, error state and scheduled navigation. The actual extracted validation prefix and core scalar constructor execute. No actual AuthStore storage/signing setup, Leptos reactive runtime, browser print/download APIs, profile request, relay or timeout navigation runs in this native suite. The readiness callback models the parent's `sheet_ready.set(true)`.

## Browser evidence

`browser-results.json` records five cases in real Chrome through the browsercontainer MCP SSE interface:

1. Plain generated HTML exposes the synthetic recovery text and preserves a synthetic fragment.
2. Same-origin `replaceState` clears the fragment.
3. Deliberately cross-origin replacement throws `SecurityError` and retains it.
4. Generated HTML with a length-valid display name `<svg onload=window.auditXss=1>` executes the marker assignment.
5. A test-only escaped-name derivative shows the literal text without executing it.

`html/plain.html` and `html/marker.html` come from the formatter test. `html/escaped-control.html` is a derivative with only the display-name payload escaped using Python's `html.escape`; it is not a production repair. All credentials and URLs are synthetic. The marker makes an assignment only: no exfiltration or external request. The screenshot shows rendered synthetic recovery material; the JSON evaluation establishes marker execution, which cannot be inferred from a screenshot alone.

`browser_probe.py` and `mcp_client.py` retain the exact scripts used. They include the observed container address, port, isolated-context name and `/tmp/vf-nf-onboarding-artifacts` output path. They are execution records, not portable unattended runners. To repeat, first read the browser skill for the current MCP endpoint, serve only the synthetic `html` directory, adapt the local URL/output paths, create a fresh isolated context, and close its page after the run. Do not run against a real signup or a secret-bearing recovery file.

`http-server.log` records the local requests without fragments and the server's keyboard-interrupt shutdown. `owned-browser-page.json` records the closed page. The server's process handle was subsequently absent. No real account, message, device registration or deployment was changed.

Browser limits: the full Leptos application, Blob download and file-origin opening, real print/save, QR scanning and relay acceptance were not run. The forced cross-origin History API error is a browser-semantics control; native mocks separately demonstrate the production helper's ignored-error behaviour. It is not evidence of a currently misconfigured deployment.

## Review and validation

The companion report records section dispositions, acceptance work and primary-source comparisons. Render, citation and text-check receipts are added after validation. Automated corpus checks do not imply semantic completion of all other diagrams or ADRs.

Validation: both affected topics render 22/22 panels; all six changed panels were visually inspected. The full citation gate checks 192 topics / 1,667 panels with 25,000 verified citations, zero warnings and nine existing allowlisted artefact citations. The freshness gate records 47 declared revision labels. See `validation.json` for scope and prose-scanner dispositions.
