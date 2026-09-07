# External GitHub Action source identity

The 2026-09-07 closeout resolves every external action reference in the declared workflow checkouts to a full 40-character commit SHA. Local relative actions keep their repository identity. Reusable workflow subpaths keep their path and receive the same repository commit identity.

[The resolution receipt](../estate-review/evidence/execution-2026-09-07/action-pins.json) records the original reference and GitHub-resolved commit. [The validation receipt](../estate-review/evidence/execution-2026-09-07/workflow-validation.json) records 299 external uses, zero mutable references and zero YAML parse errors. Changing an action requires resolving the intended release to a new commit, reviewing the change and rerunning the consuming workflow. Pinning makes source selection immutable; it does not attest that the action is vulnerability-free or that every hosted workflow succeeded.

The Agentbox image scanner referred to nonexistent `aquasecurity/trivy-action@0.28.0`. Its intended release exists as `v0.28.0`; that release is now pinned to `915b19bbe73b92a6cf82a1bc12b087c9a19a5fe2`. This is recorded separately from the ordinary tag-to-hash resolutions.
