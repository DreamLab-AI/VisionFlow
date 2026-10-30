| # | Verdict | Decisive supplied output and scope |
|---|---|---|
| 1 | **REFUTED** | `Activation: staged … installed-Obsidian acceptance, verification item 5, is not recorded`. OM-003 was not fully closed under the stated rule. The revised census corrects this. |
| 2 | **INCONCLUSIVE** | `workers-deploy.yml:319: for required in PRF_SERVER_SECRET ADMIN_PUBKEYS; do`. Deployment does check required secrets, but the excerpt does not show whether they are missing in production or whether the failed manual workflow blocks deployment. Removing the blocker assertion was justified. |
| 3 | **CONFIRMED** | The listed changed paths are exclusively under `docs/`, including `docs/adr/README.md` and `docs/adr/ADR-2154-pele-renders-mermaid-in-the-image.md`. This supports the documentation-only commit-range claim; it does not independently verify the running image. |
| 4 | **CONFIRMED** | `return process.env.VAULT_BIN || 'vault';`, `runVault(['find', …])`, and argument entries `'edit', page`, `'create', file`, `'apply', file`. The four legacy verb references remain. The excerpt does not independently establish runtime execution or the absence of a `VAULT_BIN` override. |
| 5 | **CONFIRMED** | `jq: parse error: Invalid literal at line 35, column 145`, `exit=5`. This establishes the parser’s first reported failure, not its cause or hash-chain integrity. |
| 6 | **CONFIRMED** | Package/version pairs: `nostr-bbs-core` → `1.0.0-beta.17`; `solid-pod-rs` → `0.5.0-alpha.17`; `crypto-bigint` → `0.5.5`; `k256` → `0.13.4`. No additional versions appear in the supplied output. |
| 7 | **CONFIRMED** | `"artifact_expires": "2027-01-11T10:57:32Z"`. This is the pinned **artifact’s recorded expiry**, not a demonstrated browser credential expiry. |
| 8 | **INCONCLUSIVE** | The excerpt ends at `CORPUS PROVENANCE`. With truncation acknowledged, missing attribution in the displayed portion cannot establish its absence from the complete NOTICE. |
| 9 | **INCONCLUSIVE** | `**Heads:** VisionClaw 7c953a40b · agentbox 6500582df … VisionFlow 4d361b5`, beneath `Landing 2026-10-30`. This confirms a dated, explicitly scoped snapshot; it neither reconciles the different gate SHAs nor verifies the ~660-commit/four-reader coverage. |
| 10 | **INCONCLUSIVE** | `174` confirms 174 lines matching the supplied regex. It does not establish 174 distinct parked records or that every row contains a file-level gap and reopening trigger. |

The landing provides useful, bounded evidence; it does not establish estate-wide acceptance or production security.
The corrected census and six-ledger/355-ADR gate scope resolve those reporting failures; the deployment-blocker assertion is also removed.
Remaining **FAIL:** ADR-2156 lacks the rebuild invocation and substitutes a three-crate rebuild for the prescribed single-file measurement.
Remaining **FAIL:** ADR-2025’s “once” claim exceeds what `grep -c -a` proves; change it to “one matching line” or supply occurrence-count evidence.
OM-003’s owner-ratification claim remains unsubstantiated; qualify it or provide owner-decision evidence.
LAN-IP redaction is resolved at HEAD as reported; historical exposure remains.
