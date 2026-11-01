# Information identity and inbox audit fixtures

The nine scenarios assert current counterexamples; they are not product fixes. `sources.json` records exact exported source hashes. The website revision is `b5eb1ecf96599ebe154ebf23bbe7c703d2e653a5`; consumed worker code is pinned at `9c36b6032640c89879ae52fe25da99493979a0b6`.

To reproduce, create an empty harness under an owned task worktree on an executable filesystem. Export these pinned website paths, retaining their directory structure:

- `scripts/inbox-watch/run.mjs`, `scripts/seed/provision-website-info.mjs`.
- `scripts/lib/{website-info,digest,enquiry-topic,jev,toml-lite}.mjs`.
- `scripts/__tests__/{website-info,inbox-digest}.test.mjs`.
- `forum-config/dreamlab.toml`, `.github/workflows/deploy.yml`, `scripts/contact-embed/entry.ts`, `public/.well-known/nostr.json` for the existing source-mirror tests.

Provide the website's compatible installed `node_modules` and an ESM `package.json` (`{"type":"module","private":true}`). Copy `probes.mjs`, `mock-io.mjs` and `vitest.config.mjs` from this evidence directory into the harness root. Ensure `/tmp/vf-dw-inbox-artifacts` exists for results. From the harness root:

```sh
node_modules/.bin/vitest run --config vitest.config.mjs
node probes.mjs
```

Use a new empty harness each time; the scripts refuse to overwrite `fixtures/`. The probe runner generates new synthetic monitor/operator/sender keys and real NIP-17 encrypted wraps. All HTTP and WebSocket traffic is replaced by `mock-io.mjs` before importing the actual runner/provisioner. An explicit child environment excludes real credentials; any unexpected admin-key read points to a nonexistent fixture file. Only fake transports receive the synthetic profile, probe and digest events.

The registration scenario keeps a separate unprivileged whitelist row but supplies configuration-only public disclosure. This isolates missing registry state without assuming configuration ingress authorises kind-0 authorship. An earlier fixture allowed that profile write without a whitelist row; its artefacts are retained separately, but the final result uses the stricter setup. A rerun setup command initially used the wrong working directory and then encountered the existing fixture directory; that harness failure is retained and is not product evidence.

The overflow scenario runs the watcher twice with the same thirty encrypted enquiries. The timestamp-tie case tests the real cursor helper against a modelled 500-of-501 page, not an actual relay. The TypeSafe scenario asserts the captured request's destination and plaintext payload; its deliberately malformed response falls back, so the scenario does not assert successful topic annotation. Rust merge behaviour is source-reviewed, not executed by these JavaScript fixtures.

`harness.json` records installed versions; no fresh install, complete build or deployed browser test occurred. Runtime summary is a separate read-only projection of the live saved status, with no message contents or secret material. Full fixture logs, generated keys and renders are retained locally under `/tmp/vf-dw-inbox-artifacts`; only the code, compact results and hashes are committed. Never run the real watch's `--dry-run` as a read-only test: it still publishes its probe.
