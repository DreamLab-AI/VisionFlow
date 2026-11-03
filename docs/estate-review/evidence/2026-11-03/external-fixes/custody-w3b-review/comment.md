## 🔎 deepsec found 12 findings

🔴 CRITICAL ×1 · 🟡 MEDIUM ×8 · 🟣 BUG ×3

<sub>scope: `git-diff:6dbaaddc8` · run `20261103065817-0d845aa3ac31e292`</sub>

### 🔴 CRITICAL · `config/nostr-gateway/gateway.cjs:L687-L718`

**Commander gate trusts unauthenticated rumor.pubkey from bare nip59.unwrapEvent → RCE into tab 0**
<sub>slug: `auth-bypass` · confidence: high</sub>

handleWrap() unwraps the inbound gift wrap with bare `tools.nip59.unwrapEvent(wrap, sk)` (L687) and then authorises the sender with `String(rumor.pubkey).toLowerCase() !== commanderPub` (L691). The entire fleet-control / RCE gate (documented at L18-20) rests on this check. But bare `nip59.unwrapEvent` does NOT authenticate the sender: the repo's own authenticated unwrap in management-api/lib/zone-keys.js (L413-439) exists precisely because `nip59.unwrapEvent` neither verifies the kind-13 seal signature nor enforces `rumor.pubkey === seal.pubkey` (see also the comment at junkiejarvis-agent.js…

**Recommendation:** Never trust rumor.pubkey from bare nip59.unwrapEvent. Replace the unwrap with an authenticated one that verifies the seal's BIP-340 signature and asserts rumor.pubkey === seal.pubkey (mirror zone-keys.js unwrapAny: decrypt wrap→seal, verifyEvent(seal), decrypt seal→rumor, reject unless rumor.pubkey === seal.pubkey), and gate on the authenticated seal author, not the inner rumor field. Apply the s…

---

### 🟡 MEDIUM · `config/nostr-gateway/gateway.cjs:L663-L680`

**notifyEnquiry forwards attacker-controlled content to the operator's phone with a trusted 'website enquiry' frame**
<sub>slug: `other-unauthenticated-notification-injection` · confidence: medium</sub>

notifyEnquiry (L663-682) runs BEFORE the commander check and has, by design, no sender authentication and no freshness gate. Any party able to deliver an operator-addressed gift wrap (ephemeral key, as website forms do) whose content matches `"type": "contact_signup"`/`contact_enquiry` or whose subject starts 'DreamLab website' gets arbitrary attacker-chosen name/email/message fields rendered to the operator's phone as a trusted '🔔 website enquiry/signup' notification (L680). This is a phishing / social-engineering channel into the operator (e.g. an injected 'enquiry' asking john to whitelis…

**Recommendation:** Authenticate the seal author before forwarding (even if from an allowlisted set of website signup keys), label clearly that the payload is unauthenticated, and bound the executed store by size in addition to age.

---

### 🟡 MEDIUM · `management-api/lib/junkiejarvis-agent.js:L1146-L173`

**Asker-steerable event zone (confused deputy) — member can publish JJ-signed events into a zone they lack write access to**
<sub>slug: `acl-check` · confidence: low</sub>

The zone a JJ-signed calendar event lands in is chosen from asker-controlled input: in the channel path channelZone() reads the `section` tag off the sender's own event (L173-177) and L1146 overrides spec.zone with it; in the DM path the LLM picks the zone from user text (normaliseEventDirective, L246-247, default 'friends'). JJ signs with its own identity, which is privileged across zones. A member confined to 'friends' can thus induce JJ to publish an event into 'business' or 'family' simply by tagging/asking for it — a confused-deputy privilege escalation if the relay write-gates zone writ…

**Recommendation:** Constrain the event zone to what the requesting member is actually authorised to write (verify against the member's zone membership), rather than echoing the asker-supplied section tag or the LLM's choice.

---

### 🟡 MEDIUM · `management-api/lib/junkiejarvis-agent.js:L1147-L1165`

**Calendar events bypass zone write-plan — plaintext title/zone published into encrypted zones**
<sub>slug: `other-invariant-violation` · confidence: medium</sub>

_createEvent (L1162-1175) builds the kind-31923 event and publishes it directly via `this.bridge.publish(unsigned, this.signer)` with no call to zoneKeys.writePlan / applyWritePlan / sealChannel — unlike _sendChannelReply (L1074-1091), which enforces the write plan and refuses/encrypts per zone. The module invariant (L22-28) states that a reply into an encrypted zone is encrypted to the zone key or the agent stays silent. A create_event directive produced from a message decrypted out of an E2E zone (friends/family/business) therefore leaks the event title and zone to the relay in cleartext, c…

**Recommendation:** Route calendar-event publication through the same zone write-plan path as channel replies: resolve the target zone, and for an encrypted zone either seal the event or refuse, never publish plaintext.

---

### 🟡 MEDIUM · `management-api/lib/junkiejarvis-agent.js:L695-L702`

**Exported unwrapDmRumor() returns an unauthenticated rumor whose pubkey must not be trusted**
<sub>slug: `other-unauthenticated-unwrap-helper` · confidence: medium</sub>

unwrapDmRumor (L695-706) uses bare `nip59.unwrapEvent(wrap, skBytes)` and returns the rumor on kind-14 match. Unlike the agent's own _handleDm path (which uses the authenticated zoneKeys.unwrapAny, L867), this exported helper performs no seal-signature verification and no rumor.pubkey===seal.pubkey binding, so `rumor.pubkey` is attacker-forgeable. It is exported for the nightly forum-suggestions tenant to read clarification replies; any caller that trusts rumor.pubkey for authorisation (as gateway.cjs does) inherits the critical auth bypass. The helper is a latent footgun because its return s…

**Recommendation:** Either route this helper through zoneKeys.unwrapAny so the returned sender is authenticated, or rename/annotate it to make unambiguous that rumor.pubkey is NOT authenticated and must never be used for an authorization decision. Audit scripts/dream-forum-suggestions.mjs for a trust decision on the returned pubkey.

---

### 🟡 MEDIUM · `management-api/lib/junkiejarvis-agent.js:L894-L1041`

**No per-sender rate limiting on LLM calls from channel mentions / DMs**
<sub>slug: `expensive-api-abuse` · confidence: low</sub>

Every @junkiejarvis channel mention (L1041) and every DM (L894) triggers a paid LLM completion via _think→callLlm with no per-sender rate limit or quota; dedup only prevents re-processing the same event id. The SYSTEM_PROMPT states JJ also serves anonymous dreamlab-ai.com website visitors, so the LLM-spend surface is reachable by unauthenticated parties who can address JJ. A sender can drive sustained paid-API spend by posting many distinct mentions/DMs. Fail-open design bounds crashes but not cost.

**Recommendation:** Add a per-pubkey token-bucket / cooldown before dispatching an LLM call, and a global ceiling, with a cheap canned reply when exceeded.

---

### 🟡 MEDIUM · `scripts/dream-forum-suggestions.mjs:L103-L500`

**Untrusted forum post steers LLM reply published under JunkieJarvis signing identity**
<sub>slug: `other-prompt-injection` · confidence: medium</sub>

Each candidate forum post's raw `content` is passed straight into the triage LLM (triage(), lines 98-125) as the user message. The model's returned JSON `reply` field is then published verbatim as a signed kind-42 event under the JunkieJarvis key (lines 485-501), posted publicly in the forum thread. A forum user controls the input to this pipeline, so a crafted suggestion can prompt-inject the triage brain and cause JunkieJarvis to publish attacker-influenced text (e.g. social-engineering directed at other members such as 'ask john to run X', or messages that undermine the bot's stated policy…

**Recommendation:** Treat the published reply as fully untrusted: constrain it to a small set of canned templates keyed off `verdict.decision` rather than free-form LLM text, or at minimum strip/escape control content, reject replies containing URLs/@-mentions/instructions, and keep the hard 280-char cap. Consider requiring human review before any reply is posted under the shared identity.

---

### 🟡 MEDIUM · `services/dream-engine/src/governance.rs:L859-L1105`

**Operator decisions are authorized only by the relay ACL, not by an admin allowlist in the engine**
<sub>slug: `acl-check` · confidence: medium</sub>

`ingest()` fetches kind-31403 decision events and `plan_decisions()` acts on them to resolve inbox items (approve/reject/amend), whose Amend text then becomes 'guidance for this repository's next night'. The engine's only authorization checks are: event kind == KIND_ACTION_RESPONSE, the signer is not the agent itself (`!e.pubkey.eq_ignore_ascii_case(agent_pubkey)`), and a valid BIP-340 signature (`relay::verify`). There is NO admin-pubkey allowlist — the module doc openly delegates 'only admins may decide' to the relay ('the relay already admits 31403 only from admins'). Per the review's own…

**Recommendation:** Maintain an explicit admin/operator pubkey allowlist (or verify NIP-26 delegation / panel-policy membership from signed configuration) inside `plan_decisions`, and reject any 31403 whose signer is not on it. Add `authors` to the ingest REQ as defence in depth. Require the `e` binding to be present and match `published_event_id` rather than treating a missing binding as valid. For the panel-level…

---

### 🟡 MEDIUM · `services/nostr-pod-bridge/src/identity_port/port.rs:L549-L570`

**NIP-98 allowlist check and signed `u` tag use different URL parsers, allowing sovereign-signed auth headers for off-allowlist hosts**
<sub>slug: `other-parser-differential` · confidence: medium</sub>

op_nip98 validates the request URL against the key's NIP-98 allowlist using the WHATWG-parsed form (`Url::parse(req.url.trim())` at line 549, origin/path check via `url_allowed` at line 555), but the `u` tag that actually gets signed is built from the RAW request string, merely truncated at the first `?`/`#` (lines 569-570). The two forms can disagree. WHATWG URL parsing (the `url` crate) treats backslash as a path separator for special schemes and strips internal tab/CR/LF, while an RFC 3986 verifier does neither. Example: `https://pod.example\@evil.example/x` parses (WHATWG) to host `pod.ex…

**Recommendation:** Make the validated form and the signed form byte-identical. After parsing and passing the allowlist, derive the `u` tag from the parsed URL's canonical serialization with query and fragment cleared (`url.as_str()` after `set_query(None)`/`set_fragment(None)`), OR reject the request when the raw cut differs from that canonical serialization. Additionally reject any raw URL containing `\`, tab, CR…

---

### 🟣 BUG · `management-api/lib/junkiejarvis-agent.js:L405`

**Calendar event d-tag collisions let any member overwrite another's (or seeded) events**
<sub>slug: `other-logic-bug` · confidence: medium</sub>

buildCalendarEvent (L405) derives the addressable `d` tag as `${zone}-${slug}` where slug is the lowercased, space-collapsed, non-alphanumeric-stripped title. Kind-31923 is a NIP-52 parameterised-replaceable event keyed by (kind, author, d-tag), and every event here shares the single JJ author. Two members who request events with the same title in the same zone therefore produce the same d-tag, and the later publish silently replaces the earlier event (last-write-wins). A title containing no alphanumerics slugs to `${zone}-`, so all such events collide into one. If any seeded zone events were…

**Recommendation:** Make the d-tag unique per event (append a minted URN / random suffix / creation timestamp) so replaceable-event semantics cannot clobber distinct events.

---

### 🟣 BUG · `services/dream-engine/src/digest.rs:L116-L126`

**ledger_rows_for splits on raw '|', shifting columns when a finding contains an escaped pipe**
<sub>slug: `other-parser-divergence` · confidence: high</sub>

`ledger_rows_for` (line 121) splits each LEDGER.md row on every raw `|` with no handling of escaped pipes. The sibling parser `governance::ledger_cells` deliberately treats `\|` as a literal cell character (and a governance test around line 1338 confirms real findings contain escaped pipes, e.g. 'generator parity a \| b check'). When a finding cell contains one escaped pipe, every subsequent column in the digest parser shifts right by one: `verdict` (index 7) and `witness` (index 9) read the wrong cells. The nightly digest then prints an incorrect finding text and a wrong/empty 'Evidence fing…

**Recommendation:** Reuse governance's escaping-aware cell splitter (replace `\|` with a sentinel before splitting, restore afterwards) in `ledger_rows_for` so both parsers read identical columns from the same row.

---

### 🟣 BUG · `services/dream-engine/src/governance.rs:L276-L279`

**case_context hardcodes LEDGER.md path, ignoring the repo's configured ledger_path**
<sub>slug: `other-logic-bug` · confidence: low</sub>

`case_context` reads the ledger from a hardcoded `docs/dream-cycle/LEDGER.md` (line ~279), whereas `digest::collect_ledgers` uses `cfg.ledger_path` from each repo's `dream.config.json`. A repo that configures a non-default ledger path gets empty finding/PR context on its governance cases (no deep/finding/pr_url rendered), silently degrading the case body even though the data exists.

**Recommendation:** Load `dream.config.json` for the repo and use `cfg.ledger_path` (as `collect_ledgers` does) instead of the hardcoded relative path.

---

<sub>🤖 generated by [deepsec](https://github.com/vercel/deepsec)</sub>