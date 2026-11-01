# Show-and-tell script

**Working title:** The next shift starts with the evidence. **Estimated runtime:** 07:08. Exact spoken lines below; presenter directions are not narration. Read at a natural pace, around 125–140 words per minute, and leave time to inspect the evidence. Final timing follows measured speech and footage.

**Opening disclosure:** “A hypothetical industrial workflow. Synthetic data. Staged integrations are labelled.” Keep the appropriate capture/concept/illustration label visible.

This is a proposed application story. “Software-capture-candidate” means source-backed functionality is available to investigate, not that this shot has been recorded or that the industrial workflow has passed. See [production guide](production-guide.md) and [research](research.md).

## S01 · 00:00–00:16 · A small change on a busy shift

**Picture:** Slow shoulder-height move along a quiet industrial aisle towards one blue pump skid. No sparks, emergency lights or people touching machinery.

**Narrator:** At Northbank Packaging, the next shift is about to begin. One cooling pump sounds different. Before anyone plans more work, the team needs to understand what has changed.

**Presenter cue:** Open with the operational problem. Do not introduce the repository names yet.

**Material:** H01 · generated-illustration. **On-screen copy:** The next shift starts with the evidence | Hypothetical workflow · synthetic data

## S02 · 00:16–00:39 · The evidence is scattered

**Picture:** Four readable document cards on a neutral desk-like canvas. Bring the two revision cards into the same frame; keep their dates visible.

**Narrator:** The clues are spread across a technician’s observation, an earlier work record, two versions of a maintenance reference and a trend. Finding the files is only the start. Someone must check that they concern the same asset, and decide which version applies now.

**Presenter cue:** Let the viewer read the revision letters before the next cut.

**Material:** G01 · editorial-concept. **On-screen copy:** One asset. Several records. Two document revisions.

## S03 · 00:39–01:07 · The technician reports an observation

**Picture:** Over-shoulder tablet framing, then a large transcript card. Avoid a recognisable commercial UI. Keep Maya away from moving equipment.

**Narrator:** Maya records what she has noticed. In this proposed application, her spoken report becomes an incident draft she can correct before saving. The report is an observation; it does not become a diagnosis simply because an assistant transcribed it.

**Maya (performed line):** Pump P two-oh-four sounds different again. Can we bring the records together before planning more work?

**Presenter cue:** Pause after the character line; invite the audience to notice the distinction between observation and conclusion.

**Material:** G02 · staged-application. **On-screen copy:** Application concept · staged | Reported — cause unconfirmed

## S04 · 01:07–01:30 · Give the records a common subject

**Picture:** Build a small left-to-right relationship view. Maximum seven visible nodes. Distinguish the pump, system and documents by shape as well as colour.

**Narrator:** The first connection is the asset itself: P two-oh-four, part of cooling loop two, serving packaging line two. Each record points to that same subject. This lets the team follow the evidence without relying on similar filenames or somebody remembering where a document lives.

**Presenter cue:** Say what the connections mean; avoid flying through a large unreadable graph.

**Material:** G03 · editorial-concept. **On-screen copy:** P-204 → Cooling loop 2 → Packaging Line 2

## S05 · 01:30–01:56 · Make the source inspectable

**Picture:** Record a small demo Markdown note beside its source references, then an actual validation result. Crop to the content being discussed.

**Narrator:** The foundation is editable knowledge with explicit references. Ontomark supplies tools to validate and build that material. For this pilot, the asset vocabulary and source links would be prepared deliberately. A successful structure check helps keep the model coherent; the engineer still has to judge the evidence.

**Presenter cue:** Keep the technical insert short. The point is inspectability, not CLI fluency.

**Material:** CAP01 · software-capture-candidate. **On-screen copy:** Software capture · synthetic data — only after capture acceptance

## S06 · 01:56–02:18 · Find the current reference

**Picture:** Side-by-side B and C cards. Animate one supersedes arrow from C to B. Keep the earlier work record linked to B.

**Narrator:** Here is the detail that changes the discussion. The earlier work record cites revision B. Revision C was issued later and is now the current reference in our scenario. The old record remains useful history. The next job needs the current reference.

**Presenter cue:** This is the story’s main reveal. Allow a full reading pause.

**Material:** G04 · staged-application. **On-screen copy:** B: historical reference | C: current in this scenario

## S07 · 02:18–02:42 · Explore the relationship, not a guess

**Picture:** Use VisionClaw’s real graph surface only after loading a verified demo mapping. Fixed camera; reveal one edge at a time.

**Narrator:** A graph view makes that relationship easier to inspect. Follow the pump to the previous work, then to the reference it used. Bring the newer reference alongside it. The picture helps people ask a better question; the links themselves still need owners, sources and review.

**Presenter cue:** Give the viewer a moment to follow the old-reference path.

**Material:** CAP02 · software-capture-candidate. **On-screen copy:** Software capture · synthetic data — after verification | Relations, not a physical simulation

## S08 · 02:42–02:56 · From scattered papers to a shared view

**Picture:** Slow move from the same pump housing to a nearby meeting area. Match-cut the circular housing to the selected graph node in the edit.

**Narrator:** The aim is a shared view of the situation: what happened, what the records say, and where the team still needs an answer.

**Presenter cue:** Leave a breath before the question-and-answer sequence.

**Material:** H02 · generated-illustration. **On-screen copy:** Illustrative GenAI scene

## S09 · 02:56–03:22 · Ask a bounded question

**Picture:** Start with the exact Q1 text, then show a real recorded retrieval response if available. Use an editorial annotation beside the response, never inside it.

**Narrator:** Sam asks what changed around P two-oh-four, and which records deserve attention. Loom provides retrieval and grounding information that an application can use. Our proposed screen puts the answer beside its sources, so Sam can open the earlier work record and the newer reference rather than accepting a summary on trust.

**Presenter cue:** If presenting live, ask the question once and leave the actual response visible.

**Material:** CAP03 · software-capture-candidate. **On-screen copy:** What changed around P-204, and which records should we review?

## S10 · 03:22–03:47 · Keep the unanswered question visible

**Picture:** One clear question and one short answer. Retain the two evidence references underneath; no confidence gauge.

**Narrator:** Then comes the harder question: did the last seal replacement cause the new problem? These records do not establish that. The proposed application should say so and keep the question open. An observation after a maintenance event is a reason to investigate, not proof of what caused it.

**Presenter cue:** Do not replace this moment with a confident diagnosis to make the demo look smoother.

**Material:** G05 · staged-application. **On-screen copy:** Cause not established by the available records

## S11 · 03:47–04:13 · Review the exact knowledge change

**Picture:** Show the exact NOTE-204 before/after text as a readable diff. A real proposal surface may be used only when it displays the same scope and content.

**Narrator:** Sam reviews a narrow correction to the knowledge note: retain the old reference as history, identify revision C as current, and leave the cause unconfirmed. The proposed wording is visible before any decision. The useful question is whether this exact change is justified by the sources on screen.

**Presenter cue:** Read the changed sentence aloud. Do not approve a hidden bundle of edits.

**Material:** CAP04 · software-capture-candidate. **On-screen copy:** Review this note change | Cause remains unconfirmed

## S12 · 04:13–04:40 · Separate a decision from its effects

**Picture:** Three large state cards with a separate receipt beneath each. Initially stop at reviewed; keep applied unfilled.

**Narrator:** There are three different states to show: proposed, reviewed, and applied. A review decision does not prove that a file changed, or that the next reader has the new version. The pilot should retain a separate record for each step and show a failure when one of those steps does not complete.

**Presenter cue:** Explain this in everyday language: deciding and doing are different events.

**Material:** G06 · editorial-concept. **On-screen copy:** Proposed → Reviewed → Applied | Separate evidence for each step

## S13 · 04:40–05:06 · Hand a plan to the responsible person

**Picture:** Show PLAN-204 with owner, planner, current reference and open question. End on a clearly labelled manual handoff.

**Narrator:** The outcome is an investigation plan, owned by Sam and handed to Alex for planning. The team’s normal maintenance process still governs scheduling, permits and physical work. In this first rehearsal, the handoff is manual. A connection to a maintenance-management system would be a separate integration to implement and verify.

**Presenter cue:** Answer who owns the next step; avoid implying that the software authorises physical work.

**Material:** G07 · staged-application. **On-screen copy:** Investigation plan | Manual handoff | Cause still open

## S14 · 05:06–05:30 · Verify the supported change

**Picture:** Record a disposable single-note application and its output, then show the resulting diff. Include a short separate stale-context refusal take for production review.

**Narrator:** Applying the correction should leave two things the team can inspect: the changed note and a record of the change. If the reviewed passage has changed in the meantime, the update must stop for another look. The handover should contain only the wording that was actually reviewed.

**Presenter cue:** In the live presentation, narrate the actual result rather than reading a success line automatically.

**Material:** CAP05 · software-capture-candidate. **On-screen copy:** Applied note + actual receipt — only if verified

## S15 · 05:30–05:55 · Leave a useful handover

**Picture:** Full-screen HANDOVER-204 card with five short lines and source links. No completion checkmark on the incident.

**Narrator:** The handover is now specific. The observation is still open. Revision C is the reference to use. Sam reviewed the note correction; Alex owns planning. The cause remains unconfirmed. The next shift receives that account, with links back to the records, instead of having to reconstruct the conversation.

**Presenter cue:** Hold the card. This is the practical deliverable of the story.

**Material:** G08 · staged-application. **On-screen copy:** Open observation · Current reference · Reviewed change · Next owner · Open question

## S16 · 05:55–06:21 · The next shift can ask again

**Picture:** Ask Q3 in a fresh session. Show the actual answer and an opened updated note side by side.

**Narrator:** Leah arrives on the next shift and asks what she needs to know about P two-oh-four. The intended application gives her the reviewed handover, with the unresolved question intact. She can see who owns the next step and open the evidence herself, without finding everyone who joined the earlier conversation.

**Presenter cue:** Ask a different person to explain the next action using only the handover.

**Material:** CAP06 · software-capture-candidate. **On-screen copy:** Fresh question → current handover

## S17 · 06:21–06:47 · Reveal the small technical spine

**Picture:** Four labelled component cards around a central Evidence and decisions card. Add a separate dashed Application integration band.

**Narrator:** Behind the story are a few distinct jobs. Ontomark manages the authored knowledge. Loom selects context for a request. Agentbox provides agent execution and supported change adapters. VisionClaw provides graph and review surfaces. The industrial application has to join those jobs with the site’s own rules and prove each handoff.

**Presenter cue:** This is the only stack overview. Refer technical questions to research.md.

**Material:** G09 · editorial-concept. **On-screen copy:** Authored knowledge · Grounding · Agent work · Graph and review

## S18 · 06:47–07:08 · Invite a measurable pilot

**Picture:** Return to the same quiet aisle with a slight change of daylight. End on a clean title card over the frame.

**Narrator:** Start with one asset, one recurring question and a small team. Measure how long it takes to find the right evidence, review a correction and hand it on. The next shift should inherit the reasoning, as well as the record.

**Presenter cue:** End by proposing the bounded pilot, then take questions.

**Material:** H03 · generated-illustration. **On-screen copy:** One asset. One question. A traceable handover.

## Live show-and-tell variation

Use the same sequence as chapter cards, pausing at S06, S10, S11 and S16 for discussion. Budget 10–12 minutes with questions. Rehearse a separate branch that uses the labelled editorial fallback for every unavailable service. Never switch from a failed live action to a success recording without announcing the change.

Optional questions for the room: Which record would you open next? What evidence would establish the cause? Who owns the next step? Can the incoming shift explain the decision without asking the previous team?
