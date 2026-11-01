# Production guide

This is the handoff for the later recording, voiceover and hero-generation sessions. None of the assets below has been produced by this side quest. Shot IDs, spoken lines and timing estimates live in [walkthrough.json](walkthrough.json); the [CSV](shot-list.csv) is the recording checklist.

## Direction

Make the film feel like a calm engineering conversation. The viewer should understand the incident and the decision before learning the component names. The visual turning point is the superseded reference at S06. The emotional payoff is the handover at S15: somebody else can pick up the work without guessing.

Use clean, readable document views and a small graph. Avoid a vast glowing network, a wall of terminals or a simulated control room. A graph edge should tell the viewer something useful. Keep the same source colours throughout: observation amber, historical record grey, current reference blue, proposed change violet. Use labels and shapes too; colour must not carry meaning alone.

P-204 remains an unresolved incident at the end. Do not imply that it has been repaired, that production recovered or that downtime was avoided. Those would require a different story and measured evidence.

## Capture batches

Prepare a disposable demo environment with only the synthetic fixture. Map it explicitly to the application's real input format; `fixtures/scenario.json` is not a loader schema. Preserve the mapping, generated corpus and fixture hash with the later capture receipts. Do not load it into the live estate corpus or record existing operator conversations.

| Batch | Shots / asset IDs | What to prepare and record | Acceptance / fallback |
| --- | --- | --- | --- |
| Documents and identity | S02–06 / G01–04, CAP01 | Fixture document cards, dates and single-note identity; a real Ontomark validation take if the demo corpus is ready | Source text is readable and IDs agree. If validation is not run, use a labelled editorial note, not a green terminal result. |
| Graph exploration | S07 / CAP02 | Small mapped graph in VisionClaw; record the actual available selection and inspection controls | Viewers can follow P-204 → earlier work → revision B, then see C supersede B. If fixture loading or inspection is unavailable, use G03 as an editorial diagram. |
| Retrieval and unknowns | S09–10 / CAP03, G05 | Dedicated demo corpus; Q1 and Q2; full retrieval/grounding receipts and opened sources | Use a real answer only if it matches the sources and uncertainty. Otherwise the fixture answer is a staged application card. No unverified diagnosis or invented server JSON. |
| Review and application | S11–14 / CAP04–05, G06–07 | Exact NOTE-204 diff; supported single-page path; review identity, output and receipt; separate stale-context refusal take | Match subject and content through every claimed step. A direct writer test may be a software insert but does not prove UI-to-agent integration. Use a labelled manual rehearsal when the connected path is unavailable. |
| Handover and refresh | S15–16 / G08, CAP06 | Handover card; verified consumer refresh; fresh Q3 query from a new session | New content must be served. Do not imply refresh by cutting from a saved file to an old answer. Use the staged Q3 card if necessary. |
| Technical summary | S17 / G09 | Four component cards, with application integration visibly separate | Keep the arrows conceptual; point technical reviewers to the research ledger. |
| People and setting | S01, S03, S08, S18 / H01–03, G02 | Hero continuity frames, performed technician line and optional presenter footage | Fictional setting, no private screens or site branding. Label generated shots. |

Before each take, record the exact build and configuration, reset only the disposable fixture, close notifications and clear unrelated windows. Capture at 1920×1080 or higher with a consistent 16:9 crop. The working timeline is 25 fps; retain source frame rates in the receipt and convert deliberately during editing. Frame the relevant controls large enough to read at 1080p. Leave three seconds of stillness before and after each interaction. Capture cursor movement slowly and avoid post-production acceleration that suggests a measured response time.

Suggested filename: `S09_CAP03_take01_raw.mp4`. Keep raw takes separate from trimmed edits. For every accepted take, retain a small record containing shot ID, filename/hash, recording time, build revision, fixture hash, route or command, visible result, and any redactions. Collect a still from every final shot for editorial review.

## Hero scene briefs

These are art-direction prompts for the later generation session, not submitted jobs. Use an approved reference frame to keep the plant layout, pump colour and lighting consistent. Choose the generation model and workflow after inspecting what is available then; preserve that run's model, workflow, seed and actual receipt. Add all text in the compositor. Never ask the image model to draw exact UI, document text or asset identifiers.

### H01 · Opening aisle · S01

**Purpose:** establish a recognisable working environment and a single asset without manufacturing an emergency.

**Prompt:** “Cinematic documentary view inside a modest modern packaging plant, a tidy cooling-water pump skid with a blue horizontal motor housing and stainless pipework, painted concrete floor, neutral morning daylight through high windows, restrained industrial colours, one slow shoulder-height camera move along the aisle towards the pump, believable scale and stable mechanical geometry, no readable text, no logos, no screens, no smoke, no sparks, no emergency lighting.”

**Composition:** leave clear negative space upper left for the title. Use one uninterrupted camera move. Capture or generate an 8–12 second usable section with handles; the opening title can hold on the final frame if the narration needs longer. Add the P-204 label and opening disclosure during the edit.

**Reject:** drifting pipes, changing pump geometry, impossible couplings, unmotivated motion, fake warning symbols, unreadable generated lettering or a camera move through solid equipment.

### H02 · Evidence match-cut · S08

**Purpose:** connect the physical situation to the team's shared view without presenting a fictional sensor overlay as software.

**Prompt:** “The same blue pump skid and industrial aisle as the approved reference frame, close documentary detail of the circular motor housing with stainless pipework in soft background focus, slow lateral camera movement, stable realistic surfaces, quiet neutral lighting, no interface, no holograms, no labels, no humans operating machinery.”

**Composition:** finish with the circular housing in the same screen position as P-204's node in the following deterministic diagram. Perform the transition in the editor. Aim for a clean 6–8 second generated section; use a deliberate held frame or adjacent approved detail for the remaining spoken time.

**Reject:** generated graph text, a purported live heatmap, pulsing alarm colours, shape changes or a visual suggestion that the film has diagnosed the fault.

### H03 · Next shift · S18

**Purpose:** close on continuity of knowledge and invite a bounded pilot.

**Prompt:** “Return to the same modest packaging-plant aisle and blue pump skid in the approved reference, slightly softer late-afternoon daylight, steady wide documentary composition, a gentle camera pullback, calm operational atmosphere, no repair activity, no screens, no logos, no text.”

**Composition:** leave the centre clear for the deterministic closing line. A short approved clip can hold under the end card; do not invent a new plant layout merely to vary the shot.

**Reject:** a celebratory restart, workers changing parts, a production-output counter, dramatic before/after lighting or any image suggesting the unresolved incident has been fixed.

## Voice and sound

Use a warm, precise UK-English narrator. Speak to a capable operations colleague. Aim for roughly 125–140 words per minute, with longer pauses on the document comparison and the diff. Record Maya's short line separately; a different voice is optional. A single presenter can perform it clearly as a quotation.

Pronunciation: P-204 is “P two-oh-four”; CL-2 is “cooling loop two”; Ontomark is “on-toe-mark”; Loom is the ordinary word; Agentbox is “agent box”; VisionClaw is “vision claw”. Read revision letters individually. Keep filenames, source hashes and claim IDs out of the spoken track.

Record the exact text, measure each segment, then retime the scenes. If narration runs long, extend an informative hold or edit the sentence; do not speed up speech to preserve provisional timecodes. Add low industrial room tone sparingly. Music, if used, should sit beneath speech and disappear during the uncertainty and review moments. No alarms or dramatic fault sounds that invent evidence.

Produce captions from the final spoken audio and inspect every cue. Keep two readable lines inside the safe area and clear of document references, the application-status label and graph controls. Maintain a transcript for people who cannot use the audio or spatial view.

## Assembly and fallbacks

Keep application-concept labels on staged material. Insert an explicit transition whenever moving from real software to a designed screen. A fictional answer can illustrate the desired experience; it cannot be relabelled as a returned service response. Generated imagery establishes context, not product evidence.

Pre-production uses a separate JSON schema. It is **not** a runnable `codebase-video` compositor manifest and contains no invented asset paths, speech files or generation receipts. During production, create a separate `production/plan.json` in the selected video tool's real schema, populated with actual assets and measured durations. Retain this script version as the approved creative reference.

Suggested later folders are `recordings/`, `audio/`, `generated/`, `workflows/`, `receipts/` and a versioned `delivery-v1/`. Create them when their first real artefact exists. Publishing the film is a separate later action.

## Pilot and review criteria

Compare the team's existing method with this bounded workflow on the same synthetic case, followed by a suitably authorised real case if the team chooses. Measure time to locate the current reference, incorrect-revision selections, ability to identify the unresolved question, review errors, interventions required and the incoming shift's ability to explain the next action. Record setup and authoring effort as well as task time. No numerical target or business saving has been measured by this package.

Before accepting the film, a reviewer should be able to answer: What was observed? Which source changed the discussion? What is still unknown? Who reviewed the note? Who owns planning? What can the next shift inspect? Where was the industrial application staged?

Watch every shot at delivery size. Check each caption against the actual frame, each spoken assertion against the source or hypothetical label, and every apparent success against a receipt. Review the entire soundtrack, not only isolated frames. A readable storyboard and a green manifest check are preparation; the later completed film requires its own editorial and runtime verification.
