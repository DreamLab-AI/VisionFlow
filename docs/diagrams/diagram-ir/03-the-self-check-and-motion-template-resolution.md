---
id: DI-03
title: The self-check and motion-template resolution
area: diagram-ir
governing: [../diagram-ir/README.md]
adrs: []
sources:
  - ../diagram-ir/README.md
  - ../diagram-ir/assets/template-motion.html
  - ../diagram-ir/src/bin/self_check.rs
  - ../diagram-ir/src/selfcheck/mod.rs
  - ../diagram-ir/src/selfcheck/checks.rs
  - ../diagram-ir/src/selfcheck/html.rs
  - ../diagram-ir/src/selfcheck/refs.rs
  - ../diagram-ir/tests/selfcheck.rs
verified_commit: e16337b05952cc90d1d57abdd496ea45a5678f8a
---

## For developers

`diagram-self-check` is the third binary in the crate. It does not read diagrams in the sense of the other two; it checks a generated diagram HTML file against the accessible-diagram contract: at least one accessible SVG with the right roles and labels, no remote assets beyond one approved Google Fonts stylesheet, no executable attributes, no script except one canonical motion controller, and, when motion markup is present, the structural motion contract (`../diagram-ir/src/bin/self_check.rs:1-9`). The rules are a "distilled subset" of two repository gates that stay the authority for contributions to the repository itself (`../diagram-ir/src/selfcheck/mod.rs:1-3`). Those gates are not in this repository, so the parity of the subset is not something these sources can show.

The entry point `verify` runs five groups and concatenates their findings: the unsafe-markup findings gathered while scanning, a vetting of every reference-bearing attribute, the SVG rules, the script rules and the motion rules (`../diagram-ir/src/selfcheck/checks.rs:365-378`). The scan is a hand-written tolerant HTML reader that models only start tags, end tags, character data and the raw-text mode of `script` and `style`; nothing is executed or fetched (`../diagram-ir/src/selfcheck/html.rs:1-6`, `../diagram-ir/src/selfcheck/html.rs:225`).

The script rule is where the template matters. A diagram may carry at most one script, it must be closed and carry exactly one attribute, `data-diagram-controls` with no value, and its body must equal the canonical controller after newline normalisation and trimming (`../diagram-ir/src/selfcheck/checks.rs:122-127`, `../diagram-ir/src/selfcheck/checks.rs:131-155`). The canonical controller is the single script inside `template-motion.html`, which must itself contain exactly one closed script (`../diagram-ir/src/selfcheck/checks.rs:41-45`). That file ships in the crate as `assets/template-motion.html` and is compiled into the library with `include_str!` (`../diagram-ir/src/selfcheck/mod.rs:15`); its controller is the script at `../diagram-ir/assets/template-motion.html:237`.

Where the template comes from is decided by `resolve_motion_template`: an explicit `--motion-template`, then an absolute `DIAGRAM_DESIGN_SKILL_DIR`, then the installed skill path, then the bundled copy (`../diagram-ir/src/selfcheck/mod.rs:51-61`). Every on-disk candidate the function builds itself is absolute, and a relative `DIAGRAM_DESIGN_SKILL_DIR` is dropped, so a file planted in the directory the tool runs from cannot stand in for the canonical controller (`../diagram-ir/src/selfcheck/mod.rs:35`). The change landed in commit `4a84e0d`, tracked upstream as finding COV-A-F6 (`../diagram-ir/src/selfcheck/mod.rs:22-25`), and the tests assert it (`../diagram-ir/tests/selfcheck.rs:389-408`).

## For the business

Every generated diagram page that moves, such as a stepped animation, runs one small script. The self-check is the guard that only the approved script is present, unmodified, and that the page stays readable and safe: screen-reader labels exist, nothing loads from the open internet except the one approved font stylesheet, and the page still shows a complete static picture without script, in reduced-motion settings and in print.

For a deployment the template choice is a trust decision: whichever controller the checker compares against defines what "approved" means. The design is that a standalone install needs no extra files, an installed design skill can pin its own version of the controller, and nothing found next to wherever the command happens to be run can take its place. One caveat is worth knowing: if an installed copy and the built-in copy ever differ, the installed one is the one that counts.

## DI-03.1 Where the canonical controller is found

```mermaid
flowchart TB
    S["resolve_motion_template<br/>selfcheck/mod.rs:51"] --> E{"--motion-template given?<br/>selfcheck/mod.rs:52"}
    E -->|yes| X["used exactly as written, with no existence or absolute check<br/>selfcheck/mod.rs:53"]
    E -->|no| V{"DIAGRAM_DESIGN_SKILL_DIR set and not empty?<br/>selfcheck/mod.rs:28-29"}
    V -->|yes| A{"assets/template-motion.html under it is an absolute path?<br/>selfcheck/mod.rs:35"}
    A -->|yes| C1["candidate one, from the environment<br/>selfcheck/mod.rs:36"]
    A -->|no| IG["dropped, never resolved against the working directory<br/>selfcheck/mod.rs:33-34"]
    V -->|no| C2
    C1 --> C2["candidate two, the installed skill<br/>selfcheck/mod.rs:40-42"]
    IG --> C2
    C2 --> F{"first candidate that is a file<br/>selfcheck/mod.rs:55-58"}
    F -->|found| P["that file is read when a script needs checking"]
    F -->|none| B["the bundled copy, named by a pseudo path<br/>selfcheck/mod.rs:60"]
```

**What it shows.** Four sources in strict order. The bundled copy is a constant compiled into the library (`../diagram-ir/src/selfcheck/mod.rs:15`), and its pseudo path is recognised by `canonical_controller` so that it is read from memory and not from disk (`../diagram-ir/src/selfcheck/checks.rs:30-32`). A test confirms that the bundled text equals the asset in the repository, that its controller parses, and that a compliant file passes against it (`../diagram-ir/tests/selfcheck.rs:365-386`).

**Why it is this way.** The binary no longer lives inside the skill, so it cannot assume a skill checkout; the order lets a deployment pin a controller while a standalone install still works (`../diagram-ir/src/selfcheck/mod.rs:46-50`).

**Invariant:** no automatically built candidate is relative: the environment path is pushed only when absolute and the installed path is a literal absolute path (`../diagram-ir/src/selfcheck/mod.rs:35`, `../diagram-ir/src/selfcheck/mod.rs:40-42`). The tests check every candidate and a relative override (`../diagram-ir/tests/selfcheck.rs:389-408`).

**Open:** an explicit `--motion-template` is returned as written and later opened with the ordinary file read, so a relative value is read against the working directory (`../diagram-ir/src/selfcheck/mod.rs:52-54`, `../diagram-ir/src/selfcheck/checks.rs:33`), while the README says nothing is resolved against the working directory (`../diagram-ir/README.md:121-122`). The README's reason, a planted file standing in for the controller, does not apply when the caller names the path, but the sentence is broader than the code.

**Open:** the library says the installed and bundled controllers "are expected to be byte-identical" (`../diagram-ir/src/selfcheck/mod.rs:13-14`). Nothing compares them at run time, and the only test compares the bundled text with the asset in this repository (`../diagram-ir/tests/selfcheck.rs:365-374`). When an installed copy exists it wins (`../diagram-ir/src/selfcheck/mod.rs:55-59`).

## DI-03.2 A run of the binary, file by file

```mermaid
sequenceDiagram
    participant MN as main<br/>bin/self_check.rs:35
    participant RT as resolve_motion_template<br/>selfcheck/mod.rs:51
    participant VF as verify<br/>selfcheck/checks.rs:365
    participant PS as parse<br/>selfcheck/html.rs:225
    participant CS as check_scripts<br/>selfcheck/checks.rs:113
    participant CC as canonical_controller<br/>selfcheck/checks.rs:29
    MN->>RT: the flag value or nothing (bin/self_check.rs:37)
    RT-->>MN: one path, possibly the bundled pseudo path (selfcheck/mod.rs:51)
    loop each file argument
        Note over MN: an unreadable or non-UTF-8 file becomes a failure entry,<br/>not a tool error, bin/self_check.rs:43-45
        MN->>VF: text with newlines normalised (bin/self_check.rs:48-49)
        VF->>PS: the source (selfcheck/checks.rs:366)
        PS-->>VF: scripts, svgs, motion attributes, unsafe findings, references (selfcheck/html.rs:225)
        VF->>CS: the parse and the template path (selfcheck/checks.rs:375)
        Note over CS: no script in the file means the template is never read,<br/>selfcheck/checks.rs:119
        CS->>CC: for a script carrying the exact attribute, once per file (selfcheck/checks.rs:146)
        CC-->>CS: the normalised controller, or a finding naming the missing template (selfcheck/checks.rs:33)
    end
    Note over MN: prints OK or FAIL per file, exits 1 if any failed,<br/>bin/self_check.rs:53-67
```

**What it shows.** The template path is resolved once, before any file is read, but the template is read lazily and only when a file has a script that passes the attribute test. A missing or malformed template therefore shows up as a finding on that file, not as a start-up error, and a file without scripts passes without the template existing (`../diagram-ir/src/selfcheck/checks.rs:119-121`, `../diagram-ir/tests/selfcheck.rs:318-330`).

**Why it is this way.** A diagram file is checked by value: `verify` takes source text and a template path and returns a list of strings, so the same function serves the binary, the tests and any library caller (`../diagram-ir/src/selfcheck/checks.rs:364-366`).

**Drift (README vs code):** the README gives exit code 2 as "tool error" for this binary (`../diagram-ir/README.md:127`). The exit paths in `main` itself end in 0 or 1, and the header documents only those (`../diagram-ir/src/bin/self_check.rs:11`, `../diagram-ir/src/bin/self_check.rs:63-67`); a file it cannot read or decode is a `FAIL` entry with exit 1 (`../diagram-ir/src/bin/self_check.rs:43-45`), so the most obvious tool error does not give 2. What status a rejected command line gives is decided by the argument parser, which these sources do not show.

## DI-03.3 The five rule groups in verify

```mermaid
flowchart TB
    V["verify returns every finding as a string<br/>selfcheck/checks.rs:365"] --> U
    V --> R
    V --> G
    V --> SC
    V --> M
    U["unsafe markup while scanning: base, embed, object and iframe tags, on-attributes, srcdoc<br/>selfcheck/html.rs:79-91<br/>selfcheck/checks.rs:368"]
    R["every src, href, xlink:href, poster, srcset, action and formaction value is vetted<br/>selfcheck/html.rs:36-44<br/>selfcheck/checks.rs:369-373"]
    G["svg rules: one accessible svg, role img, title first, title then desc labelling<br/>selfcheck/checks.rs:49-110"]
    SC["script rules: at most one, closed, canonical attribute, canonical body<br/>selfcheck/checks.rs:113-159"]
    M["motion rules when motion markup is present<br/>selfcheck/checks.rs:162-352"]
```

**What it shows.** The groups are independent and all run, so one file can report findings from several. The SVG rules apply to SVGs that are not `aria-hidden`; a file with none is itself a finding (`../diagram-ir/src/selfcheck/checks.rs:50-62`). Each accessible SVG needs `role="img"`, a non-empty `title` and `desc` with diagram-prefixed ids, `title` as its first child, and an `aria-labelledby` naming title then desc (`../diagram-ir/src/selfcheck/checks.rs:65-108`).

**Why it is this way.** The contract is accessibility first: the picture must be understood without sight and without script, and the single-file rule keeps a diagram a self-contained document.

**Invariant:** a script must have exactly the attribute list `data-diagram-controls` with an empty value, or the file is failed and the body is not compared (`../diagram-ir/src/selfcheck/checks.rs:134-145`).

## DI-03.4 Which references are allowed

```mermaid
flowchart TB
    IN["reference_error: tag, rel and value<br/>selfcheck/refs.rs:94"] --> EM{"empty or a fragment only?<br/>selfcheck/refs.rs:97"}
    EM -->|yes| OK["allowed"]
    EM -->|no| JS{"javascript: or data:text/html?<br/>selfcheck/refs.rs:100"}
    JS -->|yes| E1["finding: executable URL<br/>selfcheck/refs.rs:101"]
    JS -->|no| RM{"remote: http, https, scheme-relative, or any other scheme except data?<br/>selfcheck/refs.rs:108-111"}
    RM -->|no| DT{"a data URL that is not data:image?<br/>selfcheck/refs.rs:113"}
    DT -->|yes| E2["finding: non-image data URL<br/>selfcheck/refs.rs:114"]
    DT -->|no| OK
    RM -->|yes| LK{"a link element with rel stylesheet?<br/>selfcheck/refs.rs:121-126"}
    LK -->|yes| GF{"https, host fonts.googleapis.com, path css2, no port, no fragment?<br/>selfcheck/refs.rs:71-86"}
    GF -->|yes| OK
    GF -->|no| E3["finding: not the approved Google Fonts URL<br/>selfcheck/refs.rs:130"]
    LK -->|no| E4["finding: remote reference on the tag<br/>selfcheck/refs.rs:135"]
```

**What it shows.** The only network address that passes is one stylesheet link. Anything else that is remote, including a scheme the rule does not know, is a finding, as is any `data:` URL that is not an image. Local paths and fragments pass. A URL is only parsed into scheme, host, port, path and fragment, never resolved or fetched (`../diagram-ir/src/selfcheck/refs.rs:1-5`).

**Why it is this way.** Single-file safety means a diagram cannot depend on or leak to a third party. The Google Fonts stylesheet is the one exception, and the exception is narrow: an explicit port disqualifies the URL (`../diagram-ir/src/selfcheck/refs.rs:75-78`). The shipped template's own font link is this form (`../diagram-ir/assets/template-motion.html:7`).

## DI-03.5 The motion contract

```mermaid
flowchart TB
    A["motion markup present: a root, an item or a script<br/>selfcheck/checks.rs:163-167"] --> B
    B["exactly one data-motion-root<br/>selfcheck/checks.rs:168-174"] --> C
    C["mode is none, reveal, step or loop<br/>selfcheck/checks.rs:176-185"] --> D
    D["step count is a decimal integer, 0 to 8 for none, 1 to 8 otherwise<br/>selfcheck/checks.rs:186-198"] --> E
    E["at most 12 motion items<br/>selfcheck/checks.rs:200-205"] --> F
    F["per item: decimal step, semantic items carry an aria-label,<br/>decorative ones aria-hidden true and focusable false,<br/>none hidden by inline style<br/>selfcheck/checks.rs:207-250"] --> G
    G["semantic steps contiguous from 1 to the count, at most two items per step<br/>selfcheck/checks.rs:252-282"] --> H
    H{"mode and scripts"}
    H -->|none or loop| H1["no script, no controls, no actions, no status<br/>selfcheck/checks.rs:284-294"]
    H -->|step, or reveal with a script| H2["one controls group, five actions, a polite status outside it, a script<br/>selfcheck/checks.rs:295-335"]
    H1 --> Z
    H2 --> Z
    Z["any script: reduced-motion CSS, print CSS and a noscript note<br/>selfcheck/checks.rs:338-351"]
```

**What it shows.** The contract is a staged check that stops early only when there is not exactly one motion root. The five actions are `play`, `pause`, `replay`, `prev` and `next`, and the modes are the four listed (`../diagram-ir/src/selfcheck/checks.rs:13-14`). A controlled mode needs a status element with `role=status`, `aria-live=polite` and `aria-atomic=true` placed outside the controls (`../diagram-ir/src/selfcheck/checks.rs:315-331`).

**Why it is this way.** Each rule protects a reader who cannot use the animation: the static frame must be visible in the source (`../diagram-ir/src/selfcheck/checks.rs:242-249`), the semantic steps must read as a short ordered sequence, and the fallbacks cover reduced motion, print and no script.

**Invariant:** the scan counts motion items, actions, controls and status only while inside a motion root, so markup elsewhere on the page cannot satisfy or break the contract (`../diagram-ir/src/selfcheck/html.rs:106-125`).
