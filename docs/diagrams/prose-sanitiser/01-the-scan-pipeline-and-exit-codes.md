---
id: PS-01
title: The scan pipeline and its exit-code contract
area: prose-sanitiser
governing: [../prose-sanitiser/README-CLI.md, ../prose-sanitiser/crates/cli/README.md]
adrs: []
sources:
  - ../prose-sanitiser/crates/cli/src/exit.rs
  - ../prose-sanitiser/crates/cli/src/bin/slop_scan.rs
  - ../prose-sanitiser/crates/cli/src/bin/slop_detect.rs
  - ../prose-sanitiser/crates/cli/src/bin/audit_dir.rs
  - ../prose-sanitiser/crates/cli/src/audit/mod.rs
  - ../prose-sanitiser/crates/cli/src/settings.rs
  - ../prose-sanitiser/crates/slop/src/prose/mod.rs
  - ../prose-sanitiser/crates/slop/src/rules/lexicon.rs
  - ../prose-sanitiser/crates/slop/src/design/mod.rs
  - ../prose-sanitiser/crates/core/src/lib.rs
  - ../prose-sanitiser/crates/core/src/finding/mod.rs
  - ../prose-sanitiser/crates/media/src/io.rs
  - ../prose-sanitiser/crates/cli/tests/slop_scan_errors.rs
  - ../prose-sanitiser/crates/cli/tests/slop_detect_errors.rs
  - ../prose-sanitiser/README-CLI.md
  - ../prose-sanitiser/crates/cli/README.md
verified_commit: 266063408b57f52672b5b061491120373d72ffef
---
## For developers

Every binary in the workspace ends in the same three-line shim: `main` calls
`run_cli(body)` and casts the result to a byte
(`../prose-sanitiser/crates/cli/src/bin/slop_scan.rs:66`-`68`). `run_cli`
prints an error's message to stderr and returns its code
(`../prose-sanitiser/crates/core/src/lib.rs:110`-`118`). The codes themselves
live in one module: `CLEAN` is 0 (`../prose-sanitiser/crates/cli/src/exit.rs:36`),
`FINDINGS` is 1 (`:39`) and `ERROR` is 2 (`:42`). The user-facing statement of
the contract is `../prose-sanitiser/README-CLI.md:3`-`4`, and every binary prints
it under `--help` through `HELP_EPILOGUE`
(`../prose-sanitiser/crates/cli/src/exit.rs:73`-`74`).

The scan itself is `slop-scan`. It resolves a configuration, walks the path,
reads each prose file under a size cap, runs the rule table over each document
and filters the findings by the disabled-rule list. It then collapses the result
to an exit code. This topic follows that path, then compares it with the two
sibling scanners, `slop-detect` and `audit-dir`, which apply the same contract
in different ways.

Three rules carry the design:

- **A count is not an exit code.** `slop-scan` used to exit with the number of
  high-severity findings and `slop-detect` with the finding count capped at 250.
  Both now map to `FINDINGS` (`../prose-sanitiser/crates/cli/src/exit.rs:16`-`23`),
  because a count saturates at 255 and cannot be told from an error code. The
  count lives in `--format json` instead.
- **An unreadable input is never clean.** A file that cannot be read, or is over
  32 MiB, is collected as an error and the run exits 2
  (`../prose-sanitiser/crates/slop/src/prose/mod.rs:156`, then
  `../prose-sanitiser/crates/cli/src/bin/slop_scan.rs:203`-`207`).
- **A deliberate skip is not a failure.** A symlinked directory or a non-regular
  file is reported as a notice on stderr and leaves the exit code alone
  (`../prose-sanitiser/crates/cli/src/bin/slop_scan.rs:200`-`202`).

## For the business

The tool is meant to sit in a build pipeline, so what matters is what the
pipeline sees. Exit 0 means the scan finished and found nothing to report. Exit 1
means it finished and found something. Exit 2 means it could not do its job, for
example because a file was unreadable. A pipeline can therefore fail a build on
findings and page a person on a broken scan, without confusing the two.

A scan that quietly skipped a file it could not read would pass a document that
was never checked. That is the failure the contract is built to prevent for the
prose scanner, and the topic records where the sibling directory audit does not
yet follow it.

## PS-01.1 The three exit codes and who produces them

```mermaid
flowchart TB
    SHIM["main: ExitCode::from(run_cli(body) as u8)<br/>slop_scan.rs:66"]
    RUN["run_cli: Ok code is returned as is,<br/>Err prints the message and returns its code<br/>core/src/lib.rs:110"]
    C0["CLEAN = 0<br/>exit.rs:36"]
    C1["FINDINGS = 1<br/>exit.rs:39"]
    C2["ERROR = 2<br/>exit.rs:42"]
    FF["from_findings count: 0 gives CLEAN, anything else FINDINGS<br/>exit.rs:45"]
    FL["from_flag found: true gives FINDINGS<br/>exit.rs:54"]
    CE["CliError::new code, message<br/>core/src/lib.rs:88"]
    SHIM --> RUN
    FF --> C0
    FF --> C1
    FL --> C0
    FL --> C1
    CE --> C2
    RUN --> C0
    RUN --> C1
    RUN --> C2
    NOTE["DRIFT: the CliError doc names 1 for a usage or input error<br/>and 2 for a failed check, the reverse of exit.rs<br/>core/src/lib.rs:70"]
    CE --> NOTE
```

**What it shows.** Two helpers produce the success-path codes: `from_findings`
for a count and `from_flag` for a boolean. A `CliError` carries the failure
path, and every binary in this area builds it with `exit::ERROR`.

**Why it is this way.** The module header chooses the shellcheck and Vale
convention over the one `typos` uses, where findings are 2
(`../prose-sanitiser/crates/cli/src/exit.rs:13`-`14`). One shared constant set
stops twelve tools each inventing a code.

**Drift:** the `CliError.code` doc says the code is conventionally `1` for a
usage or input error and `2` for a failed check
(`../prose-sanitiser/crates/core/src/lib.rs:70`-`71`). The contract it feeds
reserves 1 for findings and 2 for errors
(`../prose-sanitiser/crates/cli/src/exit.rs:9`-`11`), so the field's own comment
describes the old Python layout, not this workspace.

**Invariant:** the three constants are distinct and ordered 0, 1, 2, and any
non-zero count collapses to 1 (`../prose-sanitiser/crates/cli/src/exit.rs:45`-`51`).

## PS-01.2 One slop-scan run

```mermaid
sequenceDiagram
    autonumber
    participant BD as body<br/>slop_scan.rs:138
    participant ST as Settings.resolve<br/>settings.rs:91
    participant SW as scan_with<br/>prose/mod.rs:809
    participant WK as iter_files_report<br/>prose/mod.rs:266
    participant RD as read_file_text<br/>prose/mod.rs:214
    participant SD as scan_document<br/>prose/mod.rs:452
    BD->>BD: path must exist, else CliError exit 2 (slop_scan.rs:143-148)
    BD->>ST: --config, or nearest .prose-sanitiser.toml (slop_scan.rs:158)
    ST-->>BD: merged config, bad file is exit 2 (settings.rs:53-60)
    BD->>SW: path, severity floor, structural flag (slop_scan.rs:168)
    SW->>WK: root (prose/mod.rs:813)
    WK-->>SW: files in a fixed order, errors, notices
    loop each file
        SW->>RD: path (prose/mod.rs:817)
        RD-->>SW: text, or a ScanError that is pushed and skipped (prose/mod.rs:819-822)
        SW->>SD: file name, text, compiled rules, floor (prose/mod.rs:826)
    end
    SW-->>BD: findings, files_scanned, errors, notices
    BD->>BD: drop findings whose rule is disabled (slop_scan.rs:169-173)
    BD->>BD: count by severity, weight, verdict (slop_scan.rs:175-191)
    BD->>BD: errors and notices to stderr, then the exit code (slop_scan.rs:195-207)
```

**What it shows.** The configuration is settled first and the scan second. The
disabled-rule filter is applied after the scan, on the findings, while the
severity floor is passed in because it also decides which rules are compiled
(`../prose-sanitiser/crates/cli/src/bin/slop_scan.rs:156`-`157`).

**Why it is this way.** Configuration follows the git and `.editorconfig`
convention: the nearest file up the tree wins, and a file that exists but does
not parse is an error, not a silent default
(`../prose-sanitiser/crates/cli/src/settings.rs:67`-`72`). Precedence is
defaults, then the file, then flags (`../prose-sanitiser/crates/cli/src/settings.rs:19`-`23`).

**Invariant:** a path that does not exist exits 2 before anything is read, so a
mistyped path in a pipeline cannot pass as clean
(`../prose-sanitiser/crates/cli/src/bin/slop_scan.rs:143`-`148`).

## PS-01.3 The file walk: reads, errors and notices

```mermaid
flowchart TB
    ROOT{"root is a single file?<br/>prose/mod.rs:267"}
    ONE["that file only, no errors or notices<br/>prose/mod.rs:268"]
    LIST{"read_dir of a directory succeeds?<br/>prose/mod.rs:279"}
    LERR["ERROR: cannot list directory<br/>prose/mod.rs:282"]
    ENT{"entry kind<br/>prose/mod.rs:297"}
    DIR{"a directory, name in SKIP_DIRS?<br/>prose/mod.rs:311"}
    SKIP["not descended into, nothing reported<br/>lexicon.rs:72"]
    SYM{"a symlink: target is a directory?<br/>prose/mod.rs:318"}
    NOTE1["NOTICE: symlinked directory not followed<br/>prose/mod.rs:321"]
    FILE{"regular file with a wanted extension?<br/>prose/mod.rs:307"}
    NOTE2["NOTICE: not a regular file, never opened<br/>prose/mod.rs:343"]
    TAKE["added to the scan, sorted by name<br/>prose/mod.rs:346"]
    ROOT -->|yes| ONE
    ROOT -->|no| LIST
    LIST -->|no| LERR
    LIST -->|yes| ENT
    ENT -->|dir| DIR
    DIR -->|yes| SKIP
    DIR -->|no| TAKE
    ENT -->|symlink| SYM
    SYM -->|yes| NOTE1
    SYM -->|a file| FILE
    ENT -->|file| FILE
    ENT -->|FIFO, socket, device| NOTE2
    FILE -->|yes| TAKE
```

**What it shows.** The walk separates three outcomes. A directory it cannot read
is an error and so exits 2. A thing it chose not to follow is a notice and
changes nothing about the exit code. Everything else is scanned in a
deterministic order.

**Why it is this way.** Symlinked directories are never followed, so a loop or a
link out of the tree cannot widen the scan, yet each is reported so a partial
tree is visible (`../prose-sanitiser/crates/slop/src/prose/mod.rs:260`-`265`).
A non-regular file is never opened because opening a FIFO with no writer blocks
forever (`../prose-sanitiser/crates/slop/src/prose/mod.rs:188`-`194`). The
scanned extensions are `md`, `markdown`, `mdx`, `txt`, `rst`, `text` and `tex`
(`../prose-sanitiser/crates/slop/src/rules/lexicon.rs:69`).

**Invariant:** a file over 32 MiB is read one byte past the cap so that
exactly-at-cap and over-cap are told apart, and the over-cap case is an error,
not a silent skip (`../prose-sanitiser/crates/slop/src/prose/mod.rs:199`-`207`).
A test pins the exit-code consequences: an unreadable file gives 2 and a
symlinked directory leaves 0
(`../prose-sanitiser/crates/cli/tests/slop_scan_errors.rs:21`, then `:58`).

## PS-01.4 From findings to a verdict and an exit code

```mermaid
flowchart TB
    F["findings after the disabled-rule filter<br/>slop_scan.rs:169"]
    W["weighted = sum of weight x count<br/>High 3, Medium 2, Low 1<br/>slop_scan.rs:187"]
    WT["Severity.weight<br/>core/src/finding/mod.rs:53"]
    V{"verdict high, weighted<br/>prose/mod.rs:720"}
    V1["high 5 or more, or weighted 20 or more:<br/>STRONG AI writing fingerprint<br/>prose/mod.rs:721"]
    V2["high 1 or more, or weighted 6 or more:<br/>Some AI tells present<br/>prose/mod.rs:724"]
    V3["weighted above 0: Mostly clean, minor tells<br/>prose/mod.rs:727"]
    V4["otherwise: Clean, no mechanical tells detected<br/>prose/mod.rs:730"]
    E{"any scan error?<br/>slop_scan.rs:203"}
    X2["exit 2<br/>slop_scan.rs:206"]
    X01["exit from_findings of the finding count: 0 or 1<br/>slop_scan.rs:204"]
    F --> W
    WT --> W
    W --> V
    V --> V1
    V --> V2
    V --> V3
    V --> V4
    F --> E
    E -->|yes| X2
    E -->|no| X01
    NOTE["the verdict wording never reaches the exit code:<br/>a STRONG and a Mostly clean scan both exit 1<br/>slop_scan.rs:204"]
    X01 --> NOTE
```

**What it shows.** The human verdict and the exit code are computed from the
same findings but separately. The verdict has four wordings chosen from the
high count and the weighted score. The exit code ignores severity: any reported
finding gives 1.

**Why it is this way.** The score separates one strong tell from a drift of weak
ones, so the verdict reads both numbers
(`../prose-sanitiser/crates/slop/src/prose/mod.rs:779`-`782`). The exit code is
deliberately coarser: the module header of `exit.rs` says a count in `$?`
reports zero for a file full of medium-severity hits
(`../prose-sanitiser/crates/cli/src/exit.rs:20`-`21`). A caller who wants a
severity gate passes `--severity`, which removes lower-severity rules before the
scan, so "findings at or above the gate" and "findings reported" are the same set
for this binary (`../prose-sanitiser/crates/cli/src/bin/slop_scan.rs:40`-`41`).

**Invariant:** the incomplete-scan text says so and exits 2 even when there are
no findings, so a run with unreadable files can never print the "Nothing
flagged" message with a zero code
(`../prose-sanitiser/crates/cli/src/bin/slop_scan.rs:291`-`304`).

## PS-01.5 How three scanners map an incomplete scan to an exit code

```mermaid
flowchart TB
    subgraph SS["slop-scan, prose"]
        S1["unreadable or oversize file: errors, exit 2<br/>slop_scan.rs:203"]
        S2["symlinked directory: notice, exit unchanged<br/>slop_scan.rs:200"]
    end
    subgraph SDT["slop-detect, design"]
        D1["unreadable or oversize file: errors, exit 2<br/>slop_detect.rs:78"]
        D2["a path that does not exist: exit 2 before scanning<br/>slop_detect.rs:65"]
    end
    subgraph AD["audit-dir, provenance"]
        A1["file over the input cap: listed as skipped, exit unchanged<br/>audit_dir.rs:56"]
        A2["unreadable file: listed as skipped, exit unchanged<br/>audit_dir.rs:61"]
        A3["a directory that cannot be listed: dropped silently<br/>audit/mod.rs:329"]
        A4["exit 1 only when an actionable file exists<br/>audit_dir.rs:91"]
    end
    SS --> SDT --> AD
    TEN["TENSION: exit.rs:11 defines 2 as unreadable input,<br/>audit-dir reports it as skipped and still exits 0 or 1<br/>audit_dir.rs:61"]
    AD --> TEN
```

**What it shows.** `slop-scan` and `slop-detect` treat anything unreadable as
exit 2. `audit-dir` records an unreadable or over-cap file in a `files_skipped`
list and decides its exit code only from whether a file is actionable.

**Why it is this way.** `audit-dir` keeps going on one bad file so a large tree
still yields a report (`../prose-sanitiser/crates/cli/src/bin/audit_dir.rs:60`).
An actionable file is one with C2PA data or a `confirmed` or `probable`
confidence level (`../prose-sanitiser/crates/cli/src/audit/mod.rs:157`-`173`).
The cap is `WATERMARKS_MAX_INPUT_BYTES`, 256 MiB by default
(`../prose-sanitiser/crates/media/src/io.rs:20`-`22`), against 32 MiB for the
prose scanners.

**Tension (exit contract vs audit-dir):** the contract says exit 2 covers
unreadable input (`../prose-sanitiser/crates/cli/src/exit.rs:11`), and
`slop-scan` honours that (`../prose-sanitiser/crates/cli/src/bin/slop_scan.rs:195`-`207`).
`audit-dir` lists the same condition as skipped and exits 0 or 1
(`../prose-sanitiser/crates/cli/src/bin/audit_dir.rs:61`-`62`, then `:91`), so a
tree whose every file is unreadable exits 0.

**Drift:** the `slop-detect` module header still says the exit code is the number
of findings, capped at 250 (`../prose-sanitiser/crates/cli/src/bin/slop_detect.rs:13`-`14`).
The code maps findings through `exit::from_findings`, which returns 0 or 1
(`../prose-sanitiser/crates/cli/src/bin/slop_detect.rs:78`-`84`).

**Debt:** `walk_files` skips a directory it cannot list without recording it
(`../prose-sanitiser/crates/cli/src/audit/mod.rs:329`-`331`), whereas the prose
walk records the same failure as an error
(`../prose-sanitiser/crates/slop/src/prose/mod.rs:279`-`287`). `audit_dir.rs:38`
also passes the literal `2` rather than `exit::ERROR`.
