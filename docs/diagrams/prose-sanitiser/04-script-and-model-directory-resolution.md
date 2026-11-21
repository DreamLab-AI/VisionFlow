---
id: PS-04
title: Script and model directory resolution
area: prose-sanitiser
governing: [../prose-sanitiser/CHANGELOG.md, ../prose-sanitiser/crates/server/README.md]
adrs: []
sources:
  - ../prose-sanitiser/crates/core/src/lib.rs
  - ../prose-sanitiser/crates/media/src/image/harness.rs
  - ../prose-sanitiser/crates/media/src/image/mod.rs
  - ../prose-sanitiser/crates/media/src/proc.rs
  - ../prose-sanitiser/crates/cli/src/rewrite/markllm.rs
  - ../prose-sanitiser/crates/cli/src/bin/clean_image.rs
  - ../prose-sanitiser/crates/cli/src/bin/rewrite_text.rs
  - ../prose-sanitiser/crates/server/src/lib.rs
  - ../prose-sanitiser/CHANGELOG.md
  - ../prose-sanitiser/crates/server/README.md
verified_commit: 266063408b57f52672b5b061491120373d72ffef
---
## For developers

Four heavy backends stay in Python because they are torch programs, not parsers:
the reverse-SynthID scorer, CtrlRegen, MarkDiffusion and MarkLLM
(`../prose-sanitiser/crates/media/src/image/harness.rs:1`-`11`). Rust locates them,
runs them as child processes under resource limits and parses their JSON back.
Locating them means resolving two kinds of directory. The **scripts directory**
holds the small Python entry points the Rust code invokes. Each backend's
**upstream directory**, a checkout holding the model code, is passed to those
scripts. This crate never downloads a model
(`../prose-sanitiser/crates/media/src/image/harness.rs:178`-`180`).

The resolution rules exist for one reason. A directory read from the environment
and resolved against the current working directory would let a hostile working
directory steer the tool at code it controls, and the scripts are executed with
the harness interpreter. So an environment override is only honoured when it is an
absolute path or begins `~/`
(`../prose-sanitiser/crates/core/src/lib.rs:139`-`151`), and the scripts directory
has no working-directory fallback at all
(`../prose-sanitiser/crates/media/src/image/harness.rs:28`-`35`).

This topic draws the rule, the scripts directory order, the per-backend
precedence and checks, one full harness run, and the places where the rule is not
applied uniformly.

## For the business

The optional image and text watermark removers are separate programs the
operator installs themselves. The tool finds them through settings, and it is
built so that a booby-trapped folder the user happens to be standing in can never
be picked up in place of the real installation. If a setting is not an absolute
location it is ignored, and the tool reports that the backend is not configured
instead of guessing.

The operator can expect clear failures. A missing checkout, a missing virtual
environment or a timed-out run is reported as "not available" with the reason, and
the rest of the cleaning carries on.

## PS-04.1 The rule for a directory override

```mermaid
flowchart TB
    NAME["env_dir NAME<br/>core/src/lib.rs:144"]
    NE["env_nonempty: unset or empty reads as unset<br/>core/src/lib.rs:133"]
    ST{"is_stable_dir_path value<br/>core/src/lib.rs:149"}
    ABS["absolute path: honoured<br/>core/src/lib.rs:150"]
    HOME["begins with ~/ : honoured<br/>core/src/lib.rs:150"]
    NO["anything else, including . and ./x and ~ and ~x:<br/>ignored, reads as unset<br/>core/src/lib.rs:145"]
    NAME --> NE --> ST
    ST -->|absolute| ABS
    ST -->|tilde slash| HOME
    ST -->|otherwise| NO
    NOTE["the filter applies to environment values only,<br/>an explicit command-line value is not passed through it<br/>harness.rs:212"]
    NAME --> NOTE
```

**What it shows.** One helper decides whether an environment variable names a
directory. It never looks at whether the directory exists and it never expands
`~`. Both of those are left to the caller.

**Why it is this way.** A relative path would resolve against whichever directory
the tool is run from. The `CHANGELOG` records the decision as one rule for all
five variables, `PROSE_SANITISER_SCRIPTS_DIR`, `REVERSE_SYNTHID_DIR`,
`NOAI_WATERMARK_DIR`, `MARKDIFFUSION_DIR` and `MARKLLM_DIR`
(`../prose-sanitiser/CHANGELOG.md:32`-`35`). A unit test lists the cases that
must be refused, including `.`, `~` and `~x`
(`../prose-sanitiser/crates/core/src/lib.rs:211`-`232`).

**Invariant:** an empty or relative override reads as unset, never as the working
directory (`../prose-sanitiser/crates/core/src/lib.rs:144`-`151`).

## PS-04.2 Where the scripts live

```mermaid
flowchart TB
    SD["scripts_dir<br/>harness.rs:36"]
    ENVS["env_dir PROSE_SANITISER_SCRIPTS_DIR<br/>harness.rs:38"]
    FROM["scripts_dir_from override, baked<br/>harness.rs:45"]
    OV{"an honoured override?<br/>harness.rs:46"}
    USE["that directory, used as given<br/>harness.rs:47"]
    BAKED["/opt/agentbox/skills/prose-sanitiser,<br/>returned even when it does not exist<br/>harness.rs:23"]
    JOIN["scripts_dir joined with the script name,<br/>for example score_synthid.py or clean_ctrlregen.py<br/>harness.rs:130"]
    NOCWD["INVARIANT: there is no working-directory fallback<br/>harness.rs:29"]
    SD --> ENVS --> FROM --> OV
    OV -->|yes| USE
    OV -->|no| BAKED
    USE --> JOIN
    BAKED --> JOIN
    BAKED --> NOCWD
```

**What it shows.** Two sources, in order: the override, then the image path the
scripts are baked at. There is no third. The answer is returned without checking
that it exists, so a missing script surfaces later as the interpreter's own error.

**Why it is this way.** The Rust binaries live outside the skill, so the location
is configuration, not the location of a source file. A development checkout sets
`PROSE_SANITISER_SCRIPTS_DIR`
(`../prose-sanitiser/crates/media/src/image/harness.rs:25`-`35`). The earlier
fallback to `skills/prose-sanitiser` in the working directory was removed
(`../prose-sanitiser/CHANGELOG.md:19`-`21`).

**Invariant:** with no override the answer is always the absolute baked path, and
a relative override is ignored. Two tests assert exactly that
(`../prose-sanitiser/crates/media/src/image/harness.rs:442`-`452`, then `:464`-`478`).

## PS-04.3 Upstream directory precedence per backend

```mermaid
flowchart TB
    subgraph SY["reverse-SynthID scorer"]
        SY1["explicit synthid_dir, else env_dir REVERSE_SYNTHID_DIR,<br/>none means no score<br/>harness.rs:123"]
        SY2["no existence check, the path goes to the script as is<br/>harness.rs:135"]
    end
    subgraph CR["CtrlRegen"]
        CR1["explicit upstream_dir, else env_dir NOAI_WATERMARK_DIR,<br/>none means not configured<br/>harness.rs:209"]
        CR2["resolve_dir: expand ~/ then canonicalise,<br/>must be a directory<br/>harness.rs:219"]
    end
    subgraph MD["MarkDiffusion"]
        MD1["explicit upstream_dir, else env_dir MARKDIFFUSION_DIR,<br/>none is allowed and the script runs without one<br/>harness.rs:310"]
        MD2["when given it must be a directory<br/>harness.rs:315"]
    end
    subgraph ML["MarkLLM"]
        ML1["explicit markllm_dir, else env_dir MARKLLM_DIR<br/>rewrite_text.rs:136"]
        ML2["expand ~/ then canonicalise, must hold a watermark directory<br/>markllm.rs:93"]
        ML3["must also hold .venv/bin/python<br/>markllm.rs:101"]
    end
    SY --> CR --> MD --> ML
    CAP["capabilities reports a backend present when env_dir is set,<br/>without checking the directory exists<br/>server/src/lib.rs:184"]
    ML --> CAP
```

**What it shows.** Every backend takes an explicit value first, then the filtered
environment variable. The checks that follow differ: CtrlRegen and MarkLLM refuse
a missing directory, MarkDiffusion tolerates having none, and the SynthID scorer
is handed the path unchecked and left to say it is unavailable.

**Why it is this way.** Each result is a small JSON object, with `available:
false` and a reason when the backend cannot run, because a missing optional
backend must never fail the cleaning around it
(`../prose-sanitiser/crates/cli/src/rewrite/markllm.rs:1`-`6`). The scorer returns
`None` for "not configured" so callers keep the default of no score
(`../prose-sanitiser/crates/media/src/image/harness.rs:118`-`121`).

**Invariant:** a relative `*_DIR` environment value yields "not configured" for
both the scorer and CtrlRegen, and a test pins that
(`../prose-sanitiser/crates/media/src/image/harness.rs:481`-`493`).

**Open:** the server reports `ctrlregen`, `diffusion`, `synthid` and `markllm` as
present from the variable alone
(`../prose-sanitiser/crates/server/src/lib.rs:184`-`188`), so a stable path that
names a missing directory reads as a capability that then fails at run time with
"dir not found" (`../prose-sanitiser/crates/media/src/image/harness.rs:221`-`224`).
Nothing states whether the capability report is meant to promise a usable backend.

## PS-04.4 One harness run

```mermaid
sequenceDiagram
    autonumber
    participant CR as run_ctrlregen_clean<br/>harness.rs:208
    participant SD as scripts_dir<br/>harness.rs:36
    participant VP as venv_python<br/>harness.rs:61
    participant RC as run_capture<br/>proc.rs:138
    participant PP as parse_payload<br/>harness.rs:86
    CR->>CR: upstream from the option or env_dir, else not configured (harness.rs:209-218)
    CR->>CR: resolve_dir, then must be a directory (harness.rs:219-225)
    CR->>SD: where is clean_ctrlregen.py (harness.rs:227)
    SD-->>CR: the override or the baked directory
    CR->>VP: the checkout (harness.rs:251)
    VP-->>CR: the checkout venv python if present, else the default interpreter (harness.rs:61-68)
    CR->>RC: interpreter, args, torch rlimits, timeout, no stdin (harness.rs:250-256)
    RC-->>CR: captured output, or a timeout or spawn error
    CR->>PP: result and the adapter name (harness.rs:257)
    PP-->>CR: JSON payload, or available false with a reason (harness.rs:86-116)
    CR->>CR: on success, available true is added to the payload (harness.rs:258-262)
```

**What it shows.** The sequence is the same for CtrlRegen and MarkDiffusion. The
checkout's own virtual environment is preferred so that torch and diffusers are
importable. Failures of any kind come back as data, not as an error.

**Why it is this way.** The payload parser reads a non-zero exit as unavailable
with up to 2000 characters of stderr, treats empty stdout as an empty object and
reports invalid JSON by adapter name
(`../prose-sanitiser/crates/media/src/image/harness.rs:100`-`115`). The scorer
differs: exit 3 is the script's own "not configured" signal and is not a failure
(`:149`-`152`).

**Invariant:** a child that exceeds its wall-clock budget is reported as
`timed out after Ns`, and the CtrlRegen and MarkDiffusion budgets default to 3600
seconds, the scorer's to 180 (`../prose-sanitiser/crates/media/src/image/harness.rs:92`-`97`,
`:202`, `:299` and `:142`).

## PS-04.5 Interpreter and resource limits

```mermaid
flowchart TB
    PY["default_python<br/>harness.rs:53"]
    P1["PROSE_SANITISER_PYTHON when non-empty, any value<br/>harness.rs:54"]
    P2["else python3 found on PATH, else the bare name python3<br/>harness.rs:57"]
    VE["venv_python: upstream/.venv/bin/python if it is a file<br/>harness.rs:63"]
    L1["default_child: 4 GiB address space, 2 GiB file size<br/>proc.rs:54"]
    L2["torch_child: 32 GiB address space, 2 GiB file size<br/>proc.rs:62"]
    L3["MarkLLM: address space unlimited unless<br/>WATERMARKS_MARKLLM_RLIMIT_AS is set<br/>markllm.rs:54"]
    USE1["SynthID scorer, with the default interpreter<br/>harness.rs:138"]
    USE2["CtrlRegen and MarkDiffusion, with the venv or default<br/>harness.rs:250"]
    USE3["MarkLLM, with its own venv only<br/>markllm.rs:122"]
    PY --> P1 --> P2
    VE --> PY
    PY --> USE1
    VE --> USE2
    L1 --> USE1
    L2 --> USE2
    L3 --> USE3
    OPEN["OPEN: the stable-path rule is not applied to this variable<br/>harness.rs:54"]
    P1 --> OPEN
```

**What it shows.** The scorer always uses the default interpreter. The two pixel
removers prefer the checkout's venv. MarkLLM requires its venv and has no
fallback, so a checkout without one is reported as missing. Memory caps differ by
backend because torch needs a large address space.

**Why it is this way.** A crafted file must not make a child exhaust host memory
or fill the disk, so each family carries a cap, but the torch ones are larger and
env-overridable (`../prose-sanitiser/crates/media/src/proc.rs:19`-`30`). The
MarkLLM address-space cap is opt-in for the same reason
(`../prose-sanitiser/crates/cli/src/rewrite/markllm.rs:52`-`53`).

**Open:** `PROSE_SANITISER_PYTHON` is taken verbatim when it is non-empty
(`../prose-sanitiser/crates/media/src/image/harness.rs:54`-`56`), whereas every
`*_DIR` variable is refused when relative. Nothing states whether a relative
interpreter value is meant to be accepted.

**Debt:** the two arms of `markllm_limits` take their file-size cap from different
variables. With an address cap it is the CtrlRegen one, `torch_child().file_size`
(`../prose-sanitiser/crates/cli/src/rewrite/markllm.rs:58`), and without one it is
`WATERMARKS_CHILD_RLIMIT_FSIZE` (`:64`).

## PS-04.6 Where a leading tilde is and is not expanded

```mermaid
flowchart TB
    ED["env_dir accepts a value that begins with ~/<br/>core/src/lib.rs:150"]
    SDIR["PROSE_SANITISER_SCRIPTS_DIR:<br/>PathBuf::from the value, no expansion<br/>harness.rs:47"]
    SYN["REVERSE_SYNTHID_DIR:<br/>passed to the script as is<br/>harness.rs:135"]
    CTR["NOAI_WATERMARK_DIR and MARKDIFFUSION_DIR:<br/>expand_user then canonicalise<br/>harness.rs:80"]
    MLL["MARKLLM_DIR:<br/>expand_user then canonicalise<br/>markllm.rs:93"]
    DR1["DRIFT: server README says ~/ paths are accepted<br/>for the scripts directory<br/>server/README.md:58"]
    DR2["DRIFT: expand_user exists but this path never calls it<br/>harness.rs:71"]
    ED --> SDIR
    ED --> SYN
    ED --> CTR
    ED --> MLL
    SDIR --> DR1
    SDIR --> DR2
```

**What it shows.** The filter lets a `~/` value through for all five variables,
and three of the five then expand it against `$HOME`. The scripts directory and
the SynthID directory do not.

**Why it is this way.** The `CHANGELOG` says absolute and `~/` values are
honoured (`../prose-sanitiser/CHANGELOG.md:34`-`35`), and the helper's doc says
`~/` expands against `$HOME`, not the working directory
(`../prose-sanitiser/crates/core/src/lib.rs:141`-`142`). The expansion lives in the
callers: `expand_user` is defined twice, once for the media crate
(`../prose-sanitiser/crates/media/src/image/harness.rs:71`-`78`) and once for
MarkLLM (`../prose-sanitiser/crates/cli/src/rewrite/markllm.rs:154`-`161`).

**Drift:** the server README documents `PROSE_SANITISER_SCRIPTS_DIR` as accepting
absolute or `~/` paths (`../prose-sanitiser/crates/server/README.md:58`). For a
`~/` value `scripts_dir_from` builds the path from the literal text
(`../prose-sanitiser/crates/media/src/image/harness.rs:47`), so the script path
passed to the interpreter begins with a relative `~` component, which is the
working-directory resolution the rule exists to prevent.

**Open:** `REVERSE_SYNTHID_DIR` with a `~/` value reaches the scorer script
unexpanded (`../prose-sanitiser/crates/media/src/image/harness.rs:125`, then `:135`).
The script is not part of this repository, so whether it expands the tilde is not
established here.

**Debt:** `expand_user` is duplicated in two crates
(`../prose-sanitiser/crates/media/src/image/harness.rs:71`, and
`../prose-sanitiser/crates/cli/src/rewrite/markllm.rs:154`), and `is_stable_dir_path`
lives in a third (`../prose-sanitiser/crates/core/src/lib.rs:149`), so the
acceptance rule and the expansion rule can diverge, as above.
