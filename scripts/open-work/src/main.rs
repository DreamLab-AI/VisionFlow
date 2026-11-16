//! Validates the estate open-work register and generates its views.
//!
//! The register (`register.jsonl`) is the one record of open, parked and
//! recently closed work across the estate. Each line is one item with one
//! canonical id. `seams.json` beside it defines the components and the seams
//! (contracts between components) an item may be filed under.
//!
//! ```text
//! open-work <dir>                 regenerate <dir>/README.md and <dir>/by-seam.md
//! open-work <dir> --check         validate, and fail if either view is stale
//! open-work <dir> --board <file>  also write the VisionClaw and agentbox board
//! ```
//!
//! Views are never edited by hand: change the register and run this tool.

use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::{env, fs, path::Path, process::ExitCode};

const KINDS: &[&str] = &["todo", "finding", "duplicate", "parked", "gap"];
const STATUSES: &[&str] = &[
    "open",
    "fixed-on-branch",
    "unverified",
    "parked",
    "fixed",
    "not-a-defect",
    "overtaken",
];
const BLOCKERS: &[&str] = &[
    "owner-decision",
    "owner-act",
    "engineering",
    "deployment",
    "device",
    "external",
    "stale",
    "untriaged",
];
/// Components whose live items make up the board written by `--board`.
const BOARD_COMPONENTS: &[&str] = &["visionclaw", "agentbox"];

/// One register line, with the fields the views need pulled out.
struct Item {
    id: String,
    aliases: Vec<String>,
    title: String,
    kind: String,
    components: Vec<String>,
    seam: String,
    blocker: String,
    status: String,
    checked: String,
    method: String,
    fix_branch: String,
    related: Vec<String>,
}

impl Item {
    /// Open work whose state was established by a check, not carried from a list.
    fn live(&self) -> bool {
        self.status == "open" || self.status == "fixed-on-branch"
    }
    fn closed(&self) -> bool {
        matches!(self.status.as_str(), "fixed" | "not-a-defect" | "overtaken")
    }
    fn waits_on_owner(&self) -> bool {
        self.live() && (self.blocker == "owner-decision" || self.blocker == "owner-act")
    }
}

struct Seam {
    id: String,
    title: String,
    between: Vec<String>,
    contract: String,
}

fn text(v: &Value, key: &str) -> String {
    v.get(key)
        .and_then(Value::as_str)
        .unwrap_or("")
        .trim()
        .to_string()
}

fn list(v: &Value, key: &str) -> Vec<String> {
    v.get(key)
        .and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default()
}

fn is_date(s: &str) -> bool {
    let b = s.as_bytes();
    b.len() == 10
        && b[4] == b'-'
        && b[7] == b'-'
        && b.iter()
            .enumerate()
            .all(|(i, c)| i == 4 || i == 7 || c.is_ascii_digit())
}

/// Reads `seams.json`: the allowed components and the seams in display order.
fn load_seams(dir: &Path) -> Result<(Vec<String>, Vec<Seam>), String> {
    let raw = fs::read_to_string(dir.join("seams.json")).map_err(|e| format!("seams.json: {e}"))?;
    let v: Value = serde_json::from_str(&raw).map_err(|e| format!("seams.json: {e}"))?;
    let components = list(&v, "components");
    let seams: Vec<Seam> = v
        .get("seams")
        .and_then(Value::as_array)
        .ok_or("seams.json: no `seams` array")?
        .iter()
        .map(|s| Seam {
            id: text(s, "id"),
            title: text(s, "title"),
            between: list(s, "between"),
            contract: text(s, "contract"),
        })
        .collect();
    if components.is_empty() || seams.is_empty() {
        return Err("seams.json: `components` and `seams` must both be non-empty".into());
    }
    if !seams.iter().any(|s| s.id == "none") {
        return Err("seams.json: a seam with id `none` is required".into());
    }
    Ok((components, seams))
}

/// Reads and validates the register. Every problem is reported, not only the first.
fn load_register(
    dir: &Path,
    components: &[String],
    seams: &[Seam],
) -> Result<Vec<Item>, Vec<String>> {
    let raw = match fs::read_to_string(dir.join("register.jsonl")) {
        Ok(r) => r,
        Err(e) => return Err(vec![format!("register.jsonl: {e}")]),
    };
    let mut errors = Vec::new();
    let mut items = Vec::new();
    let mut names: BTreeMap<String, usize> = BTreeMap::new();
    for (n, line) in raw
        .lines()
        .enumerate()
        .filter(|(_, l)| !l.trim().is_empty())
    {
        let at = n + 1;
        let v: Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(e) => {
                errors.push(format!("line {at}: not JSON: {e}"));
                continue;
            }
        };
        let verified = v.get("verified").cloned().unwrap_or(Value::Null);
        let item = Item {
            id: text(&v, "id"),
            aliases: list(&v, "aliases"),
            title: text(&v, "title"),
            kind: text(&v, "kind"),
            components: list(&v, "components"),
            seam: text(&v, "seam"),
            blocker: text(&v, "blocker"),
            status: text(&v, "status"),
            checked: text(&verified, "date"),
            method: text(&verified, "method"),
            fix_branch: text(&v, "fix_branch"),
            related: list(&v, "related"),
        };
        let who = if item.id.is_empty() {
            format!("line {at}")
        } else {
            item.id.clone()
        };
        if item.id.is_empty() {
            errors.push(format!("line {at}: no id"));
        }
        if item.title.is_empty() {
            errors.push(format!("{who}: no title"));
        }
        if !KINDS.contains(&item.kind.as_str()) {
            errors.push(format!(
                "{who}: kind `{}` is not one of {KINDS:?}",
                item.kind
            ));
        }
        if !STATUSES.contains(&item.status.as_str()) {
            errors.push(format!(
                "{who}: status `{}` is not one of {STATUSES:?}",
                item.status
            ));
        }
        if !BLOCKERS.contains(&item.blocker.as_str()) {
            errors.push(format!(
                "{who}: blocker `{}` is not one of {BLOCKERS:?}",
                item.blocker
            ));
        }
        if !seams.iter().any(|s| s.id == item.seam) {
            errors.push(format!(
                "{who}: seam `{}` is not defined in seams.json",
                item.seam
            ));
        }
        if item.components.is_empty() {
            errors.push(format!("{who}: no components"));
        }
        for c in &item.components {
            if !components.contains(c) {
                errors.push(format!(
                    "{who}: component `{c}` is not defined in seams.json"
                ));
            }
        }
        if !is_date(&item.checked) {
            errors.push(format!(
                "{who}: verified.date `{}` is not YYYY-MM-DD",
                item.checked
            ));
        }
        if item.method.is_empty() {
            errors.push(format!(
                "{who}: verified.method is empty; say how the state is known"
            ));
        }
        for name in std::iter::once(&item.id).chain(item.aliases.iter()) {
            if name.is_empty() {
                continue;
            }
            if let Some(first) = names.insert(name.clone(), at) {
                errors.push(format!(
                    "{who}: `{name}` is already an id or alias on line {first}"
                ));
            }
        }
        items.push(item);
    }
    let ids: BTreeSet<&str> = items.iter().map(|i| i.id.as_str()).collect();
    for item in &items {
        for r in &item.related {
            if !ids.contains(r.as_str()) {
                errors.push(format!(
                    "{}: related `{r}` is not an id in the register",
                    item.id
                ));
            }
        }
    }
    if errors.is_empty() {
        Ok(items)
    } else {
        Err(errors)
    }
}

/// Makes free text safe inside a Markdown table cell.
fn cell(s: &str) -> String {
    s.replace('|', "/").replace(['\n', '\r'], " ")
}

fn id_cell(i: &Item) -> String {
    if i.aliases.is_empty() {
        cell(&i.id)
    } else {
        format!("{} (also {})", cell(&i.id), cell(&i.aliases.join(", ")))
    }
}

fn state_cell(i: &Item) -> String {
    match (i.status.as_str(), i.fix_branch.is_empty()) {
        ("fixed-on-branch", false) => {
            format!("fixed on an unmerged branch: {}", cell(&i.fix_branch))
        }
        ("fixed-on-branch", true) => "fixed on an unmerged branch".to_string(),
        (s, _) => s.to_string(),
    }
}

fn blocker_label(b: &str) -> &str {
    match b {
        "owner-decision" => "Waiting on an owner decision",
        "owner-act" => "Waiting on an owner act",
        "engineering" => "Ordinary engineering: can start now",
        "deployment" => "Needs a live deployment",
        "device" => "Needs a device",
        "external" => "Waiting on an external party",
        "stale" => "Stale: close or reword",
        _ => "Not yet triaged",
    }
}

const GENERATED: &str = "<!-- Generated by scripts/open-work from register.jsonl. Do not edit: change the register and regenerate. -->";

fn live_table(out: &mut String, items: &[&Item], with_seam: bool) {
    let seam_head = if with_seam { " Seam |" } else { "" };
    let seam_rule = if with_seam { "---|" } else { "" };
    let _ = writeln!(
        out,
        "| Id | Item |{seam_head} Components | Blocked by | State | Checked |"
    );
    let _ = writeln!(out, "|---|---|{seam_rule}---|---|---|---|");
    for i in items {
        let seam = if with_seam {
            format!(" {} |", i.seam)
        } else {
            String::new()
        };
        let _ = writeln!(
            out,
            "| {} | {} |{seam} {} | {} | {} | {} |",
            id_cell(i),
            cell(&i.title),
            i.components.join(", "),
            i.blocker,
            state_cell(i),
            i.checked
        );
    }
    out.push('\n');
}

fn unverified_table(out: &mut String, items: &[&Item]) {
    let _ = writeln!(out, "| Id | Item | Components | Kind | Last recorded |");
    let _ = writeln!(out, "|---|---|---|---|---|");
    for i in items {
        let _ = writeln!(
            out,
            "| {} | {} | {} | {} | {} |",
            id_cell(i),
            cell(&i.title),
            i.components.join(", "),
            i.kind,
            i.checked
        );
    }
    out.push('\n');
}

fn readme(items: &[Item], components: &[String], seams: &[Seam]) -> String {
    let mut o = String::new();
    let count = |f: &dyn Fn(&Item) -> bool| items.iter().filter(|i| f(i)).count();
    let live = count(&|i| i.live());
    let unverified = count(&|i| i.status == "unverified");
    let parked = count(&|i| i.status == "parked");
    let closed = count(&|i| i.closed());
    let _ = writeln!(o, "{GENERATED}\n\n# Open work across the estate\n");
    let _ = writeln!(o, "One register for every open, parked and recently closed item in the estate. It replaces the to-do board, the audit findings index, the duplicate-implementations report and the open rows of the gap and park registers as the place where status is kept. Each item has one id; ids it carried in the older lists are kept as aliases.\n");
    let _ = writeln!(o, "Items are filed by **seam**: the contract between components that would have to hold, or be tested, for the item to close. An item that concerns one component alone is filed as inside that component. The seams are defined in [seams.json](seams.json) and listed in [by seam](by-seam.md).\n");
    let _ = writeln!(o, "## How to use this\n");
    let _ = writeln!(o, "- **To see what needs a decision,** read the next section. Answer in a pull request or in a session; an agent records the answer in the register.");
    let _ = writeln!(o, "- **To change an item,** edit its line in [register.jsonl](register.jsonl) and run `cargo run --locked --offline --manifest-path scripts/open-work/Cargo.toml -- docs/open-work`. A fix updates its item in the same pull request.");
    let _ = writeln!(o, "- **To add an item,** append a line. The gate refuses an id or alias used twice, a seam or component not in seams.json, and a state with no date or method.");
    let _ = writeln!(o, "- **Do not edit this page or [by seam](by-seam.md).** They are generated, and the gate fails when they differ from the register.\n");
    let _ = writeln!(o, "## Counts\n");
    let _ = writeln!(o, "| State | Items | Meaning |\n|---|---|---|");
    let _ = writeln!(o, "| Live | {live} | Open, or fixed on a branch that is not merged, as established by a check on the date shown |");
    let _ = writeln!(o, "| Unverified | {unverified} | Recorded as open in an older list and not re-checked since. Not known to be open |");
    let _ = writeln!(
        o,
        "| Parked | {parked} | Deliberately set aside with a re-open trigger |"
    );
    let _ = writeln!(o, "| Closed | {closed} | Fixed, overtaken, or found not to be a defect. Kept so the id still resolves |\n");

    let owner: Vec<&Item> = items.iter().filter(|i| i.waits_on_owner()).collect();
    let _ = writeln!(o, "## Waiting on the owner ({})\n", owner.len());
    for b in ["owner-decision", "owner-act"] {
        let rows: Vec<&Item> = owner.iter().copied().filter(|i| i.blocker == b).collect();
        if rows.is_empty() {
            continue;
        }
        let _ = writeln!(o, "### {} ({})\n", blocker_label(b), rows.len());
        live_table(&mut o, &rows, true);
    }

    let _ = writeln!(o, "## By seam\n");
    let _ = writeln!(
        o,
        "| Seam | Between | Live | Unverified |\n|---|---|---|---|"
    );
    for s in seams {
        let l = count(&|i| i.seam == s.id && i.live());
        let u = count(&|i| i.seam == s.id && i.status == "unverified");
        let between = if s.between.is_empty() {
            "one component".to_string()
        } else {
            s.between.join(", ")
        };
        let _ = writeln!(
            o,
            "| [{}](by-seam.md#{}) | {between} | {l} | {u} |",
            s.title, s.id
        );
    }
    let _ = writeln!(o, "\n## By component\n");
    let _ = writeln!(
        o,
        "An item that touches two components is counted under both.\n"
    );
    let _ = writeln!(o, "| Component | Live, on a seam | Live, inside the component | Unverified |\n|---|---|---|---|");
    for c in components {
        let has = |i: &Item| i.components.contains(c);
        let on = count(&|i| has(i) && i.live() && i.seam != "none");
        let inside = count(&|i| has(i) && i.live() && i.seam == "none");
        let u = count(&|i| has(i) && i.status == "unverified");
        if on + inside + u > 0 {
            let _ = writeln!(o, "| {c} | {on} | {inside} | {u} |");
        }
    }
    let _ = writeln!(o, "\n## Live work by what blocks it\n");
    let _ = writeln!(o, "| Blocked by | Items |\n|---|---|");
    for b in BLOCKERS {
        let n = count(&|i| i.live() && i.blocker == *b);
        if n > 0 {
            let _ = writeln!(o, "| {} | {n} |", blocker_label(b));
        }
    }
    let parked_rows: Vec<&Item> = items.iter().filter(|i| i.status == "parked").collect();
    if !parked_rows.is_empty() {
        let _ = writeln!(o, "\n## Parked\n");
        let _ = writeln!(
            o,
            "| Id | What | Components | Recorded |\n|---|---|---|---|"
        );
        for i in parked_rows {
            let _ = writeln!(
                o,
                "| {} | {} | {} | {} |",
                id_cell(i),
                cell(&i.title),
                i.components.join(", "),
                i.checked
            );
        }
    }
    o
}

fn by_seam(items: &[Item], components: &[String], seams: &[Seam]) -> String {
    let mut o = String::new();
    let _ = writeln!(o, "{GENERATED}\n\n# Open work by seam\n");
    let _ = writeln!(o, "A seam is a contract between components. Each section states the contract, then lists the live items filed under it, then the items an older list recorded as open that nobody has re-checked. Counts and the owner's decision queue are on the [front page](README.md).\n");
    for s in seams {
        let mine: Vec<&Item> = items.iter().filter(|i| i.seam == s.id).collect();
        let live: Vec<&Item> = mine.iter().copied().filter(|i| i.live()).collect();
        let unverified: Vec<&Item> = mine
            .iter()
            .copied()
            .filter(|i| i.status == "unverified")
            .collect();
        let closed = mine.iter().filter(|i| i.closed()).count();
        let _ = writeln!(o, "<a id=\"{}\"></a>\n\n## {}\n", s.id, s.title);
        let _ = writeln!(o, "{}\n", s.contract);
        if !s.between.is_empty() {
            let _ = writeln!(o, "Between: {}.\n", s.between.join(", "));
        }
        let _ = writeln!(
            o,
            "{} live, {} unverified, {} closed.\n",
            live.len(),
            unverified.len(),
            closed
        );
        if s.id != "none" {
            if !live.is_empty() {
                let _ = writeln!(o, "### Live\n");
                live_table(&mut o, &live, false);
            }
            if !unverified.is_empty() {
                let _ = writeln!(o, "### Recorded as open, not re-checked\n");
                unverified_table(&mut o, &unverified);
            }
            continue;
        }
        for c in components {
            let l: Vec<&Item> = live
                .iter()
                .copied()
                .filter(|i| i.components.first() == Some(c))
                .collect();
            let u: Vec<&Item> = unverified
                .iter()
                .copied()
                .filter(|i| i.components.first() == Some(c))
                .collect();
            if l.is_empty() && u.is_empty() {
                continue;
            }
            let _ = writeln!(o, "### {c}\n");
            if !l.is_empty() {
                live_table(&mut o, &l, false);
            }
            if !u.is_empty() {
                let _ = writeln!(o, "Recorded as open, not re-checked:\n");
                unverified_table(&mut o, &u);
            }
        }
    }
    o
}

fn board(items: &[Item]) -> String {
    let mut o = String::new();
    let rows: Vec<&Item> = items
        .iter()
        .filter(|i| {
            i.live()
                && i.components
                    .iter()
                    .any(|c| BOARD_COMPONENTS.contains(&c.as_str()))
        })
        .collect();
    let unverified = items
        .iter()
        .filter(|i| {
            i.status == "unverified"
                && i.components
                    .iter()
                    .any(|c| BOARD_COMPONENTS.contains(&c.as_str()))
        })
        .count();
    let _ = writeln!(o, "<!-- Generated by VisionFlow scripts/open-work from docs/open-work/register.jsonl. Do not edit: change the register in VisionFlow and regenerate. -->\n");
    let _ = writeln!(o, "# Unified TODO: VisionClaw and Agentbox\n");
    let _ = writeln!(o, "**This page is generated.** It is the view of the [estate open-work register](https://github.com/DreamLab-AI/VisionFlow/blob/main/docs/open-work/README.md) for items that touch VisionClaw or agentbox. To change a row, change its line in the register in the VisionFlow repository; do not edit this file.\n");
    let _ = writeln!(o, "{} live items, grouped by what blocks them. A further {unverified} items touching these components were recorded as open in older lists and have not been re-checked; they are in the register's [by-seam view](https://github.com/DreamLab-AI/VisionFlow/blob/main/docs/open-work/by-seam.md), not here.\n", rows.len());
    let _ = writeln!(o, "The board's history from 7 September to 15 November 2026, with the definition of each row id, is in the [archived board](archive/TODO-unified-2026-09-07-to-2026-11-15.md).\n");
    for b in BLOCKERS {
        let group: Vec<&Item> = rows.iter().copied().filter(|i| i.blocker == *b).collect();
        if group.is_empty() {
            continue;
        }
        let _ = writeln!(o, "## {} ({})\n", blocker_label(b), group.len());
        live_table(&mut o, &group, true);
    }
    o
}

/// Writes `content` to `path`, or with `check` reports whether it already matches.
fn emit(path: &Path, content: &str, check: bool) -> Result<(), String> {
    if check {
        let current = fs::read_to_string(path).unwrap_or_default();
        if current != content {
            return Err(format!(
                "{} is stale: regenerate it from the register",
                path.display()
            ));
        }
        return Ok(());
    }
    fs::write(path, content).map_err(|e| format!("{}: {e}", path.display()))
}

fn run(args: &[String]) -> Result<String, Vec<String>> {
    let mut dir = None;
    let mut check = false;
    let mut board_path = None;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--check" => check = true,
            "--board" => board_path = it.next().cloned(),
            other if dir.is_none() && !other.starts_with("--") => dir = Some(other.to_string()),
            other => return Err(vec![format!("unexpected argument `{other}`")]),
        }
    }
    let dir =
        dir.ok_or_else(|| vec!["usage: open-work <dir> [--check] [--board <file>]".to_string()])?;
    let dir = Path::new(&dir);
    let (components, seams) = load_seams(dir).map_err(|e| vec![e])?;
    let items = load_register(dir, &components, &seams)?;
    let mut errors = Vec::new();
    for (name, content) in [
        ("README.md", readme(&items, &components, &seams)),
        ("by-seam.md", by_seam(&items, &components, &seams)),
    ] {
        if let Err(e) = emit(&dir.join(name), &content, check) {
            errors.push(e);
        }
    }
    if let Some(p) = board_path {
        if let Err(e) = emit(Path::new(&p), &board(&items), check) {
            errors.push(e);
        }
    }
    if !errors.is_empty() {
        return Err(errors);
    }
    let live = items.iter().filter(|i| i.live()).count();
    Ok(format!(
        "open-work: {} items, {live} live, {} seams, {}",
        items.len(),
        seams.len(),
        if check {
            "views up to date"
        } else {
            "views written"
        }
    ))
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().skip(1).collect();
    match run(&args) {
        Ok(summary) => {
            println!("{summary}");
            ExitCode::SUCCESS
        }
        Err(errors) => {
            for e in &errors {
                eprintln!("open-work: {e}");
            }
            eprintln!("open-work: {} problem(s)", errors.len());
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SEAMS: &str = r#"{"components":["a","b"],"seams":[{"id":"wire","title":"Wire","between":["a","b"],"contract":"A and B agree."},{"id":"none","title":"Inside","between":[],"contract":"One component."}]}"#;

    fn line(id: &str, extra: &str) -> String {
        format!(
            r#"{{"id":"{id}","aliases":[],"title":"t","kind":"todo","components":["a"],"seam":"wire","blocker":"engineering","status":"open","verified":{{"date":"2026-11-15","method":"checked"}},"related":[]{extra}}}"#
        )
    }

    fn dir_with(register: &str) -> std::path::PathBuf {
        static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let n = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let d = env::temp_dir().join(format!("open-work-test-{}-{n}", std::process::id()));
        fs::create_dir_all(&d).unwrap();
        fs::write(d.join("seams.json"), SEAMS).unwrap();
        fs::write(d.join("register.jsonl"), register).unwrap();
        d
    }

    fn problems(register: &str) -> Vec<String> {
        let d = dir_with(register);
        let (c, s) = load_seams(&d).unwrap();
        load_register(&d, &c, &s).err().unwrap_or_default()
    }

    #[test]
    fn a_valid_register_has_no_problems() {
        assert!(problems(&format!("{}\n{}\n", line("X-1", ""), line("X-2", ""))).is_empty());
    }

    #[test]
    fn an_id_used_twice_is_refused() {
        let p = problems(&format!("{}\n{}\n", line("X-1", ""), line("X-1", "")));
        assert!(
            p.iter().any(|e| e.contains("already an id or alias")),
            "{p:?}"
        );
    }

    #[test]
    fn an_alias_that_is_another_items_id_is_refused() {
        let aliased = line("X-2", "").replace(r#""aliases":[]"#, r#""aliases":["X-1"]"#);
        let p = problems(&format!("{}\n{aliased}\n", line("X-1", "")));
        assert!(p.iter().any(|e| e.contains("`X-1` is already")), "{p:?}");
    }

    #[test]
    fn an_undefined_seam_or_component_is_refused() {
        let bad = line("X-1", "")
            .replace(r#""seam":"wire""#, r#""seam":"nope""#)
            .replace(r#"["a"]"#, r#"["z"]"#);
        let p = problems(&format!("{bad}\n"));
        assert!(p.iter().any(|e| e.contains("seam `nope`")), "{p:?}");
        assert!(p.iter().any(|e| e.contains("component `z`")), "{p:?}");
    }

    #[test]
    fn a_state_without_a_date_or_method_is_refused() {
        let bad = line("X-1", "")
            .replace("2026-11-15", "yesterday")
            .replace(r#""method":"checked""#, r#""method":"""#);
        let p = problems(&format!("{bad}\n"));
        assert!(p.iter().any(|e| e.contains("not YYYY-MM-DD")), "{p:?}");
        assert!(
            p.iter().any(|e| e.contains("verified.method is empty")),
            "{p:?}"
        );
    }

    #[test]
    fn a_related_id_must_exist() {
        let bad = line("X-1", "").replace(r#""related":[]"#, r#""related":["X-9"]"#);
        assert!(problems(&format!("{bad}\n"))
            .iter()
            .any(|e| e.contains("related `X-9`")));
    }

    #[test]
    fn check_fails_on_a_stale_view_and_passes_once_regenerated() {
        let d = dir_with(&format!("{}\n", line("X-1", "")));
        let arg = vec![d.to_string_lossy().to_string()];
        let mut checked = arg.clone();
        checked.push("--check".into());
        assert!(run(&checked).is_err(), "no views yet, so --check must fail");
        run(&arg).unwrap();
        run(&checked).unwrap();
        fs::write(
            d.join("register.jsonl"),
            format!(
                "{}\n",
                line("X-1", "").replace(r#""title":"t""#, r#""title":"changed""#)
            ),
        )
        .unwrap();
        assert!(
            run(&checked).is_err(),
            "the register changed, so the views are stale"
        );
    }

    #[test]
    fn unverified_items_are_not_counted_as_live() {
        let u = line("X-2", "").replace(r#""status":"open""#, r#""status":"unverified""#);
        let d = dir_with(&format!("{}\n{u}\n", line("X-1", "")));
        let (c, s) = load_seams(&d).unwrap();
        let items = load_register(&d, &c, &s).unwrap();
        let page = readme(&items, &c, &s);
        assert!(page.contains("| Live | 1 |"), "{page}");
        assert!(page.contains("| Unverified | 1 |"), "{page}");
    }
}
