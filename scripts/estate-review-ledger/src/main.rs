//! Inventory explicitly scoped repositories and bind manual review claims to bytes.
//! This private audit CLI neither infers semantic completion nor changes source repos.

use pulldown_cmark::{CodeBlockKind, Event, Options, Parser, Tag, TagEnd};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeSet,
    env,
    error::Error,
    fs,
    path::{Component, Path, PathBuf},
    process::Command,
};

type Result<T> = std::result::Result<T, Box<dyn Error>>;

fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn relative(path: &str) -> Result<&Path> {
    let p = Path::new(path);
    if p.as_os_str().is_empty() || p.components().any(|c| !matches!(c, Component::Normal(_))) {
        return Err(format!("not a clean relative path: {path}").into());
    }
    Ok(p)
}

fn git(root: &Path, args: &[&str]) -> Result<String> {
    let out = Command::new("git")
        .arg("-C")
        .arg(root)
        .args(args)
        .output()?;
    if !out.status.success() {
        return Err(format!(
            "git {} in {}: {}",
            args.join(" "),
            root.display(),
            String::from_utf8_lossy(&out.stderr)
        )
        .into());
    }
    Ok(String::from_utf8(out.stdout)?)
}

fn field<'a>(v: &'a Value, key: &str) -> Result<&'a str> {
    v[key]
        .as_str()
        .ok_or_else(|| format!("missing string field {key}").into())
}

fn array<'a>(v: &'a Value, key: &str) -> Result<&'a Vec<Value>> {
    v[key]
        .as_array()
        .ok_or_else(|| format!("missing array field {key}").into())
}

fn candidate(path: &str) -> bool {
    let lower = path.to_ascii_lowercase();
    let p = Path::new(&lower);
    lower.ends_with(".md")
        && (p.file_name().is_some_and(|n| {
            n.to_string_lossy()
                .split(|c: char| !c.is_ascii_alphanumeric())
                .any(|token| {
                    token == "adr"
                        || token.strip_prefix("adr").is_some_and(|rest| {
                            !rest.is_empty() && rest.bytes().all(|b| b.is_ascii_digit())
                        })
                })
        }) || p
            .components()
            .any(|c| c.as_os_str() == "adr" || c.as_os_str() == "adrs"))
}

fn category(path: &str) -> &'static str {
    let lower = path.to_ascii_lowercase();
    if lower.starts_with("docs/estate-review/")
        || lower.starts_with("docs/estate-closeout/")
        || lower.contains("/evidence/")
    {
        return "audit-evidence-or-support";
    }
    let name = Path::new(&lower)
        .file_stem()
        .unwrap_or_default()
        .to_string_lossy();
    let numbered = name
        .split('-')
        .any(|s| !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit()));
    if numbered
        && (name.starts_with("adr-")
            || (name.as_bytes().first().is_some_and(u8::is_ascii_digit)
                && Path::new(&lower)
                    .components()
                    .any(|c| c.as_os_str() == "adr" || c.as_os_str() == "adrs")))
    {
        if lower.contains("archive/") || lower.contains("archived/") {
            "historical-decision-candidate"
        } else {
            "decision-candidate"
        }
    } else {
        "support-or-unclassified"
    }
}

/// Parse rendered Markdown headings and Mermaid blocks, excluding fenced examples.
fn units(text: &str) -> (Vec<Value>, Vec<Value>) {
    let mut headings = Vec::new();
    let mut panels = Vec::new();
    let mut heading: Option<(usize, String)> = None;
    let mut current = String::new();
    let mut mermaid: Option<(usize, String)> = None;
    for (event, range) in
        Parser::new_ext(text, Options::ENABLE_YAML_STYLE_METADATA_BLOCKS).into_offset_iter()
    {
        match event {
            Event::Start(Tag::Heading { .. }) => {
                heading = Some((range.start, String::new()));
            }
            Event::End(TagEnd::Heading(_)) => {
                if let Some((start, title)) = heading.take() {
                    current = title.clone();
                    headings.push(json!({"ordinal":headings.len()+1,"line":text[..start].bytes().filter(|b|*b==b'\n').count()+1,"title":title}));
                }
            }
            Event::Start(Tag::CodeBlock(CodeBlockKind::Fenced(info)))
                if info.trim() == "mermaid" =>
            {
                mermaid = Some((range.start, String::new()));
            }
            Event::End(TagEnd::CodeBlock) => {
                if let Some((start, body)) = mermaid.take() {
                    panels.push(json!({"id":current.split_whitespace().next().unwrap_or(""),"heading":current,"ordinal":panels.len()+1,"line":text[..start].bytes().filter(|b|*b==b'\n').count()+1,"body_sha256":digest(body.as_bytes())}));
                }
            }
            Event::Text(t) | Event::Code(t) => {
                if let Some((_, title)) = &mut heading {
                    title.push_str(&t);
                }
                if let Some((_, body)) = &mut mermaid {
                    body.push_str(&t);
                }
            }
            Event::SoftBreak | Event::HardBreak => {
                if let Some((_, title)) = &mut heading {
                    title.push(' ');
                }
            }
            _ => {}
        }
    }
    (headings, panels)
}

fn claim_state(source_hash: &str, report_hash: &str, claim: &Value) -> &'static str {
    if claim["source_sha256"].as_str() != Some(source_hash) {
        "stale-source"
    } else if claim["report_sha256"].as_str() != Some(report_hash) {
        "stale-report"
    } else {
        "assessment-recorded"
    }
}

fn apply_claims(
    record: &mut Value,
    claims: &[Value],
    reports: &Path,
    kind: &str,
    seen: &mut BTreeSet<usize>,
) -> Result<()> {
    record["review"] = json!("unreconciled");
    let mut matches = Vec::new();
    for (index, claim) in claims.iter().enumerate() {
        if claim["kind"] != kind
            || claim["repo"] != record["repo"]
            || claim["path"] != record["path"]
        {
            continue;
        }
        let report_path = relative(field(claim, "report")?)?;
        let report = fs::read(reports.join(report_path))?;
        let state = claim_state(field(record, "sha256")?, &digest(&report), claim);
        let coverage = field(claim, "coverage")?;
        if (kind == "adr" && coverage != "whole-document")
            || (kind == "topic" && coverage != "selected-panels")
        {
            return Err(format!("invalid coverage {coverage} for {kind}").into());
        }
        if field(claim, "basis")?.trim().is_empty() {
            return Err("claim needs a review basis".into());
        }
        if kind == "topic" {
            let selected = array(claim, "panels")?;
            if selected.is_empty() {
                return Err("empty panel claim".into());
            }
            for id in selected {
                if !array(record, "panels")?.iter().any(|p| p["id"] == *id) {
                    return Err(format!("unknown panel {id}").into());
                }
            }
        }
        seen.insert(index);
        matches.push(json!({"report":report_path,"state":state,"coverage":coverage,"panels":claim["panels"],"basis":claim["basis"]}));
    }
    if matches.iter().any(|m| m["state"] == "assessment-recorded") {
        record["review"] = json!("assessment-recorded");
    } else if !matches.is_empty() {
        record["review"] = json!("stale-evidence");
    }
    let key = if kind == "adr" { "sections" } else { "panels" };
    for unit in record[key].as_array_mut().ok_or("missing units")? {
        let assessed: Vec<_> = matches
            .iter()
            .filter(|m| {
                m["state"] == "assessment-recorded"
                    && (kind == "adr"
                        || m["panels"]
                            .as_array()
                            .is_some_and(|a| a.contains(&unit["id"])))
            })
            .map(|m| m["report"].clone())
            .collect();
        unit["review"] = json!(if assessed.is_empty() {
            "unreconciled"
        } else {
            "assessment-recorded"
        });
        unit["reports"] = json!(assessed);
    }
    record["claims"] = json!(matches);
    if kind == "topic"
        && record["review"] == "assessment-recorded"
        && array(record, "panels")?
            .iter()
            .any(|p| p["review"] != "assessment-recorded")
    {
        record["review"] = json!("partial-assessment-recorded");
    }
    Ok(())
}

fn scan(workspace: &Path, vf: &Path, scope: &Value, claims: &[Value]) -> Result<Value> {
    let mut repos = Vec::new();
    let mut records = Vec::new();
    let mut topics = Vec::new();
    let mut seen = BTreeSet::new();
    let mut names = BTreeSet::new();
    let mut roots = BTreeSet::new();
    for repo in array(scope, "repositories")? {
        let name = field(repo, "name")?;
        let local = relative(field(repo, "local")?)?;
        if !names.insert(name) {
            return Err(format!("duplicate scope name {name}").into());
        }
        let root = if name == "VisionFlow" {
            vf.to_path_buf()
        } else {
            workspace.join(local)
        };
        if !root.exists() {
            repos.push(
                json!({"name":name,"local":local,"available":false,"reason":"checkout missing"}),
            );
            continue;
        }
        let real = fs::canonicalize(&root)?;
        if !roots.insert(real.clone()) {
            return Err(format!("duplicate canonical checkout {}", real.display()).into());
        }
        let top = fs::canonicalize(git(&root, &["rev-parse", "--show-toplevel"])?.trim())?;
        if top != real {
            return Err(format!("scope path is not a Git root: {}", root.display()).into());
        }
        let head = git(&root, &["rev-parse", "HEAD"])?;
        let status = git(&root, &["status", "--porcelain", "--untracked-files=all"])?;
        let paths = git(
            &root,
            &[
                "ls-files",
                "--cached",
                "--others",
                "--exclude-standard",
                "-z",
            ],
        )?;
        let mut scanned = 0;
        for path in paths
            .split('\0')
            .filter(|s| !s.is_empty())
            .collect::<BTreeSet<_>>()
        {
            let rp = relative(path)?;
            let diagram = name == "VisionFlow"
                && path.starts_with("docs/diagrams/")
                && path.ends_with(".md")
                && ![
                    "rendered",
                    "archive",
                    "hero",
                    "src",
                    "upgraded",
                    "regen-2026-06-14",
                    "triptych-src",
                ]
                .iter()
                .any(|x| rp.components().any(|c| c.as_os_str() == *x));
            if !candidate(path) && !diagram {
                continue;
            }
            let file = root.join(rp);
            if file.symlink_metadata()?.file_type().is_symlink() {
                return Err(
                    format!("candidate symlink needs explicit owner: {}", file.display()).into(),
                );
            }
            // Parent inventories must not absorb an independently owned nested checkout.
            if file
                .parent()
                .into_iter()
                .flat_map(Path::ancestors)
                .take_while(|p| *p != root)
                .any(|p| p.join(".git").exists())
            {
                continue;
            }
            let bytes = fs::read(&file)?;
            let text = std::str::from_utf8(&bytes)?;
            let (sections, panels) = units(text);
            if candidate(path) {
                let mut record = json!({"repo":name,"path":path,"sha256":digest(&bytes),"category":category(path),"sections":sections});
                apply_claims(&mut record, claims, vf, "adr", &mut seen)?;
                records.push(record);
                scanned += 1;
            }
            if diagram && !panels.is_empty() {
                // Catalogue topics carry an id in front matter; README diagrams are ancillary.
                let is_topic = text
                    .lines()
                    .take_while(|l| !l.starts_with("## "))
                    .any(|l| l.starts_with("id: "));
                if is_topic {
                    let mut record =
                        json!({"repo":name,"path":path,"sha256":digest(&bytes),"panels":panels});
                    apply_claims(&mut record, claims, vf, "topic", &mut seen)?;
                    topics.push(record);
                }
            }
        }
        let head_after = git(&root, &["rev-parse", "HEAD"])?;
        let status_after = git(&root, &["status", "--porcelain", "--untracked-files=all"])?;
        if head_after != head || status_after != status {
            return Err(format!("repository changed during scan: {name}").into());
        }
        repos.push(json!({"name":name,"local":local,"available":true,"head":head.trim(),"dirty":!status.trim().is_empty(),"candidate_files":scanned}));
    }
    if seen.len() != claims.len() {
        let missing: Vec<_> = (0..claims.len()).filter(|i| !seen.contains(i)).collect();
        return Err(format!("unmatched claim indices: {missing:?}").into());
    }
    let eligible: Vec<_> = records
        .iter()
        .filter(|r| {
            r["category"]
                .as_str()
                .is_some_and(|s| s.ends_with("decision-candidate"))
        })
        .collect();
    let panels: Vec<_> = topics
        .iter()
        .flat_map(|r| r["panels"].as_array().into_iter().flatten())
        .collect();
    let counts = json!({"scope_targets":repos.len(),"available_checkouts":repos.iter().filter(|r|r["available"]==true).count(),"candidate_files":records.len(),"numbered_decision_candidates":eligible.len(),"numbered_candidates_with_recorded_assessment":eligible.iter().filter(|r|r["review"]=="assessment-recorded").count(),"topics":topics.len(),"panels":panels.len(),"panels_with_recorded_assessment":panels.iter().filter(|p|p["review"]=="assessment-recorded").count(),"stale_claim_records":records.iter().chain(topics.iter()).filter(|r|r["review"]=="stale-evidence").count()});
    Ok(
        json!({"schema":1,"method":"Explicit scope; Git-owned Markdown; parsed headings exclude fenced examples; explicit human review claims require exact source/report hashes. Unreconciled is not unreviewed. Candidate counts are not a complete obligation denominator or overall task percentage.","counts":counts,"repositories":repos,"adr_records":records,"topics":topics}),
    )
}

fn main() -> Result<()> {
    let args: Vec<_> = env::args().skip(1).collect();
    if args.len() != 5 {
        return Err("usage: estate-review-ledger WORKSPACE VISIONFLOW_ROOT SCOPE_JSON CLAIMS_JSON OUTPUT_JSON".into());
    }
    let scope: Value = serde_json::from_slice(&fs::read(&args[2])?)?;
    let claims: Value = serde_json::from_slice(&fs::read(&args[3])?)?;
    let output = scan(
        Path::new(&args[0]),
        Path::new(&args[1]),
        &scope,
        claims.as_array().ok_or("claims must be an array")?,
    )?;
    let dest = PathBuf::from(&args[4]);
    let temporary = dest.with_extension("json.tmp");
    fs::write(&temporary, serde_json::to_vec_pretty(&output)?)?;
    fs::rename(temporary, dest)?;
    println!("{}", serde_json::to_string_pretty(&output["counts"])?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn markdown_examples_are_not_sections() {
        let(h,p)=units("---\ntitle: Example\n---\n# Actual\n```md\n## Fake\n```\n## Real `code`\n~~~mermaid\nflowchart LR\nA-->B\n~~~\n");
        assert_eq!(h.len(), 2);
        assert_eq!(h[1]["title"], "Real code");
        assert_eq!(p.len(), 1);
    }
    #[test]
    fn setext_and_duplicate_headings_are_retained() {
        let (h, _) = units("Title\n=====\n## Check\n## Check\n");
        assert_eq!(h.len(), 3);
        assert_ne!(h[1]["ordinal"], h[2]["ordinal"]);
    }
    #[test]
    fn examples_with_longer_fences_stay_inside() {
        let (h, _) = units("````md\n```\n## Fake\n```\n````\n## Actual\n");
        assert_eq!(h.len(), 1);
        assert_eq!(h[0]["title"], "Actual");
    }
    #[test]
    fn paths_cannot_escape_scope() {
        for p in ["../secret", "/tmp/file", "a/../../b", ""] {
            assert!(relative(p).is_err());
        }
        assert!(relative("docs/adr/A.md").is_ok());
    }
    #[test]
    fn evidence_copies_are_not_decisions() {
        assert_eq!(
            category("docs/estate-review/evidence/x/ADR-2001.md"),
            "audit-evidence-or-support"
        );
        assert_eq!(
            category("docs/archive/adr/ADR-001.md"),
            "historical-decision-candidate"
        );
        assert_eq!(category("docs/adr/TEMPLATE.md"), "support-or-unclassified");
    }
    #[test]
    fn candidates_include_nonstandard_decision_directories() {
        assert!(candidate("plugins/a/adrs/0001-contract.md"));
        assert!(candidate("docs/engineering/ADR-005.md"));
        assert!(!candidate("docs/adr/image.png"));
        assert!(!candidate("docs/26-headroom.md"));
        assert!(!candidate("knowledge/Quadratic Voting.md"));
        assert!(candidate("docs/P0-ADR071-PHASE3.md"));
        assert_eq!(
            category("docs/audits/05-phase2-and-adr054.md"),
            "support-or-unclassified"
        );
    }
    #[test]
    fn mentions_cannot_become_assessments() {
        let c = json!({"source_sha256":"a","report_sha256":"b"});
        assert_eq!(claim_state("a", "b", &c), "assessment-recorded");
        assert_eq!(claim_state("c", "b", &c), "stale-source");
        assert_eq!(claim_state("a", "c", &c), "stale-report");
    }
    #[test]
    fn sha256_known_vector() {
        assert_eq!(
            digest(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn isolated_repositories_preserve_scope_and_invalidate_changed_evidence() -> Result<()> {
        struct Scratch(PathBuf);
        impl Drop for Scratch {
            fn drop(&mut self) {
                let _ = fs::remove_dir_all(&self.0);
            }
        }
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos();
        let scratch = Scratch(env::temp_dir().join(format!(
            "estate-ledger-test-{}-{unique}",
            std::process::id()
        )));
        let vf = scratch.0.join("VisionFlow");
        fs::create_dir_all(vf.join("docs/adr"))?;
        fs::create_dir_all(vf.join("docs/diagrams/area"))?;
        let adr = "# Decision\n## Context\ntext\n```md\n## Example only\n```\n## Decision\ntext\n";
        fs::write(vf.join("docs/adr/ADR-001.md"), adr)?;
        fs::write(
            vf.join("report.md"),
            "Reviewed Context and Decision; open findings retained.",
        )?;
        let topic="---\nid: X-01\n---\n## X-01.1 One\n```mermaid\nflowchart LR\nA-->B\n```\n## X-01.2 Two\n```mermaid\nflowchart LR\nB-->C\n```\n";
        fs::write(vf.join("docs/diagrams/area/topic.md"), topic)?;
        git(&vf, &["init", "--quiet"])?;
        git(&vf, &["add", "."])?;
        git(
            &vf,
            &[
                "-c",
                "user.name=Fixture",
                "-c",
                "user.email=fixture@example.invalid",
                "-c",
                "core.hooksPath=/dev/null",
                "commit",
                "--quiet",
                "--no-gpg-sign",
                "-m",
                "fixture",
            ],
        )?;
        let nested = vf.join("nested");
        fs::create_dir_all(nested.join("docs/adr"))?;
        fs::write(nested.join("docs/adr/ADR-002.md"), "# Nested\n## Scope\n")?;
        git(&nested, &["init", "--quiet"])?;
        fs::create_dir_all(scratch.0.join("unrelated/docs/adr"))?;
        fs::write(
            scratch.0.join("unrelated/docs/adr/ADR-999.md"),
            "private unrelated",
        )?;
        let scope = json!({"repositories":[{"name":"VisionFlow","local":"VisionFlow"},{"name":"missing","local":"missing"}]});
        let report_hash = digest(&fs::read(vf.join("report.md"))?);
        let mut claims = vec![
            json!({"kind":"adr","repo":"VisionFlow","path":"docs/adr/ADR-001.md","source_sha256":digest(adr.as_bytes()),"report":"report.md","report_sha256":report_hash,"coverage":"whole-document","basis":"fixture source assessment"}),
            json!({"kind":"topic","repo":"VisionFlow","path":"docs/diagrams/area/topic.md","source_sha256":digest(topic.as_bytes()),"report":"report.md","report_sha256":report_hash,"coverage":"selected-panels","panels":["X-01.1"],"basis":"fixture panel assessment"}),
        ];
        let output = scan(&scratch.0, &vf, &scope, &claims)?;
        assert_eq!(output["counts"]["candidate_files"], 1);
        assert_eq!(output["counts"]["available_checkouts"], 1);
        assert_eq!(
            output["adr_records"][0]["sections"]
                .as_array()
                .unwrap()
                .len(),
            3
        );
        assert_eq!(output["counts"]["panels_with_recorded_assessment"], 1);
        assert_eq!(output["topics"][0]["review"], "partial-assessment-recorded");
        claims[1]["panels"] = json!(["X-01.9"]);
        assert!(scan(&scratch.0, &vf, &scope, &claims).is_err());
        claims.pop();
        fs::write(vf.join("report.md"), "changed report")?;
        assert_eq!(
            scan(&scratch.0, &vf, &scope, &claims)?["counts"]
                ["numbered_candidates_with_recorded_assessment"],
            0
        );
        claims[0]["report_sha256"] = json!(digest(b"changed report"));
        fs::write(vf.join("docs/adr/ADR-001.md"), "# Changed decision")?;
        assert_eq!(
            scan(&scratch.0, &vf, &scope, &claims)?["adr_records"][0]["claims"][0]["state"],
            "stale-source"
        );
        claims[0]["path"] = json!("docs/adr/ADR-404.md");
        assert!(scan(&scratch.0, &vf, &scope, &claims).is_err());
        Ok(())
    }
}
