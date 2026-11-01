//! Offline command-boundary fixture. Never contacts a forge.
use std::{env, fs, io::Write, path::PathBuf};
fn main() {
    let root = PathBuf::from(env::var("DREAM_AUDIT_GH_STATE").expect("fixture state required"));
    assert_eq!(
        fs::read_to_string(root.join("fixture-marker")).unwrap(),
        "offline-dream-audit"
    );
    let args: Vec<String> = env::args().skip(1).collect();
    let mut log = fs::OpenOptions::new()
        .append(true)
        .create(true)
        .open(root.join("commands.log"))
        .unwrap();
    writeln!(log, "{args:?}").unwrap();
    let mode = fs::read_to_string(root.join("mode")).unwrap();
    let first = args.first().map(String::as_str);
    let second = args.get(1).map(String::as_str);
    match (first, second) {
        (Some("pr"), Some("create")) => {
            if mode == "create-error" {
                eprintln!("fixture create error");
                std::process::exit(1);
            }
            fs::write(root.join("draft"), "true").unwrap();
            if mode != "empty-url" {
                println!("https://example.invalid/audit/repo/pull/1");
            }
        }
        (Some("pr"), Some("view")) => println!(
            "{}",
            if mode == "wrong-base" {
                "other"
            } else {
                "main"
            }
        ),
        (Some("api"), _) => {
            if mode == "api-error" {
                eprintln!("HTTP 403 fixture denied");
                std::process::exit(1);
            }
            if mode == "no-checks" {
                println!("{{\"required_status_checks\":{{\"contexts\":[]}}}}");
            } else {
                println!("{{\"required_status_checks\":{{\"contexts\":[\"ci\"]}}}}");
            }
        }
        (Some("pr"), Some("ready")) => {
            if mode == "ready-error" {
                eprintln!("fixture ready error");
                std::process::exit(1);
            }
            fs::write(
                root.join("draft"),
                if args.iter().any(|a| a == "--undo") {
                    "true"
                } else {
                    "false"
                },
            )
            .unwrap();
        }
        (Some("pr"), Some("merge")) => {
            if mode == "merge-error" {
                eprintln!("fixture merge error");
                std::process::exit(1);
            }
            fs::write(root.join("merge-requested"), "true").unwrap();
        }
        _ => {
            eprintln!("unexpected fixture command");
            std::process::exit(99);
        }
    }
}
