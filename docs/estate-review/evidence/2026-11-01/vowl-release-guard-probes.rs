    #[test]
    fn audit_source_commit_is_shape_checked_not_resolved() {
        let t=fixture(); let path=t.path().join("docs/release-inventory.json");
        let mut p:serde_json::Value=serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        p["source_commit"]=serde_json::json!("0".repeat(40));
        fs::write(path,serde_json::to_vec(&p).unwrap()).unwrap();
        check(t.path()).unwrap();
    }
    #[test]
    fn audit_release_workflow_and_guard_are_outside_source_hash_set() {
        let t=fixture();let r=t.path();
        for path in [".github/workflows/release.yml","tools/release-inventory/src/main.rs"] {
            let p=r.join(path);fs::create_dir_all(p.parent().unwrap()).unwrap();fs::write(p,"changed control input").unwrap();
        }
        check(r).unwrap();
    }
    #[test]
    fn audit_package_guard_does_not_validate_name_or_generated_wasm() {
        let t=fixture();let r=t.path();
        fs::write(r.join("pkg/package.json"),r#"{"name":"unrelated-package","version":"0.2.0","files":["vowl_wasm_bg.wasm"]}"#).unwrap();
        fs::write(r.join("pkg/vowl_wasm_bg.wasm"),"not a wasm module").unwrap();
        package(r,"v0.2.0","rustc reviewed","wasm-pack reviewed").unwrap();
        assert_eq!(fs::read(r.join("pkg/vowl_wasm_bg.wasm")).unwrap(),b"not a wasm module");
    }
