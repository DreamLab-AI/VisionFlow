#[cfg(test)]
mod mesh_backup_audit {
    use super::*;
    #[test]
    fn role_key_names_are_not_selected_by_current_backup_allowlist() {
        let d = tempfile::tempdir().unwrap();
        fs::create_dir_all(d.path().join("run/secrets/ab-mesh")).unwrap();
        for name in [
            "federation.key",
            "NODE_FEDERATION_PRIVKEY",
            "identity.env",
            ".env",
        ] {
            fs::write(
                d.path().join("run/secrets/ab-mesh").join(name),
                "synthetic-not-a-key",
            )
            .unwrap();
        }
        let selected = collect(d.path()).unwrap();
        assert_eq!(selected.len(), 2);
        assert!(selected
            .iter()
            .all(|p| ["identity.env", ".env"].contains(&p.file_name().unwrap().to_str().unwrap())));
        println!("OBS backup_collect_omits_federation_key_names_even_inside_root; identity_env_and_dotenv_controls_selected");
    }
}
