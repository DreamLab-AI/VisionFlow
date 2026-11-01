#![allow(dead_code)]
pub mod envmap;
pub mod identity;
pub mod mirror_key;
pub mod pyjson;
pub mod role_secret;
pub mod identity_port {
    pub mod acl;
    pub mod port;
    pub mod receipts;
    pub mod server;
}
mod backup;

#[cfg(test)]
mod audit {
    use super::identity_port::{
        acl::Acl,
        port::{KeyRing, Peer, Port},
        receipts::Receipts,
        server,
    };
    use nostr_bbs_mesh::{Envelope, MeshConfig, SeenIds};
    use serde_json::{json, Value};

    #[test]
    fn checked_in_mesh_block_is_not_library_compatible() {
        let text =
            std::fs::read_to_string(std::env::var("AUDIT_AGENTBOX_MANIFEST").unwrap()).unwrap();
        let err = MeshConfig::from_toml_str(&text).unwrap_err();
        assert!(err.to_string().contains("sequence"), "{err}");
        println!("OBS current_manifest_rejected: boolean honor_remote_moderation is not a list");
    }

    #[test]
    fn proposed_peers_are_ignored_and_do_not_establish_pins() {
        let text = format!("[mesh]\nmode='federated'\npeer_relays=['ws://insecure.example']\n[[mesh.peers]]\npubkey='{}'\nrelay='wss://peer.example'\nendpoint='wss://peer.example/graph'\norigin='https://viewer.example'\n", "a".repeat(64));
        let c = MeshConfig::from_toml_str(&text).unwrap();
        assert!(c.resolve().is_ok());
        assert!(c.allowed_remote_dids.is_empty());
        assert!(!c.is_allowed_remote(&"a".repeat(64)));
        assert_eq!(c.peer_relays, vec!["ws://insecure.example"]);
        println!("OBS proposed_peers_ignored; ws_relay_resolves; inbound_pin_remains_denied");
    }

    #[test]
    fn all_three_control_types_are_unknown() {
        for kind in ["dataspace_announce", "link_notice", "retraction"] {
            let e = json!({"v":1,"from":format!("did:nostr:{}", "a".repeat(64)),"to":format!("did:nostr:{}", "b".repeat(64)),"kind":kind,"body":{},"sequence":1});
            let error = Envelope::from_jcs_str(&e.to_string()).unwrap_err();
            assert!(error.to_string().contains("unknown variant"), "{error}");
        }
        println!("OBS all_three_control_types_rejected");
    }

    #[test]
    fn generic_envelope_drops_unknown_sequence_and_seen_ids_forget() {
        let e = Envelope::chat(&"a".repeat(64), &"b".repeat(64), "fixture");
        let mut v = serde_json::to_value(e).unwrap();
        v["sequence"] = json!(9);
        let decoded = Envelope::from_jcs_str(&v.to_string()).unwrap();
        assert!(serde_json::to_value(decoded)
            .unwrap()
            .get("sequence")
            .is_none());
        let mut seen = SeenIds::new(1);
        assert!(seen.insert("a"));
        assert!(!seen.insert("a"));
        assert!(seen.insert("b"));
        assert!(seen.insert("a"));
        assert!(SeenIds::new(1).insert("a"));
        println!("OBS top_level_sequence_dropped; id_cache_eviction_and_restart_allow_again");
    }

    fn fixture_port(allow_catalogue: bool) -> (tempfile::TempDir, Port) {
        let d = tempfile::tempdir().unwrap();
        std::fs::write(d.path().join("fixture.key"), "a".repeat(64)).unwrap();
        let kinds = if allow_catalogue {
            vec![30078]
        } else {
            vec![38410]
        };
        let acl = Acl::parse(&json!({"version":1,"keys":{"core":{"file":"fixture.key","required":true}},"callers":{"1000":{"name":"fixture","ops":{"sign_event":{"keys":["core"],"kinds":kinds}}}}}).to_string(),None).unwrap();
        let keys = KeyRing::load(&acl, d.path(), None).unwrap();
        let receipts = Receipts::new(d.path().join("receipts")).unwrap();
        (d, Port::new(acl, keys, receipts))
    }

    #[test]
    fn adding_kind_30078_alone_is_not_a_catalogue_signer() {
        let (_d, p) = fixture_port(true);
        let request = json!({"op":"sign_event","params":{"kind":30078,"tags":[["d","unrelated:namespace"]],"content":"not a catalogue","created_at":1}}).to_string();
        for pid in [111, 222] {
            let v = p.handle(
                Peer {
                    uid: 1000,
                    gid: 1000,
                    pid: Some(pid),
                },
                &request,
            );
            let ev: nostr_bbs_core::NostrEvent =
                serde_json::from_value(v["event"].clone()).unwrap();
            assert!(nostr_bbs_core::event::verify_event(&ev));
        }
        assert!(p
            .handle(
                Peer {
                    uid: 4242,
                    gid: 1000,
                    pid: Some(111)
                },
                &request
            )
            .get("refused")
            .is_some());
        let (_d, current) = fixture_port(false);
        assert!(current
            .handle(
                Peer {
                    uid: 1000,
                    gid: 1000,
                    pid: None
                },
                &request
            )
            .get("refused")
            .is_some());
        println!("OBS synthetic_30078_grant_signs_wrong_namespace_without_rev; same_uid_pids_equivalent; other_uid_and_ungranted_kind_refused");
    }

    #[tokio::test]
    async fn socket_parent_group_write_is_not_checked() {
        use std::os::unix::fs::PermissionsExt;
        let d = tempfile::tempdir().unwrap();
        std::fs::set_permissions(d.path(), std::fs::Permissions::from_mode(0o770)).unwrap();
        let path = d.path().join("fixture.sock");
        let l = server::bind(&server::Bind {
            path: path.clone(),
            gid: None,
        })
        .unwrap();
        drop(l);
        std::fs::remove_file(&path).unwrap();
        std::fs::set_permissions(d.path(), std::fs::Permissions::from_mode(0o777)).unwrap();
        assert!(server::bind(&server::Bind { path, gid: None }).is_err());
        println!("OBS socket_parent_0770_accepted_0777_refused; trusted_parent_group_is_external_precondition");
    }

    #[tokio::test]
    async fn real_socket_uses_kernel_credentials() {
        use std::sync::Arc;
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let (d, port) = fixture_port(true);
        // Tests execute as uid 1000 in this estate; fail visibly if that changes.
        let listener = server::bind(&server::Bind {
            path: d.path().join("real.sock"),
            gid: None,
        })
        .unwrap();
        let (tx, rx) = tokio::sync::oneshot::channel();
        let task = tokio::spawn(server::serve(Arc::new(port), listener, async {
            let _ = rx.await;
        }));
        let mut stream = tokio::net::UnixStream::connect(d.path().join("real.sock"))
            .await
            .unwrap();
        let req = json!({"op":"sign_event","params":{"kind":30078}}).to_string() + "\n";
        stream.write_all(req.as_bytes()).await.unwrap();
        let mut out = String::new();
        stream.read_to_string(&mut out).await.unwrap();
        let v: Value = serde_json::from_str(&out).unwrap();
        assert!(v.get("event").is_some(), "{v}");
        tx.send(()).unwrap();
        task.await.unwrap().unwrap();
        println!("OBS local_socket_real_SO_PEERCRED_fixture_signed; no_live_service_used");
    }
}
