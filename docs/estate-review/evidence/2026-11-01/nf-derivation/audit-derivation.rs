use nostr_bbs_core::{derive_subkey, SecretKey};

#[test]
fn tag_bump_does_not_invalidate_old_signatures() {
    let root = SecretKey::from_bytes([1; 32]).unwrap();
    let old = derive_subkey(&root, "agentbox-mirror-v1").unwrap();
    let new = derive_subkey(&root, "agentbox-mirror-v2").unwrap();
    assert_ne!(old.public_key(), new.public_key());
    let msg = [42; 32];
    let sig = old.sign(&msg).unwrap();
    old.public_key().verify(&msg, &sig).unwrap();
    assert!(new.public_key().verify(&msg, &sig).is_err());
    assert_eq!(old.as_bytes(), derive_subkey(&root, "agentbox-mirror-v1").unwrap().as_bytes());
}

#[test]
fn tags_are_exact_bytes_including_spaces_and_normalisation() {
    let root = SecretKey::from_bytes([1; 32]).unwrap();
    for (a, b) in [("agentbox-mirror-v1", " agentbox-mirror-v1 "), ("é", "e\u{301}")] {
        assert_ne!(derive_subkey(&root, a).unwrap().as_bytes(), derive_subkey(&root, b).unwrap().as_bytes());
    }
}

#[test]
fn same_root_and_tag_have_no_account_or_service_context() {
    let a = SecretKey::from_bytes([1; 32]).unwrap();
    let b = SecretKey::from_bytes([1; 32]).unwrap();
    assert_eq!(derive_subkey(&a, "device:known-id").unwrap().as_bytes(), derive_subkey(&b, "device:known-id").unwrap().as_bytes());
}
