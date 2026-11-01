
#[cfg(test)]
mod audit_lifecycle {
    use super::*;
    use nostr_bbs_core::{UnsignedEvent, verify_event_strict};
    use std::{future::Future, pin::pin, task::{Context, Poll, Waker}};

    // Local PrfSigner operations complete in one poll; a pending future fails
    // this harness rather than pretending to execute a browser extension.
    fn ready<F: Future>(future: F) -> F::Output {
        let mut future = pin!(future);
        match future.as_mut().poll(&mut Context::from_waker(Waker::noop())) {
            Poll::Ready(value) => value,
            Poll::Pending => panic!("unexpected pending local operation"),
        }
    }
    fn event(signer: &dyn Signer) -> UnsignedEvent {
        UnsignedEvent { pubkey: signer.public_key().into(), created_at: 1,
            kind: 42, tags: vec![], content: "synthetic audit action".into() }
    }
    #[test]
    fn retained_signer_produces_valid_event_after_logout() {
        let owner = Owner::new(); owner.with(|| {
            let s = BbsSigner::new(); s.login_with_key(&"01".repeat(32)).unwrap();
            let retained = s.get_signer().unwrap();
            s.logout(); assert!(!s.has_key()); assert!(s.get_signer().is_none());
            let signed = ready(retained.sign_event(event(retained.as_ref()))).unwrap();
            verify_event_strict(&signed).unwrap();
            assert_eq!(signed.pubkey, retained.public_key());
        });
    }
    #[test]
    fn retained_signer_survives_account_replacement() {
        let owner = Owner::new(); owner.with(|| {
            let s = BbsSigner::new(); s.login_with_key(&"01".repeat(32)).unwrap();
            let old = s.get_signer().unwrap();
            s.login_with_key(&"02".repeat(32)).unwrap();
            assert_ne!(s.pubkey_hex().unwrap(), old.public_key());
            let signed = ready(old.sign_event(event(old.as_ref()))).unwrap();
            verify_event_strict(&signed).unwrap();
        });
    }
    #[test]
    fn rejected_paste_leaves_previous_account_installed() {
        let owner = Owner::new(); owner.with(|| {
            let s = BbsSigner::new(); s.login_with_key(&"01".repeat(32)).unwrap();
            let before = s.pubkey_hex();
            assert!(s.login_with_key("invalid").is_err());
            assert_eq!(s.pubkey_hex(), before); assert!(s.get_signer().is_some());
        });
    }
    #[test]
    fn invalid_baked_key_leaves_previous_account_installed() {
        let owner = Owner::new(); owner.with(|| {
            let s = BbsSigner::new(); s.login_with_key(&"01".repeat(32)).unwrap();
            let before = s.pubkey_hex(); assert!(!s.adopt_baked_key(&[0; 32]));
            assert_eq!(s.pubkey_hex(), before);
        });
    }
    #[test]
    fn extension_adoption_accepts_nonhex_public_key() {
        assert_eq!(choose_adoption(false, Some(r#"{"publicKey":"not-a-key","isNip07":true}"#), true),
            AdoptChoice::Nip07("not-a-key".into()));
    }
    #[test]
    fn extension_adoption_ignores_schema_version() {
        let pk = "ab".repeat(32);
        let session = format!(r#"{{"_v":999,"publicKey":"{pk}","isNip07":true}}"#);
        assert_eq!(choose_adoption(false, Some(&session), true), AdoptChoice::Nip07(pk));
    }
    #[test]
    fn valid_local_key_is_adoptable_without_session_metadata() {
        assert_eq!(choose_adoption(true, None, false), AdoptChoice::LocalKey);
    }
    #[test]
    fn local_signer_rejects_mismatched_unsigned_author_control() {
        let owner = Owner::new(); owner.with(|| {
            let s = BbsSigner::new(); s.login_with_key(&"01".repeat(32)).unwrap();
            let signer = s.get_signer().unwrap(); let mut e = event(signer.as_ref());
            e.pubkey = parse_secret_key(&"02".repeat(32)).unwrap().public.to_hex();
            assert!(matches!(ready(signer.sign_event(e)), Err(nostr_bbs_core::signer::SignerError::PubkeyMismatch { .. })));
        });
    }
}
