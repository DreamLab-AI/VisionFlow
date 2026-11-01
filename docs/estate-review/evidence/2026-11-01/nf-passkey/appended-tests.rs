#[cfg(test)]
mod passkey_audit {
    use super::*;
    fn sample_key() -> (Vec<u8>, Vec<u8>) {
        let sk=p256::ecdsa::SigningKey::from_bytes((&[7u8;32]).into()).unwrap();
        let point=sk.verifying_key().to_sec1_point(false);
        (point.x().unwrap().to_vec(),point.y().unwrap().to_vec())
    }
    fn coords_map(extra: &[u8], extra_pairs: u8) -> Vec<u8> {
        let (x,y)=sample_key();let mut b=vec![0xa2+extra_pairs];b.extend(extra);
        b.extend([0x21,0x58,0x20]);b.extend(x);b.extend([0x22,0x58,0x20]);b.extend(y);b
    }
    fn attestation(fmt:&str,unknown:bool)->Vec<u8>{
        let mut b=vec![if unknown{0xa4}else{0xa3},0x63];b.extend(b"fmt");b.push(0x60+fmt.len() as u8);b.extend(fmt.as_bytes());
        b.push(0x68);b.extend(b"authData");b.extend([0x58,37]);b.extend([0u8;37]);b.push(0x67);b.extend(b"attStmt");b.push(0xa0);
        if unknown { b.extend([0x61,b'x',0x59,0xff,0xff]); } b
    }
    // register_verify's SHA-256 statements are inserted verbatim by prepare_tests.py.
    fn stored_salt(server_secret:&str,credential_id_b64:&str,pubkey:&str)->String {
                let mut h = Sha256::new();
        h.update(b"nostr-bbs-prf-salt-v1\0");
        h.update(server_secret.as_bytes());
        h.update(credential_id_b64.as_bytes());
        h.update(pubkey.as_bytes());
        array_to_base64url(&h.finalize())
    }
    #[test]fn changed_registration_salt_changes_identity_even_with_fixed_prf(){
        let issued=array_to_base64url(&[3u8;32]);let prf=[5u8;32];
        let registered=nostr_bbs_core::derive_from_prf(&prf,issued.as_bytes()).unwrap();
        let stored=stored_salt("synthetic-server-secret","synthetic-credential",&registered.public.to_hex());
        assert_ne!(issued,stored);
        let login=nostr_bbs_core::derive_from_prf(&prf,stored.as_bytes()).unwrap();
        assert_ne!(registered.public.to_hex(),login.public.to_hex());
        assert_eq!(registered.public.to_hex(),nostr_bbs_core::derive_from_prf(&prf,issued.as_bytes()).unwrap().public.to_hex());
    }
    #[test]fn empty_secret_unknown_salt_is_publicly_reproducible(){
        let pk="a".repeat(64);let mut h=Sha256::new();h.update(b"nostr-bbs-prf-salt-fallback-v1\0");h.update(pk.as_bytes());
        assert_eq!(deterministic_salt_for(&pk,""),array_to_base64url(&h.finalize()));
        assert_ne!(stored_salt("","synthetic-credential",&pk),deterministic_salt_for(&pk,""));
    }
    #[test]fn real_secret_changes_both_salt_constructions(){
        let pk="a".repeat(64);assert_ne!(stored_salt("one","cred",&pk),stored_salt("two","cred",&pk));
        assert_ne!(deterministic_salt_for(&pk,"one"),deterministic_salt_for(&pk,"two"));
    }
    #[test]fn cose_coordinates_without_algorithm_curve_or_type_are_accepted(){assert!(parse_cose_p256_key(&array_to_base64url(&coords_map(&[],0))).is_ok());}
    #[test]fn contradictory_cose_metadata_is_ignored(){
        // kty=RSA(3), alg=RS256(-257), crv=99, followed by valid P-256 coordinates.
        let b=coords_map(&[0x01,0x03,0x03,0x39,0x01,0x00,0x20,0x18,0x63],3);
        assert!(parse_cose_p256_key(&array_to_base64url(&b)).is_ok());
    }
    #[test]fn duplicate_x_coordinate_uses_last_value(){
        let mut extra=vec![0x21,0x58,0x20];extra.extend([0u8;32]);let b=coords_map(&extra,1);
        assert!(parse_cose_p256_key(&array_to_base64url(&b)).is_ok());
    }
    #[test]fn invalid_curve_point_is_rejected(){let mut b=vec![0xa2,0x21,0x58,0x20];b.extend([0u8;32]);b.extend([0x22,0x58,0x20]);b.extend([0u8;32]);assert!(parse_cose_p256_key(&array_to_base64url(&b)).is_err());}
    #[test]fn packed_attestation_with_empty_statement_is_parsed(){let p=parse_attestation_object(&attestation("packed",false)).unwrap();assert_eq!(p.fmt,"packed");assert_eq!(p.auth_data.len(),37);}
    #[test]fn unknown_truncated_attestation_field_is_accepted(){assert!(parse_attestation_object(&attestation("none",true)).is_ok());}
    #[test]fn known_truncated_attestation_authdata_is_rejected(){let b=[0xa2,0x63,b'f',b'm',b't',0x64,b'n',b'o',b'n',b'e',0x68,b'a',b'u',b't',b'h',b'D',b'a',b't',b'a',0x58,37];assert!(parse_attestation_object(&b).is_err());}
}
