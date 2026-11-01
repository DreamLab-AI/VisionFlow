#[actix_web::test]
async fn rdf_patch_checks_existing_and_new_policy_targets() {
    for existing in [true, false] {
        let f = Fixture::new().await;
        if !existing { f.storage.delete(POLICY).await.unwrap(); }
        else { f.storage.put(POLICY, Bytes::from(format!("<urn:owner> <{ACL}agent> <{}> .\n<urn:owner> <{ACL}accessTo> </secret.bin> .\n<urn:owner> <{ACL}mode> <{ACL}Control> .\n", f.owner)), "text/turtle").await.unwrap(); }
        let before = f.storage.get(POLICY).await.ok().map(|x| x.0);
        let deletes = if existing { format!("<urn:owner> <{ACL}accessTo> </secret.bin> .") } else { String::new() };
        let patch = format!("solid:deletes {{ {deletes} }} ; solid:inserts {{ <urn:owner> <{ACL}agent> <{}> .\n<urn:owner> <{ACL}accessTo> </elsewhere> .\n<urn:owner> <{ACL}mode> <{ACL}Control> . }}", f.owner);
        assert_eq!(f.call("PATCH", POLICY, "text/n3", &patch, &[]).await, 409, "existing {existing}");
        assert_eq!(f.storage.get(POLICY).await.ok().map(|x|x.0), before);
        // A correct-target result remains accepted, including the create-via-PATCH path.
        let valid = patch.replace("</elsewhere>", "</secret.bin>");
        assert_eq!(f.call("PATCH", POLICY, "text/n3", &valid, &[]).await, if existing {204} else {201});
        assert_eq!(f.call("GET", POLICY, "", "", &[]).await, 200);
    }
}

