use serde_json::{Value, json};
use std::{fs, path::PathBuf, process::Command};
use video_production_runtime::{comfy::sha256_hex, paths::Root, seam_qa};

#[test]
fn seam_bounds_and_colour_contract() {
    let path = PathBuf::from(std::env::var("MEDIA_AUDIT_ROOT").unwrap());
    fs::create_dir_all(&path).unwrap();
    let status = Command::new("ffmpeg")
        .args([
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=gray:s=64x64:r=25:d=2",
            "-c:v",
            "ffv1",
            "-pix_fmt",
            "yuv420p",
        ])
        .arg(path.join("fixture.mkv"))
        .status()
        .unwrap();
    assert!(status.success());
    let root = Root::new(&path).unwrap();
    let hash = sha256_hex(&fs::read(path.join("fixture.mkv")).unwrap());
    let mut config = json!({"schema":"seam-qa/1","cut":{"path":"fixture.mkv","sha256":hash},
        "colour":{"matrix":"bt709","range":"tv"}, "seams":[{"frame":25}],
        "thresholds":seam_qa::Thresholds::kinoloom_uncalibrated()});
    let normal = seam_qa::run(&root, &serde_json::from_value(config.clone()).unwrap()).unwrap();
    assert_eq!(normal["verdict"], "PASS");
    assert_eq!(normal["acceptance_backing"], false);
    let mut observations = vec![json!({"case":"ordinary_declared_colour","receipt":normal})];
    config.as_object_mut().unwrap().remove("colour");
    let err = seam_qa::run(&root, &serde_json::from_value(config.clone()).unwrap()).unwrap_err();
    observations.push(json!({"case":"untagged_without_declaration","error":err.to_string()}));
    config["colour"] = json!({"matrix":"bt709","range":"tv"});
    for frame in [49_u64, u64::MAX] {
        config["seams"] = json!([{"frame":frame}]);
        let parsed: seam_qa::Config = serde_json::from_value(config.clone()).unwrap();
        let outcome = std::panic::catch_unwind(|| seam_qa::run(&root, &parsed));
        let result: Value = match outcome {
            Err(_) => json!({"case":"seam_bound","frame":frame,"panicked":true}),
            Ok(Ok(r)) => json!({"case":"seam_bound","frame":frame,"panicked":false,"receipt":r}),
            Ok(Err(e)) => {
                json!({"case":"seam_bound","frame":frame,"panicked":false,"error":e.to_string()})
            }
        };
        if frame == u64::MAX {
            assert_eq!(result["panicked"], true);
        } else {
            assert_eq!(result["receipt"]["verdict"], "BLOCKED");
        }
        observations.push(result);
    }
    fs::write(
        path.join("bounds-results.json"),
        serde_json::to_vec_pretty(&observations).unwrap(),
    )
    .unwrap();
}
