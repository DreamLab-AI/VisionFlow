use loom_domain::ports::VectorIndex;
use loom_vector_ruvector::HnswIndex;
use ruvector_core::{VectorDB, types::{DbOptions, DistanceMetric, QuantizationConfig, VectorEntry}};
#[tokio::main]
async fn main() {
    let dir = tempfile::tempdir().unwrap();
    for (name, dimensions, metric) in [("cosine",384,DistanceMetric::Cosine),("euclidean",384,DistanceMetric::Euclidean),("wrong_width",3,DistanceMetric::Cosine)] {
        let path = dir.path().join(format!("{name}.rvdb"));
        let mut v=vec![0.0;dimensions];v[0]=0.5;
        let db=VectorDB::new(DbOptions {dimensions,distance_metric:metric,storage_path:path.to_string_lossy().into_owned(),hnsw_config:None,quantization:Some(QuantizationConfig::None)}).unwrap();
        db.insert(VectorEntry{id:Some("urn:fixture:aligned".to_owned()),vector:v,metadata:None}).unwrap();
        drop(db);
        let index=HnswIndex::open(&path);
        let mut query=vec![0.0;384];query[0]=1.0;
        match index.nearest(&query,1).await {
            Ok(found)=>println!("{}",serde_json::json!({"case":name,"ready":index.is_ready(),"score":found.first().map(|x|x.score),"results":found.len()})),
            Err(e)=>println!("{}",serde_json::json!({"case":name,"ready":index.is_ready(),"error":e.to_string()})),
        }
    }
}
