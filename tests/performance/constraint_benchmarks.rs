// Performance benchmarks for constraint translation and reasoning
// Validates performance requirements for migration

// Benchmark disabled - requires nightly Rust
// The #[bench] attribute and test::Bencher require the unstable 'test' feature
// To re-enable: uncomment #![feature(test)], extern crate test, use test::Bencher,
// and the benchmark functions below. Then compile with nightly: cargo +nightly test

// #![feature(test)]
// extern crate test;

use std::time::{Duration, Instant};
// use test::Bencher;

// Mock structures for benchmarking
#[derive(Clone, Debug)]
struct OntologyAxiom {
    axiom_type: String,
    subject: String,
    object: String,
    property: Option<String>,
}

#[derive(Clone, Debug)]
struct PhysicsConstraint {
    id: String,
    constraint_type: String,
    strength: f32,
    priority: i32,
    source_node: Option<String>,
    target_node: Option<String>,
}

impl From<OntologyAxiom> for PhysicsConstraint {
    fn from(axiom: OntologyAxiom) -> Self {
        PhysicsConstraint {
            id: format!("constraint_{}", uuid::Uuid::new_v4()),
            constraint_type: axiom.axiom_type.clone(),
            strength: match axiom.axiom_type.as_str() {
                "SubClassOf" => 0.9,
                "DisjointWith" => 1.0,
                "ObjectPropertyAssertion" => 0.7,
                _ => 0.5,
            },
            priority: 1,
            source_node: Some(axiom.subject),
            target_node: Some(axiom.object),
        }
    }
}

// Load test axioms
fn load_test_axioms(count: usize) -> Vec<OntologyAxiom> {
    (0..count)
        .map(|i| OntologyAxiom {
            axiom_type: match i % 3 {
                0 => "SubClassOf".to_string(),
                1 => "DisjointWith".to_string(),
                _ => "ObjectPropertyAssertion".to_string(),
            },
            subject: format!("Class{}", i),
            object: format!("Class{}", (i + 1) % count),
            property: Some(format!("property{}", i % 10)),
        })
        .collect()
}

// Mock inference cache
struct InferenceCache {
    cache: std::collections::HashMap<i32, Vec<String>>,
}

impl InferenceCache {
    fn new() -> Self {
        Self {
            cache: std::collections::HashMap::new(),
        }
    }

    fn get(&self, key: i32) -> Option<&Vec<String>> {
        self.cache.get(&key)
    }

    fn insert(&mut self, key: i32, value: Vec<String>) {
        self.cache.insert(key, value);
    }

    async fn get_or_compute<F>(&mut self, key: i32, compute: F) -> Result<Vec<String>, String>
    where
        F: Fn() -> Result<Vec<String>, String>,
    {
        if let Some(cached) = self.get(key) {
            return Ok(cached.clone());
        }

        let result = compute()?;
        self.insert(key, result.clone());
        Ok(result)
    }
}

// Mock reasoner
struct Reasoner;

impl Reasoner {
    fn compute_inferences(&self, _key: i32) -> Result<Vec<String>, String> {
        // Simulate reasoning work
        std::thread::sleep(Duration::from_micros(100));
        Ok(vec!["inference1".to_string(), "inference2".to_string()])
    }
}

// ============================================================================
// BENCHMARKS - Disabled (requires nightly Rust)
// ============================================================================
// Benchmark disabled - requires nightly Rust
// To re-enable benchmarks, uncomment the code below and compile with:
//   cargo +nightly test --features nightly

/*
#[bench]
fn bench_constraint_translation_100_axioms(b: &mut Bencher) {
    let axioms = load_test_axioms(100);
    b.iter(|| {
        let constraints: Vec<PhysicsConstraint> = axioms
            .iter()
            .map(|a| PhysicsConstraint::from(a.clone()))
            .collect();
        assert_eq!(constraints.len(), 100);
    });
}

#[bench]
fn bench_constraint_translation_1000_axioms(b: &mut Bencher) {
    let axioms = load_test_axioms(1000);

    b.iter(|| {
        let start = Instant::now();
        let constraints: Vec<PhysicsConstraint> = axioms
            .iter()
            .map(|a| PhysicsConstraint::from(a.clone()))
            .collect();
        let elapsed = start.elapsed();

        assert_eq!(constraints.len(), 1000);
        // Assert: <120ms for 1000 axioms
        assert!(
            elapsed < Duration::from_millis(120),
            "Translation took {:?}, expected <120ms",
            elapsed
        );
    });
}

#[bench]
fn bench_constraint_translation_10000_axioms(b: &mut Bencher) {
    let axioms = load_test_axioms(10000);
    b.iter(|| {
        let constraints: Vec<PhysicsConstraint> = axioms
            .iter()
            .map(|a| PhysicsConstraint::from(a.clone()))
            .collect();
        assert_eq!(constraints.len(), 10000);
    });
}

#[bench]
fn bench_reasoning_without_cache(b: &mut Bencher) {
    let reasoner = Reasoner;
    b.iter(|| {
        let result = reasoner.compute_inferences(1).unwrap();
        assert_eq!(result.len(), 2);
    });
}

#[bench]
fn bench_reasoning_with_cache_first_access(b: &mut Bencher) {
    let mut cache = InferenceCache::new();
    let reasoner = Reasoner;

    b.iter(|| {
        cache.cache.clear(); // Clear to simulate first access
        let rt = tokio::runtime::Runtime::new().unwrap();
        let result = rt.block_on(async {
            cache.get_or_compute(1, || reasoner.compute_inferences(1)).await
        });
        assert!(result.is_ok());
    });
}

#[bench]
fn bench_reasoning_with_cache_cached_access(b: &mut Bencher) {
    let mut cache = InferenceCache::new();
    let reasoner = Reasoner;

    // Prime the cache
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        cache.get_or_compute(1, || reasoner.compute_inferences(1)).await.unwrap();
    });

    b.iter(|| {
        let start = Instant::now();
        let rt = tokio::runtime::Runtime::new().unwrap();
        let result = rt.block_on(async {
            cache.get_or_compute(1, || reasoner.compute_inferences(1)).await
        });
        let elapsed = start.elapsed();

        assert!(result.is_ok());
        // Assert: <20ms with cache
        assert!(
            elapsed < Duration::from_millis(20),
            "Cached access took {:?}, expected <20ms",
            elapsed
        );
    });
}

#[bench]
fn bench_parallel_constraint_translation(b: &mut Bencher) {
    use rayon::prelude::*;
    let axioms = load_test_axioms(10000);

    b.iter(|| {
        let constraints: Vec<PhysicsConstraint> = axioms
            .par_iter()
            .map(|a| PhysicsConstraint::from(a.clone()))
            .collect();
        assert_eq!(constraints.len(), 10000);
    });
}

#[bench]
fn bench_constraint_filtering(b: &mut Bencher) {
    let axioms = load_test_axioms(1000);
    let constraints: Vec<PhysicsConstraint> = axioms
        .iter()
        .map(|a| PhysicsConstraint::from(a.clone()))
        .collect();

    b.iter(|| {
        let high_priority: Vec<&PhysicsConstraint> = constraints
            .iter()
            .filter(|c| c.strength > 0.8)
            .collect();
        assert!(high_priority.len() > 0);
    });
}

#[bench]
fn bench_constraint_grouping_by_type(b: &mut Bencher) {
    use std::collections::HashMap;
    let axioms = load_test_axioms(1000);
    let constraints: Vec<PhysicsConstraint> = axioms
        .iter()
        .map(|a| PhysicsConstraint::from(a.clone()))
        .collect();

    b.iter(|| {
        let mut grouped: HashMap<String, Vec<&PhysicsConstraint>> = HashMap::new();
        for constraint in &constraints {
            grouped
                .entry(constraint.constraint_type.clone())
                .or_insert_with(Vec::new)
                .push(constraint);
        }
        assert!(grouped.len() > 0);
    });
}
*/

// ============================================================================
// REGULAR TESTS (NOT BENCHMARKS)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_axiom_to_constraint_conversion() {
        let axiom = OntologyAxiom {
            axiom_type: "SubClassOf".to_string(),
            subject: "ClassA".to_string(),
            object: "ClassB".to_string(),
            property: None,
        };

        let constraint = PhysicsConstraint::from(axiom);
        assert_eq!(constraint.constraint_type, "SubClassOf");
        assert_eq!(constraint.strength, 0.9);
    }

    #[test]
    fn test_cache_miss_then_hit() {
        let mut cache = InferenceCache::new();
        let reasoner = Reasoner;

        let rt = tokio::runtime::Runtime::new().unwrap();

        // First access - cache miss
        let start_miss = Instant::now();
        let result_miss = rt.block_on(async {
            cache.get_or_compute(1, || reasoner.compute_inferences(1)).await
        });
        let elapsed_miss = start_miss.elapsed();
        assert!(result_miss.is_ok());

        // Second access - cache hit
        let start_hit = Instant::now();
        let result_hit = rt.block_on(async {
            cache.get_or_compute(1, || reasoner.compute_inferences(1)).await
        });
        let elapsed_hit = start_hit.elapsed();
        assert!(result_hit.is_ok());

        // Cache hit should be much faster
        assert!(elapsed_hit < elapsed_miss);
        println!("Cache miss: {:?}, Cache hit: {:?}", elapsed_miss, elapsed_hit);
    }

    #[test]
    fn test_constraint_translation_1000_axioms_under_120ms() {
        let axioms = load_test_axioms(1000);
        let start = Instant::now();

        let constraints: Vec<PhysicsConstraint> = axioms
            .iter()
            .map(|a| PhysicsConstraint::from(a.clone()))
            .collect();

        let elapsed = start.elapsed();

        assert_eq!(constraints.len(), 1000);
        assert!(
            elapsed < Duration::from_millis(120),
            "Translation took {:?}, expected <120ms",
            elapsed
        );
        println!("✅ 1000 axioms translated in {:?}", elapsed);
    }

    #[test]
    fn test_cached_reasoning_under_20ms() {
        let mut cache = InferenceCache::new();
        let reasoner = Reasoner;
        let rt = tokio::runtime::Runtime::new().unwrap();

        // Prime cache
        rt.block_on(async {
            cache.get_or_compute(1, || reasoner.compute_inferences(1)).await
        }).unwrap();

        // Test cached access
        let start = Instant::now();
        let result = rt.block_on(async {
            cache.get_or_compute(1, || reasoner.compute_inferences(1)).await
        });
        let elapsed = start.elapsed();

        assert!(result.is_ok());
        assert!(
            elapsed < Duration::from_millis(20),
            "Cached reasoning took {:?}, expected <20ms",
            elapsed
        );
        println!("✅ Cached reasoning completed in {:?}", elapsed);
    }
}
