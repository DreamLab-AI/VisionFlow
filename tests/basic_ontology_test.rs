//! Basic Ontology Test to verify setup works

use std::collections::HashMap;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_setup() {
        // Just a simple test to verify our test setup works
        let test_map: HashMap<String, String> = HashMap::new();
        assert!(test_map.is_empty());
        println!("Basic ontology test setup working!");
    }
}