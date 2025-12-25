# API Integration Tests

This directory contains integration tests for VisionFlow REST API endpoints.

## Test Files

### `reasoning_api_tests.rs`
Comprehensive integration tests for ontology reasoning API endpoints including:
- **Class Management**: GET, POST, PUT, DELETE operations on OWL classes
- **Property Management**: CRUD operations for OWL properties
- **Axiom Operations**: Add and remove axioms
- **Inference Results**: Store and retrieve reasoning results
- **Validation**: Ontology validation endpoints
- **Queries**: SPARQL-like ontology queries
- **Metrics**: Ontology statistics
- **Graph Operations**: Load and save ontology graphs

Tests are marked with `#[ignore]` because they require full AppState initialization including:
- Neo4jOntologyRepository
- CQRS handlers (QueryHandler, DirectiveHandler)
- Actor system runtime

Documentation tests (in `api_documentation` module) run without infrastructure and verify:
- API contract structures (request/response formats)
- Endpoint catalog (all 19 ontology routes)
- HTTP status codes

## Running Tests

### Documentation Tests (No Infrastructure Required)
```bash
cargo test --test ontology_api_test api_documentation
```

### Integration Tests (Requires Full System)
```bash
# These are ignored by default - require Neo4j, actor system, etc.
cargo test --test ontology_api_test -- --ignored
```

## API Endpoints Tested

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/ontology/graph` | Get full ontology graph |
| POST | `/ontology/graph` | Save ontology graph |
| GET | `/ontology/classes` | List all OWL classes |
| POST | `/ontology/classes` | Add new OWL class |
| GET | `/ontology/classes/{iri}` | Get specific class |
| PUT | `/ontology/classes/{iri}` | Update class |
| DELETE | `/ontology/classes/{iri}` | Delete class |
| GET | `/ontology/classes/{iri}/axioms` | Get class axioms |
| GET | `/ontology/properties` | List properties |
| POST | `/ontology/properties` | Add property |
| GET | `/ontology/properties/{iri}` | Get property |
| PUT | `/ontology/properties/{iri}` | Update property |
| POST | `/ontology/axioms` | Add axiom |
| DELETE | `/ontology/axioms/{id}` | Remove axiom |
| GET | `/ontology/inference` | Get inference results |
| POST | `/ontology/inference` | Store inference results |
| GET | `/ontology/validate` | Validate ontology |
| POST | `/ontology/query` | Query ontology |
| GET | `/ontology/metrics` | Get ontology metrics |

## Test Structure

Each test follows this pattern:

```rust
#[actix_web::test]
#[ignore = "Requires full AppState"]
async fn test_endpoint_name() {
    let app = test::init_service(create_test_app()).await;

    let req = test::TestRequest::get()
        .uri("/ontology/endpoint")
        .to_request();

    let resp = test::call_service(&app, req).await;

    assert_eq!(resp.status().as_u16(), 200);
}
```

## Notes

- Tests use `actix_web::test` utilities for proper HTTP integration testing
- All tests verify actual HTTP requests/responses, not just handler logic
- Error cases include: malformed JSON (400), not found (404), method not allowed (405)
- Tests document expected request/response structures via JSON examples
