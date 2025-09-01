//! Stress tests for concurrent partial settings updates
//!
//! Tests system behavior under heavy concurrent load,
//! race conditions, memory pressure, and edge cases
//!

use actix_web::{test, web, App, http::StatusCode};
use serde_json::{json, Value};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::time::sleep;
use futures::future::join_all;

// Import project modules
use webxr::{app_state::AppState, handlers::settings_handler};
use webxr::config::AppFullSettings;

#[cfg(test)]
mod stress_concurrent_tests {
    use super::*;
    
    /// Helper to create app for stress testing
    async fn create_stress_test_app() -> impl actix_web::dev::Service<
        actix_web::dev::ServiceRequest,
        Response = actix_web::dev::ServiceResponse,
        Error = actix_web::Error,
    > {
        let app_settings = AppFullSettings::default();
        let app_state = AppState::new_test_state(app_settings);
        
        test::init_service(
            App::new()
                .app_data(web::Data::new(app_state))
                .configure(settings_handler::config)
        ).await
    }
    
    #[tokio::test]
    async fn test_high_concurrency_granular_updates() {
        let app = Arc::new(create_stress_test_app().await);
        
        let num_clients = 50;
        let requests_per_client = 20;
        let total_requests = num_clients * requests_per_client;
        
        println!("Starting high concurrency test: {} clients, {} requests each", num_clients, requests_per_client);
        
        let start_time = Instant::now();
        let results = Arc::new(Mutex::new(Vec::new()));
        
        let mut handles = Vec::new();
        
        for client_id in 0..num_clients {
            let app_clone = Arc::clone(&app);
            let results_clone = Arc::clone(&results);
            
            let handle = tokio::spawn(async move {
                let mut client_results = Vec::new();
                
                for request_id in 0..requests_per_client {
                    let request_start = Instant::now();
                    
                    // Different types of updates to stress different paths
                    let update = match request_id % 5 {
                        0 => json!([
                            {"path": "visualisation.glow.intensity", "value": (client_id as f64 * 0.01) + (request_id as f64 * 0.001)}
                        ]),
                        1 => json!([
                            {"path": "visualisation.glow.nodeGlowStrength", "value": 1.0 + (client_id as f64 * 0.1)},
                            {"path": "visualisation.glow.edgeGlowStrength", "value": 1.0 + (request_id as f64 * 0.05)}
                        ]),
                        2 => json!([
                            {"path": "visualisation.graphs.logseq.physics.springK", "value": 0.001 + (client_id as f64 * 0.001)},
                            {"path": "visualisation.graphs.logseq.physics.repelK", "value": 50.0 + (request_id as f64 * 0.1)}
                        ]),
                        3 => json!([
                            {"path": "system.network.maxRequestSize", "value": 1048576 + (client_id * 1024)},
                            {"path": "system.websocket.maxConnections", "value": 100 + client_id}
                        ]),
                        _ => json!([
                            {"path": "xr.roomScale", "value": 1.0 + (client_id as f64 * 0.01)},
                            {"path": "xr.interactionDistance", "value": 1.5 + (request_id as f64 * 0.01)}
                        ])
                    };
                    
                    let req = test::TestRequest::post()
                        .uri("/api/settings/set")
                        .set_json(&update)
                        .insert_header(("X-Client-ID", format!("stress-client-{}", client_id)))
                        .to_request();
                        
                    let resp = test::call_service(&*app_clone, req).await;
                    let request_duration = request_start.elapsed();
                    
                    client_results.push((
                        client_id,
                        request_id,
                        resp.status(),
                        request_duration
                    ));
                    
                    // Simulate realistic client behavior with small delays
                    if request_id % 5 == 0 {
                        sleep(Duration::from_millis(1)).await;
                    }
                }
                
                // Store results thread-safely
                results_clone.lock().unwrap().extend(client_results);
            });
            
            handles.push(handle);
        }
        
        // Wait for all clients to complete
        join_all(handles).await;
        
        let total_duration = start_time.elapsed();
        let results = results.lock().unwrap();
        
        // Analyze results
        let successful = results.iter().filter(|(_, _, status, _)| *status == StatusCode::OK).count();
        let rate_limited = results.iter().filter(|(_, _, status, _)| *status == StatusCode::TOO_MANY_REQUESTS).count();
        let errors = results.iter().filter(|(_, _, status, _)| status.is_server_error()).count();
        
        let avg_duration = results.iter()
            .map(|(_, _, _, duration)| duration.as_millis() as f64)
            .sum::<f64>() / results.len() as f64;
            
        let max_duration = results.iter()
            .map(|(_, _, _, duration)| *duration)
            .max()
            .unwrap_or(Duration::ZERO);
        
        let requests_per_second = total_requests as f64 / total_duration.as_secs_f64();
        
        println!("High concurrency stress test results:");
        println!("  Total requests: {}", total_requests);
        println!("  Total duration: {:?}", total_duration);
        println!("  Requests/second: {:.2}", requests_per_second);
        println!("  Successful: {} ({:.1}%)", successful, successful as f64 / total_requests as f64 * 100.0);
        println!("  Rate limited: {} ({:.1}%)", rate_limited, rate_limited as f64 / total_requests as f64 * 100.0);
        println!("  Server errors: {} ({:.1}%)", errors, errors as f64 / total_requests as f64 * 100.0);
        println!("  Average response time: {:.2}ms", avg_duration);
        println!("  Max response time: {:?}", max_duration);
        
        // Performance assertions
        assert!(successful as f64 / total_requests as f64 > 0.7, "Success rate too low: {:.1}%", successful as f64 / total_requests as f64 * 100.0);
        assert!(errors == 0, "Server errors occurred: {}", errors);
        assert!(avg_duration < 500.0, "Average response time too high: {:.2}ms", avg_duration);
        assert!(max_duration.as_millis() < 2000, "Max response time too high: {:?}", max_duration);
        assert!(requests_per_second > 50.0, "Throughput too low: {:.2} req/s", requests_per_second);
        
        println!("✅ High concurrency stress test passed");
    }
    
    #[tokio::test]
    async fn test_race_condition_handling() {
        let app = Arc::new(create_stress_test_app().await);
        
        let num_racers = 20;
        let same_path = "visualisation.glow.intensity";
        
        println!("Testing race conditions: {} concurrent updates to same path", num_racers);
        
        let mut handles = Vec::new();
        let results = Arc::new(Mutex::new(Vec::new()));
        
        let barrier = Arc::new(tokio::sync::Barrier::new(num_racers));
        
        for racer_id in 0..num_racers {
            let app_clone = Arc::clone(&app);
            let results_clone = Arc::clone(&results);
            let barrier_clone = Arc::clone(&barrier);
            
            let handle = tokio::spawn(async move {
                // Synchronize start time to maximize race condition likelihood
                barrier_clone.wait().await;
                
                let start_time = Instant::now();
                let value = racer_id as f64 * 0.1;
                
                let update = json!([
                    {"path": same_path, "value": value}
                ]);
                
                let req = test::TestRequest::post()
                    .uri("/api/settings/set")
                    .set_json(&update)
                    .insert_header(("X-Racer-ID", format!("racer-{}", racer_id)))
                    .to_request();
                    
                let resp = test::call_service(&*app_clone, req).await;
                let duration = start_time.elapsed();
                
                results_clone.lock().unwrap().push((
                    racer_id,
                    resp.status(),
                    duration,
                    value
                ));
                
                if resp.status() == StatusCode::OK {
                    Some(value)
                } else {
                    None
                }
            });
            
            handles.push(handle);
        }
        
        let final_values: Vec<Option<f64>> = join_all(handles).await
            .into_iter()
            .map(|r| r.unwrap())
            .collect();
        
        let results = results.lock().unwrap();
        
        let successful_updates: Vec<_> = results.iter()
            .filter(|(_, status, _, _)| *status == StatusCode::OK)
            .collect();
        
        println!("Race condition test results:");
        println!("  Successful updates: {}/{}", successful_updates.len(), num_racers);
        
        for (racer_id, status, duration, value) in &*results {
            println!("  Racer {}: {:?} in {:?}ms (value: {})", racer_id, status, duration.as_millis(), value);
        }
        
        // Verify final state is consistent
        let verify_req = test::TestRequest::get()
            .uri(&format!("/api/settings/get?paths={}", same_path))
            .to_request();
            
        let verify_resp = test::call_service(&*app, verify_req).await;
        assert_eq!(verify_resp.status(), StatusCode::OK);
        
        let final_state: Value = test::read_body_json(verify_resp).await;
        let final_value = final_state["visualisation"]["glow"]["intensity"].as_f64().unwrap();
        
        println!("  Final value: {}", final_value);
        
        // The final value should be one of the successfully updated values
        let successful_values: Vec<f64> = successful_updates.iter()
            .map(|(_, _, _, value)| **value)
            .collect();
        
        if !successful_values.is_empty() {
            assert!(successful_values.contains(&final_value),
                "Final value {} not in successful updates: {:?}", final_value, successful_values);
        }
        
        // Should have at least some successful updates
        assert!(successful_updates.len() > 0, "No updates succeeded in race condition test");
        
        println!("✅ Race condition handling test passed");
    }
    
    #[tokio::test]
    async fn test_memory_pressure_resilience() {
        let app = Arc::new(create_stress_test_app().await);
        
        println!("Testing memory pressure resilience");
        
        // Create many concurrent requests with large payloads
        let num_clients = 10;
        let large_batch_size = 100;
        
        let mut handles = Vec::new();
        
        for client_id in 0..num_clients {
            let app_clone = Arc::clone(&app);
            
            let handle = tokio::spawn(async move {
                // Create large batch of updates
                let mut large_batch = Vec::new();
                
                for i in 0..large_batch_size {
                    large_batch.push(json!({
                        "path": format!("visualisation.glow.customParam{}", i),
                        "value": format!("client_{}_value_{}", client_id, i)
                    }));
                }
                
                let large_update = json!(large_batch);
                
                let req = test::TestRequest::post()
                    .uri("/api/settings/set")
                    .set_json(&large_update)
                    .insert_header(("X-Client-ID", format!("memory-test-{}", client_id)))
                    .to_request();
                    
                let start_time = Instant::now();
                let resp = test::call_service(&*app_clone, req).await;
                let duration = start_time.elapsed();
                
                (client_id, resp.status(), duration)
            });
            
            handles.push(handle);
        }
        
        let results: Vec<_> = join_all(handles).await
            .into_iter()
            .map(|r| r.unwrap())
            .collect();
        
        let successful = results.iter().filter(|(_, status, _)| *status == StatusCode::OK).count();
        let rejected = results.iter().filter(|(_, status, _)| *status == StatusCode::BAD_REQUEST || *status == StatusCode::PAYLOAD_TOO_LARGE).count();
        let avg_duration = results.iter().map(|(_, _, duration)| duration.as_millis() as f64).sum::<f64>() / results.len() as f64;
        
        println!("Memory pressure test results:");
        println!("  Successful: {}/{}", successful, num_clients);
        println!("  Rejected: {}/{}", rejected, num_clients);
        println!("  Average duration: {:.2}ms", avg_duration);
        
        for (client_id, status, duration) in results {
            println!("  Client {}: {:?} in {:?}ms", client_id, status, duration.as_millis());
        }
        
        // System should handle memory pressure gracefully (reject or succeed, but not crash)
        let server_errors = results.iter().filter(|(_, status, _)| status.is_server_error()).count();
        assert_eq!(server_errors, 0, "Server errors occurred under memory pressure");
        
        // Should have reasonable response times even under pressure
        assert!(avg_duration < 5000.0, "Average response time too high under memory pressure: {:.2}ms", avg_duration);
        
        println!("✅ Memory pressure resilience test passed");
    }
    
    #[tokio::test]
    async fn test_sustained_load_performance() {
        let app = Arc::new(create_stress_test_app().await);
        
        println!("Testing sustained load performance");
        
        let duration = Duration::from_secs(30); // 30 second sustained test
        let target_rps = 10; // requests per second
        let interval = Duration::from_millis(1000 / target_rps);
        
        let start_time = Instant::now();
        let mut request_count = 0;
        let mut successful_count = 0;
        let mut response_times = Vec::new();
        
        while start_time.elapsed() < duration {
            let request_start = Instant::now();
            
            let update = json!([
                {"path": "visualisation.glow.intensity", "value": (request_count as f64 * 0.001) % 5.0},
                {"path": "visualisation.glow.nodeGlowStrength", "value": 1.0 + (request_count as f64 * 0.01) % 3.0}
            ]);
            
            let req = test::TestRequest::post()
                .uri("/api/settings/set")
                .set_json(&update)
                .insert_header(("X-Request-ID", format!("sustained-{}", request_count)))
                .to_request();
                
            let resp = test::call_service(&*app, req).await;
            let request_duration = request_start.elapsed();
            
            response_times.push(request_duration.as_millis() as f64);
            request_count += 1;
            
            if resp.status() == StatusCode::OK {
                successful_count += 1;
            }
            
            // Log progress every 100 requests
            if request_count % 100 == 0 {
                let elapsed = start_time.elapsed();
                let current_rps = request_count as f64 / elapsed.as_secs_f64();
                println!("  Progress: {} requests in {:?} ({:.2} req/s)", request_count, elapsed, current_rps);
            }
            
            // Maintain target rate
            sleep(interval).await;
        }
        
        let total_duration = start_time.elapsed();
        let actual_rps = request_count as f64 / total_duration.as_secs_f64();
        let success_rate = successful_count as f64 / request_count as f64 * 100.0;
        let avg_response_time = response_times.iter().sum::<f64>() / response_times.len() as f64;
        let max_response_time = response_times.iter().fold(0.0, |a, &b| a.max(b));
        let p95_response_time = {
            let mut sorted = response_times.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            sorted[(sorted.len() as f64 * 0.95) as usize]
        };
        
        println!("Sustained load test results:");
        println!("  Duration: {:?}", total_duration);
        println!("  Total requests: {}", request_count);
        println!("  Target RPS: {}", target_rps);
        println!("  Actual RPS: {:.2}", actual_rps);
        println!("  Success rate: {:.1}%", success_rate);
        println!("  Average response time: {:.2}ms", avg_response_time);
        println!("  Max response time: {:.2}ms", max_response_time);
        println!("  95th percentile response time: {:.2}ms", p95_response_time);
        
        // Performance assertions for sustained load
        assert!(success_rate > 95.0, "Success rate too low under sustained load: {:.1}%", success_rate);
        assert!(avg_response_time < 200.0, "Average response time too high under sustained load: {:.2}ms", avg_response_time);
        assert!(p95_response_time < 500.0, "95th percentile response time too high: {:.2}ms", p95_response_time);
        assert!(actual_rps > target_rps as f64 * 0.8, "Actual RPS too low: {:.2}", actual_rps);
        
        println!("✅ Sustained load performance test passed");
    }
    
    #[tokio::test]
    async fn test_connection_limit_handling() {
        let app = Arc::new(create_stress_test_app().await);
        
        println!("Testing connection limit handling");
        
        // Try to create more concurrent connections than typical limits
        let num_connections = 200;
        let mut handles = Vec::new();
        
        for conn_id in 0..num_connections {
            let app_clone = Arc::clone(&app);
            
            let handle = tokio::spawn(async move {
                // Hold connection open with a slow request
                let update = json!([
                    {"path": "visualisation.glow.intensity", "value": conn_id as f64 * 0.01}
                ]);
                
                let req = test::TestRequest::post()
                    .uri("/api/settings/set")
                    .set_json(&update)
                    .insert_header(("X-Connection-ID", format!("conn-{}", conn_id)))
                    .to_request();
                    
                let start_time = Instant::now();
                
                // Add small delay to simulate slow network
                sleep(Duration::from_millis(10)).await;
                
                let resp = test::call_service(&*app_clone, req).await;
                let duration = start_time.elapsed();
                
                (conn_id, resp.status(), duration)
            });
            
            handles.push(handle);
            
            // Small delay to stagger connection creation
            sleep(Duration::from_millis(1)).await;
        }
        
        let results: Vec<_> = join_all(handles).await
            .into_iter()
            .map(|r| r.unwrap())
            .collect();
        
        let successful = results.iter().filter(|(_, status, _)| *status == StatusCode::OK).count();
        let connection_errors = results.iter().filter(|(_, status, _)| 
            *status == StatusCode::SERVICE_UNAVAILABLE || 
            *status == StatusCode::TOO_MANY_REQUESTS
        ).count();
        let server_errors = results.iter().filter(|(_, status, _)| status.is_server_error()).count();
        
        println!("Connection limit test results:");
        println!("  Attempted connections: {}", num_connections);
        println!("  Successful: {}", successful);
        println!("  Connection limited: {}", connection_errors);
        println!("  Server errors: {}", server_errors);
        
        // System should handle connection pressure gracefully
        assert_eq!(server_errors, 0, "Server errors occurred under connection pressure");
        assert!(successful > 0, "No connections succeeded");
        
        // Either connections succeed or are gracefully rejected
        assert_eq!(successful + connection_errors, num_connections, 
            "Unexpected response types under connection pressure");
        
        println!("✅ Connection limit handling test passed");
    }
    
    #[tokio::test]
    async fn test_data_consistency_under_stress() {
        let app = Arc::new(create_stress_test_app().await);
        
        println!("Testing data consistency under stress");
        
        // Multiple clients updating different but related settings
        let num_clients = 20;
        let updates_per_client = 10;
        
        // Test paths that could interact or conflict
        let test_paths = vec![
            "visualisation.glow.intensity",
            "visualisation.glow.nodeGlowStrength", 
            "visualisation.glow.edgeGlowStrength",
            "visualisation.glow.baseColor",
            "visualisation.glow.enabled"
        ];
        
        let mut handles = Vec::new();
        
        for client_id in 0..num_clients {
            let app_clone = Arc::clone(&app);
            let paths = test_paths.clone();
            
            let handle = tokio::spawn(async move {
                let mut client_results = Vec::new();
                
                for update_id in 0..updates_per_client {
                    let path_idx = (client_id + update_id) % paths.len();
                    let path = &paths[path_idx];
                    
                    let value = match path {
                        p if p.contains("intensity") || p.contains("Strength") => 
                            json!(1.0 + (client_id as f64 * 0.1) + (update_id as f64 * 0.01)),
                        p if p.contains("Color") => 
                            json!(format!("#{:02x}{:02x}{:02x}", 
                                (client_id * 10) % 256, 
                                (update_id * 20) % 256, 
                                ((client_id + update_id) * 15) % 256)),
                        p if p.contains("enabled") => 
                            json!((client_id + update_id) % 2 == 0),
                        _ => json!(client_id as f64)
                    };
                    
                    let update = json!([
                        {"path": path, "value": value}
                    ]);
                    
                    let req = test::TestRequest::post()
                        .uri("/api/settings/set")
                        .set_json(&update)
                        .to_request();
                        
                    let resp = test::call_service(&*app_clone, req).await;
                    client_results.push((client_id, update_id, path.clone(), resp.status()));
                    
                    // Random small delays to create realistic concurrency
                    sleep(Duration::from_millis((client_id % 10) as u64)).await;
                }
                
                client_results
            });
            
            handles.push(handle);
        }
        
        let all_results: Vec<Vec<_>> = join_all(handles).await
            .into_iter()
            .map(|r| r.unwrap())
            .collect();
        
        // Verify final consistency
        let verify_paths = test_paths.join(",");
        let verify_req = test::TestRequest::get()
            .uri(&format!("/api/settings/get?paths={}", verify_paths))
            .to_request();
            
        let verify_resp = test::call_service(&*app, verify_req).await;
        assert_eq!(verify_resp.status(), StatusCode::OK);
        
        let final_state: Value = test::read_body_json(verify_resp).await;
        
        // Analyze results
        let mut total_updates = 0;
        let mut successful_updates = 0;
        
        for client_results in all_results {
            for (client_id, update_id, path, status) in client_results {
                total_updates += 1;
                if status == StatusCode::OK {
                    successful_updates += 1;
                }
                
                // Log any unexpected failures
                if status.is_server_error() {
                    println!("  Server error: client {} update {} path {}: {:?}", 
                        client_id, update_id, path, status);
                }
            }
        }
        
        println!("Data consistency test results:");
        println!("  Total updates: {}", total_updates);
        println!("  Successful updates: {}", successful_updates);
        println!("  Success rate: {:.1}%", successful_updates as f64 / total_updates as f64 * 100.0);
        
        // Verify final state is valid and consistent
        for path in &test_paths {
            let value = extract_value_by_path(&final_state, path);
            println!("  Final {}: {:?}", path, value);
            
            // Ensure values are in valid ranges/formats
            match path {
                p if p.contains("intensity") || p.contains("Strength") => {
                    if let Some(v) = value.and_then(|v| v.as_f64()) {
                        assert!(v >= 0.0 && v <= 10.0, "Value out of range for {}: {}", path, v);
                    }
                },
                p if p.contains("Color") => {
                    if let Some(s) = value.and_then(|v| v.as_str()) {
                        assert!(s.starts_with("#") && s.len() == 7, "Invalid color format for {}: {}", path, s);
                    }
                },
                p if p.contains("enabled") => {
                    assert!(value.and_then(|v| v.as_bool()).is_some(), "Invalid boolean for {}", path);
                },
                _ => {}
            }
        }
        
        // No server errors should occur
        let server_error_count = all_results.iter()
            .flatten()
            .filter(|(_, _, _, status)| status.is_server_error())
            .count();
        assert_eq!(server_error_count, 0, "Server errors occurred during stress test");
        
        println!("✅ Data consistency under stress test passed");
    }
}

// Helper function to extract value by dot-notation path
fn extract_value_by_path(json: &Value, path: &str) -> Option<&Value> {
    let parts: Vec<&str> = path.split('.').collect();
    let mut current = json;
    
    for part in parts {
        match current.get(part) {
            Some(value) => current = value,
            None => return None,
        }
    }
    
    Some(current)
}