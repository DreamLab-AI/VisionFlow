"""
Load testing script for VisionFlow API using Locust
Tests system performance under concurrent load

Usage:
    locust -f tests/load/locustfile.py --users 100 --spawn-rate 10 --host http://localhost:8080
"""

from locust import HttpUser, task, between, events
from locust.contrib.fasthttp import FastHttpUser
import json
import random
import time

# Test data
TEST_NODES = [
    {"id": f"node_{i}", "label": f"Test Node {i}", "x": random.random() * 100, "y": random.random() * 100, "z": random.random() * 100}
    for i in range(100)
]

TEST_EDGES = [
    {"source": f"node_{i}", "target": f"node_{(i+1) % 100}", "label": "test_edge"}
    for i in range(100)
]

class VisionFlowUser(FastHttpUser):
    """
    Simulates a VisionFlow application user
    Uses FastHttpUser for better performance
    """
    wait_time = between(1, 3)  # Wait 1-3 seconds between tasks

    def on_start(self):
        """Called when a simulated user starts"""
        print("VisionFlow user started")

    @task(10)  # Weight: 10 (most common operation)
    def load_graph(self):
        """Load the full graph visualization"""
        with self.client.get("/api/graph", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Failed to load graph: {response.status_code}")

    @task(5)  # Weight: 5
    def get_node_details(self):
        """Get details for a specific node"""
        node_id = f"node_{random.randint(0, 99)}"
        with self.client.get(f"/api/nodes/{node_id}", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            elif response.status_code == 404:
                response.success()  # 404 is acceptable for random nodes
            else:
                response.failure(f"Failed to get node: {response.status_code}")

    @task(3)  # Weight: 3
    def update_constraint(self):
        """Update a physics constraint"""
        constraint_id = random.randint(1, 50)
        payload = {
            "strength": random.uniform(0.5, 1.0),
            "priority": random.randint(1, 5)
        }
        with self.client.put(
            f"/api/constraints/{constraint_id}",
            json=payload,
            catch_response=True
        ) as response:
            if response.status_code in [200, 201]:
                response.success()
            else:
                response.failure(f"Failed to update constraint: {response.status_code}")

    @task(8)  # Weight: 8
    def get_physics_settings(self):
        """Get current physics settings"""
        with self.client.get("/api/settings/physics", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Failed to get settings: {response.status_code}")

    @task(2)  # Weight: 2
    def update_physics_settings(self):
        """Update physics settings"""
        payload = {
            "gravity": random.uniform(9.0, 10.0),
            "damping": random.uniform(0.95, 0.99),
            "stiffness": random.uniform(0.4, 0.6),
            "iterations": random.randint(8, 12),
            "enabled": True
        }
        with self.client.post(
            "/api/settings/physics",
            json=payload,
            catch_response=True
        ) as response:
            if response.status_code in [200, 201]:
                response.success()
            else:
                response.failure(f"Failed to update settings: {response.status_code}")

    @task(4)  # Weight: 4
    def search_nodes(self):
        """Search for nodes by label"""
        search_term = random.choice(["Test", "Node", "Graph", "Entity"])
        with self.client.get(
            f"/api/nodes/search?q={search_term}",
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Failed to search nodes: {response.status_code}")

    @task(1)  # Weight: 1 (infrequent)
    def create_node(self):
        """Create a new node"""
        payload = {
            "id": f"node_load_test_{random.randint(1000, 9999)}",
            "label": f"Load Test Node {random.randint(1, 1000)}",
            "x": random.random() * 100,
            "y": random.random() * 100,
            "z": random.random() * 100
        }
        with self.client.post(
            "/api/nodes",
            json=payload,
            catch_response=True
        ) as response:
            if response.status_code in [200, 201]:
                response.success()
            else:
                response.failure(f"Failed to create node: {response.status_code}")

    @task(6)  # Weight: 6
    def get_neighbors(self):
        """Get neighbors for a node"""
        node_id = f"node_{random.randint(0, 99)}"
        with self.client.get(
            f"/api/nodes/{node_id}/neighbors",
            catch_response=True
        ) as response:
            if response.status_code in [200, 404]:
                response.success()
            else:
                response.failure(f"Failed to get neighbors: {response.status_code}")

    @task(2)  # Weight: 2
    def batch_update_positions(self):
        """Batch update node positions (physics simulation)"""
        num_updates = random.randint(5, 20)
        payload = {
            "positions": [
                {
                    "id": f"node_{random.randint(0, 99)}",
                    "x": random.random() * 100,
                    "y": random.random() * 100,
                    "z": random.random() * 100
                }
                for _ in range(num_updates)
            ]
        }
        with self.client.post(
            "/api/nodes/batch-positions",
            json=payload,
            catch_response=True
        ) as response:
            if response.status_code in [200, 201]:
                response.success()
            else:
                response.failure(f"Failed to batch update: {response.status_code}")


class WebSocketUser(HttpUser):
    """
    Simulates WebSocket connections for real-time updates
    Note: Locust's WebSocket support is limited, this is a simplified version
    """
    wait_time = between(2, 5)

    @task
    def ws_connection(self):
        """Simulate WebSocket connection overhead"""
        # In a real test, you'd use websocket-client or similar
        # This just simulates the initial HTTP upgrade request
        with self.client.get("/ws", catch_response=True) as response:
            if response.status_code in [101, 200]:
                response.success()
            else:
                response.failure(f"WebSocket connection failed: {response.status_code}")


# Custom event handlers for monitoring
@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Called when the test starts"""
    print("🚀 Load test starting...")
    print(f"Target host: {environment.host}")


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Called when the test stops"""
    print("🏁 Load test completed!")
    print(f"Total requests: {environment.stats.total.num_requests}")
    print(f"Total failures: {environment.stats.total.num_failures}")
    print(f"Average response time: {environment.stats.total.avg_response_time:.2f}ms")
    print(f"Requests per second: {environment.stats.total.total_rps:.2f}")


# Performance benchmarks
class PerformanceBenchmarkUser(FastHttpUser):
    """
    Heavy load user for performance benchmarking
    """
    wait_time = between(0.1, 0.5)  # Very aggressive

    @task
    def stress_test_graph_load(self):
        """Stress test graph loading"""
        self.client.get("/api/graph")

    @task
    def stress_test_constraint_updates(self):
        """Stress test constraint updates"""
        for i in range(10):
            payload = {"strength": random.random(), "priority": random.randint(1, 5)}
            self.client.put(f"/api/constraints/{random.randint(1, 100)}", json=payload)


# Mixed workload simulation
class MixedWorkloadUser(FastHttpUser):
    """
    Simulates realistic mixed workload with different user behaviors
    """
    wait_time = between(1, 4)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Assign user type randomly
        self.user_type = random.choice(['viewer', 'editor', 'admin'])

    @task
    def mixed_workflow(self):
        """Execute workflow based on user type"""
        if self.user_type == 'viewer':
            # Viewers mostly read
            self.client.get("/api/graph")
            self.client.get(f"/api/nodes/node_{random.randint(0, 99)}")

        elif self.user_type == 'editor':
            # Editors read and write
            self.client.get("/api/graph")
            node_id = f"node_{random.randint(0, 99)}"
            self.client.get(f"/api/nodes/{node_id}")
            self.client.put(f"/api/constraints/{random.randint(1, 50)}",
                          json={"strength": random.random()})

        else:  # admin
            # Admins do everything
            self.client.get("/api/graph")
            self.client.get("/api/settings/physics")
            self.client.post("/api/settings/physics",
                           json={"gravity": 9.8, "damping": 0.99})


if __name__ == "__main__":
    import os
    os.system("locust -f locustfile.py --users 100 --spawn-rate 10")
