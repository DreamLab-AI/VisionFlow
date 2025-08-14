#!/bin/bash

# Test GPU Clustering API
# This script tests the GPU-accelerated clustering implementation

echo "Testing GPU Clustering API"
echo "========================="

# Base URL for the API
BASE_URL="http://localhost:4000/api/analytics"

# Test spectral clustering
echo ""
echo "1. Testing Spectral Clustering..."
curl -X POST "${BASE_URL}/cluster" \
  -H "Content-Type: application/json" \
  -d '{
    "method": "spectral",
    "params": {
      "num_clusters": 5
    }
  }' | jq '.'

# Test K-means clustering
echo ""
echo "2. Testing K-means Clustering..."
curl -X POST "${BASE_URL}/cluster" \
  -H "Content-Type: application/json" \
  -d '{
    "method": "kmeans",
    "params": {
      "num_clusters": 8
    }
  }' | jq '.'

# Test Louvain community detection
echo ""
echo "3. Testing Louvain Community Detection..."
curl -X POST "${BASE_URL}/cluster" \
  -H "Content-Type: application/json" \
  -d '{
    "method": "louvain",
    "params": {
      "resolution": 1.2
    }
  }' | jq '.'

# Test DBSCAN clustering
echo ""
echo "4. Testing DBSCAN Clustering..."
curl -X POST "${BASE_URL}/cluster" \
  -H "Content-Type: application/json" \
  -d '{
    "method": "dbscan",
    "params": {
      "eps": 0.5,
      "min_samples": 5
    }
  }' | jq '.'

echo ""
echo "GPU Clustering tests complete!"