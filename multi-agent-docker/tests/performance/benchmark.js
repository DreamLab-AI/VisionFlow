/**
 * Performance benchmarks for CachyOS and Z.AI Docker system
 */

const axios = require('axios');
const { performance } = require('perf_hooks');

const BASE_URL = process.env.API_BASE_URL || 'http://localhost:9090';
const ZAI_URL = process.env.ZAI_CONTAINER_URL || 'http://localhost:9600';
const API_KEY = process.env.MANAGEMENT_API_KEY || 'change-this-secret-key';

const client = axios.create({
  baseURL: BASE_URL,
  headers: { 'X-API-Key': API_KEY }
});

class Benchmark {
  constructor(name) {
    this.name = name;
    this.results = [];
  }

  async run(fn, iterations = 100) {
    console.log(`\n📊 Benchmark: ${this.name}`);
    console.log(`Running ${iterations} iterations...\n`);

    for (let i = 0; i < iterations; i++) {
      const start = performance.now();
      await fn();
      const end = performance.now();
      this.results.push(end - start);
    }

    this.analyze();
  }

  analyze() {
    const sorted = this.results.sort((a, b) => a - b);
    const mean = sorted.reduce((a, b) => a + b) / sorted.length;
    const min = sorted[0];
    const max = sorted[sorted.length - 1];
    const p50 = sorted[Math.floor(sorted.length * 0.5)];
    const p95 = sorted[Math.floor(sorted.length * 0.95)];
    const p99 = sorted[Math.floor(sorted.length * 0.99)];

    console.log(`Results for ${this.name}:`);
    console.log(`  Mean:    ${mean.toFixed(2)}ms`);
    console.log(`  Min:     ${min.toFixed(2)}ms`);
    console.log(`  Max:     ${max.toFixed(2)}ms`);
    console.log(`  P50:     ${p50.toFixed(2)}ms`);
    console.log(`  P95:     ${p95.toFixed(2)}ms`);
    console.log(`  P99:     ${p99.toFixed(2)}ms`);
    console.log(`  Ops/sec: ${(1000 / mean).toFixed(2)}`);
  }
}

async function main() {
  console.log('🚀 Agentic Flow Performance Benchmarks\n');
  console.log('=' .repeat(50));

  // Health check latency
  const healthBench = new Benchmark('Health Check Latency');
  await healthBench.run(async () => {
    await axios.get(`${BASE_URL}/health`);
  });

  // Metrics endpoint latency
  const metricsBench = new Benchmark('Metrics Endpoint Latency');
  await metricsBench.run(async () => {
    await axios.get(`${BASE_URL}/metrics`);
  });

  // API authentication latency
  const authBench = new Benchmark('Authenticated API Call Latency');
  await authBench.run(async () => {
    await client.get('/v1/status');
  });

  // Z.AI prompt latency
  const zaiBench = new Benchmark('Z.AI Simple Prompt Latency');
  await zaiBench.run(async () => {
    await axios.post(`${ZAI_URL}/prompt`, {
      prompt: 'Say hi',
      timeout: 5000
    });
  }, 20); // Fewer iterations for expensive operations

  // Concurrent request handling
  console.log('\n📊 Benchmark: Concurrent Request Handling');
  const concurrentStart = performance.now();
  const concurrentRequests = Array(50).fill(null).map(() =>
    client.get('/v1/status')
  );
  await Promise.all(concurrentRequests);
  const concurrentEnd = performance.now();
  const concurrentDuration = concurrentEnd - concurrentStart;
  console.log(`  50 concurrent requests: ${concurrentDuration.toFixed(2)}ms`);
  console.log(`  Throughput: ${(50 / (concurrentDuration / 1000)).toFixed(2)} req/sec`);

  // Memory footprint
  console.log('\n💾 Memory Metrics:');
  const metricsResponse = await axios.get(`${BASE_URL}/metrics`);
  const heapUsed = metricsResponse.data.match(/process_heap_bytes (\d+)/);
  if (heapUsed) {
    console.log(`  Heap Used: ${(parseInt(heapUsed[1]) / 1024 / 1024).toFixed(2)}MB`);
  }

  console.log('\n' + '='.repeat(50));
  console.log('✅ Benchmarks complete!\n');
}

main().catch(console.error);
