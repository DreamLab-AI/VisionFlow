# GPU Configuration Guide

Complete guide for configuring NVIDIA GPU support in the multi-agent Docker environment.

## NVIDIA GPU Requirements

### Hardware Requirements

- NVIDIA GPU with compute capability 3.5 or higher
- Minimum 4GB GPU memory (8GB+ recommended for ML workloads)
- PCIe 3.0 or higher interface

### Supported GPU Architectures

- **Kepler** (GTX 700 series, Tesla K-series) - compute 3.5+
- **Maxwell** (GTX 900 series) - compute 5.0+
- **Pascal** (GTX 10 series, Tesla P-series) - compute 6.0+
- **Volta** (Tesla V100) - compute 7.0+
- **Turing** (RTX 20 series) - compute 7.5+
- **Ampere** (RTX 30 series, A-series) - compute 8.0+
- **Ada Lovelace** (RTX 40 series) - compute 8.9+
- **Hopper** (H100) - compute 9.0+

### Driver Requirements

- NVIDIA Driver version 525.60.13 or higher (for CUDA 12.x)
- NVIDIA Driver version 470.57.02 or higher (for CUDA 11.x)

Check driver version:
```bash
nvidia-smi
```

## Installing NVIDIA Docker Runtime

### Ubuntu/Debian

```bash
# Add NVIDIA Docker repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Install NVIDIA Container Toolkit
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Configure Docker daemon
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### RHEL/CentOS/Fedora

```bash
# Add NVIDIA Docker repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/nvidia-container-toolkit.repo | \
    sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo

# Install NVIDIA Container Toolkit
sudo yum install -y nvidia-container-toolkit

# Configure Docker daemon
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### Arch Linux

```bash
# Install from AUR
yay -S nvidia-container-toolkit

# Configure Docker daemon
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### Verification

```bash
# Test NVIDIA runtime
docker run --rm --gpus all nvidia/cuda:12.3.0-base-ubuntu22.04 nvidia-smi

# Check runtime configuration
docker info | grep -i runtime
```

## GPU Device Mapping in docker-compose.yml

### Basic GPU Configuration

Enable GPU access in `docker-compose.yml`:

```yaml
services:
  workstation:
    image: multi-agent-workstation:latest
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu, compute, utility]
```

### Single GPU Mapping

```yaml
services:
  workstation:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['0']
              capabilities: [gpu, compute, utility]
```

### Specific GPU by UUID

```bash
# List GPU UUIDs
nvidia-smi -L
# GPU 0: NVIDIA GeForce RTX 3090 (UUID: GPU-12345678-1234-1234-1234-123456789012)
```

```yaml
services:
  workstation:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['GPU-12345678-1234-1234-1234-123456789012']
              capabilities: [gpu, compute, utility]
```

### GPU Capabilities

Available capabilities:
- `gpu` - Basic GPU access
- `compute` - CUDA compute operations
- `utility` - nvidia-smi and management tools
- `graphics` - OpenGL/Vulkan graphics
- `video` - Video encode/decode
- `display` - Display output

Common combinations:
```yaml
# ML/AI workloads
capabilities: [gpu, compute, utility]

# Graphics + ML
capabilities: [gpu, compute, utility, graphics]

# Video processing
capabilities: [gpu, compute, utility, video]
```

## GPU Environment Variables

### CUDA Configuration

```yaml
services:
  workstation:
    environment:
      # CUDA device visibility
      CUDA_VISIBLE_DEVICES: "0,1"  # GPUs 0 and 1

      # Force specific GPU
      CUDA_DEVICE_ORDER: "PCI_BUS_ID"

      # Memory management
      CUDA_LAUNCH_BLOCKING: "0"  # Async launches (default)
      CUDA_CACHE_MAXSIZE: "268435456"  # 256MB kernel cache

      # Debugging
      CUDA_DEVICE_DEBUG: "0"  # Disable debug mode
```

### cuDNN Configuration

```yaml
environment:
  # Algorithm selection
  CUDNN_BENCHMARK: "1"  # Auto-tune algorithms
  CUDNN_DETERMINISTIC: "0"  # Non-deterministic (faster)

  # Workspace limits
  CUDNN_WORKSPACE_LIMIT: "2147483648"  # 2GB
```

### TensorFlow GPU Settings

```yaml
environment:
  # Memory growth
  TF_FORCE_GPU_ALLOW_GROWTH: "true"

  # GPU device placement
  TF_GPU_THREAD_MODE: "gpu_private"
  TF_GPU_THREAD_COUNT: "2"

  # XLA compilation
  TF_XLA_FLAGS: "--tf_xla_auto_jit=2"

  # Memory allocation
  TF_GPU_ALLOCATOR: "cuda_malloc_async"
```

### PyTorch GPU Settings

```yaml
environment:
  # CUDA allocation
  PYTORCH_CUDA_ALLOC_CONF: "max_split_size_mb:512"

  # cuDNN settings
  TORCH_CUDNN_V8_API_ENABLED: "1"

  # Memory management
  PYTORCH_NO_CUDA_MEMORY_CACHING: "0"
```

### NVIDIA Driver Settings

```yaml
environment:
  # Driver capabilities
  NVIDIA_VISIBLE_DEVICES: "all"
  NVIDIA_DRIVER_CAPABILITIES: "compute,utility,graphics"

  # Disable modes
  NVIDIA_DISABLE_REQUIRE: "false"

  # MIG configuration (A100/H100)
  NVIDIA_MIG_CONFIG_DEVICES: "all"
  NVIDIA_MIG_MONITOR_DEVICES: "all"
```

## Multi-GPU Configuration

### Multiple GPUs for Single Service

```yaml
services:
  workstation:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 2  # Use 2 GPUs
              capabilities: [gpu, compute, utility]
    environment:
      CUDA_VISIBLE_DEVICES: "0,1"
```

### GPU Distribution Across Services

```yaml
services:
  training:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['0', '1']
              capabilities: [gpu, compute, utility]
    environment:
      CUDA_VISIBLE_DEVICES: "0,1"

  inference:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['2']
              capabilities: [gpu, compute, utility]
    environment:
      CUDA_VISIBLE_DEVICES: "0"  # Mapped to physical GPU 2
```

### Multi-Instance GPU (MIG)

For A100/H100 GPUs with MIG support:

```bash
# Enable MIG mode
sudo nvidia-smi -mig 1

# Create MIG instances (A100 example)
sudo nvidia-smi mig -cgi 9,9,9,9,9,9,9 -C  # 7x 1g.5gb instances

# List MIG devices
nvidia-smi -L
```

```yaml
services:
  workstation:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['MIG-GPU-12345678-1234-1234-1234-123456789012/0/0']
              capabilities: [gpu, compute, utility]
```

### GPU Scheduling Strategies

```yaml
# Round-robin GPU assignment
services:
  worker_1:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
    environment:
      CUDA_DEVICE_ORDER: "PCI_BUS_ID"

  worker_2:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
    environment:
      CUDA_DEVICE_ORDER: "PCI_BUS_ID"
```

## GPU Monitoring and Metrics

### nvidia-smi Commands

```bash
# Basic GPU status
nvidia-smi

# Continuous monitoring (1 second interval)
nvidia-smi -l 1

# Detailed query
nvidia-smi --query-gpu=timestamp,name,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv -l 1

# Process-specific monitoring
nvidia-smi pmon -i 0 -s m -c 100

# Power monitoring
nvidia-smi --query-gpu=power.draw,power.limit --format=csv -l 1
```

### Container GPU Monitoring

```bash
# Monitor GPU usage in container
docker exec workstation nvidia-smi

# Stream GPU metrics
docker exec workstation nvidia-smi -l 1

# Check GPU allocation
docker exec workstation bash -c 'echo $CUDA_VISIBLE_DEVICES'
```

### NVML Python Monitoring

```python
# Install: pip install nvidia-ml-py
import pynvml

pynvml.nvmlInit()
device_count = pynvml.nvmlDeviceGetCount()

for i in range(device_count):
    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    util = pynvml.nvmlDeviceGetUtilizationRates(handle)

    print(f"GPU {i}:")
    print(f"  Memory: {info.used / 1024**3:.2f}GB / {info.total / 1024**3:.2f}GB")
    print(f"  GPU Util: {util.gpu}%")
    print(f"  Mem Util: {util.memory}%")

pynvml.nvmlShutdown()
```

### Prometheus Metrics Export

```yaml
services:
  nvidia-exporter:
    image: utkuozdemir/nvidia_gpu_exporter:latest
    restart: unless-stopped
    ports:
      - "9835:9835"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [utility]
```

### Grafana Dashboard

Sample Prometheus queries:
```promql
# GPU utilization
nvidia_gpu_utilization_ratio

# GPU memory usage
nvidia_gpu_memory_used_bytes / nvidia_gpu_memory_total_bytes

# GPU temperature
nvidia_gpu_temperature_celsius

# Power draw
nvidia_gpu_power_draw_watts
```

## Troubleshooting GPU Issues

### GPU Not Detected

**Symptom**: `nvidia-smi` fails or shows no GPUs

**Solutions**:
```bash
# Check driver installation
lsmod | grep nvidia

# Verify PCI devices
lspci | grep -i nvidia

# Reinstall driver
sudo apt-get purge nvidia-*
sudo apt-get install nvidia-driver-535

# Check secure boot (must be disabled for unsigned drivers)
mokutil --sb-state
```

### Docker Cannot Access GPU

**Symptom**: `docker: Error response from daemon: could not select device driver`

**Solutions**:
```bash
# Verify NVIDIA runtime
docker run --rm --gpus all nvidia/cuda:12.3.0-base-ubuntu22.04 nvidia-smi

# Check Docker daemon configuration
cat /etc/docker/daemon.json

# Should contain:
{
  "runtimes": {
    "nvidia": {
      "path": "nvidia-container-runtime",
      "runtimeArgs": []
    }
  }
}

# Restart Docker
sudo systemctl restart docker
```

### CUDA Version Mismatch

**Symptom**: `CUDA driver version is insufficient for CUDA runtime version`

**Solutions**:
```bash
# Check driver CUDA version
nvidia-smi | grep "CUDA Version"

# Check container CUDA version
docker run --rm --gpus all nvidia/cuda:12.3.0-base-ubuntu22.04 nvcc --version

# Use compatible CUDA version in Dockerfile
# Driver 525+ supports CUDA 12.x
# Driver 470+ supports CUDA 11.x
```

### Out of Memory Errors

**Symptom**: `CUDA out of memory` or `RuntimeError: CUDA error: out of memory`

**Solutions**:
```python
# PyTorch: Clear cache
import torch
torch.cuda.empty_cache()

# Reduce batch size
batch_size = 16  # Reduce from 32

# Enable gradient checkpointing
model.gradient_checkpointing_enable()

# Use mixed precision
from torch.cuda.amp import autocast
with autocast():
    output = model(input)
```

```yaml
# Limit container memory
services:
  workstation:
    deploy:
      resources:
        limits:
          memory: 32G
```

### GPU Process Conflicts

**Symptom**: GPU already in use by another process

**Solutions**:
```bash
# List GPU processes
nvidia-smi pmon -i 0

# Kill specific process
sudo kill -9 <PID>

# Set compute mode to exclusive
sudo nvidia-smi -c EXCLUSIVE_PROCESS

# Reset compute mode
sudo nvidia-smi -c DEFAULT
```

### Performance Degradation

**Symptom**: GPU utilization low despite heavy workload

**Solutions**:
```bash
# Check thermal throttling
nvidia-smi --query-gpu=temperature.gpu,clocks.current.graphics,clocks.max.graphics --format=csv

# Disable power management
sudo nvidia-smi -pm 1

# Set maximum power limit (adjust for your GPU)
sudo nvidia-smi -pl 350  # 350W for RTX 3090

# Check PCIe link speed
nvidia-smi -q | grep -A 3 "PCIe"
# Should show Gen3 or Gen4, x16 lanes
```

### Container Persistence Issues

**Symptom**: GPU state lost between container restarts

**Solutions**:
```bash
# Enable persistence mode on host
sudo nvidia-smi -pm 1

# Make persistent (systemd)
sudo systemctl enable nvidia-persistenced
sudo systemctl start nvidia-persistenced
```

## Performance Optimization

### Memory Optimization

```python
# PyTorch memory management
import torch

# Set memory allocator
torch.cuda.set_per_process_memory_fraction(0.8, device=0)

# Enable TF32 (Ampere+)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Use channels_last format
model = model.to(memory_format=torch.channels_last)
```

### Compute Optimization

```python
# Enable cuDNN auto-tuner
torch.backends.cudnn.benchmark = True

# Compile models (PyTorch 2.0+)
model = torch.compile(model, mode="max-autotune")

# Use fused operations
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
```

### Data Transfer Optimization

```python
# Pin memory for faster transfers
data_loader = DataLoader(
    dataset,
    batch_size=32,
    pin_memory=True,
    num_workers=4
)

# Async data loading
for batch in data_loader:
    batch = batch.to(device, non_blocking=True)
    output = model(batch)
```

### Multi-GPU Training

```python
# PyTorch DataParallel (simple but less efficient)
model = torch.nn.DataParallel(model, device_ids=[0, 1])

# PyTorch DistributedDataParallel (recommended)
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

dist.init_process_group(backend='nccl')
model = DDP(model, device_ids=[local_rank])
```

### Profiling

```python
# PyTorch profiler
from torch.profiler import profile, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True
) as prof:
    model(inputs)

print(prof.key_averages().table(sort_by="cuda_time_total"))
prof.export_chrome_trace("trace.json")
```

### Docker Performance Settings

```yaml
services:
  workstation:
    # Use host network for IPC performance
    ipc: host

    # Increase shared memory
    shm_size: '16gb'

    # Set ulimits
    ulimits:
      memlock: -1
      stack: 67108864

    # Security options for GPU access
    security_opt:
      - seccomp:unconfined

    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu, compute, utility]
```

## Best Practices

1. **Always specify GPU capabilities** explicitly based on workload requirements
2. **Monitor GPU temperature** and throttling during heavy workloads
3. **Use specific CUDA versions** in base images to match driver compatibility
4. **Enable persistence mode** for production deployments
5. **Profile before optimizing** to identify actual bottlenecks
6. **Implement proper cleanup** with `torch.cuda.empty_cache()` for long-running processes
7. **Test GPU access** after every Docker or driver update
8. **Use mixed precision** (FP16/BF16) for compatible workloads
9. **Leverage tensor cores** on Volta+ GPUs with proper data formats
10. **Implement proper error handling** for OOM and CUDA errors

## References

- [NVIDIA Container Toolkit Documentation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/index.html)
- [Docker GPU Access](https://docs.docker.com/compose/gpu-support/)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [PyTorch Performance Tuning](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
- [TensorFlow GPU Guide](https://www.tensorflow.org/guide/gpu)
