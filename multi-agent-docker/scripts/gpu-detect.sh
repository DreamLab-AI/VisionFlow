#!/bin/bash
# GPU Detection and Configuration Script
# Automatically detects and configures GPU acceleration

set -e

log_info() {
    echo "[GPU] $1"
}

log_success() {
    echo "[GPU] ✅ $1"
}

log_warning() {
    echo "[GPU] ⚠️  $1"
}

detect_gpu() {
    log_info "Detecting GPU hardware..."

    # Check for NVIDIA GPU
    if command -v nvidia-smi &>/dev/null; then
        if nvidia-smi &>/dev/null; then
            export CUDA_VISIBLE_DEVICES=0
            export USE_GPU=true
            export GPU_VENDOR=nvidia

            # Get GPU info
            GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
            GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
            CUDA_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)

            log_success "NVIDIA GPU detected: $GPU_NAME"
            log_info "GPU Memory: ${GPU_MEMORY}MB"
            log_info "CUDA Driver: $CUDA_VERSION"

            # Configure CUDA paths
            export CUDA_HOME=/usr/local/cuda
            export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
            export PATH=/usr/local/cuda/bin:$PATH

            return 0
        fi
    fi

    # Check for AMD GPU
    if command -v rocm-smi &>/dev/null; then
        if rocm-smi &>/dev/null; then
            export USE_GPU=true
            export GPU_VENDOR=amd

            log_success "AMD GPU detected"
            return 0
        fi
    fi

    # No GPU detected
    log_warning "No GPU detected, falling back to CPU"
    export USE_GPU=false
    export GPU_VENDOR=none
    return 1
}

configure_gpu_tools() {
    if [ "$USE_GPU" = "true" ] && [ "$GPU_VENDOR" = "nvidia" ]; then
        log_info "Configuring GPU-accelerated tools..."

        # PyTorch GPU check
        if python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
            log_success "PyTorch CUDA support enabled"
        else
            log_warning "PyTorch CUDA support not available"
        fi

        # TensorFlow GPU check
        if python3 -c "import tensorflow as tf; print(len(tf.config.list_physical_devices('GPU')))" 2>/dev/null | grep -q -v "^0$"; then
            log_success "TensorFlow GPU support enabled"
        else
            log_warning "TensorFlow GPU support not available"
        fi
    fi
}

configure_resource_limits() {
    if [ "$USE_GPU" = "true" ]; then
        log_info "Configuring GPU resource limits..."

        # Set GPU memory growth to avoid OOM
        export TF_FORCE_GPU_ALLOW_GROWTH=true
        export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

        log_success "GPU resource limits configured"
    fi
}

save_gpu_config() {
    local config_file="/workspace/.gpu_config"

    cat > "$config_file" << EOF
# GPU Configuration
# Generated: $(date)
USE_GPU=$USE_GPU
GPU_VENDOR=$GPU_VENDOR
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-none}
GPU_NAME=${GPU_NAME:-none}
GPU_MEMORY=${GPU_MEMORY:-0}
CUDA_VERSION=${CUDA_VERSION:-none}
EOF

    chmod 644 "$config_file"
    log_success "GPU configuration saved to $config_file"
}

main() {
    log_info "Starting GPU detection..."

    if detect_gpu; then
        configure_gpu_tools
        configure_resource_limits
        save_gpu_config

        log_success "GPU configuration complete"
        return 0
    else
        save_gpu_config
        log_info "Running in CPU mode"
        return 0
    fi
}

# Run detection if sourced with --detect flag
if [ "$1" = "--detect" ] || [ "${BASH_SOURCE[0]}" = "${0}" ]; then
    main
fi