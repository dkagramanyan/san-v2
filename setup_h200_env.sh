#!/bin/bash
# Environment setup script for H200 GPU training
# Run this before your training script: source setup_h200_env.sh

echo "Setting up environment for H200 GPU training..."

# 1. Set CUDA architecture for H200 (Hopper - SM 9.0)
export TORCH_CUDA_ARCH_LIST="9.0"
echo "✓ Set TORCH_CUDA_ARCH_LIST=9.0 (Hopper architecture)"

# 2. Clear cached CUDA extensions (they may be compiled for wrong architecture)
CACHE_DIR="$HOME/.cache/torch_extensions"
if [ -d "$CACHE_DIR" ]; then
    echo "Clearing cached CUDA extensions in $CACHE_DIR..."
    rm -rf "$CACHE_DIR"
    echo "✓ Cleared CUDA extension cache"
fi

# 3. NCCL optimizations for multi-GPU training
export NCCL_DEBUG=WARN  # Set to INFO for debugging
export NCCL_P2P_LEVEL=NVL  # Use NVLink for P2P if available
export NCCL_SHM_DISABLE=0  # Enable shared memory
export NCCL_IB_DISABLE=0   # Enable InfiniBand if available
echo "✓ Set NCCL environment variables"

# 4. CUDA performance settings
export CUDA_LAUNCH_BLOCKING=0  # Allow async kernel launches
export CUDA_DEVICE_MAX_CONNECTIONS=1  # Can help with serialization issues
echo "✓ Set CUDA performance variables"

# 5. PyTorch settings
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"  # Better memory allocation
echo "✓ Set PyTorch memory allocation config"

# 6. Disable debug features that slow things down
export TORCH_DISTRIBUTED_DEBUG=OFF
echo "✓ Disabled distributed debug mode"

# Print summary
echo ""
echo "Environment configured. Key settings:"
echo "  TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
echo "  NCCL_P2P_LEVEL=$NCCL_P2P_LEVEL"
echo "  CUDA_LAUNCH_BLOCKING=$CUDA_LAUNCH_BLOCKING"
echo ""
echo "IMPORTANT: If you're using GCC 14, consider downgrading to GCC 11 or 12:"
echo "  conda install -c conda-forge gcc=12 gxx=12"
echo ""
echo "Now run your training script."
