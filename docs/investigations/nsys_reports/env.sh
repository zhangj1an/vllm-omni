#!/bin/bash
# Source this to enable FlashInfer on sm_120 with the userspace CUDA 13.2 toolchain.
# Tested on RTX Pro 6000 Blackwell (sm_120) with torch 2.11.0+cu130, cuDNN 9.19.0,
# FlashInfer 0.6.8.post1.
export CUDA_HOME=/tmp/cuda13
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
export FLASHINFER_CUDA_ARCH_LIST="12.0f"
