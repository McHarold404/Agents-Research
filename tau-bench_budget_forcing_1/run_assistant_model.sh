  #!/bin/bash

# Point to the shared Hugging Face cache
export HF_HOME="/mnt/shared/shared_hf_home"
export TRANSFORMERS_CACHE="/mnt/shared/shared_hf_home"
export HF_DATASETS_CACHE="/mnt/shared/shared_hf_home"

export CUDA_VISIBLE_DEVICES=0,1


vllm serve Qwen/Qwen3-8B \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.9 \
  --pipeline-parallel-size 2 \
  --tensor-parallel-size 1 \
  --port 8005

