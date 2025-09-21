#!/bin/bash

 export CUDA_VISIBLE_DEVICES=2

# export HF_HOME=/data/data/amir/.cache/
# export TRANSFORMERS_CACHE=/data/data/amir/.cache/
# export HF_DATASETS_CACHE=/data/data/amir/.cache/

export HF_HOME="/mnt/shared/shared_hf_home"
export TRANSFORMERS_CACHE="/mnt/shared/shared_hf_home"
export HF_DATASETS_CACHE="/mnt/shared/shared_hf_home"

# python test.py
# python run.py --num-trials 5 --agent-strategy irma-retail --env retail --model phi-4 --model-provider vllm --user-model Qwen2.5-72B-Instruct --user-model-provider vllm --user-strategy llm --max-concurrency 2 

# python run.py --num-trials 5 --agent-strategy react --env airline --model Qwen/Qwen3-4B-Thinking-2507 --model-provider vllm --user-model gpt-4o --user-model-provider openai --user-strategy llm --max-concurrency 1
# python run.py --num-trials 5 --agent-strategy react --env retail --model Qwen3-8B --model-provider vllm --user-model Qwen2.5-72B-Instruct  --user-model-provider vllm --user-strategy llm --max-concurrency 4

### Naman 
# python run.py --num-trials 5 --agent-strategy react --env retail --model Qwen3-8B --model-provider vllm --user-model Qwen2.5-72B-Instruct --user-model-provider vllm --user-strategy llm --max-concurrency 4 

python run.py --num-trials 5 --agent-strategy react --env retail --model Qwen3-8B --model-provider vllm --user-model Qwen3-8B --user-model-provider vllm --user-strategy llm --max-concurrency 1  --task-ids 0 
