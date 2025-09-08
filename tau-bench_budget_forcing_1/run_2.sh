#!/bin/bash

export CUDA_VISIBLE_DEVICES=2,3

export HF_HOME=/data/data/amir/.cache/
export TRANSFORMERS_CACHE=/data/data/amir/.cache/
export HF_DATASETS_CACHE=/data/data/amir/.cache/



# python test.py
python run.py --num-trials 5 --agent-strategy react --env airline --model Qwen3-14B --model-provider vllm --user-model Qwen2.5-72B-Instruct  --user-model-provider vllm --user-strategy llm --max-concurrency 3

# python run.py --num-trials 5 --agent-strategy irma-airline --env airline --model phi-4 --model-provider vllm --user-model Qwen2.5-72B-Instruct --user-model-provider vllm --user-strategy llm --max-concurrency 2