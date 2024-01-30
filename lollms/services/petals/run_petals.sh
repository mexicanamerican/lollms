#!/bin/bash

cd ~/vllm
PATH="$HOME/miniconda3/bin:$PATH"
export PATH
conda activate vllm && python -m vllm.entrypoints.openai.api_server --model %1

# Wait for all background processes to finish
wait