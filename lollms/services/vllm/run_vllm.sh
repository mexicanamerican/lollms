#!/bin/bash

cd ~/vllm

python -m vllm.entrypoints.openai.api_server --model %1

# Wait for all background processes to finish
wait