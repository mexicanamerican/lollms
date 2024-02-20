#!/bin/bash

PATH="$HOME/miniconda3/bin:$PATH"
export PATH
echo "Initializing conda"
$HOME/miniconda3/bin/conda init --all
source activate vllm && python -m vllm.entrypoints.openai.api_server --model "$1"

# Wait for all background processes to finish
wait