#!/bin/sh
conda create -n myenv python=3.9 -y && conda activate myenv && pip install vllm --user