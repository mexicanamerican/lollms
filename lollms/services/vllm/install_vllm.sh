#!/bin/bash

# Check if miniconda3/bin/conda exists
if [ -e "$HOME/miniconda3/bin/conda" ]; then
    echo "Conda is installed!"
else
    echo "Conda is not installed. Please install it first."
    echo Installing conda
    curl -LOk https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    ./Miniconda3-latest-Linux-x86_64.sh -b
    $HOME/miniconda3/bin/conda init --all
    rm ./Miniconda3-latest-Linux-x86_64.sh
    echo Done
fi
PATH="$HOME/miniconda3/bin:$PATH"
export PATH
conda create -n vllm python=3.9 -y && conda activate vllm && pip install vllm --user