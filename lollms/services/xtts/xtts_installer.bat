@echo off
conda deactivate && conda create --name xtts --yes && conda activate xtts && pip install xtts-api-server --user