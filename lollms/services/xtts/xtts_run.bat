@echo off
conda activate xtts && python -m xtts_api_server -o %1 -sf %2
