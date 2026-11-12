#!/usr/bin/env bash
set -e

# Quantize 14B model
uv run python -m src.utils.quantization --model Qwen/Qwen2.5-14B --quant_bits 8

# DGP figure
uv run python main.py --dgp

# Experiment 1: 7B model entropy scaling (asymptotic analysis)
uv run python main.py --models 7B --max_train 300 --num_train 64 --num_contexts 256 --num_val 64 --num_entropy_levels 10

# Experiment 2: Model ladder scaling at H=0.6931 (asymptotic analysis)
uv run python main.py --max_train 300 --num_train 64 --num_contexts 256 --num_val 64 --entropies 0.6931

# Experiment 3: 7B model entropy scaling (scaling bound analysis)
uv run python main.py --models 7B --max_train 1000 --num_train 500 --num_contexts 100 --num_val 100 --entropies 0.3600 0.4000 0.4400 0.4700 0.5100 0.5500 0.5800 0.6200 0.6600 0.6931

# Experiment 4: Model ladder scaling at H=0.6931 (scaling bound analysis)
uv run python main.py --max_train 1000 --num_train 500 --num_contexts 100 --num_val 100 --entropies 0.6931
