#!/bin/bash
export CUDA_VISIBLE_DEVICES=5
python nf_graphs_batched.py --config_name=0
python nf_graphs_batched.py --config_name=3
python nf_graphs_batched.py --config_name=6
python nf_graphs_batched.py --config_name=9
python nf_graphs_batched.py --config_name=12
python nf_graphs_batched.py --config_name=15
python nf_graphs_batched.py --config_name=18
python nf_graphs_batched.py --config_name=21
