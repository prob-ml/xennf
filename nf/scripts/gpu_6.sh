#!/bin/bash
export CUDA_VISIBLE_DEVICES=7
python nf_graphs_batched.py --config_name=2
python nf_graphs_batched.py --config_name=5
python nf_graphs_batched.py --config_name=8
python nf_graphs_batched.py --config_name=11
python nf_graphs_batched.py --config_name=14
python nf_graphs_batched.py --config_name=17
python nf_graphs_batched.py --config_name=20
python nf_graphs_batched.py --config_name=23
