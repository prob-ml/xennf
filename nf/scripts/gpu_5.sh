#!/bin/bash
export CUDA_VISIBLE_DEVICES=6
python nf_graphs_batched.py --config_name=1
python nf_graphs_batched.py --config_name=4
python nf_graphs_batched.py --config_name=7
python nf_graphs_batched.py --config_name=10
python nf_graphs_batched.py --config_name=13
python nf_graphs_batched.py --config_name=16
python nf_graphs_batched.py --config_name=19
python nf_graphs_batched.py --config_name=22
