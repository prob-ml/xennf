#!/bin/bash
export CUDA_VISIBLE_DEVICES=6
python nf_graphs_custom_loss.py --config_name=2
python nf_graphs_custom_loss.py --config_name=5
python nf_graphs_custom_loss.py --config_name=8
python nf_graphs_custom_loss.py --config_name=11
python nf_graphs_custom_loss.py --config_name=14
python nf_graphs_custom_loss.py --config_name=17
python nf_graphs_custom_loss.py --config_name=20
python nf_graphs_custom_loss.py --config_name=23
python nf_graphs_custom_loss.py --config_name=26
python nf_graphs_custom_loss.py --config_name=29
