#!/bin/bash
export CUDA_VISIBLE_DEVICES=5
python nf_graphs_custom_loss.py --config_name=1
python nf_graphs_custom_loss.py --config_name=4
python nf_graphs_custom_loss.py --config_name=7
python nf_graphs_custom_loss.py --config_name=10
python nf_graphs_custom_loss.py --config_name=13
python nf_graphs_custom_loss.py --config_name=16
python nf_graphs_custom_loss.py --config_name=19
python nf_graphs_custom_loss.py --config_name=22
python nf_graphs_custom_loss.py --config_name=25
python nf_graphs_custom_loss.py --config_name=28
