#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 python infer_retrained_WAD.py --range 0 250
CUDA_VISIBLE_DEVICES=0 python infer_retrained_WAD.py --range 251 500
CUDA_VISIBLE_DEVICES=1 python infer_retrained_WAD.py --range 501 750
CUDA_VISIBLE_DEVICES=1 python infer_retrained_WAD.py --range 751 1000
CUDA_VISIBLE_DEVICES=2 python infer_retrained_WAD.py --range 1001 1250
CUDA_VISIBLE_DEVICES=2 python infer_retrained_WAD.py --range 1251 1500
CUDA_VISIBLE_DEVICES=3 python infer_retrained_WAD.py --range 1501 1700
CUDA_VISIBLE_DEVICES=3 python infer_retrained_WAD.py --range 1701 1917