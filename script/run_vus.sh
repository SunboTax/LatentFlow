#!/bin/bash

mkdir -p logs/vus/

python main.py --model_name LatentFlow --dataset MSL --batch_size 256 --lradj type4 --epochs 30 --patience 5 --dropout 0.3 --lr 1e-3 --d_model 128 --window_size 256 --patch_size 32 --patch_stride 16 --k 3 --metric_mode vus > logs/vus/LatentFlow_MSL.log

python main.py --model_name LatentFlow --dataset SMD --batch_size 256 --lradj type4 --epochs 30 --patience 5 --dropout 0.3 --lr 1e-3 --d_model 64 --window_size 196 --patch_size 96 --patch_stride 4 --k 15 --metric_mode vus > logs/vus/LatentFlow_SMD.log

python main.py --model_name LatentFlow --dataset SWAT --batch_size 256 --lradj type4 --epochs 30 --patience 5 --dropout 0.3 --lr 1e-3 --d_model 128 --window_size 128 --patch_size 64 --patch_stride 8 --k 15 --metric_mode vus > logs/vus/LatentFlow_SWAT.log

python main.py --model_name LatentFlow --dataset PSM --batch_size 256 --lradj type3 --epochs 30 --patience 5 --dropout 0.3 --lr 1e-4 --d_model 128 --window_size 256 --patch_size 32 --patch_stride 16 --k 3 --metric_mode vus > logs/vus/LatentFlow_PSM.log

python main.py --model_name LatentFlow --dataset WADI --batch_size 64 --lradj type4 --epochs 30 --patience 5 --dropout 0.3 --lr 5e-3 --d_model 64 --window_size 196 --patch_size 128 --patch_stride 12 --k 15 --metric_mode vus > logs/vus/LatentFlow_WADI.log

python main.py --model_name LatentFlow --dataset Kitsune --batch_size 64 --lradj type3 --epochs 30 --patience 5 --dropout 0.3 --lr 1e-3 --d_model 128 --window_size 128 --patch_size 96 --patch_stride 4 --k 15 --metric_mode vus > logs/vus/LatentFlow_Kitsune.log

python main.py --model_name LatentFlow --dataset Creditcard --batch_size 64 --lradj type4 --epochs 30 --patience 5 --dropout 0.3 --lr 1e-4 --d_model 64 --window_size 256 --patch_size 128 --patch_stride 12 --k 15 --metric_mode vus > logs/vus/LatentFlow_Creditcard.log

python main.py --model_name LatentFlow --dataset UNSW --batch_size 64 --lradj type5 --epochs 30 --patience 5 --dropout 0.3 --lr 4e-4 --d_model 64 --window_size 256 --patch_size 32 --patch_stride 8 --k 15 --metric_mode vus > logs/vus/LatentFlow_UNSW.log
