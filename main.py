import argparse

import torch

from exp.exp import Exp
from utils.seed import setSeed

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # data and experiment setup
    parser.add_argument('--dataset', type=str, default='SMD', help='dataset')
    parser.add_argument('--data_dir', type=str, default='./dataset/', help='path of the data')
    parser.add_argument('--model_dir', type=str, default='./checkpoint/', help='path of the checkpoint')
    parser.add_argument('--model_name', type=str, default='LatentFlow', choices=['LatentFlow'], help='model name')
    parser.add_argument('--save_model', action='store_true', help='keep the best checkpoint after evaluation')
    parser.add_argument('--itr', type=int, default=3, help='num of evaluation')

    # optimization
    parser.add_argument('--epochs', type=int, default=30, help='epoch of train')
    parser.add_argument('--patience', type=int, default=5, help='patience of early stopping')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of data')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate of optimizer')
    parser.add_argument('--lradj', type=str, default='6', help='adjust learning rate')
    parser.add_argument('--train_rate', type=float, default=0.8, help='proportion of training data')

    # LatentFlow model
    parser.add_argument('--period', type=int, default=1440, help='approximate period of time series')
    parser.add_argument('--window_size', type=int, default=64, help='size of sliding window')
    parser.add_argument('--d_model', type=int, default=64, help='dimension of hidden layer')
    parser.add_argument('--dropout', type=float, default=0.6, help='dropout')
    parser.add_argument('--patch_size', type=int, default=32, help='patch size')
    parser.add_argument('--patch_stride', type=int, default=2, help='patch stride')
    parser.add_argument('--k', type=int, default=15, help='number of top-k cross-channel neighbors')
    parser.add_argument('--lambda_sparsity', type=float, default=1e-3, help='sparsity regularization coefficient in LatentFlow')

    # evaluation
    parser.add_argument('--q', type=float, default=None,
                        help='SPOT q value for threshold-based Affiliation metrics')
    parser.add_argument('--metric_mode', type=str, default='aff', choices=['aff', 'vus', 'all'],
                        help='metrics to compute and print')
    parser.add_argument('--select_metric', type=str, default=None,
                        choices=['Affiliation_F1', 'VUS_ROC', 'VUS_PR'],
                        help='metric used to select the best q during evaluation')

    # runtime
    parser.add_argument('--random_seed', type=int, default=42, help='random seed')
    parser.add_argument('--use_gpu', type=int, default=1, help='use gpu')
    parser.add_argument('--use_multi_gpu', type=int, default=0, help='use multiple gpus')
    parser.add_argument('--gpu_id', type=int, default=0, help='device ids of gpus')
    parser.add_argument('--devices', type=str, default='0', help='device ids of multile gpus')

    config = parser.parse_args()
    if config.use_gpu and torch.cuda.is_available():
        torch.cuda.set_device(config.gpu_id)
    elif config.use_gpu:
        print('CUDA is not available. Falling back to CPU.')
        config.use_gpu = 0
        config.use_multi_gpu = 0

    if config.select_metric is None:
        config.select_metric = 'VUS_ROC' if config.metric_mode == 'vus' else 'Affiliation_F1'

    if config.metric_mode in {'aff', 'all'} and config.q is None:
        parser.error('--q is required when --metric_mode is aff or all')

    valid_select_metrics = {
        'aff': {'Affiliation_F1'},
        'vus': {'VUS_ROC', 'VUS_PR'},
        'all': {'Affiliation_F1', 'VUS_ROC', 'VUS_PR'},
    }
    if config.select_metric not in valid_select_metrics[config.metric_mode]:
        valid = ', '.join(sorted(valid_select_metrics[config.metric_mode]))
        parser.error(f'--select_metric must be one of {{{valid}}} when --metric_mode {config.metric_mode}')

    if config.use_gpu and config.use_multi_gpu:
        config.devices = config.devices.replace(' ', '')
        device_ids = config.devices.split(',')
        config.device_ids = [int(id_) for id_ in device_ids]
        config.gpu = config.device_ids[0]
    print(config)

    for ii in range(config.itr):
        setSeed(config.random_seed+ii)
        exp = Exp(config)
        exp.train()
        exp.test()
