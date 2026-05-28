
import os
import sys
from time import time

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import Dataset
from data.preprocess import getData
from models import LatentFlow
from utils.earlystop import EarlyStop
from utils.vus_evaluate import evaluate
from utils.tools import adjust_learning_rate
from utils.tools import RG_EAD_Loss

import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)


def progress(iterable):
    return tqdm(iterable, disable=not sys.stderr.isatty())

def evaluate_wrapper(q, init_score, test_score, test_label, window_size, metric_mode):
    return evaluate(
        init_score.reshape(-1),
        test_score.reshape(-1),
        test_label=test_label,
        q=q,
        slidingWindow=window_size,
        metric_mode=metric_mode,
    )

def aggregate_reconstruction(recon_seq, original_len, stride=1):

    B, S, N, P = recon_seq.shape
    device = recon_seq.device
    
    recon_canvas = torch.zeros((B, original_len, N), device=device)
    count_canvas = torch.zeros((B, original_len, N), device=device)
    
    for t in range(S):
        start_idx = t * stride
        end_idx = start_idx + P
        
        patch_pred = recon_seq[:, t, :, :].permute(0, 2, 1) 
        
        recon_canvas[:, start_idx:end_idx, :] += patch_pred
        count_canvas[:, start_idx:end_idx, :] += 1.0
        
    final_recon = recon_canvas / (count_canvas + 1e-8)
    
    return final_recon


class Exp:
    def __init__(self, config):
        self.config = config
        self.config.enc_in = self._get_data()
        self._get_model()
        self._select_criterion()
        self._select_optimizer()
        self.best_model = None

        if not os.path.exists(self.config.model_dir):
            os.makedirs(self.config.model_dir)

    def _get_data(self):
        data = getData(
            path=self.config.data_dir,
            dataset=self.config.dataset,
            period=self.config.period,
            train_rate=self.config.train_rate
        )

        self.feature_num = data['train_data'].shape[1]
        print('\ndata shape: ')
        for k, v in data.items():
            print(k, ': ', v.shape)

        self.train_set = Dataset(data=data['train_data'], stable=data['train_stable'], label=data['train_label'], window_size=self.config.window_size)
        self.valid_set = Dataset(data=data['valid_data'], stable=data['valid_stable'], label=data['valid_label'], window_size=self.config.window_size)
        self.init_set = Dataset(data=data['init_data'], stable=data['init_stable'], label=data['init_label'], window_size=self.config.window_size)
        self.test_set = Dataset(data=data['test_data'], stable=data['test_stable'], label=data['test_label'], window_size=self.config.window_size)

        self.train_loader = DataLoader(self.train_set, batch_size=self.config.batch_size, shuffle=True, drop_last=False)
        self.valid_loader = DataLoader(self.valid_set, batch_size=self.config.batch_size, shuffle=False, drop_last=False)
        self.init_loader = DataLoader(self.init_set, batch_size=self.config.batch_size, shuffle=False, drop_last=False)
        self.test_loader = DataLoader(self.test_set, batch_size=self.config.batch_size, shuffle=False, drop_last=False)

        channel = self.train_set.data.shape[1]
        return channel
    
    def _acquire_device(self):
        if self.config.use_gpu and torch.cuda.is_available():
            device = torch.device('cuda:{}'.format(self.config.gpu_id))
            print('\ndevice: Use GPU: cuda:{}'.format(self.config.gpu_id))
        else:
            self.config.use_gpu = 0
            self.config.use_multi_gpu = 0
            device = torch.device('cpu')
            print('device: Use CPU')
        return device

    def _get_model(self):
        self.device = self._acquire_device()

        model_dict = {
            "LatentFlow": LatentFlow
        }

        self.model = model_dict[self.config.model_name].Model(self.config).float().to(self.device)
        self.checkpoint_path = os.path.join(self.config.model_dir, f"{self.config.dataset}_{self.config.window_size}_{self.config.patch_size}_{self.config.patch_stride}_{self.config.k}_{self.config.lr}_{self.config.batch_size}_{self.config.random_seed}_{self.config.lambda_sparsity}.pth")

        if self.config.use_multi_gpu and self.config.use_gpu:
            self.model = nn.DataParallel(self.model, device_ids=self.config.device_ids)
        
    def _select_optimizer(self):
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr, weight_decay=1e-4)
        
    def _select_criterion(self):
        self.criterion = RG_EAD_Loss(lambda_sparsity = self.config.lambda_sparsity)    
            
    def _process_one_batch(self, batch_data, batch_stable, train, return_adj=False):
        batch_data = batch_data.float().to(self.device)
        batch_stable = batch_stable.float().to(self.device)

        # training
        if train:
            recon, truth, adjs = self.model(batch_data, inference_inertia=False)
            loss, l_rec, l_spar = self.criterion(recon, truth, adjs)
            return loss    
        else:
            recon, _, adj_seq = self.model(
                batch_data,
                inference_inertia=True,
                return_adj=return_adj,
            )
            recon = aggregate_reconstruction(recon, original_len=self.config.window_size, stride=self.config.patch_stride)
            
            return recon, adj_seq

    def train(self):

        early_stopping = EarlyStop(patience=self.config.patience, path=self.checkpoint_path)

        for e in range(self.config.epochs):
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()

            start = time()
            
            self.model.train()
            train_loss = []
            for (batch_data, batch_stable, _) in progress(self.train_loader):
                self.optimizer.zero_grad()
            
                loss = self._process_one_batch(batch_data, batch_stable, train=True)
                train_loss.append(loss.item())
                loss.backward()
                self.optimizer.step()

            with torch.no_grad():
                self.model.eval()
                valid_loss = []
                for (batch_data, batch_stable, _) in progress(self.valid_loader):
                    
                    loss = self._process_one_batch(batch_data, batch_stable, train=True)
                    valid_loss.append(loss.item())
                
            train_loss, valid_loss = np.average(train_loss), np.average(valid_loss)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
                peak_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
            else:
                peak_memory_mb = 0.0

            end = time()
            train_time_epoch = end - start
            print(
                f'Epoch: {e} || Train Loss: {train_loss:.6f} Valid Loss: {valid_loss:.6f} '
                f'|| Train Time: {train_time_epoch:.4f} s || Mem: {peak_memory_mb:.2f} MB'
            )

            early_stopping(valid_loss, self.model)
            if early_stopping.early_stop:
                break
            
            adjust_learning_rate(self.optimizer, e + 1, self.config)

        self.model.load_state_dict(torch.load(self.checkpoint_path, map_location=self.device))

    def test(self):
        self.model.load_state_dict(torch.load(self.checkpoint_path, map_location=self.device))

        with torch.no_grad():
            self.model.eval()
           
            init_scores = []
            for (batch_data, batch_stable, batch_label) in progress(self.init_loader):            
                recon, _ = self._process_one_batch(batch_data, batch_stable, train=False)
                batch_score = torch.mean(
                    (batch_data.float().to(self.device)[:, -1, :] - recon[:, -1, :]) ** 2,
                    dim=-1,
                    keepdim=True,
                )
                init_scores.append(batch_score.detach().cpu().numpy().astype(np.float32, copy=False))
            
            test_label, test_scores = [], []
            iter_id = 0

            for (batch_data, batch_stable, batch_label) in progress(self.test_loader):                
                recon, _ = self._process_one_batch(batch_data, batch_stable, train=False)
                batch_score = torch.mean(
                    (batch_data.float().to(self.device)[:, -1, :] - recon[:, -1, :]) ** 2,
                    dim=-1,
                    keepdim=True,
                )
                test_scores.append(batch_score.detach().cpu().numpy().astype(np.float32, copy=False))
                test_label.append(batch_label.detach().cpu().numpy()[:, -1, :].astype(np.int8, copy=False))
                
                iter_id += 1

            test_label = np.concatenate(test_label, axis=0)
            init_score = np.concatenate(init_scores, axis=0)
            test_score = np.concatenate(test_scores, axis=0)
                
            str_len = (80 - len(self.config.dataset)) // 2
            print(f"\n{'='*str_len} {self.config.dataset} {'='*str_len}")

            print(f"Metric mode: {self.config.metric_mode}")

            if self.config.metric_mode == 'vus':
                print("Q candidates: not used")
            else:
                print(f"Q: {self.config.q}")

            res = evaluate_wrapper(
                self.config.q, init_score, test_score, test_label,
                self.config.window_size, self.config.metric_mode
            )
            print(res)
            print(f"{'='*80}\n")

            if not self.config.save_model and os.path.exists(self.checkpoint_path):
                os.remove(self.checkpoint_path)
