import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from layers.RevIN import RevIN

class TemporalFeatureExtractor(nn.Module):
    def __init__(self, input_len, d_model, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=d_model, 
                               kernel_size=kernel_size, padding=kernel_size//2)
        self.conv_gate = nn.Conv1d(in_channels=1, out_channels=d_model, 
                                   kernel_size=kernel_size, padding=kernel_size//2)
        self.norm = nn.LayerNorm(d_model)
        
        self.flatten_dim = d_model * input_len
        self.final_proj = nn.Sequential(
            nn.Linear(self.flatten_dim, d_model),
            nn.GELU(), 
            nn.Dropout(0.1)
        )

    def forward(self, x):
        
        B, N, P = x.shape
        x_in = x.reshape(B * N, 1, P)
        
        feature = self.conv1(x_in)       
        gate = torch.sigmoid(self.conv_gate(x_in)) 
        out = feature * gate 
        
        out = out.reshape(out.shape[0], -1)
        out = self.final_proj(out) 
        out = self.norm(out)
        
        out = out.reshape(B, N, -1) 
        return out

class SparseGraphLearner(nn.Module):
    def __init__(self, d_model, k_neighbors=10):
        super().__init__()
        self.k = k_neighbors
        self.query = nn.Linear(d_model, d_model // 4)
        self.key = nn.Linear(d_model, d_model // 4)

    def forward(self, x):

        Q = self.query(x)
        K = self.key(x)
        
        logits = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(Q.size(-1))
        
        B, N, _ = x.shape
        diag_mask = torch.eye(N, device=x.device).bool()
        logits.masked_fill_(diag_mask.unsqueeze(0), -1e9)
        

        mask = torch.full_like(logits, -float('inf'))
        curr_k = min(self.k, N - 1) 
        topk_values, topk_indices = torch.topk(logits, k=curr_k, dim=-1)
        logits_sparse = torch.scatter(mask, -1, topk_indices, topk_values)
        adj = torch.softmax(logits_sparse, dim=-1)
        
        return adj

class StructureEvolutionCell(nn.Module):
    def __init__(self, d_model, k_neighbors=10):
        super().__init__()
        self.graph_learner = SparseGraphLearner(d_model, k_neighbors)
        self.gate_net = nn.Sequential(
            nn.Linear(d_model + 1, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x_t, prev_adj, force_inertia=False):
        B, N, D = x_t.shape
        adj_candidate = self.graph_learner(x_t)
        prev_degree = prev_adj.sum(dim=-1, keepdim=True) / (N + 1e-6)
        
        gate_input = torch.cat([x_t, prev_degree], dim=-1)
        alpha = self.gate_net(gate_input)
        alpha = alpha.expand(-1, -1, N)
        
        if force_inertia:
            alpha = alpha * 0.5 
            
        adj_t = alpha * adj_candidate + (1 - alpha) * prev_adj
        
        diag_mask = torch.eye(N, device=x_t.device).bool().unsqueeze(0)
        adj_t = adj_t.masked_fill(diag_mask, 0.0)
        
        row_sum = adj_t.sum(dim=-1, keepdim=True) + 1e-6
        adj_t = adj_t / row_sum
        
        return adj_t

class GraphAggregator(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj):
        
        h = self.proj(x)
        h_agg = torch.bmm(adj, h) 
        
        out = self.norm(h_agg + self.dropout(x))
        
        return out

class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.n_nodes = configs.enc_in      
        self.input_len = configs.window_size 
        self.d_model = configs.d_model
        
        self.patch_size = configs.patch_size
        self.stride = configs.patch_stride 
        self.seq_len = (self.input_len - self.patch_size) // self.stride + 1
        self.k = getattr(configs, 'k', 10)
        
        self.encoder = TemporalFeatureExtractor(self.patch_size, self.d_model)
        self.feature_rnn = nn.GRUCell(input_size=self.d_model, hidden_size=self.d_model)
        self.structure_cell = StructureEvolutionCell(self.d_model, k_neighbors=self.k)
        self.gnn = GraphAggregator(self.d_model, configs.dropout)
        
        self.bottleneck_dim = self.d_model // 2 
        
        self.decoder = nn.Sequential(
            nn.Linear(self.d_model, self.bottleneck_dim), 
            nn.GELU(),
            nn.Linear(self.bottleneck_dim, self.patch_size) 
        )

    def forward(self, x, init_adj=None, inference_inertia=False, return_adj=False):
        
        B, L, N = x.shape
        x_perm = x.permute(0, 2, 1) 
        x_patches = x_perm.unfold(dimension=2, size=self.patch_size, step=self.stride)
        x_seq = x_patches.permute(0, 2, 1, 3) 
        
        if init_adj is None:
            curr_adj = torch.zeros(B, N, N, device=x.device)
        else:
            curr_adj = init_adj
            
        curr_h = torch.zeros(B * N, self.d_model, device=x.device)
        recon_list = []
        adj_list = []
        
        for t in range(self.seq_len):
            x_t_raw = x_seq[:, t, :, :].contiguous() 
            x_t_emb = self.encoder(x_t_raw)          
            
            x_t_flat = x_t_emb.reshape(B * N, -1)
            curr_h = self.feature_rnn(x_t_flat, curr_h) 
            x_t_context = curr_h.reshape(B, N, -1)
            
            curr_adj = self.structure_cell(x_t_context, curr_adj, force_inertia=inference_inertia)
            
            h_spatial = self.gnn(x_t_context, curr_adj)
        
            x_t_recon = self.decoder(h_spatial)
            
            recon_list.append(x_t_recon)
            adj_list.append(curr_adj)
            
        recon_seq = torch.stack(recon_list, dim=1) 
        adj_seq = torch.stack(adj_list, dim=1)     
        
        return recon_seq, x_seq, adj_seq
