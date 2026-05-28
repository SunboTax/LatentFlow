import torch
import torch.nn as nn


def adjust_learning_rate(optimizer, epoch, args, printout=True):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.lr * (0.5 ** ((epoch) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.lr if epoch < 3 else args.lr * (0.7 ** ((epoch - 3) // 1))}
    elif args.lradj == 'type4':
        lr_adjust = {epoch: args.lr if epoch < 10 else args.lr * (0.5 ** ((epoch // 10) // 1))}
    elif args.lradj == 'type5':
        lr_adjust = {epoch: args.lr if epoch < 5 else args.lr * (0.5 ** ((epoch // 5) // 1))}
    elif args.lradj == 'type6':
        lr_adjust = {20: args.lr * 0.5 , 40: args.lr * 0.01, 60:args.lr * 0.01,8:args.lr * 0.01,100:args.lr * 0.01 }
    elif args.lradj == 'constant':
        lr_adjust = {epoch: args.lr}
    elif args.lradj == '3':
        lr_adjust = {epoch: args.lr if epoch < 10 else args.lr*0.1}
    elif args.lradj == '4':
        lr_adjust = {epoch: args.lr if epoch < 15 else args.lr*0.1}
    elif args.lradj == '5':
        lr_adjust = {epoch: args.lr if epoch < 25 else args.lr*0.1}
    elif args.lradj == '6':
        lr_adjust = {epoch: args.lr if epoch < 5 else args.lr*0.1}  
    
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if printout: print('Updating learning rate to {}'.format(lr))

class RG_EAD_Loss(nn.Module):
    def __init__(self, lambda_sparsity=1e-3):
        super().__init__()
        self.lambda_s = lambda_sparsity
        self.mse = nn.MSELoss()
        
    def forward(self, recon_seq, true_seq, adj_seq):
        """
        recon_seq: (B, S, N, P)
        true_seq:  (B, S, N, P)
        adj_seq:   (B, S, N, N)
        """
        # 1. Reconstruction Loss (Main Objective)
        rec_loss = self.mse(recon_seq, true_seq)
        
        # 2. Sparsity Loss (Regularization)
        # Encourage the graph to be sparse (L1 Norm)
        sparsity_loss = torch.mean(torch.abs(adj_seq))
        
        total_loss = rec_loss + self.lambda_s * sparsity_loss
        return total_loss, rec_loss, sparsity_loss
