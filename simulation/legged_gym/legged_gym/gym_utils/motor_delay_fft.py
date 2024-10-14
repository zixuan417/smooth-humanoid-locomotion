import torch
import torch.nn as nn


class MotorDelay_80(nn.Module):
    def __init__(self, num_envs, num_actions, device="cuda:0"):
        super(MotorDelay_80, self).__init__()
        self.a = 1.2766
        self.b = 12.13208
        # self.alpha = 1.0
        self.alpha = torch.exp(torch.tensor([-1 / self.b]).to(device))
        self.beta = self.a / self.b
        # self.y_pre = 0.0
        self.y_pre = torch.zeros(num_envs, num_actions, dtype = torch.float, device=device)

        
    def forward(self, x):
        if x.dim() ==1:
            x = x.unsqueeze(1)
        
        # if self.y_pre is None:
        #     self.y_pre = torch.zeros(x.size(0), x.size(1), dtype = x.dtype, device=x.device)

        y = self.alpha * self.y_pre + self.beta * x
        self.y_pre = y
        return y
    
    def reset(self, env_idx):
        self.y_pre[env_idx] = 0
    
    
class MotorDelay_130(nn.Module):
    def __init__(self, num_envs, num_actions, device="cuda:0"):
        super(MotorDelay_130, self).__init__()
        self.a = 0.91
        self.b = 11.28
        # self.alpha = 1.0
        self.alpha = torch.exp(torch.tensor([-1 / self.b]).to(device))
        self.beta = self.a / self.b
        # self.y_pre = 0.0
        self.y_pre = torch.zeros(num_envs, num_actions, dtype = torch.float, device=device)


    def forward(self, x):
        if x.dim() ==1:
            x = x.unsqueeze(1)
        
        # if self.y_pre is None:
        #     self.y_pre = torch.zeros(x.size(0), x.size(1), dtype = x.dtype, device=x.device)

        y = self.alpha * self.y_pre + self.beta * x
        self.y_pre = y
        return y
    
    def reset(self, env_idx):
        self.y_pre[env_idx] = 0
