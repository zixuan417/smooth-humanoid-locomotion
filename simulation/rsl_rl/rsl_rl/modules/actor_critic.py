# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import numpy as np

import code
import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.modules import rnn
from torch.nn.modules.activation import ReLU


class Actor(nn.Module):
    def __init__(self, num_prop,
                 num_actions,
                 actor_hidden_dims, 
                 activation, 
                 tanh_encoder_output=False, **kwargs) -> None:
        super().__init__()
        # prop
        self.num_prop = num_prop
        self.num_actions = num_actions

        actor_layers = []
        actor_layers.append(nn.Linear(num_prop, 
                                      actor_hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(activation)
        if tanh_encoder_output:
            actor_layers.append(nn.Tanh())
        self.actor_backbone = nn.Sequential(*actor_layers)

    def forward(self, obs_all, hist_encoding=False, eval=False, scandots_latent=None):
        obs = obs_all
        obs_prop = obs[:, :self.num_prop]
        backbone_input = obs_prop
        backbone_output = self.actor_backbone(backbone_input)
        return backbone_output


class ActorCritic(nn.Module):
    is_recurrent = False
    def __init__(self,  num_prop,
                        num_critic_obs,
                        num_priv_latent, 
                        num_priv_explicit,
                        num_hist,
                        num_actions,
                        actor_hidden_dims=[256, 256, 256],
                        critic_hidden_dims=[256, 256, 256],
                        activation='elu',
                        init_noise_std=1.0,
                        fix_action_std=False,
                        **kwargs):
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super().__init__()

        self.fix_action_std = fix_action_std
        
        self.kwargs = kwargs
        priv_encoder_dims= kwargs['priv_encoder_dims']
        activation = get_activation(activation)
        
        self.actor = Actor(num_prop=num_prop, 
                           num_actions=num_actions, 
                           actor_hidden_dims=actor_hidden_dims, 
                           activation=activation, tanh_encoder_output=kwargs['tanh_encoder_output'])
        
        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(num_critic_obs, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        # Action noise
        if self.fix_action_std:
            # action_std = torch.tensor([0.5, 0.25, 0.25, 0.25, 0.2, 0.2] * 2 + [0.2, 0.2, 0.2] + [0.3] * 8)
            action_std = torch.tensor([0.3, 0.3, 0.3, 0.4, 0.2] * 2 + [0.25, 0.25, 0.25] + [0.5] * 8)
            self.std = nn.Parameter(action_std, requires_grad=False)
        else:
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False
        
    
    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError
    
    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        mean = self.actor(observations)
        # has_nan_mean = torch.isnan(mean).any().item()
        # if has_nan_mean:
        #     print("mean has nan")
        #     has_nan_obs = torch.isnan(observations).any().item()
        #     if has_nan_obs:
        #         print("has nan obs: ", has_nan_obs)
        #     obs_array = observations.cpu().detach().numpy()
        #     np.savetxt('nan_obs.txt', obs_array)
        #     exit()
        self.distribution = Normal(mean, mean*0. + self.std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations, eval=False, **kwargs):
        if not eval:
            actions_mean = self.actor(observations, eval)
            return actions_mean
        else:
            actions_mean = self.actor(observations, eval=True)
            return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value
    
    def reset_std(self, std, num_actions, device):
        new_std = std * torch.ones(num_actions, device=device)
        self.std.data = new_std.data

def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
