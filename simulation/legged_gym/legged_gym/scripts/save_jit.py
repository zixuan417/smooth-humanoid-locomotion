import os, sys
from statistics import mode
sys.path.append("../../../rsl_rl")
import torch
import torch.nn as nn
from rsl_rl.modules.actor_critic_rma import Actor, StateHistoryEncoder, get_activation
import argparse
import code
import shutil

def get_load_path(root, load_run=-1, checkpoint=-1, model_name_include="model"):
    if not os.path.isdir(root):  # use first 4 chars to mactch the run name
        model_name_cand = os.path.basename(root)
        model_parent = os.path.dirname(root)
        model_names = os.listdir(model_parent)
        model_names = [name for name in model_names if os.path.isdir(os.path.join(model_parent, name))]
        for name in model_names:
            if len(name) >= 6:
                if name[:6] == model_name_cand:
                    root = os.path.join(model_parent, name)
    if checkpoint==-1:
        models = [file for file in os.listdir(root) if model_name_include in file]
        models.sort(key=lambda m: '{0:0>15}'.format(m))
        model = models[-1]
        checkpoint = model.split("_")[-1].split(".")[0]
    else:
        model = "model_{}.pt".format(checkpoint) 

    load_path = os.path.join(root, model)
    return load_path, checkpoint

class HardwareRefNN(nn.Module):
    def __init__(self,  num_prop,
                        num_priv_latent, 
                        num_hist,
                        critic_obs_extra,
                        num_actions,
                        actor_hidden_dims=[512, 256, 128],
                        activation='elu',
                        priv_encoder_dims=[64, 20],
                        ):
        super().__init__()

        self.num_prop = num_prop
        self.num_hist = num_hist
        self.num_actions = num_actions
        self.num_priv_latent = num_priv_latent
        num_obs = num_prop + num_hist*num_prop + num_priv_latent + critic_obs_extra
        self.num_obs = num_obs
        activation = get_activation(activation)
        
        num_priv_explicit = 0
        
        self.normalizer = None
        
        self.actor = Actor(num_prop, 
                           num_actions, 
                           actor_hidden_dims, 
                           priv_encoder_dims, num_priv_latent, num_hist, 
                           activation, tanh_encoder_output=False)

    def load_normalizer(self, normalizer):
        self.normalizer = normalizer
        self.normalizer.eval()

    def forward(self, obs):
        assert obs.shape[1] == self.num_obs, f"Expected {self.num_obs} but got {obs.shape[1]}"
        obs = self.normalizer(obs)
        return self.actor(obs, hist_encoding=True, eval=False)
    
def play(args):
    load_run = "../../logs/{}/{}".format(args.proj_name, args.exptid)
    checkpoint = args.checkpoint
    critic_obs_extra = 0
    
    if args.robot == "gr1":
        n_priv_latent = 4 + 1 + 21*2 + 3
        num_scan = 0
        num_actions = 19
        num_dofs = 21
        
        n_proprio = 2 + 3 + 3 + 2 + 2*num_dofs + num_actions
    elif args.robot == "h1":
        critic_obs_extra = 3
        n_priv_latent = 4 + 1 + 19*2
        num_scan = 0
        num_actions = 19
        
        n_proprio = 2 + 3 + 3 + 2 + 3*num_actions
    elif args.robot == "berkeley":
        n_priv_latent = 4 + 1 + 12*2 + 3
        num_scan = 0
        num_actions = 12
        
        n_proprio = 2 + 3 + 3 + 2 + 3*num_actions
    elif args.robot == "g1":
        n_priv_latent = 4 + 1 + 21*2 + 3
        num_scan = 0
        num_actions = 21
        
        n_proprio = 2 + 3 + 3 + 2 + 3*num_actions
    else:
        raise ValueError(f"Robot {args.robot} not supported!")
    
    history_len = 10

    device = torch.device('cpu')
    policy = HardwareRefNN(n_proprio, 
                           n_priv_latent, history_len, critic_obs_extra,
                           num_actions).to(device)
    load_path, checkpoint = get_load_path(root=load_run, checkpoint=checkpoint)
    load_run = os.path.dirname(load_path)
    print(f"Loading model from: {load_path}")
    ac_state_dict = torch.load(load_path, map_location=device)
    policy.load_state_dict(ac_state_dict['model_state_dict'], strict=False)
    policy.load_normalizer(ac_state_dict['normalizer'])
    
    policy = policy.to(device)#.cpu()
    if not os.path.exists(os.path.join(load_run, "traced")):
        os.mkdir(os.path.join(load_run, "traced"))

    # Save the traced actor
    policy.eval()
    with torch.no_grad(): 
        num_envs = 2
        
        obs_input = torch.ones(num_envs, n_proprio + n_priv_latent + history_len*n_proprio + critic_obs_extra, device=device)
        print("obs_input shape: ", obs_input.shape)
        
        traced_policy = torch.jit.trace(policy, obs_input)
        
        # traced_policy = torch.jit.script(policy)
        save_path = os.path.join(load_run, "traced", args.exptid + "-" + str(checkpoint) + "-jit.pt")
        traced_policy.save(save_path)
        print("Saved traced_actor at ", os.path.abspath(save_path))
        print("Robot: ", args.robot)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--proj_name', type=str)
    parser.add_argument('--exptid', type=str)
    parser.add_argument('--checkpoint', type=int, default=-1)
    parser.add_argument('--robot', type=str, default="gr1") # options: gr1, h1, g1

    args = parser.parse_args()
    play(args)
