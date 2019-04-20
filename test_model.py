import argparse
import os
import sys
import time

import numpy as np
import torch

from a2c_ppo_acktr.envs import VecPyTorch, make_vec_envs
from a2c_ppo_acktr.utils import get_render_func, get_vec_normalize
import gym
import gym_cassie

sys.path.append('a2c_ppo_acktr')

parser = argparse.ArgumentParser(description='RL')
parser.add_argument(
    '--seed', 
    type=int, 
    default=1, 
    help='random seed (default: 1)')
parser.add_argument(
    '--log-interval',
    type=int,
    default=10,
    help='log interval, one log per n updates (default: 10)')
parser.add_argument(
    '--env-name',
    default='Cassie-v0',
    help='environment to train on')
parser.add_argument(
    '--load-dir',
    default='./trained_models/ppo/',
    help='directory to save agent logs (default: ./trained_models/)')
parser.add_argument(
    '--non-det',
    action='store_true',
    default=False,
    help='whether to use a non-deterministic policy')
args = parser.parse_args()

args.det = not args.non_det

env = make_vec_envs(
    args.env_name,
    args.seed + 1000,
    1,
    None,
    None,
    device='cpu',
    allow_early_resets=False)

# Get a render function
render_func = get_render_func(env)

# We need to use the same statistics for normalization as used in training
#actor_critic, ob_rms = torch.load(os.path.join(args.load_dir, args.env_name + ".pt"))
#actor_critic, ob_rms = torch.load("archived_models/Cassie-v0_sharp.pt")

actor_critic.eval()
vec_norm = get_vec_normalize(env)
if vec_norm is not None:
    vec_norm.eval()
    vec_norm.ob_rms = ob_rms

recurrent_hidden_states = torch.zeros(1, actor_critic.recurrent_hidden_state_size)
masks = torch.zeros(1, 1)
print("Hidden size: ", actor_critic.recurrent_hidden_state_size)

obs = env.reset()

while True:
    with torch.no_grad():
        value, action, _, recurrent_hidden_states = actor_critic.act(
            obs, recurrent_hidden_states, masks, deterministic=True)

    # Obser reward and next obs
    obs, reward, done, _ = env.step(action)
    env.render()
    time.sleep(0.025)

    masks.fill_(0.0 if done else 1.0)
