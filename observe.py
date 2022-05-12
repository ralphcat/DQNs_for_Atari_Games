import os, random
import gym
import numpy as np
import torch
from torch import nn
import itertools
from baselines_wrappers import DummyVecEnv
from pytorch_wrappers import make_atari_deepmind, BatchedPytorchFrameStack, PytorchLazyFrames
import time
from Dueling_DQN import Network


name_games = ['Breakout-v4', 'SpaceInvaders-v4']
algos = ['DQN', 'Double_DQN', 'Dueling_DQN']

name_game = name_games[0]
algo = algos[2]

SAVE_PATH = f'./models/{name_game[:-3]}_{algo}.pack'
print(SAVE_PATH)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device:', device)


make_env = lambda: make_atari_deepmind(name_game)
vec_env = DummyVecEnv([make_env for _ in range(1)])
env = BatchedPytorchFrameStack(vec_env, k=4)

net = Network(env, device)
net = net.to(device)
net.load(SAVE_PATH)

obs = env.reset()
beginning_episode = True
for t in itertools.count():
    if isinstance(obs[0], PytorchLazyFrames):
        act_obs = np.stack([o.get_frames() for o in obs])
        action = net.act(act_obs, 0.0)
    else:
        action = net.act(obs, 0.0)

    if beginning_episode:
        action = [1]
        beginning_episode = False

    obs, rew, done, _ = env.step(action)
    #env.render()
    time.sleep(0.01)
    
    if done[0]:
        obs = env.reset()
        beginning_episode = True
        #env.close()
        #break
