import gym
import configparser
import torch
import numpy as np
import copy
import sys
import os
import argparse
from importlib import import_module

from pytorchBaselines.a2c_ppo_acktr.envs import make_vec_envs

from matplotlib import pyplot as plt

from ast_toolbox.simulators import ASTSimulator
from crowd_sim import *

def make_env(config, ax):
    env = gym.make(config.env.env_name)
    env.configure(config)
    env.thisSeed = config.env.seed
    env.phase = 'test'
    env.render_axis = ax
    env.nenv = 1

    return env

# env_name = 'CrowdSimDict-v0'
# env = gym.make(env_name)
# config = 'data/example_model/configs/config.py'

# model_dir_string = 'data.example_model.configs.config'
# model_arguments = import_module(model_dir_string)
# Config = getattr(model_arguments, 'Config')
# config = Config()


# env.configure(config)
# env.render()


model_dir_string = 'data.example_model.configs.config'
model_arguments = import_module(model_dir_string)
Config = getattr(model_arguments, 'Config')
config = Config()

fig, ax = plt.subplots(figsize=(7, 7))
ax.set_xlim(-6, 6)
ax.set_ylim(-6, 6)
ax.set_xlabel('x(m)', fontsize=16)
ax.set_ylabel('y(m)', fontsize=16)
plt.ion()
plt.show()

test_env = make_env(config, ax)
print(test_env)
print(test_env.robot.vy)
test_env.reset()
print(test_env.robot.vy)

for i in range(100):
    test_env.render()

# eval_dir = 'data'
# torch.set_num_threads(1)
# device = torch.device("cuda" if config.training.cuda else "cpu")

# envs = make_vec_envs(env_name, config.env.seed, 1,
#                      config.reward.gamma, eval_dir, device, allow_early_resets=True,
#                      config=config, ax=ax, test_case=-1)

# print(envs)