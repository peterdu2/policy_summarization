import argparse
import configparser
import copy
import os
import sys

import crowd_sim
import gym
import numpy as np
import torch

from ast_toolbox.simulators import ASTSimulator
from importlib import import_module
from matplotlib import pyplot as plt
from pytorchBaselines.a2c_ppo_acktr.envs import make_vec_envs
from pytorchBaselines.a2c_ppo_acktr.model import Policy


class DSRNNCoupledSimulator(ASTSimulator):

    def __init__(self, model_dirs, config_names, model_names):
        assert len(model_dirs) == 2, 'Provide two models for simulator'
        assert len(config_names) == 2, 'Provide two models for simulator'

        # Load each config object
        self.config_filepaths = []
        self.configs = []
        for i in range(len(model_dirs)):
            path = model_dirs[i].replace('/', '.') + 'configs.' + config_names[i]
            self.config_filepaths.append(path)
            self.configs.append(self.import_config(self.config_filepaths[i]))

        # Create gym environments (device)
        self.envs = []
        for i in range(len(self.configs)):
            # Only create plot for first environment
            if i == 0:
                ax = self.create_render_axis()
            else:
                ax = None
            # Create env on device
            device = torch.device("cuda" if self.configs[i].training.cuda else "cpu")
            env = make_vec_envs(env_name=self.configs[i].env.env_name,
                                seed=self.configs[i].env.seed,
                                num_processes=1,
						        gamma=self.configs[i].reward.gamma,
                                log_dir=None, 
                                device=device,
                                allow_early_resets=True,
						        config=self.configs[0],
                                ax=ax)
            self.envs.append(env)

        # Load each DSRNN model
        self.model_filepaths = []
        self.models = []
        for i in range(len(model_dirs)):
            load_path = os.path.join(model_dirs[i],'checkpoints', model_names[i])
            print('[DSRNNCoupledSimulator] Loading model:', load_path)

            actor_critic = Policy(
            self.envs[i].observation_space.spaces,  # pass the Dict into policy to parse
            self.envs[i].action_space,
            base_kwargs=self.configs[i],
            base=self.configs[i].robot.policy)

            device = torch.device("cuda" if self.configs[i].training.cuda else "cpu")
            actor_critic.load_state_dict(torch.load(load_path, map_location=device))
            actor_critic.base.nenv = 1

            # allow the usage of multiple GPUs to increase the number of examples processed simultaneously
            torch.nn.DataParallel(actor_critic).to(device)

            self.models.append(actor_critic)

        # Reset simulation and initialize hidden states
        self.reset()

    
    def import_config(self, config_filepath):
        config_module = import_module(config_filepath)
        config_class = getattr(config_module, 'Config')
        config = config_class()
        return config


    def create_render_axis(self):
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        ax.set_xlabel('x(m)', fontsize=16)
        ax.set_ylabel('y(m)', fontsize=16)
        return ax


    def make_env(self, config, ax):
        env = gym.make(config.env.env_name)
        env.configure(config)
        env.thisSeed = config.env.seed
        env.phase = 'test'
        env.render_axis = ax
        env.nenv = 1
        return env


    def render(self):
        self.envs[0].render()


    def init_hidden_states(self):
        self.eval_recurrent_hidden_states = []
        self.eval_masks = []
        for i in range(len(self.models)):
            device = torch.device("cuda" if self.configs[i].training.cuda else "cpu")
            num_processes = 1
            rnn_factor = 1
            node_num = 1
            edge_num = self.models[i].base.human_num + 1

            eval_recurrent_hidden_states = {}
            eval_recurrent_hidden_states['human_node_rnn'] = torch.zeros(num_processes,
                                                                    node_num,
                                                                    self.configs[i].SRNN.human_node_rnn_size * rnn_factor,
                                                                    device=device)
            eval_recurrent_hidden_states['human_human_edge_rnn'] = torch.zeros(num_processes,
                                                                          edge_num,
                                                                          self.configs[i].SRNN.human_human_edge_rnn_size * rnn_factor,
                                                                          device=device)
            eval_masks = torch.zeros(num_processes, 1, device=device)

            self.eval_recurrent_hidden_states.append(eval_recurrent_hidden_states)
            self.eval_masks.append(eval_masks)


    def reset(self):
        self.observations = []

        # Reset simulation environments and observations
        for env in self.envs:
            obs = env.reset()
            self.observations.append(obs)

        # Reset hidden states
        self.init_hidden_states()


    def step(self):
        for i in range(len(self.models)):
            # Compute action for each robot policy
            with torch.no_grad():
                _, action, _, eval_recurrent_hidden_states = self.models[i].act(
                    self.obs_test,
                    self.eval_recurrent_hidden_states[i],
                    self.eval_masks[i],
                    deterministic=True)
            
            # Step simulation forward
            #obs, rew, done, infos = self.envs[i].step(action)
            obs, rew, done, infos = self.envs_test.step(action)

            # Update observation
            self.observations[i] = copy.copy(obs)

            # Update hidden state
            self.eval_recurrent_hidden_states[i] = copy.copy(eval_recurrent_hidden_states)



# env_name = 'CrowdSimDict-v0'
# env = gym.make(env_name)
# config = 'data/example_model/configs/config.py'

# model_dir_string = 'data.example_model.configs.config'
# model_arguments = import_module(model_dir_string)
# Config = getattr(model_arguments, 'Config')
# config = Config()


# env.configure(config)
# env.render()


# model_dir_string = 'data.example_model.configs.config'
# model_arguments = import_module(model_dir_string)
# Config = getattr(model_arguments, 'Config')
# config = Config()

# fig, ax = plt.subplots(figsize=(7, 7))
# ax.set_xlim(-6, 6)
# ax.set_ylim(-6, 6)
# ax.set_xlabel('x(m)', fontsize=16)
# ax.set_ylabel('y(m)', fontsize=16)
# plt.ion()
# plt.show()

# test_env = make_env(config, ax)
# print(test_env)
# print(test_env.robot.vy)
# test_env.reset()
# print(test_env.robot.vy)

# for i in range(100):
#     test_env.render()

# eval_dir = 'data'
# torch.set_num_threads(1)
# device = torch.device("cuda" if config.training.cuda else "cpu")

# envs = make_vec_envs(env_name, config.env.seed, 1,
#                      config.reward.gamma, eval_dir, device, allow_early_resets=True,
#                      config=config, ax=ax, test_case=-1)

# print(envs)