import argparse
import configparser
import copy
import os
import sys

import crowd_sim
import gym
import numpy as np
import random
import torch

from ast_toolbox.simulators import ASTSimulator
from crowd_sim.envs.utils.human import Human
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

        # Create gym environments
        self.envs = []
        for i in range(len(self.configs)):
            # Only create plot for first environment
            if i == 0:
                ax = self.create_render_axis()
            else:
                ax = None
            env = self.make_env(self.configs[i], ax)
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


    def generate_obs_tensor(self, obs):
        obs['robot_node'] = torch.tensor([[obs['robot_node']]],
                                         dtype=torch.float32,
                                         device='cuda')
        obs['temporal_edges'] = torch.tensor([[obs['temporal_edges']]],
                                        dtype=torch.float32,
                                        device='cuda')   
        obs['spatial_edges'] = torch.tensor([obs['spatial_edges']],
                                        dtype=torch.float32,
                                        device='cuda')
        return obs


    def reset(self):
        self.observations = []

        # Reset simulation environments and observations
        for i in range(len(self.envs)):
            # Reset general simulator params
            self.envs[i].desiredVelocity = [0.0, 0.0]
            self.envs[i].humans = []
            self.envs[i].global_time = 0

            # Reset robot
            robot = self.envs[i].robot
            robot_config = self.configs[i].robot
            robot.set(robot_config.init_pos[0], robot_config.init_pos[1],
                      robot_config.goal[0], robot_config.goal[1],
                      0., 0., np.pi/2)

            # Reset humans
            human_config = self.configs[i].humans
            for j in range(self.configs[i].sim.human_num):
                human = Human(self.configs[i], 'humans')
                human.set(human_config.init_pos[j][0],
                          human_config.init_pos[j][1],
                          0., 0., 0., 0., 0.)
                self.envs[i].humans.append(copy.copy(human))
            
            # Generate observation
            obs = self.envs[i].generate_ob(reset=True)
            self.observations.append(self.generate_obs_tensor(obs))

            # Reset potential
            self.envs[i].potential = -abs(np.linalg.norm(np.array([robot.px, robot.py]) - np.array([robot.gx, robot.gy])))

        # Reset hidden states
        self.init_hidden_states()

        # Reset done variable for each env
        self.dones = [False for i in range(len(self.models))]


    def step(self):

        new_accels = [[random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5)] for i in range(10)]

        for i in range(len(self.models)):
            device = torch.device("cuda" if self.configs[i].training.cuda else "cpu")

            # Compute action for each robot policy
            with torch.no_grad():
                _, action, _, eval_recurrent_hidden_states = self.models[i].act(
                    self.observations[i],
                    self.eval_recurrent_hidden_states[i],
                    self.eval_masks[i],
                    deterministic=True)
            
            #obs, rew, done, infos = self.envs[i].step(action.cpu().numpy()[0])
            obs, rew, done, infos = self.envs[i].ast_step(action.cpu().numpy()[0], 'DIRECT_ACTION', new_accels)

            # Update masks
            if done:
                self.eval_masks[i] = torch.tensor(
                    [[0.0]],
                    dtype=torch.float32,
                    device=device)
            else:
                self.eval_masks[i] = torch.tensor(
                    [[1.0]],
                    dtype=torch.float32,
                    device=device)
            
            # Update observation
            self.observations[i] = copy.copy(self.generate_obs_tensor(obs))

            # Update hidden state
            self.eval_recurrent_hidden_states[i] = copy.copy(eval_recurrent_hidden_states)

            # Update done indicator
            self.dones[i] = copy.copy(done)

            print('Simulator', i, ' ', self.envs[i].robot.get_observable_state())
        print(' ')