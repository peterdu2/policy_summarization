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
from crowd_sim.envs.utils.info import *
from importlib import import_module
from matplotlib import pyplot as plt
from pytorchBaselines.a2c_ppo_acktr.envs import make_vec_envs
from pytorchBaselines.a2c_ppo_acktr.model import Policy


class DSRNNCoupledSimulator(ASTSimulator):

    def __init__(self, model_dirs, config_names, model_names, s_0, mode, **kwargs):
        assert len(model_dirs) == 2, 'Provide two models for simulator'
        assert len(config_names) == 2, 'Provide two models for simulator'

        # AST Parameters
        self.s_0 = s_0
        self.goal = False
        self.mode = mode

        # Configs
        self.config_filepaths = []
        self.configs = []
        self.device_flags = []

        # Enviornments
        self.envs = []
        self.observations = []
        self.sim_infos = []

        # Models
        self.model_filepaths = []
        self.models = []
        self.eval_recurrent_hidden_states = []
        self.eval_masks = []

        # Reward info
        self.robot_actions = []

        # Load each config object
        for i in range(len(model_dirs)):
            path = model_dirs[i].replace('/', '.') + 'configs.' + config_names[i]
            self.config_filepaths.append(path)
            self.configs.append(self.import_config(self.config_filepaths[i]))

            # Set device flag for current config
            device = torch.device("cuda" if self.configs[i].training.cuda else "cpu")
            self.device_flags.append(device)

        # Create gym environments
        for i in range(len(self.configs)):
            # Only create plot for first environment
            if i == 0:
                ax = self.create_render_axis()
            else:
                ax = None
            env = self.make_env(self.configs[i], ax)
            self.envs.append(env)

        # Load each DSRNN model
        for i in range(len(model_dirs)):
            load_path = os.path.join(model_dirs[i],'checkpoints', model_names[i])
            print('[DSRNNCoupledSimulator] Loading model:', load_path)

            actor_critic = Policy(
            self.envs[i].observation_space.spaces,  # pass the Dict into policy to parse
            self.envs[i].action_space,
            base_kwargs=self.configs[i],
            base=self.configs[i].robot.policy)

            actor_critic.load_state_dict(torch.load(load_path, map_location=self.device_flags[i]))
            actor_critic.base.nenv = 1

            # allow the usage of multiple GPUs to increase the number of examples processed simultaneously
            torch.nn.DataParallel(actor_critic).to(self.device_flags[i])

            self.models.append(actor_critic)

        # Initialize the base simulator
        super().__init__(**kwargs)

        # Reset simulation and initialize hidden states
        self.reset(self.s_0)

    
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
            num_processes = 1
            rnn_factor = 1
            node_num = 1
            edge_num = self.models[i].base.human_num + 1

            eval_recurrent_hidden_states = {}
            eval_recurrent_hidden_states['human_node_rnn'] = torch.zeros(num_processes,
                                                                    node_num,
                                                                    self.configs[i].SRNN.human_node_rnn_size * rnn_factor,
                                                                    device=self.device_flags[i])
            eval_recurrent_hidden_states['human_human_edge_rnn'] = torch.zeros(num_processes,
                                                                          edge_num,
                                                                          self.configs[i].SRNN.human_human_edge_rnn_size * rnn_factor,
                                                                          device=self.device_flags[i])
            eval_masks = torch.zeros(num_processes, 1, device=self.device_flags[i])

            self.eval_recurrent_hidden_states.append(eval_recurrent_hidden_states)
            self.eval_masks.append(eval_masks)


    def generate_obs_tensor(self, obs):
        obs['robot_node'] = torch.tensor([[obs['robot_node']]],
                                         dtype=torch.float32,
                                         device=self.device_flags[0])
        obs['temporal_edges'] = torch.tensor([[obs['temporal_edges']]],
                                        dtype=torch.float32,
                                        device=self.device_flags[0])   
        obs['spatial_edges'] = torch.tensor([obs['spatial_edges']],
                                        dtype=torch.float32,
                                        device=self.device_flags[0])
        return obs


    def reset(self, s_0):
        # s_0 format: [[robot_init_state], [human_0_init_state], [human_1_init_state], ...]
        #             [robot_init_state] = [pos_x, pos_y, goal_x, goal_y]
        #             [human_x_state] = [pos_x, pos_y]

        assert len(s_0) == self.configs[0].sim.human_num + 1, 'Length of s_0 does not match number of agents'
        super(DSRNNCoupledSimulator, self).reset(s_0)

        self.observations = []
        self.robot_actions = []
        self.sim_infos = [{'info': Nothing()} for i in range(len(self.envs))]

        # Reset simulation environments and observations
        for i in range(len(self.envs)):
            # Reset general simulator params
            self.envs[i].desiredVelocity = [0.0, 0.0]
            self.envs[i].humans = []
            self.envs[i].global_time = 0

            # Reset robot
            robot = self.envs[i].robot
            robot.set(px=s_0[0][0], py=s_0[0][1], gx=s_0[0][2], gy=s_0[0][3],
                      vx=0., vy=0., theta=np.pi/2)

            # Reset humans
            for j in range(self.configs[i].sim.human_num):
                human = Human(self.configs[i], 'humans')
                human.set(px=s_0[j+1][0], py=s_0[j+1][1],
                          gx=0. ,gy=0., vx=0., vy=0., theta=0.)
                self.envs[i].humans.append(copy.copy(human))
            
            # Generate observation
            obs = self.envs[i].generate_ob(reset=True)
            self.observations.append(obs)
            # Reset potential
            self.envs[i].potential = -abs(np.linalg.norm(np.array([robot.px, robot.py]) - np.array([robot.gx, robot.gy])))
            # Reset robot action log
            self.robot_actions.append([0., 0.,])

        # Reset hidden states
        self.init_hidden_states()

        # Reset done variable for each env
        self.dones = [False for i in range(len(self.models))]

        # Reset AST Class items
        self.observation = copy.copy(self.observations)
        self.goal = False

        return self.observation_return()


    def closed_loop_step(self, action):
        # Reshape env_action to following format:
        # env_action = [[human_0_action], [human_0_action], ...]
        #              where [human_x_action] = [x_action, y_action]
        action = action.reshape(len(self.envs[0].humans), 2)

        for i in range(len(self.models)):
            if self.mode == 'OBSERVATION_NOISE':
                # Add noise to each sptial edge in observation
                self.observations[i]['spatial_edges'] = self.observations[i]['spatial_edges'] + action

            # Compute action for each robot policy
            with torch.no_grad():
                _, robot_action, _, eval_recurrent_hidden_states = self.models[i].act(
                    self.generate_obs_tensor(self.observations[i]),
                    self.eval_recurrent_hidden_states[i],
                    self.eval_masks[i],
                    deterministic=True)
            obs, rew, done, infos = self.envs[i].ast_step(robot_action.cpu().numpy()[0], self.mode, action)

            # Update masks
            if done:
                self.eval_masks[i] = torch.tensor(
                    [[0.0]],
                    dtype=torch.float32,
                    device=self.device_flags[i])
            else:
                self.eval_masks[i] = torch.tensor(
                    [[1.0]],
                    dtype=torch.float32,
                    device=self.device_flags[i])
            
            # Update observation
            self.observations[i] = copy.copy(obs)
            # Update hidden state
            self.eval_recurrent_hidden_states[i] = copy.copy(eval_recurrent_hidden_states)
            # Update done indicator
            self.dones[i] = copy.copy(done)
            # Record robot action
            self.robot_actions[i] = copy.copy(robot_action.cpu().numpy()[0])
            # Record crowd nav simulator status
            self.sim_infos[i] = copy.copy(infos)

        self.observation = copy.copy(self.observations)
        return self.observation_return() 


    def is_goal(self):
        for state in self.sim_infos:
            if isinstance(state['info'], Collision):
                return True
        return False
        

    def get_reward_info(self):
        robot_positions = []
        for i in range(len(self.models)):
            robot_positions.append([self.envs[i].robot.px, self.envs[i].robot.py])

        return{
            'is_terminal': self.is_terminal(),
            'is_goal': self.is_goal(),
            'sim_infos': self.sim_infos,
            'robot_actions': self.robot_actions,
            'robot_positions': robot_positions}