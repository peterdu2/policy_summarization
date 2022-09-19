import gym
import ast_toolbox
import logging
import argparse
import configparser
import os
import shutil
import torch
import numpy as np
from reward import CNReward
from space import CNSpaces
from ast_toolbox.simulators import ASTSimulator
from crowd_sim.envs.utils.robot import Robot
from crowd_nav.utils.trainer import Trainer
from crowd_nav.utils.memory import ReplayMemory
from crowd_nav.utils.explorer import Explorer
from crowd_nav.policy.policy_factory import policy_factory
from crowd_sim.envs.utils.info import *
from crowd_nav.policy.policy_factory import policy_factory

from ast_toolbox.rewards import ExampleAVReward
# action_space, observation_space, simulate, clone_state
class AST_simulator_wrapper(ASTSimulator):
    def __init__(self, config_file, max_path_length):
        env_config = configparser.RawConfigParser()
        env_config.read(config_file)
        self.space = CNSpaces()
        self.reward = CNReward()
        self.human_locations = []
        self.action_space = self.space.action_space
        self.observation_space = self.space.observation_space
        self.env = gym.make('CrowdSim-v0')
        self.env.configure(env_config)
        robot = Robot(env_config, 'robot')
        policy = policy_factory['sarl']()
        device = torch.device("cuda:0")
        #device = torch.device("cpu")
        policy_config = configparser.RawConfigParser()
        policy_config.read('policy.config')
        policy.configure(policy_config)
        policy.set_device(device)
        policy.set_env(self.env)
        model_weights = '/home/ubuntu/policy_benchmarking/crowd_nav/data/output_sarl/il_model.pth' # Add later
        if policy.trainable:
            policy.get_model().load_state_dict(torch.load(model_weights))
        robot.set_policy(policy)
        self.env.set_robot(robot)
        self.env.robot.policy.phase = 'test'
        self.env.reset()
        self.max_path_length = max_path_length
        self.current_step = 0
        self.robot_ob = [human.get_observable_state() for human in self.env.humans]
        self.robot_state = [self.env.robot.px, self.env.robot.py]
        self.current_reward = 0
        self.terminal = False
        self.goal_info = Nothing()

    def clone_state(self):
        return self.robot_ob

    def step(self, new_human_accelerations):
        new_accel = np.reshape(new_human_accelerations, (len(self.env.humans), 2))
        for index in range(0, len(self.env.humans)):
            self.env.humans[index].set(self.env.humans[index].px, self.env.humans[index].py, self.env.humans[index].gx, self.env.humans[index].gy, self.env.humans[index].vx + new_accel[index][0], self.env.humans[index].vy + new_accel[index][1], np.pi/2)

        self.robot_ob = [human.get_observable_state() for human in self.env.humans]
        action = self.env.robot.act(self.robot_ob)
        self.robot_ob, self.current_reward, self.terminal, self.goal_info = self.env.step(action)
        self.current_step += 1
        if self.current_step == self.max_path_length:
            self.terminal = True
        self.robot_state = [self.env.robot.px, self.env.robot.py]
        reward_args = {'info': self.get_reward_info()}
        reward_output = self.reward.give_reward(new_human_accelerations, **reward_args)
        out_bool = self.is_goal() or self.is_terminal()
        return self.robot_ob, reward_output, out_bool, self.goal_info

    def simulate(self, actions, s_0):
        total_reward = 0
        time_step = 0
        self.robot_ob = self.env.reset()
        self.human_locations = []
        location_list = []
        for index in range(0, len(s_0)):
            self.env.humans[index].px = s_0[index][0]
            self.env.humans[index].py = s_0[index][1]
            location_list.append(tuple([self.env.humans[index].px, self.env.humans[index].py]))
        self.human_locations.append(location_list)
        for action in actions:
            time_step = time_step + 1
            if self.is_goal() == True:
                break
            if self.is_terminal() == True:
                time_step = -1
                break
            self.robot_ob, current_reward, self.terminal, self.goal_info = self.step(action)
            location_list = []
            for human in self.env.humans:
                location_list.append(tuple([human.px, human.py]))
            self.human_locations.append(location_list)
            total_reward = total_reward + current_reward
        self.env.render(mode='video')
        return time_step

    def reset(self, s_0):
        self.robot_ob = self.env.reset()
        self.env.reset()
        for index in range(0, len(s_0)):
            self.env.humans[index].px = s_0[index][0]
            self.env.humans[index].py = s_0[index][1]
        self.robot_ob = [human.get_observable_state() for human in self.env.humans]
        self.current_step = 0
        self.robot_state = [self.env.robot.px, self.env.robot.py]
        self.current_reward = 0
        self.terminal = False
        self.goal_info = Nothing()
        return self.robot_ob

    def get_reward_info(self):
        return{
                "terminal": self.is_terminal(),
                "goal": self.is_goal(),
                "human_positions": self.robot_ob,
                "robot_position": self.robot_state}
    
    def is_goal(self):
        if self.terminal == True and str(self.goal_info) == "Collision":
            return True
        else:
            return False

    def is_terminal(self):
        if self.terminal == True or (str(self.goal_info) == "Reaching goal" or str(self.goal_info) == 'Timeout' or self.current_step == self.max_path_length):
            return True
        else:
            return False       	

