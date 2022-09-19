import gym
import configparser
import torch
import numpy as np
import copy
import time
from reward import CNReward
from space import CNSpaces
from ast_toolbox.simulators import ASTSimulator
from crowd_sim.envs.utils.robot import Robot
from crowd_nav.policy.policy_factory import policy_factory
from crowd_sim.envs.utils.action import ActionXY
from crowd_sim.envs.utils.info import *

from ast_toolbox.rewards import ExampleAVReward
# config_file_paths: list of paths to each simulators configuation file


class ASTCrowdNavCoupledSimulator(ASTSimulator):
    def __init__(self, config_file_paths, max_path_length, **kwargs):

        # Read in all configuration files
        config_files = [configparser.RawConfigParser()
                        for i in range(len(config_file_paths))]
        for i in range(len(config_file_paths)):
            config_files[i].read(config_file_paths[i])

        # Create list of simulators
        self.n_policies = len(config_file_paths)
        self.simulators = []

        for i in range(self.n_policies):
            algo = config_files[i].get('robot', 'policy')
            model_weights = config_files[i].get('robot', 'model_weights')

            # Create the Gym environment
            env = gym.make('CrowdSim-v0')
            env.configure(config_files[i])

            # Create the robot
            robot = Robot(config_files[i], 'robot')

            # Set the robot policy
            policy_config = configparser.RawConfigParser()
            policy_config.read('policy.config')
            policy = policy_factory[algo]()
            policy.configure(policy_config)
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            policy.set_device(device)
            policy.set_env(env)
            if policy.trainable:
                policy.get_model().load_state_dict(torch.load(model_weights))
            robot.set_policy(policy)

            # Set robot into simulation env
            env.set_robot(robot)
            env.robot.policy.phase = 'test'
            env.reset()

            # Add environment to simulators list
            self.simulators.append(env)

        # Simulator parameters
        # Number of humans per environment
        self.n_humans = len(self.simulators[0].humans)
        self.max_path_length = max_path_length          # Max path length of AST search

        # System observations and states
        self.robot_obs = []
        self.robot_states = []
        self.robot_actions = None
        for i in range(self.n_policies):
            self.robot_obs.append([human.get_observable_state()
                                  for human in self.simulators[i].humans])
            self.robot_states.append(
                self.simulators[i].robot.get_observable_state())


        # Other simulator properties
        self.space = CNSpaces()
        self.reward = CNReward()
        self.human_locations = []
        self.sim_info = [Nothing() for i in range(self.n_policies)]
 

        # Initialize the base simulator
        super().__init__(**kwargs)

    def clone_state(self):
        return np.array(self.robot_obs).flatten()

    def collision_check(self):
        collision = [0 for policy in range(self.n_policies)]

        for i in range(self.n_policies):
            # Get the robot's position
            pos = self.robot_states[i]
            robot_pos = np.array([pos.px, pos.py])

            for j in range(self.n_humans):
                # Get the human's position
                pos = self.observation[i*self.n_humans + j]
                human_pos = np.array([pos.px, pos.py])
                min_radius = self.simulators[i].robot.radius + self.simulators[i].humans[j].radius

                # Check for collision
                if np.linalg.norm(robot_pos-human_pos) < min_radius:
                    collision[i] = 1
                    break

        return collision
        
    def closed_loop_step(self, action):
        new_accel = np.reshape(action, (self.n_humans, 2))

        # Step the robot in all environments
        self.robot_actions = []
        for sim in range(self.n_policies):
            robot_ob = [human.get_observable_state()
                        for human in self.simulators[sim].humans]
            robot_action = self.simulators[sim].robot.act(robot_ob)
            obs, reward, done, info = self.simulators[sim].step(robot_action)
            self.simulators[sim].robot.step(robot_action)
            self.sim_info[sim] = info
            self.robot_actions.append(copy.copy(robot_action))

        # Step each human using the newly given accelerations
        for sim in range(self.n_policies):
            for i in range(self.n_humans):
                # Calculate new velocities using acceleration action
                vx = self.simulators[sim].humans[i].vx + new_accel[i][0]*self.simulators[sim].humans[i].time_step
                vy = self.simulators[sim].humans[i].vy + new_accel[i][1]*self.simulators[sim].humans[i].time_step
                human_action = ActionXY(vx, vy)
                self.simulators[sim].humans[i].step(human_action)

        # Step the AST simulation
        self._path_length += 1
        if self._path_length >= self.max_path_length:
            self._is_terminal = True

        # Collect observations
        self.robot_obs = []
        self.robot_states = []
        for i in range(self.n_policies):
            self.robot_obs.append([human.get_observable_state()
                                  for human in self.simulators[i].humans])
            self.robot_states.append(
                self.simulators[i].robot.get_observable_state())
        self.observation = np.array(self.robot_obs).flatten()

        # Check for collision 
        #collision = [1 if isinstance(x, Collision) else 0 for x in self.sim_info]
        # collision = self.collision_check()
        # if np.sum(collision) == 1:
        #     self.goal = True
        # else:
        #     self.goal = False
    
        return self.observation_return()

    def reset(self, s_0):
        super(ASTCrowdNavCoupledSimulator, self).reset(s_0)

        # Reset all environemnts
        for i in range(self.n_policies):
            self.simulators[i].reset()
            for idx in range(len(s_0)):
                self.simulators[i].humans[idx].px = s_0[idx][0]
                self.simulators[i].humans[idx].py = s_0[idx][1]

        # System observations and states
        self.robot_obs = []
        self.robot_states = []
        for i in range(self.n_policies):
            self.robot_obs.append([human.get_observable_state()
                                  for human in self.simulators[i].humans])
            self.robot_states.append(
                self.simulators[i].robot.get_observable_state())
        self.observation = np.array(self.robot_obs).flatten()

        self.goal = False

        return self.observation_return()

    def get_reward_info(self):
        human_positions = [[] for i in range(self.n_policies)]
        robot_positions = []
        for i in range(self.n_policies):
            pos = self.robot_states[i]
            robot_positions.append([pos.px, pos.py])
            for j in range(self.n_humans):
                pos = self.observation[i*self.n_humans + j]
                human_positions[i].append([pos.px, pos.py])

        # for i in range(self.n_policies):
        #     print(human_positions[i])
        #     print(robot_positions[i])
        # print(' ')

        # print(self.sim_info)

        return{
            'is_terminal': self.is_terminal(),
            'is_goal': self.is_goal(),
            'sim_info': self.sim_info,
            'robot_actions': self.robot_actions,
            'human_positions': human_positions,
            'robot_positions': robot_positions}

    def is_goal(self):
        # Check for collision
        collision = self.collision_check()
        if np.sum(collision) == 1:
            return True
        else:
            return False
        #return self.goal
        # collision = [1 if isinstance(x, Collision) else 0 for x in self.sim_info]
        # if np.sum(collision) == 1:
        #     return True
        # else:
        #     return False
