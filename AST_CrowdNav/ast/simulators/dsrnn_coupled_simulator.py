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

    def __init__(self, model_dirs, config_names, model_names, s_0,
                 mode, goal_mode, single_render_mode=False, **kwargs):
        assert len(model_dirs) == 2, 'Provide two models for simulator'
        assert len(config_names) == 2, 'Provide two models for simulator'

        # AST Parameters
        self.s_0 = s_0
        self.goal = False
        self.mode = mode
        self.goal_mode = goal_mode

        # Configs
        self.config_filepaths = []
        self.configs = []
        self.device_flags = []
        self.config_names = config_names

        # Rendering
        self.render_frame = 0
        self.coupled_axes = None
        self.single_axes = []

        # Enviornments
        self.envs = []
        self.observations = []
        self.sim_infos = []

        # Models
        self.model_filepaths = []
        self.models = []
        self.eval_recurrent_hidden_states = []
        self.eval_masks = []
        self.model_names = model_names

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
            env = self.make_env(self.configs[i], None)
            self.envs.append(env)

        if single_render_mode:
            # Create single render axis
            self.single_axis = self.create_single_render_axis()
        else:
            # Create coupled render axes
            self.coupled_axes = self.create_coupled_render_axis()

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

    
    def create_single_render_axis(self):
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        #ax.set_xlabel('x(m)', fontsize=16)
        #ax.set_ylabel('y(m)', fontsize=16)
        return ax


    def create_coupled_render_axis(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
        ax1.set_xlim(-10, 10)
        ax1.set_ylim(-10, 10)
        #ax1.set_xlabel('x(m)', fontsize=16)
        #ax1.set_ylabel('y(m)', fontsize=16)
        ax2.set_xlim(-10, 10)
        ax2.set_ylim(-10, 10)
        #ax2.set_xlabel('x(m)', fontsize=16)
        #ax2.set_ylabel('y(m)', fontsize=16)
        return ax1, ax2


    def make_env(self, config, ax):
        env = gym.make(config.env.env_name)
        env.configure(config)
        env.thisSeed = config.env.seed
        env.phase = 'test'
        env.render_axis = ax
        env.nenv = 1
        return env


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
                          gx=-s_0[j+1][0] ,gy=-s_0[j+1][1],
                          vx=0., vy=0., theta=0.)
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

        # Reset rendering frame (used for saving frames)
        self.render_frame = 0

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
        assert self.goal_mode == 'COLLISION' or self.goal_mode == 'REACHGOAL'
        if self.goal_mode == 'COLLISION':
            # Stop search when one policy results in collision
            for state in self.sim_infos:
                if isinstance(state['info'], Collision):
                    return True
        else:
            # Stop search when one policy reaches goal
            for state in self.sim_infos:
                if isinstance(state['info'], ReachGoal):
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


    def clone_state(self):
        # Observation format (dictionary of lists): 
        # {robot_node, temporal_edges_spatial_edges}
        # robot_node = [px, py, radius, gx. gy, vpref, theta] 
        # spatial_edges = [[v0], [v1], ... [v_num_humans]] where
        #                 each v is vector pointing from robot pos to human pos
        cloned_state = []
        for i in range(len(self.models)):
            cloned_state.extend(self.observation[i]['robot_node'])
            cloned_state.extend(self.observation[i]['temporal_edges'])
            cloned_state.extend(self.observation[i]['spatial_edges'].flatten())

        return np.array(cloned_state)

    def render_single(self, save_render, render_path, env_id, title=None, pause=0.05):
        import matplotlib.pyplot as plt
        import matplotlib.lines as mlines
        from matplotlib import patches

        plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'

        robot_color = 'yellow'
        goal_color = 'red'
        arrow_color = 'red'
        arrow_style = patches.ArrowStyle("->", head_length=4, head_width=2)

        def calcFOVLineEndPoint(ang, point, extendFactor):
            # choose the extendFactor big enough
            # so that the endPoints of the FOVLine is out of xlim and ylim of the figure
            FOVLineRot = np.array([[np.cos(ang), -np.sin(ang), 0],
                                   [np.sin(ang), np.cos(ang), 0],
                                   [0, 0, 1]])
            point.extend([1])
            # apply rotation matrix
            newPoint = np.matmul(FOVLineRot, np.reshape(point, [3, 1]))
            # increase the distance between the line start point and the end point
            newPoint = [extendFactor * newPoint[0, 0], extendFactor * newPoint[1, 0], 1]
            return newPoint


        artists=[]
        ax=self.single_axis
        if title == None:
            ax.set_title(self.model_names[env_id])
        else:
            ax.set_title(title)

        # add goal
        goal=mlines.Line2D([self.envs[env_id].robot.gx], [self.envs[env_id].robot.gy], color=goal_color, marker='*', linestyle='None', markersize=15, label='Goal')
        ax.add_artist(goal)
        artists.append(goal)

        # add robot
        robotX,robotY=self.envs[env_id].robot.get_position()

        robot=plt.Circle((robotX,robotY), self.envs[env_id].robot.radius, fill=True, color=robot_color)
        ax.add_artist(robot)
        artists.append(robot)

        plt.legend([robot, goal], ['Robot', 'Goal'], fontsize=16)

        # compute orientation in each step and add arrow to show the direction
        radius = self.envs[env_id].robot.radius
        arrowStartEnd=[]

        robot_theta = self.envs[env_id].robot.theta if self.envs[env_id].robot.kinematics == 'unicycle' else np.arctan2(self.envs[env_id].robot.vy, self.envs[env_id].robot.vx)

        arrowStartEnd.append(((robotX, robotY), (robotX + radius * np.cos(robot_theta), robotY + radius * np.sin(robot_theta))))

        for i, human in enumerate(self.envs[env_id].humans):
            theta = np.arctan2(human.vy, human.vx)
            arrowStartEnd.append(((human.px, human.py), (human.px + radius * np.cos(theta), human.py + radius * np.sin(theta))))

        arrows = [patches.FancyArrowPatch(*arrow, color=arrow_color, arrowstyle=arrow_style)
                for arrow in arrowStartEnd]
        for arrow in arrows:
            ax.add_artist(arrow)
            artists.append(arrow)


        # draw FOV for the robot
        # add robot FOV
        if self.envs[env_id].robot_fov < np.pi * 2:
            FOVAng = self.envs[0].robot_fov / 2
            FOVLine1 = mlines.Line2D([0, 0], [0, 0], linestyle='--')
            FOVLine2 = mlines.Line2D([0, 0], [0, 0], linestyle='--')


            startPointX = robotX
            startPointY = robotY
            endPointX = robotX + radius * np.cos(robot_theta)
            endPointY = robotY + radius * np.sin(robot_theta)

            # transform the vector back to world frame origin, apply rotation matrix, and get end point of FOVLine
            # the start point of the FOVLine is the center of the robot
            FOVEndPoint1 = calcFOVLineEndPoint(FOVAng, [endPointX - startPointX, endPointY - startPointY], 20. / self.envs[env_id].robot.radius)
            FOVLine1.set_xdata(np.array([startPointX, startPointX + FOVEndPoint1[0]]))
            FOVLine1.set_ydata(np.array([startPointY, startPointY + FOVEndPoint1[1]]))
            FOVEndPoint2 = calcFOVLineEndPoint(-FOVAng, [endPointX - startPointX, endPointY - startPointY], 20. / self.envs[env_id].robot.radius)
            FOVLine2.set_xdata(np.array([startPointX, startPointX + FOVEndPoint2[0]]))
            FOVLine2.set_ydata(np.array([startPointY, startPointY + FOVEndPoint2[1]]))

            ax.add_artist(FOVLine1)
            ax.add_artist(FOVLine2)
            artists.append(FOVLine1)
            artists.append(FOVLine2)

        # add humans and change the color of them based on visibility
        human_circles = [plt.Circle(human.get_position(), human.radius, fill=False) for human in self.envs[env_id].humans]


        for i in range(len(self.envs[env_id].humans)):
            ax.add_artist(human_circles[i])
            artists.append(human_circles[i])

            # green: visible; red: invisible
            if self.envs[env_id].detect_visible(self.envs[env_id].robot, self.envs[env_id].humans[i], robot1=True):
                human_circles[i].set_color(c='g')
            else:
                human_circles[i].set_color(c='r')
            #plt.text(self.humans[i].px - 0.1, self.humans[i].py - 0.1, str(i), color='black', fontsize=12)
            ax.text(self.envs[env_id].humans[i].px - 0.1, self.envs[env_id].humans[i].py - 0.1, str(i), color='black', fontsize=12)

        if save_render:
            plt.savefig(render_path+'/'+format(self.render_frame, '04d')+'.png')

        if pause > 0:
            plt.pause(pause)

        for item in artists:
            item.remove() # there should be a better way to do this. For example,
            # initially use add_artist and draw_artist later on
        for t in ax.texts:
            t.set_visible(False)

        self.render_frame += 1

    def render_coupled(self, save_render, render_path, titles, pause=0.05):
        import matplotlib.pyplot as plt
        import matplotlib.lines as mlines
        from matplotlib import patches

        plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'

        robot_color = 'yellow'
        goal_color = 'red'
        arrow_color = 'red'
        arrow_style = patches.ArrowStyle("->", head_length=4, head_width=2)

        def calcFOVLineEndPoint(ang, point, extendFactor):
            # choose the extendFactor big enough
            # so that the endPoints of the FOVLine is out of xlim and ylim of the figure
            FOVLineRot = np.array([[np.cos(ang), -np.sin(ang), 0],
                                   [np.sin(ang), np.cos(ang), 0],
                                   [0, 0, 1]])
            point.extend([1])
            # apply rotation matrix
            newPoint = np.matmul(FOVLineRot, np.reshape(point, [3, 1]))
            # increase the distance between the line start point and the end point
            newPoint = [extendFactor * newPoint[0, 0], extendFactor * newPoint[1, 0], 1]
            return newPoint


        artists=[]
        for cur_axis in range(2):
            ax=self.coupled_axes[cur_axis]
            ax.set_title(titles[cur_axis], fontsize=20, color='blue', fontweight='bold')

            # add goal
            goal=mlines.Line2D([self.envs[cur_axis].robot.gx], [self.envs[cur_axis].robot.gy], color=goal_color, marker='*', linestyle='None', markersize=15, label='Goal')
            ax.add_artist(goal)
            artists.append(goal)

            # add robot
            robotX,robotY=self.envs[cur_axis].robot.get_position()

            robot=plt.Circle((robotX,robotY), self.envs[cur_axis].robot.radius, fill=True, color=robot_color)
            ax.add_artist(robot)
            artists.append(robot)

            plt.legend([robot, goal], ['Robot', 'Goal'], fontsize=16)

            # compute orientation in each step and add arrow to show the direction
            radius = self.envs[cur_axis].robot.radius
            arrowStartEnd=[]

            robot_theta = self.envs[cur_axis].robot.theta if self.envs[cur_axis].robot.kinematics == 'unicycle' else np.arctan2(self.envs[cur_axis].robot.vy, self.envs[cur_axis].robot.vx)

            arrowStartEnd.append(((robotX, robotY), (robotX + radius * np.cos(robot_theta), robotY + radius * np.sin(robot_theta))))

            for i, human in enumerate(self.envs[cur_axis].humans):
                theta = np.arctan2(human.vy, human.vx)
                arrowStartEnd.append(((human.px, human.py), (human.px + radius * np.cos(theta), human.py + radius * np.sin(theta))))

            arrows = [patches.FancyArrowPatch(*arrow, color=arrow_color, arrowstyle=arrow_style)
                    for arrow in arrowStartEnd]
            for arrow in arrows:
                ax.add_artist(arrow)
                artists.append(arrow)


            # draw FOV for the robot
            # add robot FOV
            if self.envs[cur_axis].robot_fov < np.pi * 2:
                FOVAng = self.envs[0].robot_fov / 2
                FOVLine1 = mlines.Line2D([0, 0], [0, 0], linestyle='--')
                FOVLine2 = mlines.Line2D([0, 0], [0, 0], linestyle='--')


                startPointX = robotX
                startPointY = robotY
                endPointX = robotX + radius * np.cos(robot_theta)
                endPointY = robotY + radius * np.sin(robot_theta)

                # transform the vector back to world frame origin, apply rotation matrix, and get end point of FOVLine
                # the start point of the FOVLine is the center of the robot
                FOVEndPoint1 = calcFOVLineEndPoint(FOVAng, [endPointX - startPointX, endPointY - startPointY], 20. / self.envs[cur_axis].robot.radius)
                FOVLine1.set_xdata(np.array([startPointX, startPointX + FOVEndPoint1[0]]))
                FOVLine1.set_ydata(np.array([startPointY, startPointY + FOVEndPoint1[1]]))
                FOVEndPoint2 = calcFOVLineEndPoint(-FOVAng, [endPointX - startPointX, endPointY - startPointY], 20. / self.envs[cur_axis].robot.radius)
                FOVLine2.set_xdata(np.array([startPointX, startPointX + FOVEndPoint2[0]]))
                FOVLine2.set_ydata(np.array([startPointY, startPointY + FOVEndPoint2[1]]))

                ax.add_artist(FOVLine1)
                ax.add_artist(FOVLine2)
                artists.append(FOVLine1)
                artists.append(FOVLine2)

            # add humans and change the color of them based on visibility
            human_circles = [plt.Circle(human.get_position(), human.radius, fill=False) for human in self.envs[cur_axis].humans]


            for i in range(len(self.envs[cur_axis].humans)):
                ax.add_artist(human_circles[i])
                artists.append(human_circles[i])

                # green: visible; red: invisible
                if self.envs[cur_axis].detect_visible(self.envs[cur_axis].robot, self.envs[cur_axis].humans[i], robot1=True):
                    human_circles[i].set_color(c='g')
                else:
                    human_circles[i].set_color(c='r')
                ax.text(self.envs[cur_axis].humans[i].px - 0.1, self.envs[cur_axis].humans[i].py - 0.1, str(i), color='black', fontsize=12)

            # # Label state if collision or timeout
            # state = self.sim_infos[cur_axis]
            # if isinstance(state['info'], Collision):
            #     ax.text(-9,7, 'COLLISION', color='red', fontsize=20)

            ax.get_yaxis().set_visible(False)
            ax.get_xaxis().set_visible(False)

        if save_render:
            plt.savefig(render_path+'/'+format(self.render_frame, '04d')+'.png')

        if pause > 0:
            plt.pause(pause)

        for item in artists:
            item.remove() # there should be a better way to do this. For example,
            # initially use add_artist and draw_artist later on
        for ax in self.coupled_axes:
            for t in ax.texts:
                t.set_visible(False)

        self.render_frame += 1