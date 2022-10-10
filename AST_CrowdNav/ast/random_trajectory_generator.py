import numpy as np
import os
import pickle
import time

from crowd_sim.envs.utils.info import *

from simulators.dsrnn_coupled_simulator import DSRNNCoupledSimulator
from spaces.dsrnn_spaces import DSRNNSpaces

model_dirs = ['dsrnn_models/policy_summarization_10_humans/', 'dsrnn_models/policy_summarization_10_humans/']
config_name = ['config', 'config']
model_names = ['14000.pt', '34400.pt']

s_0 = []
s_0.append([-5., -4., 7., 2.])
s_0.append([0.9764521104695284, 5.673416349134316])
s_0.append([-0.3916318531059858, -6.311028515571641])
s_0.append([6.16749983681399, -1.8436072010801905])
s_0.append([2.382823965553917, 0.9261922797511946])
s_0.append([-1.7508964915665364, -4.244077560533835])
s_0.append([1.8364308394984743, -4.473900760196651])
s_0.append([5.561697860813725, 0.13723735189354574])
s_0.append([4.213509392947646, 4.356352793759812])
s_0.append([4.22327303238429, 2.3592114112982367])
s_0.append([-2.5587340425394998, 1.7638743741024])

mode = 'OBSERVATION_NOISE'
mode = 'DIRECT_ACTION'
goal_mode = 'REACHGOAL'

num_samples = 10

policy_titles = ['Policy A', 'Policy C']
policy_log_folder = 'AC'
log_folder_name = 'Random_sample_data/human_position_set_8'
render_path = '/home/peterdu2/policy_summarization/AST_CrowdNav/ast/results/data/' \
              + log_folder_name + '/' + policy_log_folder

if __name__ == '__main__':

    # Create simulator 
    sim = DSRNNCoupledSimulator(model_dirs=model_dirs,
                                config_names=config_name,
                                model_names=model_names,
                                s_0=s_0,
                                mode=mode,
                                goal_mode=goal_mode,
                                single_render_mode=False,
                                max_path_length=100,
                                blackbox_sim_state=False,
                                open_loop=False)

    # Create spaces
    num_humans = 10
    x_accel_low=-0.25
    x_accel_high=0.25
    y_accel_low=-0.25
    y_accel_high=0.25
    x_noise_low=-0.25
    x_noise_high=0.25
    y_noise_low=-0.25
    y_noise_high=0.25
    spaces = DSRNNSpaces(num_humans=num_humans,
                         x_accel_low=x_accel_low,
                         x_accel_high=x_accel_high,
                         y_accel_low=y_accel_low,
                         y_accel_high=y_accel_high,
                         x_noise_low=x_noise_low,
                         x_noise_high=x_noise_high,
                         y_noise_low=y_noise_low,
                         y_noise_high=y_noise_high,
                         mode=mode)

    for i in range(num_samples):
        # Create the path to store renderings
        save_render_path = render_path + '/' + format(i, '03d')
        if not os.path.exists(save_render_path):
            os.makedirs(save_render_path)

        sim.reset(s_0=s_0)
        sim.render_coupled(save_render=True,
                          render_path=save_render_path,
                          titles=policy_titles,
                          pause=0.)

        done_traj = False
        while not sim.is_terminal():
            action = spaces.action_space.sample()
            sim.step(action)
            sim.render_coupled(save_render=True,
                              render_path=save_render_path,
                              titles=policy_titles,
                              pause=0.)
            for state in sim.sim_infos:
                if isinstance(state['info'], ReachGoal):
                    done_traj = True
                    break
            if done_traj:
                break