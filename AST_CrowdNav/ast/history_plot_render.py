import numpy as np
import os
import pickle
import time

from crowd_sim.envs.utils.info import *

from simulators.dsrnn_coupled_simulator import DSRNNCoupledSimulator

model_dirs = ['dsrnn_models/policy_summarization_10_humans/', 'dsrnn_models/policy_summarization_10_humans/']
config_name = ['config', 'config']
model_names = ['34400.pt', '20600.pt']

s_0 = []
s_0.append([-5., 4., 7., -2.])
s_0.append([-1.9764521104695284, 3.673416349134316])
s_0.append([-1.3916318531059858, -4.311028515571641])
s_0.append([2.56749983681399, -2.8436072010801905])
s_0.append([0.382823965553917, 0.9261922797511946])
s_0.append([-3.5008964915665364, -1.244077560533835])
s_0.append([1.8364308394984743, -6.473900760196651])
s_0.append([3.561697860813725, -0.53723735189354574])
s_0.append([4.213509392947646, 1.356352793759812])
s_0.append([5.92327303238429, 1.5992114112982367])
s_0.append([-2.5587340425394998, 1.7638743741024])

mode = 'OBSERVATION_NOISE'
mode = 'DIRECT_ACTION'
goal_mode = 'REACHGOAL'

policy_titles = ['Robot 1', 'Robot 3']
log_folder_name = 'ast_dsrnn_31'
render_path = '/home/peterdu2/policy_summarization/AST_CrowdNav/ast/results/data/' \
              + log_folder_name + '/history_plots'

if __name__ == '__main__':

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
                                
    result_path = 'results/data/' + log_folder_name + '/top_actions.pkl'
    ast_results = pickle.load(open(result_path, 'rb'))

    i = 0
    for (action_seq, reward_predict) in ast_results:
        if i > 0:
            break
        print('Trajectory ID:', i)
        print('EXPECTED REWARD:', reward_predict)

        # Create the path to store renderings
        save_render_path = None
        save_render_path = render_path + '/' + format(i, '03d')
        if not os.path.exists(save_render_path):
            os.makedirs(save_render_path)

        sim.reset(s_0=s_0)
        sim.render_coupled_hist_traj(save_render=True,
                          render_path=save_render_path,
                          titles=policy_titles,
                          num_frames = 18,
                          pause=0)

        done_traj = False
        step = 0
        
        for action in action_seq:
            sim.step(action.action)
            if step % 5 == 0:
                sim.render_coupled_hist_traj(save_render=True,
                                render_path=save_render_path,
                                titles=policy_titles,
                                num_frames = 18,
                                pause=0)

            step += 1
            for state in sim.sim_infos:
                if isinstance(state['info'], ReachGoal):
                    done_traj = True
                    break
            if done_traj:
                break
        
        print('Trajectory ended with states:', sim.sim_infos)
        print(' ')
        i += 1