import numpy as np
import math
import matplotlib.pyplot as plt
import pickle
from ast_crowd_nav_coupled_simulator import ASTCrowdNavCoupledSimulator
import time

num_policies = 3
num_humans = 5


def grapher(sim, s_0, best_actions):
    traj = 0
    for (action_seq, reward_predict) in best_actions:
        sim.reset(s_0=s_0)
        robot_locations = [[[], []] for i in range(num_policies)]
        human_locations = [[[[], []] for j in range(
            num_humans)] for i in range(num_policies)]
        print(reward_predict)
        print(len(action_seq))
        for action in action_seq:
            # Run a simulation step
            observation = sim.step(action.action)
            # state = sim.clone_state()
            # angle = state[0] - 2*np.pi * np.floor(state[0]/(2*np.pi))
            # print(angle)
            print(sim.collision_check())
            print('goal:', sim.is_goal(), 'terminal:', sim.is_terminal())
            for i in range(num_policies):
                robot_pos = sim.robot_states[i]
                robot_locations[i][0].append(robot_pos.px)
                robot_locations[i][1].append(robot_pos.py)

                human_pos_list = sim.robot_obs[i]
                for j in range(num_humans):
                    human_pos = human_pos_list[j]
                    human_locations[i][j][0].append(human_pos.px)
                    human_locations[i][j][1].append(human_pos.py)

        # sim.simulators[0].render(mode='video')
        # sim.simulators[1].render(mode='video')
        # sim.simulators[2].render(mode='video')
        for i in range(num_policies):
            if i == 0:
                plt.plot(robot_locations[i][0], robot_locations[i]
                         [1], label='SARL', marker='o', color='#0083a1')
            elif i == 1:
                plt.plot(robot_locations[i][0], robot_locations[i]
                         [1], label='LSTM', marker='o', color='#00a878')
            else:
                plt.plot(robot_locations[i][0], robot_locations[i]
                         [1], label='CADRL', marker='o', color='#c4af0c')
            for j in [0, 1, 4]:
                if j == 0 and i == 0:
                    plt.plot(human_locations[i][j][0], human_locations[i]
                             [j][1], label='Human', color='grey', marker='o')
                else:
                    plt.plot(
                        human_locations[i][j][0], human_locations[i][j][1], color='grey', marker='o')
        # frame1 = plt.gca()
        # frame1.axes.xaxis.set_ticklabels([])
        # frame1.axes.yaxis.set_ticklabels([])
        plt.grid()
        plt.plot([0], [-4], marker='D', markersize=10, color='black',
                 markerfacecolor='darkorange', label='Robot Starting Position', linestyle='None')
        plt.plot([1], [0], marker='D', markersize=10, color='black',
                 markerfacecolor='grey', label='Human Starting Position', linestyle='None')
        plt.plot([-3.1], [1.4], marker='D', markersize=10,
                 color='black', markerfacecolor='grey')
        plt.plot([3.6], [-0.1], marker='D', markersize=10,
                 color='black', markerfacecolor='grey')
        plt.xlim(-4, 7)
        plt.ylim(-5, 2)
        plt.xlabel('meters')
        plt.ylabel('meters')
        plt.legend(loc='lower right')
        plt.savefig('graphs/sd_1/crowdnav_sd_1_'+str(traj)+'.png')

        plt.clf()
        traj += 1


config_file_paths = ['/home/ubuntu/policy_benchmarking/ast_config_files/sarl.config',
                     '/home/ubuntu/policy_benchmarking/ast_config_files/lstm.config',
                     '/home/ubuntu/policy_benchmarking/ast_config_files/cadrl.config']

best_actions = pickle.load(
    open('data/crowdnav_var_3/top_actions.pkl', 'rb'))

sim_args = {'blackbox_sim_state': False,
            'open_loop': False,
            'fixed_initial_state': True,
            'config_file_paths': [config_file_paths[0], config_file_paths[1], config_file_paths[2]],
            'max_path_length': 100, }
sim = ASTCrowdNavCoupledSimulator(**sim_args)
#s_0 = [(0.0, 0.0), (-4.1, 1.4), (-2.7, 3.3), (-3.1, -2.4), (-3.6, -0.1)]
# s_0 = [(0.0, 0.0), (-4.1, 1.4), (2.0, 2.7), (-2.8, -2.4), (3.6, -0.1)]
#s_0 = [(0.0, 0.0), (-2.1, 1.4), (2.0, 2.7), (-2.8, -2.4), (2.6, -0.1)]
s_0 = [(1.0, 0.0), (-3.1, 1.4), (2.0, 2.7), (-1.8, -2.4), (3.6, -0.1)]
s_0 = [(1.0, 0.0), (-3.1, 1.4), (2.0, 2.7), (0.0, 1.0), (3.6, -0.1)]

grapher(sim, s_0, best_actions)

# for (action_seq, reward_predict) in best_actions:
#     sim.reset(s_0=s_0)
#     time.sleep(1)
#     print(reward_predict)
#     print(len(action_seq))
#     for action in action_seq:
#         # Run a simulation step
#         observation = sim.step(action.action)
#         # state = sim.clone_state()
#         # angle = state[0] - 2*np.pi * np.floor(state[0]/(2*np.pi))
#         # print(angle)
#         print(sim.collision_check())
#         print('goal:', sim.is_goal(), 'terminal:', sim.is_terminal())
#     sim.simulators[0].render(mode='video')
#     sim.simulators[1].render(mode='video')
#     sim.simulators[2].render(mode='video')
#########################################################################################
