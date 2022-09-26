import numpy as np
import random
import argparse
from gym.spaces.box import Box
import pickle
from ast_toolbox.spaces import ASTSpaces
from ast_toolbox.mcts.BoundedPriorityQueues import BoundedPriorityQueue
from simulator import AST_simulator_wrapper

if __name__ == '__main__':
    np.random.seed(101)
    parser = argparse.ArgumentParser('Parse fuction arguments')
    parser.add_argument('--results_dir', type=str, default='/home/ubuntu/crowd_nav_ast/results/sarl_results.pkl')
    parser.add_argument('--env_config', type=str, default='/home/ubuntu/crowd_nav_ast/env.config')
    test_args = parser.parse_args()
    objects = []
    with (open(test_args.results_dir, "rb")) as openfile:
        while True:
            try:
                objects.append(pickle.load(openfile))
            except EOFError:
                break
    for item in objects:
        for entry in item:
            action_list = []
            for sub_entry in entry[0]:
                action_list.append(list(sub_entry.action))
            sim_1 = AST_simulator_wrapper(test_args.env_config, 50)
            sim_1.env.seed(0)
            s_0 = [(0.0, 0.0), (-4.1, 1.4), (-2.7, 3.3), (-3.1, -2.4), (-3.6, -0.1)]
            sim_1.simulate(action_list, s_0)
        
    