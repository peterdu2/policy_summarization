from ast_toolbox import rewards
import numpy as np
import itertools

from ast_toolbox.rewards import ASTReward  # import base class
from crowd_sim.envs.utils.info import *


# Define the class, inherit from the base
class CNReward(ASTReward):
    def __init__(self,
                 num_peds=5,
                 cov_x=1.0,
                 cov_y=1.0,
                 cov_sensor_noise=0.1,
                 use_heuristic=True):

        self.c_num_peds = num_peds
        self.c_cov_x = cov_x
        self.c_cov_y = cov_y
        self.c_cov_sensor_noise = cov_sensor_noise
        self.use_heuristic = use_heuristic
        super().__init__()

    def closest_distance(self, human_positions, robot_positions, n_policies):
        closest_dists = []
        for policy in range(n_policies):
            dists = []
            for human in human_positions[policy]:
                dists.append(np.linalg.norm(np.array(human) -
                             np.array(robot_positions[policy])))
            closest_dists.append(min(dists))
        return closest_dists

    def give_reward(self, action, **kwargs):
        info = kwargs['info']
        is_terminal = info['is_terminal']
        is_goal = info['is_goal']
        sim_info = info['sim_info']
        robot_actions = info['robot_actions']
        robot_positions = info['robot_positions']
        human_positions = info['human_positions']
        
        # Action Variance Reward
        if is_terminal:
            return 0. 

        # Calculate variance of robot actions
        n_policies = len(robot_actions)
        actions = []
        actions.append([act.vx for act in robot_actions])
        actions.append([act.vy for act in robot_actions])
        vars = [np.var(act_list) for act_list in actions]
        reward = np.sum(vars)
        
        if is_goal:
            reward += 100.

        return reward

        # # Safety Distance Reward
        # if is_terminal:
        #     return 0. 

        # # Calculate closet distance from human to robot for each policy
        # n_policies = len(robot_positions)
        # safety_dists = self.closest_distance(
        #     human_positions, robot_positions, n_policies)

        # # Find largest pairwise difference in safety distances
        # diffs = []
        # for pair in itertools.combinations(safety_dists, 2):
        #     diffs.append(abs(pair[0]-pair[1]))

        # max_safety_diff = max(diffs)

        # reward = max_safety_diff
        # if is_goal:
        #     reward += 100.

        # return reward

    def mahalanobis_d(self, action):
        # Mean action is 0
        mean = np.zeros((2 * self.c_num_peds, 1))
        # Assemble the diagonal covariance matrix
        cov = np.zeros((self.c_num_peds, 2))
        cov[:, 0:10] = np.array([self.c_cov_x, self.c_cov_y])
        big_cov = np.diagflat(cov)

        # subtract the mean from our actions
        dif = np.copy(action)
        dif[::2] -= mean[0, 0]
        dif[1::2] -= mean[1, 0]
        # calculate the Mahalanobis distance
        dist = np.dot(np.dot(dif.T, np.linalg.inv(big_cov)), dif)

        return np.sqrt(dist)
