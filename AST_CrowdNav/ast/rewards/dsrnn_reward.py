import numpy as np  # useful packages for math

from ast_toolbox.rewards import ASTReward  # import base class
from crowd_sim.envs.utils.info import *


class DSRNNReward(ASTReward):
    def __init__(self,
                 collision_reward,
                 goal_mode,
                 num_humans=10,
                 use_heuristic=True):

        assert goal_mode == 'COLLISION' or goal_mode == 'REACHGOAL'

        self.collision_reward = collision_reward
        self.goal_mode = goal_mode
        self.c_num_humans = num_humans
        self.use_heuristic = use_heuristic
        super().__init__()


    def give_reward(self, action, **kwargs):
        # get the info from the simulator
        info = kwargs['info']
        is_terminal = info['is_terminal']
        is_goal = info['is_goal']
        sim_infos = info['sim_infos']
        robot_positions = info['robot_positions']
        robot_actions = info['robot_actions']

        if (is_goal):
            reward = 100 * self.get_robot_separation(robot_positions)
            if self.goal_mode == 'COLLISION':
                reward += self.collision_reward
        elif (is_terminal):
            if self.use_heuristic:
                heuristic_reward = self.get_robot_separation(robot_positions)
            else:
                heuristic_reward = 0
            reward = -100000 + 100 * heuristic_reward 
        else:
            # Calculate action separation
            reward = np.linalg.norm(robot_actions[0] - robot_actions[1])
            # Check for collisions
            for state in sim_infos:
                if isinstance(state['info'], Collision):
                    reward += self.collision_reward
                    break

        return reward


    def get_robot_separation(self, robot_positions):
        assert len(robot_positions) == 2, 'Number of robot positions is not two'
        robot_positions = np.array(robot_positions)
        return np.linalg.norm(robot_positions[0]-robot_positions[1])
