import numpy as np  # useful packages for math

from ast_toolbox.rewards import ASTReward  # import base class
from crowd_sim.envs.utils.info import *


class DSRNNReward(ASTReward):
    def __init__(self,
                 collision_reward,
                 num_humans=10,
                 use_heuristic=True):

        self.collision_reward = collision_reward
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

        # if (is_terminal):
        #     if self.use_heuristic:
        #         heuristic_reward = self.get_robot_separation(robot_positions)
        #     else:
        #         heuristic_reward = 0
        #     reward = 100 * heuristic_reward 
        # else:
        #     action_separation = np.linalg.norm(robot_actions[0] - robot_actions[1])
        #     reward = action_separation
        #     # Check for collision
        #     for state in sim_infos:
        #         if isinstance(state['info'], Collision):
        #             # If collision detected overwrite reward
        #             reward = self.collision_reward + self.get_robot_separation(robot_positions)
        #             break

        if (is_goal):  # At least one of the robots has crashed
            reward = self.collision_reward + self.get_robot_separation(robot_positions)
        elif (is_terminal):
            if self.use_heuristic:
                heuristic_reward = self.get_robot_separation(robot_positions)
            else:
                heuristic_reward = 0
            reward = -100000 + 100 * heuristic_reward 
        else:
            action_separation = np.linalg.norm(robot_actions[0] - robot_actions[1])
            reward = action_separation

        return reward


    def get_robot_separation(self, robot_positions):
        assert len(robot_positions) == 2, 'Number of robot positions is not two'
        robot_positions = np.array(robot_positions)
        return np.linalg.norm(robot_positions[0]-robot_positions[1])
