"""An example implementation of an ASTReward for an AV validation scenario."""
import numpy as np  # useful packages for math

from ast_toolbox.rewards import ASTReward  # import base class


class DSRNNReward(ASTReward):
    def __init__(self,
                 num_humans=10,
                 use_heuristic=True):

        self.c_num_humans = num_humans
        self.use_heuristic = use_heuristic
        super().__init__()


    def give_reward(self, action, **kwargs):
        # get the info from the simulator
        info = kwargs['info']
        is_terminal = info['is_terminal']
        is_goal = info['is_goal']
        robot_positions = info['robot_positions']
        robot_actions = info['robot_actions']

        if (is_goal):  # At least one of the robots has crashed
            reward = 100. + self.get_robot_separation(robot_positions)
        elif (is_terminal):
            if self.use_heuristic:
                heuristic_reward = self.get_robot_separation(robot_positions)
            else:
                heuristic_reward = 0
            reward = -100000 + 10000 * heuristic_reward 
        else:
            action_separation = np.linalg.norm(robot_actions[0] - robot_actions[1])
            reward = action_separation

        return reward


    def get_robot_separation(self, robot_positions):
        assert len(robot_positions) == 2, 'Number of robot positions is not two'
        robot_positions = np.array(robot_positions)
        return np.linalg.norm(robot_positions[0]-robot_positions[1])
