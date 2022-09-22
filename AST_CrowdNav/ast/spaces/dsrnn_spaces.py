"""Class to define the action and observation spaces for an example AV validation task."""
import numpy as np
from gym.spaces.box import Box

from ast_toolbox.spaces import ASTSpaces


class DSRNNSpaces(ASTSpaces):

    def __init__(self,
                 num_humans,
                 x_accel_low,
                 x_accel_high,
                 y_accel_low,
                 y_accel_high,
                 x_noise_low,
                 x_noise_high,
                 y_noise_low,
                 y_noise_high,
                 mode
                 ):
        self.c_num_humans = num_humans

        self.c_x_accel_low = x_accel_low
        self.c_x_accel_high = x_accel_high
        self.c_y_accel_low = y_accel_low
        self.c_y_accel_high = y_accel_high

        self.c_x_noise_low = x_noise_low
        self.c_x_noise_high = x_noise_high
        self.c_y_noise_low = y_noise_low
        self.c_y_noise_high = y_noise_high

        assert mode == 'DIRECT_ACTION' or mode == 'OBSERVATION_NOISE', \
            'mode must be either DIRECT_ACTION or OBSERVATION_NOISE'
        self.c_mode = mode

        super().__init__()

    @property
    def action_space(self):

        if self.c_mode == 'DIRECT_ACTION':
            low = np.array([self.c_x_accel_low, self.c_y_accel_low])
            high = np.array([self.c_x_accel_high, self.c_y_accel_high])

            for i in range(1, self.c_num_humans):
                low = np.hstack((low, np.array([self.c_x_accel_low, self.c_y_accel_low])))
                high = np.hstack((high, np.array([self.c_x_accel_high, self.c_y_accel_high])))

            return Box(low=low, high=high, dtype=np.float32)
        
        elif self.c_mode == 'OBSERVATION_NOISE':
            low = np.array([self.c_x_noise_low, self.c_y_noise_low])
            high = np.array([self.c_x_noise_high, self.c_y_noise_high])

            for i in range(1, self.c_num_humans):
                low = np.hstack((low, np.array([self.c_x_noise_low, self.c_y_noise_low])))
                high = np.hstack((high, np.array([self.c_x_noise_high, self.c_y_noise_high])))

            return Box(low=low, high=high, dtype=np.float32)

    @property
    def observation_space(self):
        """Returns a definition of the observation space of the reinforcement learning problem.

        Returns
        -------
        : `gym.spaces.Space <https://gym.openai.com/docs/#spaces>`_
            The observation space of the reinforcement learning problem.
        """

        low = np.array([self.c_x_accel_low, self.c_y_accel_low])
        high = np.array([self.c_x_accel_high, self.c_y_accel_high])

        for i in range(1, self.c_num_humans):
            low = np.hstack((low, np.array([self.c_x_accel_low, self.c_y_accel_low])))
            high = np.hstack((high, np.array([self.c_x_accel_high, self.c_y_accel_high])))

        return Box(low=low, high=high, dtype=np.float32)
