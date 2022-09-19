import numpy as np
from gym.spaces.box import Box

from ast_toolbox.spaces import ASTSpaces


class CNSpaces(ASTSpaces):
    def __init__(self,
             x_accel_low=-1.0,
             y_accel_low=-1.0,
             x_accel_high=1.0,
             y_accel_high=1.0,
             ):

    # Constant hyper-params -- set by user
        self.c_x_accel_low = x_accel_low
        self.c_y_accel_low = y_accel_low
        self.c_x_accel_high = x_accel_high
        self.c_y_accel_high = y_accel_high
        self.c_num_peds = 5
        super().__init__()

    @property
    def action_space(self):
        """
        Returns a Space object
        """
        low = np.array([self.c_x_accel_low, self.c_y_accel_low])
        high = np.array([self.c_x_accel_high, self.c_y_accel_high])

        for i in range(1, self.c_num_peds):
            low = np.hstack((low, np.array([self.c_x_accel_low, self.c_y_accel_low])))
            high = np.hstack((high, np.array([self.c_x_accel_high, self.c_y_accel_high])))

        return Box(low=low, high=high, dtype=np.float32)

    def observation_space(self):

        low = np.array([self.c_x_accel_low, self.c_y_accel_low, -3.0, -3.0, -3.0, -3.0])
        high = np.array([self.c_x_accel_high, self.c_y_accel_high, 3.0, 3.0, 3.0, 3.0])

        for i in range(1, self.c_num_peds):
            low = np.hstack((low, np.array([self.c_x_accel_low, self.c_y_accel_low, 0.0, 0.0, 0.0, 0.0])))
            high = np.hstack((high, np.array([self.c_x_accel_high, self.c_y_accel_high, 1.0, 1.0, 1.0, 1.0])))

        return Box(low=low, high=high, dtype=np.float32)
