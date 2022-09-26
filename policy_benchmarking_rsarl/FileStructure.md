# crowd_nav_ast
Repository for Adaptive Stress Testing of Crowd Navigation code

### All of the following files were built off of the existing Crowd_Nav and AST_Toolbox Libraries:

## simulator.py:

This file contains the code for the modified Crowd_Nav simulator. The functions in this file wrap around the existing functions in crowd_sim to allow for Adaptive Stress Testing.

#### __init__:

```
def __init__(self, config_file, max_path_length):

```

The initialization fuction for the simulator. It takes the path to a environment configuration file as an argument as well as a maximum path length. A sample configuration file has been provided in the repository (env.config)

#### clone_state:

```
def clone_state(self):

```

This function returns the current state of the environment and takes no arguments.

#### step:

```
def step(self, new_human_accelerations):

```

This function steps the simulation forward by one timestep. The argument the function is a list of new human accelerations which are used to modify the enironment before each step. This function returns the current state, the current total reward, a boolean which represents if the agent has reached a terminal state, and the relevent reward information about the current state.

#### simulate:

```
def simulate(self, actions, s_0):

```

This function steps through the simulation until the max_path_length is reached or another terminal condition has been acheived. The function takes a list of actions as well as a inital condition (s_0). This function returns the total number of timesteps of the simulation.

#### reset:

```
def reset(self, s_0):

```

This function resets the simulator environment to a given intitial state (s_0). The function returns the robot observation of the current state. 



#### get_reward_info:

```
def get_reward_info(self):

```

This function returns all nessesary reward info for the reward function. This includes booleans which represent if the robot has reached its goal or the time horizon (is_terminal) and if the robot has had a collision (is_goal). In addition, this function returns a list of the human positions (human_positions) and the robot's position (robot_position).


#### is_goal:

```
def is_goal(self):

```

Returns true of the robot has had a collision. Returns false otherwise.

#### is_terminal:

```
def is_terminal(self):

```

Returns true of the robot has reached the time horizon, max path length, or its goal. Returns false otherwise.

## reward.py:

This file contains the reward functions for the the modified Crowd_Nav simulator. The functions in this file are called by the simulator when recording rewards for an episode.

#### give_reward:

```
def give_reward(self, **kwargs):
```

This function returns a reward for a given state of the simulator. The argument for this function is a dictionary containing the position of the robot (robot_position), the position of the humans (human_positions), wheather or not the robot has collided (goal), and wheather or not the robot has reached an alternate terminal condition (terminal). The function returns a reward value dependent on the state information.

## space.py:

This file contains the reward functions for the the modified Crowd_Nav simulator. The functions in this file are called by the simulator when recording rewards for an episode.

#### __init__:

```
def __init__(self,
             x_accel_low=-5.0,
             y_accel_low=-5.0,
             x_accel_high=5.0,
             y_accel_high=5.0,
             ):
```

This function sets up the action space for the AST system. It takes bound values for accelerations in the x and y directions for the arguments which are used in subsequent functions.

#### action_space:

```
def action_space(self):
```

This function returns a Box object which contains the set of accelerations. This Box represents the action space for the environment and is used by the simulator step function.

#### observation_space:

```
def observation_space(self):
```

This function is not used by the modified program and therfore returns the same box object as action_space.

## runner.py:

This file contains the AST runner.

#### run_task:

```
def run_task(snapshot_config, *_):
```

This is the primary function of the runner class. This method is identical to the function described in the AST tutorial (https://ast-toolbox.readthedocs.io/en/latest/tutorial.html#creating-a-runner), but with the modified simulator, reward, and space classes.

## batch_runner.py:

This file contains the batch runner for the AST toolbox. This class is very similar to the class outlined in example_batch_runner_av.py within the AST toolbox examples. The primary difference are the differences in the runner, simulator, reward, and space arguments, which are changed to fit the modified classes.
