# crowd_nav_ast
Repository for Adaptive Stress Testing of Crowd Navigation code

# Important Information:

This repository builds off of and combines code from Crowd_Nav and AST_Toolbox. The commit version of Crowd_Nav used in this repository is 24be7ad39fed0658c5631c0820c67f9105e457bc. For information on modified files, classes, and functions, please refer to FileStructure.md.

# Download Instructions:

1. Click the green "Code" button next to the "Add File" button underneath the upper repository tool bar.
2. Click the "Download as Zip" option
3. Expand the zip file to obtain the folder for the repository

# Pre-Requistes for Use

Before running the code, the AST_toolbox package must be installed. Instructions for installation can be found in the AST documentation website (https://ast-toolbox.readthedocs.io/en/latest/installation.html). The version of AST_toolbox that was used in this repository was version 2020.9.1.0.

After the ast_toolbox class has been installed, some changes need to be made with the package code to ensure smooth running:

1. Comment out lines 120-125 of ast_toolbox/algos/mcts.py and add the following line:

```
pickle.dump(result, open("path_to_directory/results.pkl", 'wb')
```

Replace the path_to_directory with a path to the folder in which the results file is to be stored. 

# Instructions for Running:

1. Open the terminal and use the cd command to enter the crowd_nav_ast folder
2. If desired, change the base parameters for the batch_runner environment. These can be found in the env.config file
3. If desired, train a policy using the following command in the crowd_nav folder
```
python train.py --policy 'desired_policy'
```
4. After train.py has finished running, modify the env.config file to include the path to the model weight file
5. Return to the main repository folder and run the following command in the terminal:
```
python batch_runner.py --env_config '/path/to/config_file'
```
6. After AST finishes running, you can test the failure trajectories by running the following command in terminal:
```
python load_file.py --results_dir '/path/to/fail_trajectory_file' --env_config '/path/to/config_file'
```

### If assistance with debugging is required: please email suryakm2@illinois.edu
