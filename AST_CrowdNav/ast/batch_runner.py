import pickle
import argparse
import configparser
import ast
from runner import runner as mcts_runner

# Simulator args
model_dir_path = 'dsrnn_models/policy_summarization_10_humans/'
model_dirs = [model_dir_path, model_dir_path]
config_names = ['config', 'config']
model_names = ['34400.pt', '34400.pt']
mode = 'DIRECT_ACTION'

# Spaces args
num_humans = 10
x_accel_low=-0.5
x_accel_high=0.5
y_accel_low=-0.5
y_accel_high=0.5
x_noise_low=-0.25
x_noise_high=0.25
y_noise_low=-0.25
y_noise_high=0.25

# Env args
max_path_length = 50
s_0 = []
s_0.append([-5., -4., 7., 2.])
s_0.append([0.9764521104695284, 5.673416349134316])
s_0.append([-0.3916318531059858, -6.311028515571641])
s_0.append([6.16749983681399, -1.8436072010801905])
s_0.append([6.382823965553917, 0.9261922797511946])
s_0.append([-1.7508964915665364, -6.244077560533835])
s_0.append([3.8364308394984743, -4.473900760196651])
s_0.append([5.561697860813725, 0.13723735189354574])
s_0.append([4.213509392947646, 4.356352793759812])
s_0.append([4.92327303238429, 2.5992114112982367])
s_0.append([-2.5587340425394998, 5.7638743741024])

# Logging args
base_log_dir = '/home/peterdu2/policy_summarization/AST_CrowdNav/ast/results/data'
log_folder_name = 'ast_dsrnn_test'


if __name__ == '__main__':
    # Which algorithms to run
    RUN_MCTS = True

    # experiment settings
    run_experiment_args = {'snapshot_mode': 'last',
                           'snapshot_gap': 1,
                           'log_dir': None,
                           'exp_name': None,
                           'seed': 0,
                           'n_parallel': 1,
                           'tabular_log_file': 'progress.csv'
                           }

    # runner settings
    runner_args = {'n_epochs': 100,
                   'batch_size': 100,
                   'plot': False
                   }

    # env settings
    env_args = {'id': 'CrowdSimDict-v0',
                'blackbox_sim_state': False,
                'open_loop': False,
                'fixed_init_state': True,
                's_0': s_0,
                }

    # simulation settings
    sim_args = {# AST standard args
                'blackbox_sim_state': False,
                'open_loop': False,
                'fixed_initial_state': True,
                'max_path_length': max_path_length,
                # DSRNN simulator args
                'model_dirs': model_dirs,
                'model_names': model_names,
                'config_names': config_names,
                's_0': s_0,
                'mode': mode
                }


    # reward settings
    reward_args = {}

    # spaces settings
    spaces_args = {# DSRNN reward args
                   'num_humans': num_humans,
                   'mode': mode,
                   'x_accel_low': x_accel_low,
                   'x_accel_high': x_accel_high,
                   'y_accel_low': y_accel_low,
                   'y_accel_high': y_accel_high,
                   'x_noise_low': x_noise_low,
                   'x_noise_high': x_noise_high,
                   'y_noise_low': y_noise_low,
                   'y_noise_high': y_noise_high
                   }


    # MCTS ----------------------------------------------------------------------------------
    if RUN_MCTS:
        # MCTS Settings

        mcts_type = 'mcts'

        mcts_sampler_args = {}

        mcts_algo_args = {'max_path_length': max_path_length,
                          'stress_test_mode': 1,
                          'ec': 100.0,
                          'n_itr': 20,
                          'k': 0.5,
                          'alpha': 0.5,
                          'clear_nodes': True,
                          'log_interval': 100,
                          'plot_tree': False,
                          'plot_path': None,
                          'log_dir': None,
                          }

        mcts_bpq_args = {'N': 10}

        # MCTS settings
        print("base_dir: ", base_log_dir)
        run_experiment_args['log_dir'] = base_log_dir + '/' + log_folder_name
        run_experiment_args['exp_name'] = 'mcts'

        mcts_algo_args['max_path_length'] = max_path_length
        mcts_algo_args['log_dir'] = run_experiment_args['log_dir']
        mcts_algo_args['plot_path'] = run_experiment_args['log_dir']

        mcts_runner(
            mcts_type=mcts_type,
            env_args=env_args,
            run_experiment_args=run_experiment_args,
            sim_args=sim_args,
            reward_args=reward_args,
            spaces_args=spaces_args,
            algo_args=mcts_algo_args,
            runner_args=runner_args,
            bpq_args=mcts_bpq_args,
            sampler_args=mcts_sampler_args,
            save_expert_trajectory=False,
        )
