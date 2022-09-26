import pickle
import argparse
import configparser
import ast
from runner import runner as mcts_runner

config_file_paths = ['/home/peterdu2/research_2022/policy_benchmarking/policy_benchmarking_rsarl/ast_config_files/sarl.config',
                     '/home/peterdu2/research_2022/policy_benchmarking/policy_benchmarking_rsarl/ast_config_files/rsarl.config',
                     '/home/peterdu2/research_2022/policy_benchmarking/policy_benchmarking_rsarl/ast_config_files/cadrl.config']

if __name__ == '__main__':
    # Which algorithms to run
    RUN_MCTS = True
    parser = argparse.ArgumentParser('Parse fuction arguments')
    parser.add_argument('--env_config', type=str,
                        default='/home/peterdu2/research_2022/policy_benchmarking/policy_benchmarking_rsarl/env.config')
    test_args = parser.parse_args()
    env_config = configparser.RawConfigParser()
    env_config.read(test_args.env_config)
    # Overall settings
    max_path_length = int(env_config.get('ast', 'max_path_length'))
    s_0 = ast.literal_eval(env_config.get('ast', 's_0'))
    base_log_dir = './data'
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
    env_args = {'id': 'CrowdSim-v0',
                'blackbox_sim_state': False,
                'open_loop': False,
                'fixed_init_state': True,
                's_0': s_0,
                }

    # simulation settings
    sim_args = {'blackbox_sim_state': False,
                'open_loop': False,
                'fixed_initial_state': True,
                'max_path_length': max_path_length,
                'config_file_paths': [config_file_paths[0], config_file_paths[1], config_file_paths[2]],}


    # reward settings
    reward_args = {}
    # reward_args = {'use_heuristic': True}

    # spaces settings
    spaces_args = {}

    # DRL ----------------------------------------------------------------------------------

    # MCTS ----------------------------------------------------------------------------------

    if RUN_MCTS:
        # MCTS Settings

        mcts_type = 'mcts'

        mcts_sampler_args = {}

        mcts_algo_args = {'max_path_length': max_path_length,
                          'stress_test_mode': 1,
                          'ec': 100.0,
                          'n_itr': 3000,
                          'k': 0.5,
                          'alpha': 0.5,
                          'clear_nodes': True,
                          'log_interval': 100,
                          'plot_tree': False,
                          'plot_path': None,
                          'log_dir': None,
                          }

        mcts_bpq_args = {'N': 20}

        # MCTS settings
        print("base_dir: ", base_log_dir)
        run_experiment_args['log_dir'] = base_log_dir + \
            '/rsarl_test'
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
