from simulators.dsrnn_coupled_simulator import DSRNNCoupledSimulator
from spaces.dsrnn_spaces import DSRNNSpaces
from rewards.dsrnn_reward import DSRNNReward
import random

if __name__ == '__main__':
    model_dirs = ['dsrnn_models/policy_summarization_10_humans/', 'dsrnn_models/policy_summarization_10_humans/']
    config_name = ['config', 'config']
    model_names = ['34400.pt', '34400.pt']

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

    mode = 'OBSERVATION_NOISE'
    mode = 'DIRECT_ACTION'

    sim = DSRNNCoupledSimulator(model_dirs=model_dirs,
                                config_names=config_name,
                                model_names=model_names,
                                s_0=s_0,
                                mode=mode,
                                max_path_length=100,
                                blackbox_sim_state=False,
                                open_loop=False)

    reward = DSRNNReward(num_humans=10,
                         use_heuristic=True)

    spaces = DSRNNSpaces(num_humans=10,
                         x_accel_low=-0.5,
                         x_accel_high=0.5,
                         y_accel_low=-0.5,
                         y_accel_high=0.5,
                         x_noise_low=-0.25,
                         x_noise_high=0.25,
                         y_noise_low=-0.25,
                         y_noise_high=0.25,
                         mode=mode)

    counter = 0
    while True:
        if not sim.is_terminal() and not sim.is_goal():
            env_action = [[random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5)] for i in range(10)]

            test_env_action = spaces.action_space.sample()
            obs = sim.step(test_env_action)

            r_info = sim.get_reward_info()
            r = reward.give_reward(env_action, info=r_info)

            sim.render()
        else:
            sim.reset(s_0)
            counter += 1
            if counter >= 10:
                break
