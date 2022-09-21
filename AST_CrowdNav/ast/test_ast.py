from simulators.coupled_crowd_sim import DSRNNCoupledSimulator

if __name__ == '__main__':
    model_dirs = ['data/policy_summarization_10_humans/', 'data/policy_summarization_10_humans/']
    config_name = ['config', 'config']
    model_names = ['34400.pt', '34400.pt']
    sim = DSRNNCoupledSimulator(model_dirs, config_name, model_names)


    counter = 0
    while True:
        if not True in sim.dones:
            sim.step()
            sim.render()
        else:
            sim.reset()
            counter += 1
            if counter >= 3:
                break
