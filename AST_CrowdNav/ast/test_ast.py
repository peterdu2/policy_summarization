from simulators.coupled_crowd_sim import DSRNNCoupledSimulator

if __name__ == '__main__':
    model_dirs = ['data/example_model/', 'data/example_model/']
    config_name = ['config', 'config']
    model_names = ['27776.pt', '27776.pt']
    sim = DSRNNCoupledSimulator(model_dirs, config_name, model_names)

    for i in range(10):
        sim.render()
        sim.step()