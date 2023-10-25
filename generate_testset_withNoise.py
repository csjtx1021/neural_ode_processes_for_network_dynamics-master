import numpy as np
from load_dynamics_solution2and3 import *

from NDP4ND_testNoise import make_batch, set_rand_seed


def generate_testset_1():
    rseed = 666
    set_rand_seed(rseed)

    N = 225
    # N = 400

    test_num_trials = 4

    for dynamics_name in [
        'mutualistic_interaction_dynamics',

    ]:
        for topo_type in [
            'grid',
            'random',
            'power_law',
            'small_world',
            'community'
        ]:
            #for noise in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            for noise in [0.1, 0.2, 0.3, 0.4, 0.5]:

                saved_test_set = {}

                time_steps = 100
                x_dim = 1
                num_graphs = 1

                dataset = dynamics_dataset(dynamics_name, topo_type,
                                           num_graphs_samples=num_graphs, time_steps=time_steps, x_dim=x_dim,
                                           make_test_set=True)
                test_time_steps = time_steps

                for i in range(test_num_trials):

                    test_data = dataset.query(seed=rseed + i * 10, num_graph=1, max_time_points=test_time_steps,
                                              query_all_node=True, query_all_t=True,
                                              N=N,
                                              make_test_set=True,
                                              obs_noise=noise,
                                              )
                                              
#                    import matplotlib.pyplot as plt
#                    ATask = test_data['tasks'][0]  
#                    points = ATask['points']  
#                    
#                    for node_id in [0, 50,100,200]:
#                        x = []
#                        y = []
#                        y_w_noise = []
#                        for point in points:
#                            x.append(point['t'])
#                            y.append(point['x_self'][node_id,0])
#                            y_w_noise.append(point['x_self_w_noise'][node_id,0])
#                        plt.plot(x, y, 'k:')
#                        plt.scatter(x, y_w_noise, c='r', alpha=0.5)
#                    plt.savefig('test_noise%s_%s.png'%(noise,i))
#                    plt.close()
                                    
                    #print()
                    # print(test_data)
                    #for bound_t_context in [0., 0.25, 0.5, 0.75]:
                    for bound_t_context in [0.5]:

                        set_rand_seed(rseed + i * 100)

                        gen_batch = make_batch(test_data, 1, is_shuffle=True, bound_t_context=bound_t_context, is_test=True,
                                               is_shuffle_target=False, max_x_dim=4)
                        for batch_data in gen_batch:
                            saved_test_set[(dynamics_name, topo_type, bound_t_context, i)] = batch_data

                # save test data
                import pickle

                fname = 'data/DynamicsData/test_data_on_dynamics_%s_topo_%s_dataset_x%s_numgraph%s_timestep%s_seed%s_num_nodes=%s_noise=%s_split_train_and_test.pickle' % (
                    dynamics_name, topo_type, dataset.x_dim, 1, time_steps, rseed,
                    N, noise)
                f = open(fname, 'wb')
                pickle.dump(saved_test_set, f)
                f.close()

# test
if __name__ == '__main__':
    generate_testset_1()
