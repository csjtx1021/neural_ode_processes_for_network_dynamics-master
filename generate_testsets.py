from load_dynamics_solution2and3 import *

from NDP4ND import make_batch, set_rand_seed


def generate_testset_1():
    rseed = 666
    set_rand_seed(rseed)

    N = 225
    # N = 400

    test_num_trials = 20

    for dynamics_name in [
        # 'heat_diffusion_dynamics',
        'mutualistic_interaction_dynamics',
        # 'gene_regulatory_dynamics',
        # 'combination_dynamics',
        # 'combination_dynamics_vary_coeff',
        # 'vary_dynamics_with_vary_type_and_coeff'

    ]:
        for topo_type in [
            'grid',
            'random',
            'power_law',
            'small_world',
            'community'
        ]:
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
                                          make_test_set=True
                                          )
                # print(test_data)

                for bound_t_context in [0., 0.25, 0.5, 0.75]:

                    set_rand_seed(rseed + i * 100)

                    gen_batch = make_batch(test_data, 1, is_shuffle=True, bound_t_context=bound_t_context, is_test=True,
                                           is_shuffle_target=False, max_x_dim=4)
                    for batch_data in gen_batch:
                        saved_test_set[(dynamics_name, topo_type, bound_t_context, i)] = batch_data

            # save test data
            import pickle

            fname = 'data/DynamicsData/test_data_on_dynamics_%s_topo_%s_dataset_x%s_numgraph%s_timestep%s_seed%s_num_nodes=%s_split_train_and_test.pickle' % (
                dynamics_name, topo_type, dataset.x_dim, 1, time_steps, rseed,
                N)
            f = open(fname, 'wb')
            pickle.dump(saved_test_set, f)
            f.close()


def generate_testset_2_3():
    rseed = 666
    set_rand_seed(rseed)
    # N = 20
    # N = 30
    N = 200
    # N = -1
    # N = 50
    # N = 250

    is_shuffle = True

    test_num_trials = 100

    for dynamics_name in [
        'SIS_Individual_dynamics',
        'SIR_Individual_dynamics',
        'SEIS_Individual_dynamics',
    ]:
        for topo_type in [
            'power_law',
        ]:
            saved_test_set = {}

            time_steps = 100
            x_dim = 4
            num_graphs = 1

            if dynamics_name == 'SI_Individual_dynamics' or dynamics_name == 'SIS_Individual_dynamics':
                x_dim = 2
            elif dynamics_name == 'SIR_Individual_dynamics' or dynamics_name == 'SEIS_Individual_dynamics':
                x_dim = 3
            else:
                x_dim = 4

            dataset = dynamics_dataset(dynamics_name, topo_type,
                                       num_graphs_samples=num_graphs, time_steps=time_steps, x_dim=x_dim,
                                       make_test_set=True)

            test_time_steps = time_steps

            for i in range(test_num_trials):
                if N == -1:
                    test_data = dataset.query(seed=rseed + i * 10, num_graph=1, max_time_points=test_time_steps,
                                              query_all_node=True, query_all_t=True,
                                              make_test_set=True,
                                              )
                else:
                    if dynamics_name == 'SIR_Individual_dynamics':
                        if i == 0:
                            test_data = dataset.query(seed=rseed + i * 10, num_graph=1, max_time_points=test_time_steps,
                                                      query_all_node=True, query_all_t=True,
                                                      N=N,
                                                      make_test_set=True,
                                                      fixed_param_case1=True,
                                                      )
                        elif i == 1:
                            test_data = dataset.query(seed=rseed + i * 10, num_graph=1, max_time_points=test_time_steps,
                                                      query_all_node=True, query_all_t=True,
                                                      N=N,
                                                      make_test_set=True,
                                                      fixed_param_case2=True,
                                                      )
                        elif i == 2:
                            test_data = dataset.query(seed=rseed + i * 10, num_graph=1, max_time_points=test_time_steps,
                                                      query_all_node=True, query_all_t=True,
                                                      N=N,
                                                      make_test_set=True,
                                                      fixed_param_case3=True,
                                                      )
                        elif i == 3:
                            test_data = dataset.query(seed=rseed + i * 10, num_graph=1, max_time_points=test_time_steps,
                                                      query_all_node=True, query_all_t=True,
                                                      N=N,
                                                      make_test_set=True,
                                                      fixed_param_case4=True,
                                                      )
                        else:

                            test_data = dataset.query(seed=rseed + i * 10, num_graph=1, max_time_points=test_time_steps,
                                                      query_all_node=True, query_all_t=True,
                                                      N=N,
                                                      make_test_set=True,
                                                      )
                    elif dynamics_name == 'SEIR_Individual_dynamics':
                        if i == 0:
                            test_data = dataset.query(seed=rseed + i * 10, num_graph=1, max_time_points=test_time_steps,
                                                      query_all_node=True, query_all_t=True,
                                                      N=N,
                                                      make_test_set=True,
                                                      fixed_param_case1=True,
                                                      )
                        else:
                            test_data = dataset.query(seed=rseed + i * 10, num_graph=1, max_time_points=test_time_steps,
                                                      query_all_node=True, query_all_t=True,
                                                      N=N,
                                                      make_test_set=True,
                                                      )
                    else:
                        test_data = dataset.query(seed=rseed + i * 10, num_graph=1, max_time_points=test_time_steps,
                                                  query_all_node=True, query_all_t=True,
                                                  N=N,
                                                  make_test_set=True,
                                                  )

                print("%s/%s " % (i, test_num_trials))

                for bound_t_context in [0., 0.25, 0.5, 0.75]:
                    # for bound_t_context in [0., 0.2, 0.4]:

                    set_rand_seed(rseed + i * 100)

                    gen_batch = make_batch(test_data, 1, is_shuffle=is_shuffle, bound_t_context=bound_t_context,
                                           is_test=True,
                                           is_shuffle_target=False, max_x_dim=4)
                    for batch_data in gen_batch:
                        saved_test_set[(dynamics_name, topo_type, bound_t_context, i)] = batch_data

            # save test data
            import pickle

            fname = 'data/DynamicsData/test_data_on_dynamics_%s_topo_%s_dataset_x%s_numgraph%s_timestep%s_seed%s_num_nodes=%s_split_train_and_test_is_shuffle%s.pickle' % (
                dynamics_name, topo_type, dataset.x_dim, 1, time_steps, rseed, N, str(is_shuffle))
            f = open(fname, 'wb')
            pickle.dump(saved_test_set, f)
            f.close()


def generate_testset_brain():
    rseed = 666
    set_rand_seed(rseed)

    N = 200

    test_num_trials = 3

    for dynamics_name in [
        'brain_FitzHugh_Nagumo_dynamics',
    ]:
        for topo_type in [
            # 'community',
            'power_law',
        ]:
            saved_test_set = {}

            time_steps = 100
            x_dim = 2
            num_graphs = 1

            dataset = dynamics_dataset(dynamics_name, topo_type,
                                       num_graphs_samples=num_graphs, time_steps=time_steps, x_dim=x_dim,
                                       make_test_set=True)
            test_time_steps = time_steps

            for i in range(test_num_trials):
                if N == -1:
                    test_data = dataset.query(seed=rseed + i * 10, num_graph=1, max_time_points=test_time_steps,
                                              query_all_node=True, query_all_t=True,
                                              make_test_set=True
                                              )
                else:
                    test_data = dataset.query(seed=rseed + i * 10, num_graph=1, max_time_points=test_time_steps,
                                                  query_all_node=True, query_all_t=True,
                                                  N=N,
                                                  make_test_set=True,
                                                  )
                                                  
                print("%s/%s " % (i, test_num_trials))

                for bound_t_context in [0.5]:
                    # for bound_t_context in [0., 0.2, 0.4]:

                    set_rand_seed(rseed + i * 100)

                    gen_batch = make_batch(test_data, 1, is_shuffle=True, bound_t_context=bound_t_context,
                                           is_test=True,
                                           is_shuffle_target=False, max_x_dim=4)
                    for batch_data in gen_batch:
                        saved_test_set[(dynamics_name, topo_type, bound_t_context, i)] = batch_data

            # save test data
            import pickle

            fname = 'data/DynamicsData/test_data_on_dynamics_%s_topo_%s_dataset_x%s_numgraph%s_timestep%s_seed%s_num_nodes=%s_split_train_and_test.pickle' % (
                dynamics_name, topo_type, dataset.x_dim, 1, time_steps, rseed, N)
            f = open(fname, 'wb')
            pickle.dump(saved_test_set, f)
            f.close()

def generate_testset_phototaxis_dynamics():
    rseed = 666
    set_rand_seed(rseed)

    N = 40

    test_num_trials = 3

    for dynamics_name in [
        'phototaxis_dynamics',
    ]:
        for topo_type in [
            # 'community',
            'full_connected',
        ]:
            saved_test_set = {}

            time_steps = 100
            x_dim = 5
            num_graphs = 1

            dataset = dynamics_dataset(dynamics_name, topo_type,
                                       num_graphs_samples=num_graphs, time_steps=time_steps, x_dim=x_dim,
                                       make_test_set=True)
            test_time_steps = time_steps

            for i in range(test_num_trials):
                if N == -1:
                    test_data = dataset.query(seed=rseed + i * 10, num_graph=1, max_time_points=test_time_steps,
                                              query_all_node=True, query_all_t=True,
                                              make_test_set=True
                                              )
                else:
                    test_data = dataset.query(seed=rseed + i * 10, num_graph=1, max_time_points=test_time_steps,
                                                  query_all_node=True, query_all_t=True,
                                                  N=N,
                                                  make_test_set=True,
                                                  )
                                                  
                print("%s/%s " % (i, test_num_trials))

                for bound_t_context in [0.5]:
                    # for bound_t_context in [0., 0.2, 0.4]:

                    set_rand_seed(rseed + i * 100)

                    gen_batch = make_batch(test_data, 1, is_shuffle=True, bound_t_context=bound_t_context,
                                           is_test=True,
                                           is_shuffle_target=False, max_x_dim=5)
                    for batch_data in gen_batch:
                        saved_test_set[(dynamics_name, topo_type, bound_t_context, i)] = batch_data

            # save test data
            import pickle

            fname = 'data/DynamicsData/test_data_on_dynamics_%s_topo_%s_dataset_x%s_numgraph%s_timestep%s_seed%s_num_nodes=%s_split_train_and_test.pickle' % (
                dynamics_name, topo_type, dataset.x_dim, 1, time_steps, rseed, N)
            f = open(fname, 'wb')
            pickle.dump(saved_test_set, f)
            f.close()
            
# test
if __name__ == '__main__':
    generate_testset_1()
    generate_testset_phototaxis_dynamics()
    generate_testset_brain()
    generate_testset_2_3()