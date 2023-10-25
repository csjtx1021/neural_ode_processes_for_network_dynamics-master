import random
import numpy as np
import logging
import os
import time
import warnings
import sys
import torch
import torch.nn as nn
import matplotlib
from load_dynamics_solution2and3 import *


def set_rand_seed(rseed):
    torch.manual_seed(rseed)  # 为CPU设置随机种子
    torch.cuda.manual_seed(rseed)  # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(rseed)  # 为所有GPU设置随机种子
    # random.seed(rseed)
    np.random.seed(rseed)

def get_mask_dim(dim, max_dim):
    assert dim <= max_dim
    res = []
    for i in range(max_dim):
        if i < dim:
            res.append(1)
        else:
            res.append(0)
    return np.array(res)


##=====================================================================
##
##                  make batch
##
##====================================================================

def make_batch(data, batch_size, is_shuffle=True, bound_t_context=None, is_test=False, is_shuffle_target=True, max_x_dim=None, max_num_obs=None):
    if max_x_dim is None:
        max_x_dim = x_dim
    tasks_data = data['tasks']
    if is_shuffle:
        points_data_index_shuffle = []
        for idx in range(len(data['tasks'])):
            points_data = tasks_data[idx]['points']
            index_shuffle = torch.randperm(len(points_data) - 1)
            index_shuffle = torch.cat([torch.tensor([0]).long(),
                                       index_shuffle + torch.tensor([1]).long()], dim=-1)
            points_data_index_shuffle.append(index_shuffle)
        tasks_data_index_shuffle = torch.randperm(len(tasks_data))
    else:
        points_data_index_shuffle = []
        for idx in range(len(data['tasks'])):
            points_data = tasks_data[idx]['points']
            index_shuffle = torch.linspace(0, len(points_data) - 1, len(points_data)).long()
            points_data_index_shuffle.append(index_shuffle)
        tasks_data_index_shuffle = torch.linspace(0, len(tasks_data) - 1, len(tasks_data)).long()

    # print(points_data_index_shuffle)
    # we got tasks_data_shuffle,
    # which is [(adj,
    #            x0,
    #            task_info,
    #            [{"t":, "x_self":, "mask":},...,{}]),
    #           (adj,
    #            x0,
    #            task_info,
    #            [{},...,{}]),
    #          ...]

    start_idx = 0
    while start_idx < len(tasks_data_index_shuffle):
        start_time = time.time()

        batch_adj = []  #
        batch_x0 = []  #
        batch_task_info = []  #

        contexts_batch_t = []
        contexts_batch_x_self = []
        contexts_batch_mask = []
        contexts_batch_point_info = []
        contexts_batch_adj = []
        contexts_batch_task_info = []

        targets_batch_t = []
        targets_batch_x_self = []
        targets_batch_mask = []
        targets_batch_point_info = []
        targets_batch_adj = []
        targets_batch_task_info = []

        for task_i_idx in tasks_data_index_shuffle[start_idx:start_idx + batch_size]:

            saved_obs_pool = []
            saved_unobs_pool = []

            # make batch for adj, x0 and task_info
            task_i_adj_in_batch = tasks_data[task_i_idx]['adj']
            task_i_x0_in_batch = tasks_data[task_i_idx]['X0']
            task_i_task_info_in_batch = tasks_data[task_i_idx]['task_info']
            task_i_points_in_batch = tasks_data[task_i_idx]['points']

            num_nodes_last = 0
            for x0_ in batch_x0:
                num_nodes_last += x0_.shape[0]
            if len(task_i_adj_in_batch) > 2:
                batch_adj.append(np.concatenate([task_i_adj_in_batch[:2, :] + num_nodes_last, task_i_adj_in_batch[2:,:]], axis=0))
            else:
                batch_adj.append(task_i_adj_in_batch + num_nodes_last)
            # padding zeros to x0
            task_i_x0_in_batch = np.concatenate([task_i_x0_in_batch, np.zeros((task_i_x0_in_batch.shape[0], max_x_dim - task_i_x0_in_batch.shape[-1]))],axis=-1)
            batch_x0.append(task_i_x0_in_batch)
            batch_task_info.append(task_i_task_info_in_batch + len(batch_task_info))

            # make context points and target points
            num_targets = len(task_i_points_in_batch)  ## number of targets = 50
            ## number of contexts, at least #nodes (i.e., the number of points with t=0)
            # num_contexts = np.random.randint(2, max(num_targets // 5, 4))  # [2, num_targets // 5)
            # num_contexts = np.random.randint(1, int(num_targets * 0.6))  # [2, num_targets // 5)
            num_contexts = num_targets - 1

            # num_contexts = num_targets
            contexts_index = points_data_index_shuffle[task_i_idx][:num_contexts]
            # print("context num = %s, target num = %s" % (num_contexts, num_targets))
            if is_shuffle_target:
                targets_index = points_data_index_shuffle[task_i_idx][:num_targets]
            else:
                targets_index = torch.linspace(0,
                                               len(task_i_points_in_batch) - 1,
                                               len(task_i_points_in_batch)).long()[:num_targets]
            # print("num_contexts=%s, num_targets=%s" % (num_contexts, num_targets))
            # assert num_contexts < num_targets

            contexts_index_selected = []
            contexts_point_mask = []
            for contexts_point_idx in contexts_index:
                contexts_point_i = task_i_points_in_batch[contexts_point_idx]
                if is_test and contexts_point_i['t'] > 0.:
                    num_nodes = contexts_point_i['mask'].shape[0]
                    if len(saved_obs_pool) < max_num_obs:
                        num_sampling_points_per_time = np.random.randint(1, int(num_nodes/2 + 1))  ## [1, N/2]
                        sampled_idxs = np.random.choice(a=list(range(num_nodes)), size=num_sampling_points_per_time,
                                                        replace=False)
                        if contexts_point_i['t'] <= bound_t_context:
                            if num_sampling_points_per_time + len(saved_obs_pool) > max_num_obs:
                                sampled_idxs = sampled_idxs[:max_num_obs - len(saved_obs_pool)]
                            unsampled_idxs = list(set(list(range(num_nodes)))-set(list(sampled_idxs)))
                            saved_obs_pool += [(len(contexts_index_selected), contexts_point_i['t'], n_i) for n_i in sampled_idxs]
                            saved_unobs_pool += [(len(contexts_index_selected), contexts_point_i['t'], n_i) for n_i in unsampled_idxs]

                        new_mask = np.zeros(num_nodes)
                        new_mask[sampled_idxs] = 1.
                        contexts_index_selected.append(contexts_point_idx)
                        contexts_point_mask.append(new_mask)
                else:
                    saved_obs_pool += [ (len(contexts_index_selected), contexts_point_i['t'], n_i) for n_i in range(contexts_point_i['mask'].shape[0])]
                    contexts_index_selected.append(contexts_point_idx)
                    contexts_point_mask.append(contexts_point_i['mask'])
                    
                # contexts_point_mask.append(contexts_point_i['mask'])
                
            # fill values
            #print('saved_unobs_pool=',saved_unobs_pool)
            
            if len(saved_obs_pool) < max_num_obs:
                sampled_fill = np.random.choice(a=list(range(len(saved_unobs_pool))), size=min(len(saved_unobs_pool), max_num_obs-len(saved_obs_pool)), replace=False)
                for sampled_fill_idx in sampled_fill:
                    idx, context_t_i, n_i = saved_unobs_pool[sampled_fill_idx]
                    contexts_point_mask[idx][n_i] = 1.

            # make contexts
            for idx in range(len(contexts_index_selected)):
                contexts_point_idx = contexts_index_selected[idx]
                contexts_point_i = task_i_points_in_batch[contexts_point_idx]
                if bound_t_context is not None:
                    if contexts_point_i['t'] > bound_t_context:
                        continue

                # print(contexts_point_i['t'])

                contexts_batch_t.append(torch.from_numpy(contexts_point_i['t']).float())
                # contexts_batch_x_self.append(torch.from_numpy(contexts_point_i['x_self']).float())
                # padding zeros to x_self
                contexts_batch_x_self.append(torch.from_numpy(np.concatenate(
                    [contexts_point_i['x_self'],
                     np.zeros((contexts_point_i['x_self'].shape[0], max_x_dim - contexts_point_i['x_self'].shape[-1]))],
                    axis=-1)).float())
                contexts_batch_mask.append(torch.from_numpy(contexts_point_mask[idx].reshape(-1,1) * get_mask_dim(contexts_point_i['x_self'].shape[-1], max_x_dim).reshape(1,-1)).long())
                contexts_batch_adj.append(torch.from_numpy(contexts_point_i['adj']))
                contexts_batch_point_info.append(torch.from_numpy(contexts_point_i['point_info']).long())
                contexts_batch_task_info.append(torch.tensor([len(batch_task_info) - 1]).long())

            if is_test:
                print("number of context observations = ",
                      sum([contexts_batch_mask[i].sum() for i in range(len(contexts_batch_mask))]).item())

            # make targets
            for targets_point_idx in targets_index:
                targets_point_i = task_i_points_in_batch[targets_point_idx]

                targets_batch_t.append(torch.from_numpy(targets_point_i['t']).float())
                # targets_batch_x_self.append(torch.from_numpy(targets_point_i['x_self']).float())
                # padding zeros to x_self
                targets_batch_x_self.append(torch.from_numpy(np.concatenate(
                    [targets_point_i['x_self'],
                     np.zeros((targets_point_i['x_self'].shape[0], max_x_dim - targets_point_i['x_self'].shape[-1]))],
                    axis=-1)).float())
                # targets_batch_mask.append(torch.from_numpy(targets_point_i['mask']).long())
                targets_batch_mask.append(torch.from_numpy(targets_point_i['mask'].reshape(-1,1) * get_mask_dim(targets_point_i['x_self'].shape[-1], max_x_dim).reshape(1,-1)).long())
                targets_batch_adj.append(torch.from_numpy(targets_point_i['adj']))
                targets_batch_point_info.append(torch.from_numpy(targets_point_i['point_info']).long())
                targets_batch_task_info.append(torch.tensor([len(batch_task_info) - 1]).long())

        batch_adj = np.concatenate(batch_adj, axis=-1)
        batch_x0 = np.concatenate(batch_x0, axis=0)
        batch_task_info = np.concatenate(batch_task_info, axis=0)

        start_idx = start_idx + batch_size
        batch_data = {"adj": torch.from_numpy(batch_adj),
                      "x0": torch.from_numpy(batch_x0).float(),
                      "task_info": torch.from_numpy(batch_task_info).long(),
                      "contexts": {"t": contexts_batch_t,
                                   "x_self": contexts_batch_x_self,
                                   "mask": contexts_batch_mask,
                                   "adj": contexts_batch_adj,
                                   "point_info": contexts_batch_point_info,
                                   "task_info": contexts_batch_task_info
                                   },
                      "targets": {"t": targets_batch_t,
                                  "x_self": targets_batch_x_self,
                                  "mask": targets_batch_mask,
                                  "adj": targets_batch_adj,
                                  "point_info": targets_batch_point_info,
                                  "task_info": targets_batch_task_info
                                  }
                      }
        print("make batch cost = %.2f" % (time.time() - start_time))
        yield batch_data



def generate_testset_1(sparsity=0.01):
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

                # for bound_t_context in [0., 0.25, 0.5, 0.75]:
                for bound_t_context in [0.5]:

                    set_rand_seed(rseed + i * 100)

                    gen_batch = make_batch(test_data, 1, is_shuffle=True, bound_t_context=bound_t_context, is_test=True,
                                           is_shuffle_target=False, max_x_dim=4, max_num_obs=round(N*test_time_steps*sparsity))
                    for batch_data in gen_batch:
                    
                        
                        saved_test_set[(dynamics_name, topo_type, bound_t_context, i)] = batch_data

            # save test data
            import pickle

            fname = 'data/DynamicsData/test_data_on_dynamics_%s_topo_%s_dataset_x%s_numgraph%s_timestep%s_seed%s_num_nodes=%s_sparsity=%s_split_train_and_test.pickle' % (
                dynamics_name, topo_type, dataset.x_dim, 1, time_steps, rseed,
                N,sparsity)
            f = open(fname, 'wb')
            pickle.dump(saved_test_set, f)
            f.close()



def generate_testset_2_3(sparsity=0.01):
    rseed = 666
    set_rand_seed(rseed)
    # N = 20
    # N = 30
    N = 200
    # N = -1
    # N = 50
    # N = 250

    is_shuffle = False

    test_num_trials = 20

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

                # for bound_t_context in [0., 0.25, 0.5, 0.75]:
                for bound_t_context in [0.75]:
                    # for bound_t_context in [0., 0.2, 0.4]:

                    set_rand_seed(rseed + i * 100)

                    gen_batch = make_batch(test_data, 1, is_shuffle=False, bound_t_context=bound_t_context,
                                           is_test=True,
                                           is_shuffle_target=False, max_x_dim=4, max_num_obs=round(N*test_time_steps*sparsity))
                    for batch_data in gen_batch:
                        saved_test_set[(dynamics_name, topo_type, bound_t_context, i)] = batch_data

            # save test data
            import pickle

            fname = 'data/DynamicsData/test_data_on_dynamics_%s_topo_%s_dataset_x%s_numgraph%s_timestep%s_seed%s_num_nodes=%s_sparsity=%s_split_train_and_test_is_shuffle%s.pickle' % (
                dynamics_name, topo_type, dataset.x_dim, 1, time_steps, rseed, N, sparsity,str(is_shuffle))
            f = open(fname, 'wb')
            pickle.dump(saved_test_set, f)
            f.close()


def generate_testset_RealEpidemicData(sparsity=0.01,case_no=1):
    rseed = 666
    set_rand_seed(rseed)
   

    is_shuffle = True

    test_num_trials = 150

    for dynamics_name in [
        'RealEpidemicData_mix',
        # 'RealEpidemicData',
        #'RealEpidemicData_12',
        #'RealEpidemicData_13',
        #'RealEpidemicData_23',
    ]:
        """
        if '12' in dynamics_name:
            N = 17
        elif '13' in dynamics_name:
            N = 5
        elif '23' in dynamics_name:
            N = 42
        else:
            print('ERROR unknow dynamics_name [%s]'%dynamics_name)
            exit(1)
        """
        N = 234
        for topo_type in [
            'power_law',
        ]:
            saved_test_set = {}
            
            if case_no == 1:
                time_steps = 6
                bound_t_context = 0.2 #0.9 # 5/6
            elif case_no == 2:
                time_steps = 10
                bound_t_context = 0.2 #0.5 #5/10
            elif case_no == 3:
                time_steps = 11
                bound_t_context = 0.43 #0.91 #10/11
            elif case_no == 4:
                time_steps = 20
                bound_t_context = 0.43 #0.5 #10/20
            x_dim = 1
            num_graphs = 1

            dataset = dynamics_dataset(dynamics_name, topo_type,
                                       num_graphs_samples=num_graphs, time_steps=time_steps, x_dim=x_dim,
                                       make_test_set=True)

            test_time_steps = time_steps

            for i in range(test_num_trials):
                if i < 50:
                    N = 130
                    if case_no == 1:
                        test_data = dataset.query(seed=rseed + i * 10, num_graph=1, max_time_points=test_time_steps,
                                                  query_all_node=True, query_all_t=True,
                                                  N=N,
                                                  make_test_set=True,
                                                  case_1_h1n1=i%50
                                                  )
                                                  
                    elif case_no == 2:
                        test_data = dataset.query(seed=rseed + i * 10, num_graph=1, max_time_points=test_time_steps,
                                                  query_all_node=True, query_all_t=True,
                                                  N=N,
                                                  make_test_set=True,
                                                  case_2_h1n1=i%50
                                                  )
                    elif case_no == 3:
                        test_data = dataset.query(seed=rseed + i * 10, num_graph=1, max_time_points=test_time_steps,
                                                  query_all_node=True, query_all_t=True,
                                                  N=N,
                                                  make_test_set=True,
                                                  case_3_h1n1=i%50
                                                  )
                    elif case_no == 4:
                        test_data = dataset.query(seed=rseed + i * 10, num_graph=1, max_time_points=test_time_steps,
                                                  query_all_node=True, query_all_t=True,
                                                  N=N,
                                                  make_test_set=True,
                                                  case_4_h1n1=i%50
                                                  )
                                                  
                elif i >=50 and i < 100:
                    N = 37
                    if case_no == 1:
                        test_data = dataset.query(seed=rseed + i * 10, num_graph=1, max_time_points=test_time_steps,
                                                  query_all_node=True, query_all_t=True,
                                                  N=N,
                                                  make_test_set=True,
                                                  case_1_sars=i%50
                                                  )
                                                  
                    elif case_no == 2:
                        test_data = dataset.query(seed=rseed + i * 10, num_graph=1, max_time_points=test_time_steps,
                                                  query_all_node=True, query_all_t=True,
                                                  N=N,
                                                  make_test_set=True,
                                                  case_2_sars=i%50
                                                  )
                    elif case_no == 3:
                        test_data = dataset.query(seed=rseed + i * 10, num_graph=1, max_time_points=test_time_steps,
                                                  query_all_node=True, query_all_t=True,
                                                  N=N,
                                                  make_test_set=True,
                                                  case_3_sars=i%50
                                                  )
                    elif case_no == 4:
                        test_data = dataset.query(seed=rseed + i * 10, num_graph=1, max_time_points=test_time_steps,
                                                  query_all_node=True, query_all_t=True,
                                                  N=N,
                                                  make_test_set=True,
                                                  case_4_sars=i%50
                                                  )
                elif i >=100 and i < 150:
                    N = 174
                    if case_no == 1:
                        test_data = dataset.query(seed=rseed + i * 10, num_graph=1, max_time_points=test_time_steps,
                                                  query_all_node=True, query_all_t=True,
                                                  N=N,
                                                  make_test_set=True,
                                                  case_1_covid=i%50
                                                  )
                                                  
                    elif case_no == 2:
                        test_data = dataset.query(seed=rseed + i * 10, num_graph=1, max_time_points=test_time_steps,
                                                  query_all_node=True, query_all_t=True,
                                                  N=N,
                                                  make_test_set=True,
                                                  case_2_covid=i%50
                                                  )
                    elif case_no == 3:
                        test_data = dataset.query(seed=rseed + i * 10, num_graph=1, max_time_points=test_time_steps,
                                                  query_all_node=True, query_all_t=True,
                                                  N=N,
                                                  make_test_set=True,
                                                  case_3_covid=i%50
                                                  )
                    elif case_no == 4:
                        test_data = dataset.query(seed=rseed + i * 10, num_graph=1, max_time_points=test_time_steps,
                                                  query_all_node=True, query_all_t=True,
                                                  N=N,
                                                  make_test_set=True,
                                                  case_4_covid=i%50
                                                  )
                
               
                       
                else:
                    print('ERROR: no case')
                    exit(1)

                print("%s/%s " % (i, test_num_trials))
                
                    

                if True:
                
                
                    # for bound_t_context in [0., 0.2, 0.4]:

                    set_rand_seed(rseed + i * 1000)

                    gen_batch = make_batch(test_data, 1, is_shuffle=is_shuffle, bound_t_context=bound_t_context,
                                           is_test=True,
                                           is_shuffle_target=False, max_x_dim=4, max_num_obs=round(N*time_steps*sparsity))
                    for batch_data in gen_batch:
                        
                        saved_test_set[(dynamics_name, topo_type, bound_t_context, i)] = batch_data
            
            N = 234
            # save test data
            import pickle

            fname = 'data/DynamicsData/test_data_on_dynamics_%s_topo_%s_dataset_x%s_numgraph%s_timestep%s_seed%s_num_nodes=%s_sparsity=%s_split_train_and_test_is_shuffle%s.pickle' % (
                dynamics_name, topo_type, dataset.x_dim, 1, time_steps, rseed, N, sparsity, str(is_shuffle))
            f = open(fname, 'wb')
            pickle.dump(saved_test_set, f)
            f.close()



         
# test
if __name__ == '__main__':
    # generating test sets for empirical systems
    for sparsity in [0.1, 0.2, 0.5]:
        for case_no in [1,2,3,4]:
           generate_testset_RealEpidemicData(sparsity, case_no)

    # generating test sets for testing sparsity
    for sparsity in [0.012, 0.014, 0.016, 0.018, 0.02, 0.03, 0.04, 0.05, 0.1]:
        generate_testset_1(sparsity)