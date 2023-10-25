import pickle
import torch
import scipy
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import fastdtw



def compute_metric(gt, pre_mean, pre_std=None):
    T, N, D = gt.shape

    metric = {}

    pearsonr_net = 0
    spearmanr_net = 0
    kendalltau_net = 0
    fastdtw_net = 0
    nl1_net = 0
    mse_net = 0
    negative_loglikelihood_net = 0
    for nn in range(N):
        pearsonr_per_node = 0
        spearmanr_per_node = 0
        kendalltau_per_node = 0
        fastdtw_per_node = 0
        nl1_per_node = 0
        mse_per_node = 0
        negative_loglikelihood_per_node = 0
        for dd in range(D):
            gt_ = gt[:, nn, dd]
            pre_mean_ = pre_mean[:, nn, dd]

            pearsonr = scipy.stats.pearsonr(gt_, pre_mean_)[0]
            spearmanr = scipy.stats.spearmanr(gt_, pre_mean_)[0]
            kendalltau = scipy.stats.kendalltau(gt_, pre_mean_)[0]

            if np.isnan(pearsonr):
                pearsonr = 0
            if np.isnan(spearmanr):
                spearmanr = 0
            if np.isnan(kendalltau):
                kendalltau = 0

            pearsonr_per_node += pearsonr
            spearmanr_per_node += spearmanr
            kendalltau_per_node += kendalltau
            
            fastdtw_per_node += fastdtw.fastdtw(pre_mean_, gt_)[0]

            nl1_per_node += np.mean(np.abs(pre_mean_ - gt_))
            #nl1_per_node += np.mean(np.abs(gt_ - pre_mean_)/np.array([max(i_, 1e-6) for i_ in np.abs(gt_)]))
            mse_per_node += np.mean((pre_mean_ - gt_) ** 2)

            if pre_std is not None:
                pre_std_ = pre_std[:, nn, dd]
                pre_dist = torch.distributions.Normal(torch.from_numpy(pre_mean_), torch.from_numpy(pre_std_))
                negative_loglikelihood_per_node += torch.mean(-pre_dist.log_prob(torch.from_numpy(gt_))).item()

        pearsonr_net += pearsonr_per_node / D
        spearmanr_net += spearmanr_per_node / D
        kendalltau_net += kendalltau_per_node / D
        
        fastdtw_net += fastdtw_per_node / D

        nl1_net += nl1_per_node / D
        mse_net += mse_per_node / D

        negative_loglikelihood_net += negative_loglikelihood_per_node / D

    pearsonr_net = pearsonr_net / N
    spearmanr_net = spearmanr_net / N
    kendalltau_net = kendalltau_net / N
    
    fastdtw_net = fastdtw_net / N

    nl1_net = nl1_net / N
    mse_net = mse_net / N

    negative_loglikelihood_net = negative_loglikelihood_net / N

    metric['pearsonr'] = pearsonr_net
    metric['spearmanr'] = spearmanr_net
    metric['Kendalltau'] = kendalltau_net
    metric['negative_loglikelihood'] = negative_loglikelihood_net
    metric['MAE'] = nl1_net
    metric['mse'] = mse_net
    
    metric['dtw'] = fastdtw_net

    # print(kendalltau_net)

    return metric
    


def get_ndcn_results(dynamics, topo, bound_t_context, idx, groundtruth):
    #### load ndcn results
    if 'brain' in dynamics or 'phototaxis' in dynamics:
        fname = 'compared_methods/ndcn_%s_on_%s_ndcn_norm_adj.pickle'%(dynamics, topo)
    else:
        fname = 'compared_methods/ndcn_all_%s_on_%s_ndcn_norm_adj.pickle'%(dynamics, topo)
                
    with open(fname, 'rb') as f:
        ndcn_results_data = pickle.load(f)
        ndcn_results_data_dict = {}
        for dd in ndcn_results_data:
            ndcn_results_data_dict[list(dd.keys())[0]] = list(dd.values())[0]
                        
        
    ndcn_results_data_one = ndcn_results_data_dict[(dynamics, topo, bound_t_context, idx)]
    predictions = ndcn_results_data_one['pred_y']
    
    # predictions [100,225]
    if len(predictions.shape) == 2:
        predictions = predictions.unsqueeze(-1)

    metric_1 = compute_metric(groundtruth.cpu().numpy()[:int(predictions.shape[0]/2)], predictions.cpu().numpy()[:int(predictions.shape[0]/2)], pre_std=None)
    metric_2 = compute_metric(groundtruth.cpu().numpy()[int(predictions.shape[0]/2):], predictions.cpu().numpy()[int(predictions.shape[0]/2):], pre_std=None)
    
    return metric_1, metric_2

def get_dnnd_results(dynamics, topo, bound_t_context, idx, groundtruth):
    
    if idx == 3 or idx == 5:
        idx = 4

    #### load dnnd results
    fname = 'compared_methods/dnnd_ijcai23_all_%s_on_%s_ndcn_norm_adj.pickle'%(dynamics, topo)
                
    with open(fname, 'rb') as f:
        ndcn_results_data = pickle.load(f)
        ndcn_results_data_dict = {}
        for dd in ndcn_results_data:
            ndcn_results_data_dict[list(dd.keys())[0]] = list(dd.values())[0]
                        
        
    ndcn_results_data_one = ndcn_results_data_dict[(dynamics, topo, bound_t_context, idx)]
    predictions = ndcn_results_data_one['pred_y']
    
    # predictions [100,225]
    if len(predictions.shape) == 2:
        predictions = predictions.unsqueeze(-1)

    metric_1 = compute_metric(groundtruth.cpu().numpy()[:int(predictions.shape[0]/2)], predictions.cpu().numpy()[:int(predictions.shape[0]/2)], pre_std=None)
    metric_2 = compute_metric(groundtruth.cpu().numpy()[int(predictions.shape[0]/2):], predictions.cpu().numpy()[int(predictions.shape[0]/2):], pre_std=None)
    
    return metric_1, metric_2
    
def get_lg_ode_results(dynamics, topo, bound_t_context, idx, groundtruth):
    if idx == 10:
        idx = 9 
    #### load lg_ode results
    fname = 'compared_methods/LG_ODE_%s_%s.pickle'%(dynamics, topo)
    with open(fname, 'rb') as f:
        ndcn_results_data = pickle.load(f)
        
        #print(ndcn_results_data[(dynamics, topo, bound_t_context, idx)]['pred_y'].size())

    predictions = torch.mean(ndcn_results_data[(dynamics, topo, bound_t_context, idx)]['pred_y'], dim=0)
    predictions = torch.mean(predictions, dim=-1, keepdim=True)
    predictions = torch.transpose(predictions, 0, 1)
    predictions = torch.cat([predictions, predictions[-1].unsqueeze(0)], dim=0)
    
    #print(predictions.size(),groundtruth.size())
    #exit(1)
    # predictions = ndcn_results_data_one['pred_y']
    
    # predictions [100,225]
    if len(predictions.shape) == 2:
        predictions = predictions.unsqueeze(-1)

    metric_1 = compute_metric(groundtruth.cpu().numpy()[:int(predictions.shape[0]/2)], predictions.cpu().numpy()[:int(predictions.shape[0]/2)], pre_std=None)
    metric_2 = compute_metric(groundtruth.cpu().numpy()[int(predictions.shape[0]/2):], predictions.cpu().numpy()[int(predictions.shape[0]/2):], pre_std=None)
    
    return metric_1, metric_2




def get_results(method,dynamics_name,topo_list,t,  x_dim, num_nodes, fname_tamplate, ndcn_or_dnnd='none'):

    deter_flag = True
    if 'wo_deterpath' in method:
        deter_flag = False

        
    MAE_1_list = []
    Kendalltau_1_list = []
    MAE_1_list_ndcn = []
    Kendalltau_1_list_ndcn = []
    MAE_2_list = []
    Kendalltau_2_list = []
    MAE_2_list_ndcn = []
    Kendalltau_2_list_ndcn = []
    for topo in topo_list:
        fname = fname_tamplate % (method,str(deter_flag),dynamics_name,topo,x_dim,epoch,t,num_nodes)
        with open(fname, 'rb') as f:
            saved_results_data = pickle.load(f)
        
        #for idx in range(len(saved_results_data['test_id'])):
        
        #idx_list = [0, 1, 2, 4, 6,7,8,9, 10, 11,12,13,14,15,16,17,18,19]
        idx_list = [0, 1, 2, 3, 4, 5, 6,7,8,9, 10, 11,12,13,14,15,16,17,18,19]
        for idx in idx_list:
            if idx not in list(range(len(saved_results_data['test_id']))):
                print('idx [%s] is not in idx_list'%idx)
                continue
    
            groundtruth = saved_results_data['groundtruth'][idx]
            predictions = saved_results_data['predictions'][idx]

            groundtruth = torch.transpose(groundtruth, 0, 1)
            predictions['mean'] = torch.transpose(predictions['mean'], 1, 2)
            predictions['std'] = torch.transpose(predictions['std'], 1, 2)
            
            
            observations = saved_results_data['observations'][idx]
            t_list_obs = observations['t']
            t_n_x_list = []
            for t_idx in range(len(t_list_obs)):
                for n_idx in range(observations['x_self'][t_idx].size(0)):
                    if observations['mask'][t_idx][n_idx].sum() >= 1:
                        t_n_x_list.append((t_list_obs[t_idx].numpy() * 100, n_idx, observations['x_self'][t_idx][n_idx][observations['mask'][t_idx][n_idx]==1].numpy()))
            
            print('#obs=',len(t_n_x_list))


            metric_1 = compute_metric(groundtruth.numpy()[:int(groundtruth.size(0)/2)], torch.mean(predictions['mean'], dim=0).numpy()[:int(groundtruth.size(0)/2)],
                                            pre_std=None)
            metric_2 = compute_metric(groundtruth.numpy()[int(groundtruth.size(0)/2):], torch.mean(predictions['mean'], dim=0).numpy()[int(groundtruth.size(0)/2):],
                                            pre_std=None)
                                            
            MAE_1_list.append(metric_1['MAE'])
            Kendalltau_1_list.append(metric_1['dtw'])   
            MAE_2_list.append(metric_2['MAE'])
            Kendalltau_2_list.append(metric_2['dtw'])     
            
            if ndcn_or_dnnd == 'ndcn':
                if 'all' == topo[:3]:
                    topo_ = topo[3:]
                else:
                    topo_ = topo
                metric_1_ndcn, metric_2_ndcn = get_ndcn_results(dynamics_name, topo_, t, idx, groundtruth)        
                MAE_1_list_ndcn.append(metric_1_ndcn['MAE'])
                Kendalltau_1_list_ndcn.append(metric_1_ndcn['dtw'])  
                MAE_2_list_ndcn.append(metric_2_ndcn['MAE'])
                Kendalltau_2_list_ndcn.append(metric_2_ndcn['dtw'])   
            elif ndcn_or_dnnd == 'dnnd': 
                if 'all' == topo[:3]:
                    topo_ = topo[3:]
                else:
                    topo_ = topo
                metric_1_ndcn, metric_2_ndcn = get_dnnd_results(dynamics_name, topo_, t, idx, groundtruth)        
                MAE_1_list_ndcn.append(metric_1_ndcn['MAE'])
                Kendalltau_1_list_ndcn.append(metric_1_ndcn['dtw'])  
                MAE_2_list_ndcn.append(metric_2_ndcn['MAE'])
                Kendalltau_2_list_ndcn.append(metric_2_ndcn['dtw'])    
            elif ndcn_or_dnnd == 'lg_ode': 
                if 'all' == topo[:3]:
                    topo_ = topo[3:]
                else:
                    topo_ = topo
                metric_1_ndcn, metric_2_ndcn = get_lg_ode_results(dynamics_name, topo_, t, idx, groundtruth)        
                MAE_1_list_ndcn.append(metric_1_ndcn['MAE'])
                Kendalltau_1_list_ndcn.append(metric_1_ndcn['dtw'])  
                MAE_2_list_ndcn.append(metric_2_ndcn['MAE'])
                Kendalltau_2_list_ndcn.append(metric_2_ndcn['dtw'])    
    
    if ndcn_or_dnnd == 'ndcn' or ndcn_or_dnnd == 'dnnd' or ndcn_or_dnnd == 'lg_ode':
        return MAE_1_list_ndcn, Kendalltau_1_list_ndcn, MAE_2_list_ndcn, Kendalltau_2_list_ndcn

    return MAE_1_list, Kendalltau_1_list, MAE_2_list, Kendalltau_2_list


def plot_onechangjing(dynamics_name, data):
    figsize = (6, 6)

    sns.set(context='notebook', style='whitegrid', font_scale=2, palette="pastel")
    plt.figure(figsize=figsize)

    size = 200
    plt.scatter(data['MAE'][0], data['Kendalltau'][0], c='k', s=size, marker='x', alpha=0.8, linewidths=3)
    plt.scatter(data['MAE'][1], data['Kendalltau'][1], c='b', s=size, marker='^', alpha=0.8, linewidths=1)
    plt.scatter(data['MAE'][2], data['Kendalltau'][2], c='c', s=size, marker='^', alpha=0.8, linewidths=1)
    plt.scatter(data['MAE'][3], data['Kendalltau'][3], c='tan', s=size, marker='v', alpha=0.8, linewidths=1)
    plt.scatter(data['MAE'][4], data['Kendalltau'][4], c='orange', s=size, marker='v', alpha=0.8, linewidths=1)
    plt.scatter(data['MAE'][5], data['Kendalltau'][5], c='g', s=size, marker='o', alpha=0.8, linewidths=1)
    plt.scatter(data['MAE'][6], data['Kendalltau'][6], c='m', s=size, marker='o', alpha=0.8, linewidths=1)
    plt.scatter(data['MAE'][7], data['Kendalltau'][7], c='r', s=300, marker='*', alpha=0.8, linewidths=1)

    plt.xlabel('MAE')
    plt.ylabel('Kendalltau')
    plt.legend(['NDCN', r'NP rel N', r'NP irrel N', r'NDP rel N', r'NDP irrel N', r'GraphNDP w/o ode', r'GraphNDP w/o z', r'GraphNDP'],
               fontsize=18)

    # sns.scatterplot(x="MAE", y="Kendalltau",
    #             hue="Methods",
    #             style='Methods',
    #             data=data,
    #             # markers=['x', '^', '^', 'v', 'v', 's', 's', 'p'],
    #             legend=None,
    #                 )

    plt.tight_layout()

    #plt.show()



if __name__ == '__main__':

    print('------------------------------')
    print("Ablation 1")
     
    t_list = [0.5]

    for dynamics_name in ['mutualistic_interaction_dynamics', 'SIS_Individual_dynamics', 'SIR_Individual_dynamics', 'SEIS_Individual_dynamics', 'brain_FitzHugh_Nagumo_dynamics', 'phototaxis_dynamics']:
        data = []
        
        epoch = 30
        if 'opinion' in dynamics_name:
                x_dim, num_nodes = 2, 200
                topo_list = ['small_world']
        elif 'SI_' in dynamics_name or 'SIS_' in dynamics_name:
                x_dim, num_nodes = 2, 200
                topo_list = ['power_law']
        elif 'SIR_' in dynamics_name or 'SEIS_' in dynamics_name:
                x_dim, num_nodes = 3, 200
                topo_list = ['power_law']
        elif 'SEIR_' in dynamics_name:
                x_dim, num_nodes = 4, 200
                topo_list = ['power_law']
        elif 'brain_' in dynamics_name:
                x_dim, num_nodes = 2, 200
                topo_list = ['power_law']
                epoch = 200
        elif 'phototaxis' in dynamics_name:
                x_dim, num_nodes = 5, 40
                topo_list = ['full_connected']
                epoch = 200
        else:
                x_dim, num_nodes = 1, 225
                topo_list = ['allgrid', 'allpower_law','allrandom','allsmall_world','allcommunity']
                epoch = 30
                            
        print('==========================    %s    =============================='%(dynamics_name))
        
        
        if dynamics_name == 'mutualistic_interaction_dynamics':
            print('**-----  LG-ODE  -----**')
            
            MAE_1_list = []
            Kendalltau_1_list = []
            MAE_2_list = []
            Kendalltau_2_list = []
            for t in t_list:
                
                MAE_1_list_, Kendalltau_1_list_, MAE_2_list_, Kendalltau_2_list_ = get_results('', dynamics_name, topo_list, t,  x_dim, num_nodes,  fname_tamplate = 'results/saved_test_results_GNDP_OneForAll%s_MLlossFalse_deter%s_uncerTrue_%s_%s_x%s_numgraph1000_timestep100_epoch%s_bound_t_context%s_seed1_num_nodes%s.pkl',ndcn_or_dnnd='lg_ode')
                
                MAE_1_list += MAE_1_list_
                Kendalltau_1_list += Kendalltau_1_list_
                MAE_2_list += MAE_2_list_
                Kendalltau_2_list += Kendalltau_2_list_
                
                print(MAE_1_list_)
                print(MAE_2_list_)
                    
                print('** MAE 1', t, np.median(MAE_1_list_), np.mean(MAE_1_list_), np.std(MAE_1_list_))
                print('** dtw 1', t, np.median(Kendalltau_1_list_), np.mean(Kendalltau_1_list_), np.std(Kendalltau_1_list_))
                print('** MAE 2', t, np.median(MAE_2_list_), np.mean(MAE_2_list_), np.std(MAE_2_list_))
                print('** dtw 2', t, np.median(Kendalltau_2_list_), np.mean(Kendalltau_2_list_), np.std(Kendalltau_2_list_))
                
            print(MAE_1_list)
            print(MAE_2_list)
            print(Kendalltau_1_list)
            print(Kendalltau_2_list)
                
            print('** MAE 1', t, np.median(MAE_1_list), np.mean(MAE_1_list), np.std(MAE_1_list))
            print('** dtw 1', t, np.median(Kendalltau_1_list), np.mean(Kendalltau_1_list), np.std(Kendalltau_1_list))
            print('** MAE 2', t, np.median(MAE_2_list), np.mean(MAE_2_list), np.std(MAE_2_list))
            print('** dtw 2', t, np.median(Kendalltau_2_list), np.mean(Kendalltau_2_list), np.std(Kendalltau_2_list))
            data.append([dynamics_name, 'LG-ODE', np.median(MAE_1_list), np.median(Kendalltau_1_list)])
        
        print('**-----  ndcn  -----**')
        
        MAE_1_list = []
        Kendalltau_1_list = []
        MAE_2_list = []
        Kendalltau_2_list = []
        for t in t_list:
            
            MAE_1_list_, Kendalltau_1_list_, MAE_2_list_, Kendalltau_2_list_ = get_results('', dynamics_name, topo_list, t,  x_dim, num_nodes,  fname_tamplate = 'results/saved_test_results_GNDP_OneForAll%s_MLlossFalse_deter%s_uncerTrue_%s_%s_x%s_numgraph1000_timestep100_epoch%s_bound_t_context%s_seed1_num_nodes%s.pkl',ndcn_or_dnnd='ndcn')
            
            MAE_1_list += MAE_1_list_
            Kendalltau_1_list += Kendalltau_1_list_
            MAE_2_list += MAE_2_list_
            Kendalltau_2_list += Kendalltau_2_list_
            
            print(MAE_1_list_)
            print(MAE_2_list_)
                
            print('** MAE 1', t, np.median(MAE_1_list_), np.mean(MAE_1_list_), np.std(MAE_1_list_))
            print('** dtw 1', t, np.median(Kendalltau_1_list_), np.mean(Kendalltau_1_list_), np.std(Kendalltau_1_list_))
            print('** MAE 2', t, np.median(MAE_2_list_), np.mean(MAE_2_list_), np.std(MAE_2_list_))
            print('** dtw 2', t, np.median(Kendalltau_2_list_), np.mean(Kendalltau_2_list_), np.std(Kendalltau_2_list_))
            
        print(MAE_1_list)
        print(MAE_2_list)
        print(Kendalltau_1_list)
        print(Kendalltau_2_list)
            
        print('** MAE 1', t, np.median(MAE_1_list), np.mean(MAE_1_list), np.std(MAE_1_list))
        print('** dtw 1', t, np.median(Kendalltau_1_list), np.mean(Kendalltau_1_list), np.std(Kendalltau_1_list))
        print('** MAE 2', t, np.median(MAE_2_list), np.mean(MAE_2_list), np.std(MAE_2_list))
        print('** dtw 2', t, np.median(Kendalltau_2_list), np.mean(Kendalltau_2_list), np.std(Kendalltau_2_list))
        data.append([dynamics_name, 'NDCN', np.median(MAE_1_list), np.median(Kendalltau_1_list)])
        
        print('**-----  dnnd  -----**')
        
        MAE_1_list = []
        Kendalltau_1_list = []
        MAE_2_list = []
        Kendalltau_2_list = []
        for t in t_list:
            
            MAE_1_list_, Kendalltau_1_list_, MAE_2_list_, Kendalltau_2_list_ = get_results('', dynamics_name, topo_list, t,  x_dim, num_nodes,  fname_tamplate = 'results/saved_test_results_GNDP_OneForAll%s_MLlossFalse_deter%s_uncerTrue_%s_%s_x%s_numgraph1000_timestep100_epoch%s_bound_t_context%s_seed1_num_nodes%s.pkl',ndcn_or_dnnd='dnnd')
            
            MAE_1_list += MAE_1_list_
            Kendalltau_1_list += Kendalltau_1_list_
            MAE_2_list += MAE_2_list_
            Kendalltau_2_list += Kendalltau_2_list_
            
            print(MAE_1_list_)
            print(MAE_2_list_)
                
            print('** MAE 1', t, np.median(MAE_1_list_), np.mean(MAE_1_list_), np.std(MAE_1_list_))
            print('** dtw 1', t, np.median(Kendalltau_1_list_), np.mean(Kendalltau_1_list_), np.std(Kendalltau_1_list_))
            print('** MAE 2', t, np.median(MAE_2_list_), np.mean(MAE_2_list_), np.std(MAE_2_list_))
            print('** dtw 2', t, np.median(Kendalltau_2_list_), np.mean(Kendalltau_2_list_), np.std(Kendalltau_2_list_))
            
        print(MAE_1_list)
        print(MAE_2_list)
        print(Kendalltau_1_list)
        print(Kendalltau_2_list)
            
        print('** MAE 1', t, np.median(MAE_1_list), np.mean(MAE_1_list), np.std(MAE_1_list))
        print('** dtw 1', t, np.median(Kendalltau_1_list), np.mean(Kendalltau_1_list), np.std(Kendalltau_1_list))
        print('** MAE 2', t, np.median(MAE_2_list), np.mean(MAE_2_list), np.std(MAE_2_list))
        print('** dtw 2', t, np.median(Kendalltau_2_list), np.mean(Kendalltau_2_list), np.std(Kendalltau_2_list))
        data.append([dynamics_name, 'DNND', np.median(MAE_1_list), np.median(Kendalltau_1_list)])
        


        method = ''
        print('**-----  GraphNDP %s  -----**'%method)
        MAE_1_list = []
        Kendalltau_1_list = []
        MAE_2_list = []
        Kendalltau_2_list = []
        for t in t_list:
            MAE_1_list_, Kendalltau_1_list_, MAE_2_list_, Kendalltau_2_list_ = get_results(method, dynamics_name, topo_list, t,  x_dim, num_nodes,   fname_tamplate = 'results/saved_test_results_GNDP_OneForAll%s_MLlossFalse_deter%s_uncerTrue_%s_%s_x%s_numgraph1000_timestep100_epoch%s_bound_t_context%s_seed1_num_nodes%s.pkl')
            
            MAE_1_list += MAE_1_list_
            Kendalltau_1_list += Kendalltau_1_list_
            MAE_2_list += MAE_2_list_
            Kendalltau_2_list += Kendalltau_2_list_
            
            print(MAE_1_list_)
            print(MAE_2_list_)
                
            print('** MAE 1', t, np.median(MAE_1_list_), np.mean(MAE_1_list_), np.std(MAE_1_list_))
            print('** dtw 1', t, np.median(Kendalltau_1_list_), np.mean(Kendalltau_1_list_), np.std(Kendalltau_1_list_))
            print('** MAE 2', t, np.median(MAE_2_list_), np.mean(MAE_2_list_), np.std(MAE_2_list_))
            print('** dtw 2', t, np.median(Kendalltau_2_list_), np.mean(Kendalltau_2_list_), np.std(Kendalltau_2_list_))
        
        print(MAE_1_list)
        print(MAE_2_list)
        print(Kendalltau_1_list)
        print(Kendalltau_2_list)
            
        print('** MAE 1', t, np.median(MAE_1_list), np.mean(MAE_1_list), np.std(MAE_1_list))
        print('** dtw 1', t, np.median(Kendalltau_1_list), np.mean(Kendalltau_1_list), np.std(Kendalltau_1_list))
        print('** MAE 2', t, np.median(MAE_2_list), np.mean(MAE_2_list), np.std(MAE_2_list))
        print('** dtw 2', t, np.median(Kendalltau_2_list), np.mean(Kendalltau_2_list), np.std(Kendalltau_2_list))
        data.append([dynamics_name, 'GraphNDP', np.median(MAE_1_list), np.median(Kendalltau_1_list)])
        
        if dynamics_name == 'mutualistic_interaction_dynamics':
            method = '_wo_ode_in_dec_l'
            print('**-----  %s  -----**'%method)
            MAE_1_list = []
            Kendalltau_1_list = []
            MAE_2_list = []
            Kendalltau_2_list = []
            for t in t_list:        
                MAE_1_list_, Kendalltau_1_list_, MAE_2_list_, Kendalltau_2_list_ = get_results(method, dynamics_name, topo_list, t,  x_dim, num_nodes,   fname_tamplate = 'results/saved_test_results_GNDP_OneForAll%s_MLlossFalse_deter%s_uncerTrue_%s_%s_x%s_numgraph1000_timestep100_epoch%s_bound_t_context%s_seed1_num_nodes%s.pkl')
                
                MAE_1_list += MAE_1_list_
                Kendalltau_1_list += Kendalltau_1_list_
                MAE_2_list += MAE_2_list_
                Kendalltau_2_list += Kendalltau_2_list_
                
                print(MAE_1_list_)
                print(MAE_2_list_)
                    
                print('** MAE 1', t, np.median(MAE_1_list_), np.mean(MAE_1_list_), np.std(MAE_1_list_))
                print('** dtw 1', t, np.median(Kendalltau_1_list_), np.mean(Kendalltau_1_list_), np.std(Kendalltau_1_list_))
                print('** MAE 2', t, np.median(MAE_2_list_), np.mean(MAE_2_list_), np.std(MAE_2_list_))
                print('** dtw 2', t, np.median(Kendalltau_2_list_), np.mean(Kendalltau_2_list_), np.std(Kendalltau_2_list_))
            
            print(MAE_1_list)
            print(MAE_2_list)
            print(Kendalltau_1_list)
            print(Kendalltau_2_list) 
                       
            print('** MAE 1', t, np.median(MAE_1_list), np.mean(MAE_1_list), np.std(MAE_1_list))
            print('** dtw 1', t, np.median(Kendalltau_1_list), np.mean(Kendalltau_1_list), np.std(Kendalltau_1_list))
            print('** MAE 2', t, np.median(MAE_2_list), np.mean(MAE_2_list), np.std(MAE_2_list))
            print('** dtw 2', t, np.median(Kendalltau_2_list), np.mean(Kendalltau_2_list), np.std(Kendalltau_2_list))
            data.append([dynamics_name, r'GraphNDP w/o ode', np.median(MAE_1_list), np.median(Kendalltau_1_list)])
    
            method = '_wo_z_in_dec_l'
            print('**-----  %s  -----**'%method)
            MAE_1_list = []
            Kendalltau_1_list = []
            MAE_2_list = []
            Kendalltau_2_list = []
            for t in t_list:          
                MAE_1_list_, Kendalltau_1_list_, MAE_2_list_, Kendalltau_2_list_  = get_results(method, dynamics_name, topo_list, t,  x_dim, num_nodes,   fname_tamplate = 'results/saved_test_results_GNDP_OneForAll%s_MLlossFalse_deter%s_uncerTrue_%s_%s_x%s_numgraph1000_timestep100_epoch%s_bound_t_context%s_seed1_num_nodes%s.pkl')
                
                MAE_1_list += MAE_1_list_
                Kendalltau_1_list += Kendalltau_1_list_
                MAE_2_list += MAE_2_list_
                Kendalltau_2_list += Kendalltau_2_list_
                
                print(MAE_1_list_)
                print(MAE_2_list_)
                    
                print('** MAE 1', t, np.median(MAE_1_list_), np.mean(MAE_1_list_), np.std(MAE_1_list_))
                print('** dtw 1', t, np.median(Kendalltau_1_list_), np.mean(Kendalltau_1_list_), np.std(Kendalltau_1_list_))
                print('** MAE 2', t, np.median(MAE_2_list_), np.mean(MAE_2_list_), np.std(MAE_2_list_))
                print('** dtw 2', t, np.median(Kendalltau_2_list_), np.mean(Kendalltau_2_list_), np.std(Kendalltau_2_list_))
                
            print(MAE_1_list)
            print(MAE_2_list)
            print(Kendalltau_1_list)
            print(Kendalltau_2_list)
        
            print('** MAE 1', t, np.median(MAE_1_list), np.mean(MAE_1_list), np.std(MAE_1_list))
            print('** dtw 1', t, np.median(Kendalltau_1_list), np.mean(Kendalltau_1_list), np.std(Kendalltau_1_list))
            print('** MAE 2', t, np.median(MAE_2_list), np.mean(MAE_2_list), np.std(MAE_2_list))
            print('** dtw 2', t, np.median(Kendalltau_2_list), np.mean(Kendalltau_2_list), np.std(Kendalltau_2_list))
            data.append([dynamics_name, r'GraphNDP w/o z', np.median(MAE_1_list), np.median(Kendalltau_1_list)])
            
            method = '_NP_compare_undepend_N'
            print('**-----  %s  -----**'%method)
            MAE_1_list = []
            Kendalltau_1_list = []
            MAE_2_list = []
            Kendalltau_2_list = []
            for t in t_list:          
                MAE_1_list_, Kendalltau_1_list_, MAE_2_list_, Kendalltau_2_list_ = get_results(method, dynamics_name, topo_list, t, x_dim, num_nodes,    fname_tamplate = 'results/saved_test_results%s_MLlossFalse_deter%s_uncerTrue_%s_%s_x%s_numgraph1000_timestep100_epoch%s_bound_t_context%s_seed1_num_nodes%s.pkl')
                
                MAE_1_list += MAE_1_list_
                Kendalltau_1_list += Kendalltau_1_list_
                MAE_2_list += MAE_2_list_
                Kendalltau_2_list += Kendalltau_2_list_
                
                print(MAE_1_list_)
                print(MAE_2_list_)
                    
                print('** MAE 1', t, np.median(MAE_1_list_), np.mean(MAE_1_list_), np.std(MAE_1_list_))
                print('** dtw 1', t, np.median(Kendalltau_1_list_), np.mean(Kendalltau_1_list_), np.std(Kendalltau_1_list_))
                print('** MAE 2', t, np.median(MAE_2_list_), np.mean(MAE_2_list_), np.std(MAE_2_list_))
                print('** dtw 2', t, np.median(Kendalltau_2_list_), np.mean(Kendalltau_2_list_), np.std(Kendalltau_2_list_))
            
            print(MAE_1_list)
            print(MAE_2_list)
            print(Kendalltau_1_list)
            print(Kendalltau_2_list)
        
            print('** MAE 1', t, np.median(MAE_1_list), np.mean(MAE_1_list), np.std(MAE_1_list))
            print('** dtw 1', t, np.median(Kendalltau_1_list), np.mean(Kendalltau_1_list), np.std(Kendalltau_1_list))
            print('** MAE 2', t, np.median(MAE_2_list), np.mean(MAE_2_list), np.std(MAE_2_list))
            print('** dtw 2', t, np.median(Kendalltau_2_list), np.mean(Kendalltau_2_list), np.std(Kendalltau_2_list))
            data.append([dynamics_name, r'NP irrel N', np.median(MAE_1_list), np.median(Kendalltau_1_list)])
    
            method = '_NP_compare_depend_N'
            print('**-----  %s  -----**'%method)
            MAE_1_list = []
            Kendalltau_1_list = []
            MAE_2_list = []
            Kendalltau_2_list = []
            for t in t_list:         
                MAE_1_list_, Kendalltau_1_list_, MAE_2_list_, Kendalltau_2_list_ = get_results(method, dynamics_name, topo_list, t, x_dim*num_nodes, num_nodes,    fname_tamplate = 'results/saved_test_results%s_MLlossFalse_deter%s_uncerTrue_%s_%s_x%s_numgraph1000_timestep100_epoch%s_bound_t_context%s_seed1_num_nodes%s.pkl')
                
                MAE_1_list += MAE_1_list_
                Kendalltau_1_list += Kendalltau_1_list_
                MAE_2_list += MAE_2_list_
                Kendalltau_2_list += Kendalltau_2_list_
                
                 
                print(MAE_1_list_)
                print(MAE_2_list_)
                    
                print('** MAE 1', t, np.median(MAE_1_list_), np.mean(MAE_1_list_), np.std(MAE_1_list_))
                print('** dtw 1', t, np.median(Kendalltau_1_list_), np.mean(Kendalltau_1_list_), np.std(Kendalltau_1_list_))
                print('** MAE 2', t, np.median(MAE_2_list_), np.mean(MAE_2_list_), np.std(MAE_2_list_))
                print('** dtw 2', t, np.median(Kendalltau_2_list_), np.mean(Kendalltau_2_list_), np.std(Kendalltau_2_list_))
                
            print(MAE_1_list)
            print(MAE_2_list)
            print(Kendalltau_1_list)
            print(Kendalltau_2_list)
        
            print('** MAE 1', t, np.median(MAE_1_list), np.mean(MAE_1_list), np.std(MAE_1_list))
            print('** dtw 1', t, np.median(Kendalltau_1_list), np.mean(Kendalltau_1_list), np.std(Kendalltau_1_list))
            print('** MAE 2', t, np.median(MAE_2_list), np.mean(MAE_2_list), np.std(MAE_2_list))
            print('** dtw 2', t, np.median(Kendalltau_2_list), np.mean(Kendalltau_2_list), np.std(Kendalltau_2_list))
            data.append([dynamics_name, r'NP rel N', np.median(MAE_1_list), np.median(Kendalltau_1_list)])
    
            method = '_NDP_compare_undepend_N'
            print('**-----  %s  -----**'%method)
            MAE_1_list = []
            Kendalltau_1_list = []
            MAE_2_list = []
            Kendalltau_2_list = []
            for t in t_list:     
                MAE_1_list_, Kendalltau_1_list_, MAE_2_list_, Kendalltau_2_list_ = get_results(method, dynamics_name, topo_list, t,  x_dim, num_nodes,   fname_tamplate = 'results/saved_test_results%s_MLlossFalse_deter%s_uncerTrue_%s_%s_x%s_numgraph1000_timestep100_epoch%s_bound_t_context%s_seed1_num_nodes%s.pkl')
                
                MAE_1_list += MAE_1_list_
                Kendalltau_1_list += Kendalltau_1_list_
                MAE_2_list += MAE_2_list_
                Kendalltau_2_list += Kendalltau_2_list_
                
                 
                print(MAE_1_list_)
                print(MAE_2_list_)
                    
                print('** MAE 1', t, np.median(MAE_1_list_), np.mean(MAE_1_list_), np.std(MAE_1_list_))
                print('** dtw 1', t, np.median(Kendalltau_1_list_), np.mean(Kendalltau_1_list_), np.std(Kendalltau_1_list_))
                print('** MAE 2', t, np.median(MAE_2_list_), np.mean(MAE_2_list_), np.std(MAE_2_list_))
                print('** dtw 2', t, np.median(Kendalltau_2_list_), np.mean(Kendalltau_2_list_), np.std(Kendalltau_2_list_))
               
            print(MAE_1_list)
            print(MAE_2_list)
            print(Kendalltau_1_list)
            print(Kendalltau_2_list)
         
            print('** MAE 1', t, np.median(MAE_1_list), np.mean(MAE_1_list), np.std(MAE_1_list))
            print('** dtw 1', t, np.median(Kendalltau_1_list), np.mean(Kendalltau_1_list), np.std(Kendalltau_1_list))
            print('** MAE 2', t, np.median(MAE_2_list), np.mean(MAE_2_list), np.std(MAE_2_list))
            print('** dtw 2', t, np.median(Kendalltau_2_list), np.mean(Kendalltau_2_list), np.std(Kendalltau_2_list))
            data.append([dynamics_name, r'NDP irrel N', np.median(MAE_1_list), np.median(Kendalltau_1_list)])
    
            method = '_NDP_compare_depend_N'
            print('**-----  %s  -----**'%method)
            MAE_1_list = []
            Kendalltau_1_list = []
            MAE_2_list = []
            Kendalltau_2_list = []
            for t in t_list:     
                MAE_1_list_, Kendalltau_1_list_, MAE_2_list_, Kendalltau_2_list_ = get_results(method, dynamics_name, topo_list, t,x_dim*num_nodes, num_nodes,     fname_tamplate = 'results/saved_test_results%s_MLlossFalse_deter%s_uncerTrue_%s_%s_x%s_numgraph1000_timestep100_epoch%s_bound_t_context%s_seed1_num_nodes%s.pkl')
                
                MAE_1_list += MAE_1_list_
                Kendalltau_1_list += Kendalltau_1_list_
                MAE_2_list += MAE_2_list_
                Kendalltau_2_list += Kendalltau_2_list_
                
                 
                print(MAE_1_list_)
                print(MAE_2_list_)
                    
                print('** MAE 1', t, np.median(MAE_1_list_), np.mean(MAE_1_list_), np.std(MAE_1_list_))
                print('** dtw 1', t, np.median(Kendalltau_1_list_), np.mean(Kendalltau_1_list_), np.std(Kendalltau_1_list_))
                print('** MAE 2', t, np.median(MAE_2_list_), np.mean(MAE_2_list_), np.std(MAE_2_list_))
                print('** dtw 2', t, np.median(Kendalltau_2_list_), np.mean(Kendalltau_2_list_), np.std(Kendalltau_2_list_))
             
            print(MAE_1_list)
            print(MAE_2_list)
            print(Kendalltau_1_list)
            print(Kendalltau_2_list)
           
            print('** MAE 1', t, np.median(MAE_1_list), np.mean(MAE_1_list), np.std(MAE_1_list))
            print('** dtw 1', t, np.median(Kendalltau_1_list), np.mean(Kendalltau_1_list), np.std(Kendalltau_1_list))
            print('** MAE 2', t, np.median(MAE_2_list), np.mean(MAE_2_list), np.std(MAE_2_list))
            print('** dtw 2', t, np.median(Kendalltau_2_list), np.mean(Kendalltau_2_list), np.std(Kendalltau_2_list))
            data.append([dynamics_name, r'NDP rel N', np.median(MAE_1_list_), np.median(Kendalltau_1_list_)])
            
            print('------------------------------')
    
            df = pd.DataFrame(data, columns=['dynamics_name', 'Methods', 'MAE', 'Kendalltau'], dtype=float)
