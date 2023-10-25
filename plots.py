import numpy as np
import torch
import scipy
import seaborn as sns
import pickle
import pandas as pd
from matplotlib import pyplot as plt

color_list = [(0.5, 0.5, 0.5),
              (0.3, 0.3, 0.7),
              #(0.55, 0.55, 1.0),
              #(1.0, 0.55, 0.55),
              (0.3, 0.7, 0.3),
              (0.7, 0.3, 0.3)]


def compute_metric(gt, pre_mean, pre_std=None):
    T, N, D = gt.shape

    metric = {}

    pearsonr_net = 0
    spearmanr_net = 0
    kendalltau_net = 0
    nl1_net = 0
    mse_net = 0
    negative_loglikelihood_net = 0
    for nn in range(N):
        pearsonr_per_node = 0
        spearmanr_per_node = 0
        kendalltau_per_node = 0
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

            nl1_per_node += np.mean(np.abs(pre_mean_ - gt_))
            mse_per_node += np.mean((pre_mean_ - gt_) ** 2)

            if pre_std is not None:
                pre_std_ = pre_std[:, nn, dd]
                pre_dist = torch.distributions.Normal(torch.from_numpy(pre_mean_), torch.from_numpy(pre_std_))
                negative_loglikelihood_per_node += torch.mean(-pre_dist.log_prob(torch.from_numpy(gt_))).item()

        pearsonr_net += pearsonr_per_node / D
        spearmanr_net += spearmanr_per_node / D
        kendalltau_net += kendalltau_per_node / D

        nl1_net += nl1_per_node / D
        mse_net += mse_per_node / D

        negative_loglikelihood_net += negative_loglikelihood_per_node / D

    pearsonr_net = pearsonr_net / N
    spearmanr_net = spearmanr_net / N
    kendalltau_net = kendalltau_net / N

    nl1_net = nl1_net / N
    mse_net = mse_net / N

    negative_loglikelihood_net = negative_loglikelihood_net / N

    metric['pearsonr'] = pearsonr_net
    metric['spearmanr'] = spearmanr_net
    metric['Kendalltau'] = kendalltau_net
    metric['negative_loglikelihood'] = negative_loglikelihood_net
    metric['MAE'] = nl1_net
    metric['mse'] = mse_net

    # print(kendalltau_net)

    return metric


def plot_functions(ax, target_x, target_y, context_x, context_y, pred_y, std, is_2D=False):
    """Plots the predicted mean and variance and the context points.

  Args:
    target_x: An array of shape [B,num_targets,1] that contains the
        x values of the target points.
    target_y: An array of shape [B,num_targets,1] that contains the
        y values of the target points.
    context_x: An array of shape [B,num_contexts,1] that contains
        the x values of the context points.
    context_y: An array of shape [B,num_contexts,1] that contains
        the y values of the context points.
    pred_y: An array of shape [B,num_targets,1] that contains the
        predicted means of the y values at the target points in target_x.
    std: An array of shape [B,num_targets,1] that contains the
        predicted std dev of the y values at the target points in target_x.
      """
    # Plot everything
    if not is_2D:
        if len(target_y.size()) == 3:

            draw_target_x_sorted, draw_target_sorted_index = torch.sort(target_x[0], dim=0)
            draw_context_x_sorted, draw_context_sorted_index = torch.sort(context_x[0], dim=0)

            # groundtruth
            ax.plot(draw_target_x_sorted, target_y[0][draw_target_sorted_index.view(-1), :], 'k:', linewidth=1)

            # observations
            ax.plot(draw_context_x_sorted, context_y[0][draw_context_sorted_index.view(-1), :], 'kx', markersize=10)

            # predictions
            for i in range(pred_y.size(0)):
                ax.plot(draw_target_x_sorted, pred_y[i, 0, draw_target_sorted_index.view(-1), :], 'b', linewidth=1,
                        alpha=0.1)

                ax.fill_between(
                    draw_target_x_sorted[:, 0],
                    pred_y[i, 0, draw_target_sorted_index.view(-1), 0] - std[
                        i, 0, draw_target_sorted_index.view(-1), 0],
                    pred_y[i, 0, draw_target_sorted_index.view(-1), 0] + std[
                        i, 0, draw_target_sorted_index.view(-1), 0],
                    # alpha=0.05,
                    alpha=0.05,
                    facecolor='b',
                    interpolate=True)

        elif len(target_y.size()) == 4:
            for j in range(target_x.size(0)):
                draw_target_x_sorted, draw_target_sorted_index = torch.sort(target_x[j][0], dim=0)
                draw_context_x_sorted, draw_context_sorted_index = torch.sort(context_x[j][0], dim=0)

                # groundtruth
                ax.plot(draw_target_x_sorted, target_y[j][0][draw_target_sorted_index.view(-1), :], 'k:', linewidth=1,
                        alpha=0.2)

                # observations
                # ax.plot(draw_context_x_sorted, context_y[j][0][draw_context_sorted_index.view(-1), :], 'kx', markersize=10)

                # predictions
                i = j
                ax.plot(draw_target_x_sorted, pred_y[i, 0, draw_target_sorted_index.view(-1), :], 'b', linewidth=1,
                        alpha=0.2)

                ax.fill_between(
                    draw_target_x_sorted[:, 0],
                    pred_y[i, 0, draw_target_sorted_index.view(-1), 0] - std[
                        i, 0, draw_target_sorted_index.view(-1), 0],
                    pred_y[i, 0, draw_target_sorted_index.view(-1), 0] + std[
                        i, 0, draw_target_sorted_index.view(-1), 0],
                    # alpha=0.05,
                    alpha=0.05,
                    facecolor='b',
                    interpolate=True)
        else:
            print('wrong dim for target_y!!!')
            exit(1)



    else:
        # 2D draw
        # groundtruth

        draw_target_x = target_x[0]
        draw_target_y = target_y[0]
        draw_context_x = context_x[0]
        draw_context_y = context_y[0]

        n = int(torch.sqrt(torch.tensor(draw_target_x.size(0))).item())
        X = draw_target_x[:, 0].view(n, n)
        Y = draw_target_x[:, 1].view(n, n)
        Z = draw_target_y.view(n, n)
        ax0 = ax[0].contourf(X, Y, Z, 25, cmap='jet', vmin=-2, vmax=2)
        # ax[0].contour(X, Y, Z)

        # predictions
        Z = pred_y[0, 0, :, :].view(n, n)
        ax1 = ax[1].contourf(X, Y, Z, 25, cmap='jet', vmin=-2, vmax=2)

        Z = std[0, 0, :, :].view(n, n)
        ax2 = ax[2].contourf(X, Y, Z, 25, cmap='jet', vmin=0, vmax=3)

        # observations
        ax[0].plot(draw_context_x[:, 0], draw_context_x[:, 1], 'kx', markersize=10)
        ax[1].plot(draw_context_x[:, 0], draw_context_x[:, 1], 'kx', markersize=10)
        ax[2].plot(draw_context_x[:, 0], draw_context_x[:, 1], 'kx', markersize=10)

        return [ax0, ax1, ax2]

    # Make the plot pretty
    # plt.yticks([-6, 0, 6], fontsize=16)
    # plt.xticks([-6, 0, 6], fontsize=16)
    # plt.ylim([-6, 6])
    ax.grid(False)
    # ax = plt.gca()


def plot_violinplot(save_fig=True, add_str='', plot_type='violin', dynamics_list=[], topo_list=[],
                    bound_t_context_list=[], test_N=-1, test_num_trials=10,
                    exp_type='3dynamics5topo_onedynamics_onetopo', ndcn_flag=False,
                    x_dim=1,
                    train_topo=''):
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd
    import pickle

    figsize = (6, 5.2)

    # sns.set_theme(style="whitegrid", palette="pastel")
    sns.set(context='notebook', style='whitegrid', font_scale=2, palette="pastel")

    data_ndcn_all_dynamics = []
    data_all_dynamics = []

    for dynamics in dynamics_list:
        data_ndcn = []
        data = []
        for topo in topo_list:
        
            if ndcn_flag:
                #### load ndcn results
                fname = 'compared_methods/ndcn_all_%s_on_%s_ndcn_norm_adj.pickle'%(dynamics, topo)
                
        
                with open(fname, 'rb') as f:
                    ndcn_results_data = pickle.load(f)
                    ndcn_results_data_dict = {}
                    for dd in ndcn_results_data:
                        ndcn_results_data_dict[list(dd.keys())[0]] = list(dd.values())[0]

            for bound_t_context in bound_t_context_list:

                if exp_type == '3dynamics5topo_onedynamics_onetopo':
                    fname = 'results/saved_test_results_%s_MLlossFalse_deterTrue_uncerTrue_%s_%s_x1_numgraph1000_timestep100_epoch30_bound_t_context%s_seed1_num_nodes%s.pkl' % (
                        model_name, dynamics, topo, bound_t_context, test_N)
                elif exp_type == '3dynamics5topo_onedynamics_alltopo':
                    fname = 'results/saved_test_results_%s_MLlossFalse_deterTrue_uncerTrue_%s_all%s_x1_numgraph1000_timestep100_epoch20_bound_t_context%s_seed1_num_nodes%s.pkl' % (
                        model_name, dynamics, topo, bound_t_context, test_N)
                elif exp_type == '3dynamics5topo_alldynamics_alltopo':
                    fname = 'results/saved_test_results_%s_MLlossFalse_deterTrue_uncerTrue_all%s_all%s_x1_numgraph1000_timestep100_epoch20_bound_t_context%s_seed1_num_nodes%s.pkl' % (
                        model_name, dynamics, topo, bound_t_context, test_N)
                elif exp_type == '3dynamics5topo_onedynamics_onetopo_difftopo':
                    fname = 'results/saved_test_results_%s_MLlossFalse_deterTrue_uncerTrue_%s_%s%s_x1_numgraph1000_timestep100_epoch30_bound_t_context%s_seed1_num_nodes%s.pkl' % (
                        model_name, dynamics, train_topo, topo, bound_t_context, test_N)

                elif exp_type == '5dynamics1topo_onedynamics_onetopo_all_epidemic':
                    # fname = 'results/saved_test_results_GNDP_MLlossFalse_deterTrue_uncerTrue_all_epidemic%s_%s_x4_numgraph1000_timestep100_epoch20_bound_t_context%s_seed1_num_nodes%s.pkl' % (
                    #     dynamics, topo, bound_t_context, test_N)
                    # fname = 'results/saved_test_results_GNDP_MLlossFalse_deterTrue_uncerTrue_all_epidemic%s_%s_x4_numgraph1000_timestep100_epoch40_bound_t_context%s_seed1_num_nodes%s.pkl' % (
                    #     dynamics, topo, bound_t_context, test_N)
                    # fname = 'results/saved_test_results_GNDP_MLlossFalse_deterTrue_uncerTrue_all_epidemic%s_%s_x4_numgraph1000_timestep100_epoch60_bound_t_context%s_seed1_num_nodes%s.pkl' % (
                    #     dynamics, topo, bound_t_context, test_N)
                    if 'SI_' in dynamics or 'SIS_' in dynamics:
                        x_dim_ = 2
                    elif 'SIR_' in dynamics or 'SEIS_' in dynamics:
                        x_dim_ = 3
                    else:
                        x_dim_ = 4
                    fname = 'results/saved_test_results_%s_MLlossFalse_deterTrue_uncerTrue_%s_%s_x%s_numgraph1000_timestep100_epoch30_bound_t_context%s_seed1_num_nodes%s.pkl' % (
                        model_name, dynamics, topo, x_dim_,bound_t_context, test_N)
                    # fname = 'results/saved_test_results_GNDP_MLlossFalse_deterTrue_uncerTrue_all_epidemic%s_%s_x4_numgraph1000_timestep100_epoch40_bound_t_context%s_seed1_num_nodes%s_True.pkl' % (
                    #     dynamics, topo, bound_t_context, test_N)
                else:
                    # fname = 'results/saved_test_results_GNDP_MLlossFalse_deterTrue_uncerTrue_%s_%s_x1_numgraph1000_timestep100_epoch30_bound_t_context%s_seed1_num_nodes%s.pkl' % (
                    #     dynamics + dynamics, topo + topo, bound_t_context, test_N)
                    # fname = 'results/saved_test_results_GNDP_MLlossFalse_deterTrue_uncerTrue_%s_%s_x1_numgraph1000_timestep100_epoch50_bound_t_context%s_seed1_num_nodes-1.pkl' % (
                    #     dynamics + dynamics, topo + topo, bound_t_context)
                    # fname = 'results/saved_test_results_GNDP_MLlossFalse_deterTrue_uncerTrue_%s_%s_x1_numgraph1000_timestep100_epoch50_bound_t_context%s_seed1_num_nodes%s.pkl' % (
                    #     dynamics + dynamics, topo + topo, bound_t_context, test_N)

                    # fname = 'results/saved_test_results_GNDP_MLlossFalse_deterTrue_uncerTrue_%s_%s_x1_numgraph400_timestep100_epoch20_bound_t_context%s_seed1_num_nodes%s.pkl' % (
                    #     dynamics + dynamics, topo + topo, bound_t_context, test_N)
                    # fname = 'results/saved_test_results_GNDP_MLlossFalse_deterTrue_uncerTrue_%s_%s_x1_numgraph5000_timestep100_epoch20_bound_t_context%s_seed1_num_nodes%s.pkl' % (
                    #     dynamics + dynamics, topo + topo, bound_t_context, test_N)
                    # fname = 'results/saved_test_results_%s_MLlossFalse_deterTrue_uncerTrue_%s_%s_x%s_numgraph1000_timestep100_epoch60_bound_t_context%s_seed1_num_nodes%s.pkl' % (
                    #     model_name, dynamics, topo, x_dim, bound_t_context, test_N)
                    fname = 'results/saved_test_results_%s_MLlossFalse_deterTrue_uncerTrue_%s_%s_x%s_numgraph1000_timestep100_epoch30_bound_t_context%s_seed1_num_nodes%s_with_2nd_phase.pkl' % (
                        model_name, dynamics, topo, x_dim, bound_t_context, test_N)

                with open(fname, 'rb') as f:
                    saved_results_data = pickle.load(f)
                print('hhah')
                groundtruth_saved = []
                for idx in range(len(saved_results_data['test_id']) - test_num_trials,
                                 len(saved_results_data['test_id'])):
                    test_id = saved_results_data['test_id'][idx]

                    observations = saved_results_data['observations'][idx]
                    groundtruth = saved_results_data['groundtruth'][idx]
                    # groundtruth_sum = saved_results_data['groundtruth_sum'][idx].view(-1)
                    predictions = saved_results_data['predictions'][idx]
                    # predictions_sum = saved_results_data['predictions_sum'][idx].view(-1, len(groundtruth_sum))
                    # nl1_error = saved_results_data['nl1_error'][idx]
                    # l2_error = saved_results_data['l2_error'][idx]

                    groundtruth = torch.transpose(groundtruth, 0, 1)
                    predictions['mean'] = torch.transpose(predictions['mean'], 1, 2)
                    predictions['std'] = torch.transpose(predictions['std'], 1, 2)

                    groundtruth_saved.append(groundtruth)

                    metric = compute_metric(groundtruth.numpy(), torch.mean(predictions['mean'], dim=0).numpy(),
                                            pre_std=None)

                    data.append([metric['MAE'], metric['Kendalltau'], 'GMNND', bound_t_context, dynamics, topo])
                    data_all_dynamics.append(
                        [metric['MAE'], metric['Kendalltau'], 'GMNND', bound_t_context, dynamics.split('_')[0],
                         topo])

                    if idx in [0, 1, 2, 3, 4]:
                        print("************run_idx=%s, Ours (%s-%s-%s), MAE=%s, Kendalltau=%s" % (
                        idx, dynamics, topo, bound_t_context, metric['MAE'], metric['Kendalltau']))

                if ndcn_flag:
                    # get ndcn results data

                    for idx in range(test_num_trials):
                        ndcn_results_data_one = ndcn_results_data_dict[(dynamics, topo, bound_t_context, idx)]

                        predictions = ndcn_results_data_one['pred_y']
                        predictions_sum = torch.sum(ndcn_results_data_one['pred_y'], dim=1)
                        #nl1_error = ndcn_results_data_one['normalized_l1'].item()
                        #l2_error = ndcn_results_data_one['mse'].item()

                        groundtruth = groundtruth_saved[idx]

                        # predictions [100,225]
                        if len(predictions.shape) == 2:
                            predictions = predictions.unsqueeze(-1)

                        metric = compute_metric(groundtruth.cpu().numpy(), predictions.cpu().numpy(), pre_std=None)

                        data_ndcn.append(
                            [metric['MAE'], metric['Kendalltau'], 'NDCN', bound_t_context, dynamics.split('_')[0],
                             topo])
                        data_ndcn_all_dynamics.append(
                            [metric['MAE'], metric['Kendalltau'], 'NDCN', bound_t_context, dynamics.split('_')[0],
                             topo])
                        if idx in [0, 1, 2, 3, 4]:
                            print("************run_idx=%s, NDCN (%s-%s-%s), MAE=%s, Kendalltau=%s" % (
                                idx, dynamics, topo, bound_t_context, metric['MAE'], metric['Kendalltau']))

    df_ndcn_all_dynamics = pd.DataFrame(data_ndcn_all_dynamics,
                                        columns=['MAE', 'Kendalltau', 'Methods', 'bound_t_context',
                                                 'Dynamics types', 'Topology types'],
                                        dtype=float)

    df_all_dynamics = pd.DataFrame(data_all_dynamics,
                                   columns=['MAE', 'Kendalltau', 'Methods', 'bound_t_context', 'Dynamics types',
                                            'Topology types'],
                                   dtype=float)

    df_all_all_dynamics = pd.concat([df_all_dynamics, df_ndcn_all_dynamics], axis=0)

    if dynamics == 'opinion_dynamics_Baumann2021_2topic':
        exp_name = 'EXP2_opinion_dynamics'
    elif dynamics == 'SI_Individual_dynamics' or \
            dynamics == 'SIS_Individual_dynamics' or \
            dynamics == 'SIR_Individual_dynamics' or \
            dynamics == 'SEIS_Individual_dynamics' or dynamics == 'SEIR_Individual_dynamics' or dynamics == 'Coupled_Epidemic_dynamics':
        exp_name = 'EXP3_all_epidemic'
    elif dynamics == 'SIR_meta_pop_dynamics':
        exp_name = 'EXP4_real_epidemic'
    else:
        print('unknown dynamics [%s]' % dynamics)

    for key_metric in ['MAE', 'Kendalltau']:
        # Draw
        plt.figure(figsize=figsize)
        if plot_type == 'box':
            color_palette = sns.color_palette(
                sns.color_palette('pastel')[2:3] + sns.color_palette('pastel')[3:4])
            sns.boxplot(x="Methods", y=key_metric, showfliers=False,
                        # hue="Methods",
                        data=df_all_all_dynamics,
                        palette=color_palette
                        )
            sns.stripplot(x="Methods", y=key_metric,
                          data=df_all_all_dynamics,
                          size=5, linewidth=0, alpha=0.5,
                          # color=".3",
                          hue='bound_t_context',
                          legend=False,
                          palette=sns.color_palette(color_list)
                          )

        else:
            sns.violinplot(x="Methods", y=key_metric,
                           # hue="Methods",
                           data=df_all_all_dynamics)

        # add mid line for each context_t_bound
        idx_t = 0
        for bound_t_context in bound_t_context_list:
            if bound_t_context == 0:
                idx_t = 0
            else:
                idx_t += 1
            sns.boxplot(x="Methods", y=key_metric, showfliers=False, showcaps=False,
                        whiskerprops={'color': 'w', 'alpha': 0},
                        # hue="Methods",
                        linewidth=2,
                        data=df_all_all_dynamics[df_all_all_dynamics['bound_t_context'] == bound_t_context],
                        boxprops={"facecolor": (1, 1, 1, 1), 'alpha': 0},
                        medianprops={'marker': '>', 'markevery': 2, 'markersize': 10, 'markeredgecolor': 'w',
                                     'linestyle': 'none', "color": color_list[idx_t], 'alpha': 0.9},
                        )

        if dynamics == 'heat_diffusion_dynamics':
            plt.ylim(0, 60)
        elif dynamics == 'mutualistic_interaction_dynamics':
            plt.ylim(0, 100)
        elif dynamics == 'gene_regulatory_dynamics':
            plt.ylim(0, 200)
        elif dynamics == 'opinion_dynamics':
            plt.ylim(0, 10)
        elif dynamics == 'opinion_dynamics_Baumann2021':
            plt.ylim(0, 100)
        elif dynamics == 'opinion_dynamics_Baumann2021_2topic':
            plt.ylim(0, 3)
            if key_metric == 'MAE':
                import matplotlib.ticker as mtick
                plt.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))

        elif dynamics == 'SI_Individual_dynamics' or \
                dynamics == 'SIS_Individual_dynamics' or \
                dynamics == 'SIR_Individual_dynamics' or \
                dynamics == 'SEIS_Individual_dynamics' or dynamics == 'SEIR_Individual_dynamics':
            plt.ylim(0, 5)
            if key_metric == 'MAE':
                import matplotlib.ticker as mtick
                plt.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
        else:
            plt.ylim(0, 10)

        if key_metric == 'Kendalltau':
            plt.ylim(-1, 1)

        xticks = plt.gca().get_xticklabels()
        plt.gca().set_xticklabels(xticks, rotation=0)
        plt.tight_layout()

        #
        if save_fig:
            plt.savefig(
                'results/%s_%s_%s_all_topo_together.png' % (exp_name, key_metric, add_str))
            plt.close()
        else:
            plt.show()

    ###
    ###
    for key_metric in ['Kendalltau', 'MAE']:
        for dynamics_name in dynamics_list:
            df_i = df_all_all_dynamics[df_all_all_dynamics['Dynamics types'] == dynamics_name.split('_')[0]]

            plt.figure(figsize=figsize)
            if plot_type == 'box':
                color_palette = sns.color_palette(
                    sns.color_palette('pastel')[2:3] + sns.color_palette('pastel')[3:4])
                sns.boxplot(x="Methods", y=key_metric, showfliers=False,
                            # hue="Methods",
                            data=df_i,
                            palette=color_palette)
                sns.stripplot(x="Methods", y=key_metric,
                              data=df_i,
                              size=5, linewidth=0, alpha=0.6,
                              # color='0.3',
                              hue='bound_t_context',
                              edgecolor='w',
                              legend=False,
                              # palette="rainbow")
                              palette=sns.color_palette(color_list))

            else:
                sns.violinplot(x="Methods", y=key_metric,
                               # hue="Methods",
                               data=df_i)

            # add mid line for each context_t_bound
            idx_t = 0
            for bound_t_context in bound_t_context_list:
                if bound_t_context == 0:
                    idx_t = 0
                else:
                    idx_t += 1
                sns.boxplot(x="Methods", y=key_metric, showfliers=False, showcaps=False,
                            whiskerprops={'color': 'w', 'alpha': 0},
                            # hue="Methods",
                            linewidth=2,
                            data=df_i[df_i['bound_t_context'] == bound_t_context],
                            boxprops={"facecolor": (1, 1, 1, 1), 'alpha': 0},
                            medianprops={'marker': '>', 'markevery': 2, 'markersize': 10, 'markeredgecolor': 'w',
                                         'linestyle': 'none', "color": color_list[idx_t], 'alpha': 0.9},
                            )

            if dynamics == 'heat_diffusion_dynamics':
                plt.ylim(0, 60)
            elif dynamics == 'mutualistic_interaction_dynamics':
                plt.ylim(0, 100)
            elif dynamics == 'gene_regulatory_dynamics':
                plt.ylim(0, 200)
            elif dynamics == 'opinion_dynamics':
                plt.ylim(0, 10)
            elif dynamics == 'opinion_dynamics_Baumann2021':
                plt.ylim(0, 100)
            elif dynamics == 'opinion_dynamics_Baumann2021_2topic':
                plt.ylim(0, 3)
                if key_metric == 'MAE':
                    import matplotlib.ticker as mtick
                    plt.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))

            elif dynamics == 'SI_Individual_dynamics' or \
                    dynamics == 'SIS_Individual_dynamics' or \
                    dynamics == 'SIR_Individual_dynamics' or \
                    dynamics == 'SEIS_Individual_dynamics' or dynamics == 'SEIR_Individual_dynamics':
                plt.ylim(0, 5)
                if key_metric == 'MAE':
                    import matplotlib.ticker as mtick
                    plt.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
            else:
                plt.ylim(0, 10)

            if key_metric == 'Kendalltau':
                plt.ylim(-1, 1.1)

            xticks = plt.gca().get_xticklabels()
            plt.gca().set_xticklabels(xticks, rotation=0)
            plt.tight_layout()

            #
            if save_fig:
                plt.savefig(
                    'results/%s_%s_%s_for_each_%s_all_topo_together.png' % (
                    exp_name, key_metric, dynamics_name, add_str))
                plt.close()
            else:
                plt.show()


def plot_violinplot_new(save_fig=True, add_str='', plot_type='violin', dynamics_topo_list=[],
                        bound_t_context_list=[], test_N=-1, test_num_trials=10, ndcn_flag=False,
                        x_dim=1,
                        train_topo=''):
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd
    import pickle

    # figsize = (14, 10)
    #figsize = (6, 6)
    figsize = (5, 6)
    # sns.set_theme(style="whitegrid", palette="pastel")
    sns.set(context='notebook', style='whitegrid', font_scale=2, palette="pastel")

    data = []

    for dynamics, topo in dynamics_topo_list:
            
            #### load ndcn results
            if ndcn_flag:
                fname = 'compared_methods/ndcn_all_%s_on_%s_ndcn_norm_adj.pickle'%(dynamics, topo)
                
                ndcn_results_data_dict = {}
                
                with open(fname, 'rb') as f:
                    ndcn_results_data = pickle.load(f)
                    for dd in ndcn_results_data:
                        ndcn_results_data_dict[list(dd.keys())[0]] = list(dd.values())[0]
                
            
            test_num_trials = 100
            add_str_topo = ''
            if 'SI_' in dynamics or 'SIS_' in dynamics or 'opinion_' in dynamics:
                test_N = 200
                x_dim = 2
            elif 'SEIS_' in dynamics or 'SIR_' in dynamics:
                test_N = 200
                x_dim = 3
            elif 'SEIR_' in dynamics:
                test_N = 200
                x_dim = 4
            else:
                test_N = 225
                x_dim = 1  
                test_num_trials = 20
                add_str_topo = 'all'
    

            for bound_t_context in bound_t_context_list:

                        
                #for exp_type in [ 'GMNND', 'GraphNDP',]:
                for exp_type in [ 'GNND',]:
                    if exp_type == 'GraphNDP':
                        fname = 'results/saved_test_results_%s_MLlossFalse_deterTrue_uncerTrue_%s_%s_x%s_numgraph1000_timestep100_epoch30_bound_t_context%s_seed1_num_nodes%s.pkl' % (
                            model_name, dynamics, add_str_topo+topo, x_dim, bound_t_context, test_N)
                    elif exp_type == 'GNND':
                        fname = 'results/saved_test_results_%s_MLlossFalse_deterTrue_uncerTrue_%s_%s_x%s_numgraph1000_timestep100_epoch30_bound_t_context%s_seed1_num_nodes%s_with_2nd_phase.pkl' % (
                            model_name, dynamics, add_str_topo+topo,  x_dim, bound_t_context, test_N)
                    else:
                        print("Unknown [%s]"%exp_type)
                        exit(1)
                            
                    with open(fname, 'rb') as f:
                        saved_results_data = pickle.load(f)
                    print('loading ...', dynamics, topo, bound_t_context, exp_type)

                    groundtruth_saved = []

                    for idx in range(len(saved_results_data['test_id']) - test_num_trials,
                                     len(saved_results_data['test_id'])):
                        test_id = saved_results_data['test_id'][idx]

                        # observations = saved_results_data['observations'][idx]
                        groundtruth = saved_results_data['groundtruth'][idx]
                        # groundtruth_sum = saved_results_data['groundtruth_sum'][idx].view(-1)
                        predictions = saved_results_data['predictions'][idx]
                        # predictions_sum = saved_results_data['predictions_sum'][idx].view(-1, len(groundtruth_sum))
                        # nl1_error = saved_results_data['nl1_error'][idx]
                        # l2_error = saved_results_data['l2_error'][idx]

                        # groundtruth [225, 100, 1]; predictions['mean'] [20, 225, 100, 1]; predictions['std'] [20, 225, 100, 1]
                        #  |
                        #  v  using transpose
                        #  v
                        # groundtruth [100, 225, 1]; predictions['mean'] [20, 100, 225, 1]; predictions['std'] [20, 100, 225, 1]
                        groundtruth = torch.transpose(groundtruth, 0, 1)
                        predictions['mean'] = torch.transpose(predictions['mean'], 1, 2)
                        predictions['std'] = torch.transpose(predictions['std'], 1, 2)

                        groundtruth_saved.append(groundtruth)

                        metric = compute_metric(groundtruth.numpy(),
                                                torch.mean(predictions['mean'], dim=0).numpy(),
                                                (torch.mean(
                                                    torch.pow(predictions['std'], 2) + torch.pow(predictions['mean'],
                                                                                                 2),
                                                    dim=0) \
                                                 - torch.pow(torch.mean(predictions['mean'], dim=0), 2)).numpy())
                        if bound_t_context == 0:
                            bound_t_context_str = '$=0$'
                        else:
                            bound_t_context_str = '$(0,%s]$' % bound_t_context
                        data.append(
                            list(metric.values()) + [exp_type, bound_t_context_str, dynamics[:4], topo])

                        #data.append(
                        #    list(metric.values()) + ['GMNND', bound_t_context_str, dynamics[:4], topo])

                if ndcn_flag:
                    # get ndcn results data

                    for idx in range(test_num_trials):
                        ndcn_results_data_one = ndcn_results_data_dict[(dynamics, topo, bound_t_context, idx)]

                        predictions = ndcn_results_data_one['pred_y'].cpu()
                        # predictions_sum = torch.sum(ndcn_results_data_one['pred_y'], dim=1)
                        # nl1_error = ndcn_results_data_one['normalized_l1'].item()
                        # l2_error = ndcn_results_data_one['mse'].item()

                        groundtruth = groundtruth_saved[idx]

                        # predictions [100,225] -> [100,225,1]
                        if predictions.dim() == 2:
                            predictions = predictions.unsqueeze(-1)

                        metric = compute_metric(groundtruth.numpy(),
                                                predictions.numpy())

                        if bound_t_context == 0:
                            bound_t_context_str = '$=0$'
                        else:
                            bound_t_context_str = '$(0,%s]$' % bound_t_context
                        data.append(list(metric.values()) + ['NDCN', bound_t_context_str, dynamics[:4], topo])

    df = pd.DataFrame(data,
                      columns=list(metric.keys()) + ['Methods', 'max $t_{obs}$',
                                                     'Dynamics types', 'Topology types'],
                      dtype=float)

    # metric['pearsonr'] = pearsonr_net
    # metric['spearmanr'] = spearmanr_net
    # metric['kendalltau'] = kendalltau_net
    # metric['negative_loglikelihood'] = negative_loglikelihood_net
    # metric['nl1_net'] = nl1_net
    # metric['mse'] = mse_net

    # Draw

    for key_metric in ['Kendalltau', 'negative_loglikelihood', 'MAE', 'mse']:
        idx_t = 0
        for bound_t_context in bound_t_context_list:
            if bound_t_context == 0:
                    idx_t = 0
                    bound_t_context_str = '$=0$'
            else:
                    idx_t += 1
                    bound_t_context_str = '$(0,%s]$' % bound_t_context
                    
        
            plt.figure(figsize=figsize)
            if plot_type == 'box':
                sns.boxplot(x="Methods", y=key_metric, showfliers=False,
                            # hue="Methods",
                            data=df[df['max $t_{obs}$'] == bound_t_context_str])
                sns.stripplot(x="Methods", y=key_metric,
                              data=df[df['max $t_{obs}$'] == bound_t_context_str],
                              size=3, linewidth=0, alpha=0.3,
                              # color='0.3',
                              hue='max $t_{obs}$',
                              edgecolor='w',
                              legend=False,
                              # palette="rainbow")
                              palette=sns.color_palette(color_list[idx_t:idx_t+1]))
    
            else:
                sns.violinplot(x="Methods", y=key_metric,
                               # hue="Methods",
                               data=df)
            """
            # add mid line for each context_t_bound
            idx_t = 0
            for bound_t_context in bound_t_context_list:
                if bound_t_context == 0:
                    idx_t = 0
                    bound_t_context_str = '$=0$'
                else:
                    idx_t += 1
                    bound_t_context_str = '$(0,%s]$' % bound_t_context
                sns.boxplot(x="Methods", y=key_metric, showfliers=False, showcaps=False,
                            whiskerprops={'color': 'w', 'alpha': 0},
                            # hue="Methods",
                            linewidth=2,
                            data=df[df['max $t_{obs}$'] == bound_t_context_str],
                            boxprops={"facecolor": (1, 1, 1, 1), 'alpha': 0},
                            medianprops={'marker': '>', 'markevery': 2, 'markersize': 10, 'markeredgecolor': 'w',
                                         'linestyle': 'none', "color": color_list[idx_t], 'alpha': 0.9},
                            )
            """
    
            if key_metric == 'Kendalltau':
                plt.ylim(-1, 1.1)
            elif key_metric == 'negative_loglikelihood':
                plt.ylim(0, 100)
            elif key_metric == 'MAE':
                plt.ylim(0, 10)
            elif key_metric == 'mse':
                plt.ylim(0, 20)
    
            xticks = plt.gca().get_xticklabels()
            plt.gca().set_xticklabels(xticks, rotation=0)
            plt.tight_layout()
    
            #
            if save_fig:
                plt.savefig(
                    'results/EXP1_%s_%s_%s.png' % (add_str, key_metric, bound_t_context))
                plt.close()
            else:
                plt.show()

    ###
    for key_metric in ['Kendalltau', 'negative_loglikelihood', 'MAE', 'mse']:
        for dynamics_name in [dd[0] for dd in dynamics_topo_list]:
        
            idx_t = 0
            for bound_t_context in bound_t_context_list:
                if bound_t_context == 0:
                        idx_t = 0
                        bound_t_context_str = '$=0$'
                else:
                        idx_t += 1
                        bound_t_context_str = '$(0,%s]$' % bound_t_context
            
                df_i = df[df['Dynamics types'] == dynamics_name[:4]]

                plt.figure(figsize=figsize)
                if plot_type == 'box':
                    sns.boxplot(x="Methods", y=key_metric, showfliers=False,
                                # hue="Methods",
                                data=df_i[df_i['max $t_{obs}$'] == bound_t_context_str])
                    sns.stripplot(x="Methods", y=key_metric,
                                  data=df_i[df_i['max $t_{obs}$'] == bound_t_context_str],
                                  size=3, linewidth=0, alpha=0.3,
                                  # color='0.3',
                                  hue='max $t_{obs}$',
                                  edgecolor='w',
                                  legend=False,
                                  # palette="rainbow")
                                  palette=sns.color_palette(color_list[idx_t:idx_t+1]))
    
                else:
                    sns.violinplot(x="Methods", y=key_metric,
                                   # hue="Methods",
                                   data=df_i)
                """
                # add mid line for each context_t_bound
                idx_t = 0
                for bound_t_context in bound_t_context_list:
                    if bound_t_context == 0:
                        idx_t = 0
                        bound_t_context_str = '$=0$'
                    else:
                        idx_t += 1
                        bound_t_context_str = '$(0,%s]$' % bound_t_context
                    sns.boxplot(x="Methods", y=key_metric, showfliers=False, showcaps=False,
                                whiskerprops={'color': 'w', 'alpha': 0},
                                # hue="Methods",
                                linewidth=2,
                                data=df_i[df_i['max $t_{obs}$'] == bound_t_context_str],
                                boxprops={"facecolor": (1, 1, 1, 1), 'alpha': 0},
                                medianprops={'marker': '>', 'markevery': 2, 'markersize': 10, 'markeredgecolor': 'w',
                                             'linestyle': 'none', "color": color_list[idx_t], 'alpha': 0.9},
                                )
                """
                if key_metric == 'Kendalltau':
                    plt.ylim(-1, 1.1)
                elif key_metric == 'negative_loglikelihood':
                    plt.ylim(0, 100)
                elif key_metric == 'MAE':
                    if 'opinion' in dynamics_name or 'SI_' in dynamics_name or 'SIS_' in dynamics_name or 'SIR_' in dynamics_name or 'SEIS_' in dynamics_name or 'SEIR_' in dynamics_name:
                        plt.ylim(0, 5)
                    else:
                        plt.ylim(0, 10)
                elif key_metric == 'mse':
                    plt.ylim(0, 20)
    
                xticks = plt.gca().get_xticklabels()
                plt.gca().set_xticklabels(xticks, rotation=0)
                plt.tight_layout()
    
                #
                if save_fig:
                    plt.savefig(
                        'results/EXP1_%s_%s_%s_%s.png' % (add_str, dynamics_name, key_metric, bound_t_context))
                    plt.close()
                else:
                    plt.show()

        # plt.close()


def plot_sum_state_compare(test_ids, save_fig=False, dynamics_topo_list=[],
                           show_run_no_indexs_per_dynamic_topo_dict=None):
    import pickle
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd

    figsize = (6, 3.5)

    sns.set(context='notebook', style='whitegrid', font_scale=2)
    color_palette = sns.color_palette("pastel")
    # color_palette.insert(0,(0,0,0))
    # sns.color_palette(color_palette)

    for dynamics, topo in dynamics_topo_list:
            
            #### load ndcn results
            ndcn_results_data_dict = {}
            fname = 'compared_methods/ndcn_all_%s_on_%s_ndcn_norm_adj.pickle'%(dynamics, topo)
            with open(fname, 'rb') as f:
                ndcn_results_data = pickle.load(f)
                for dd in ndcn_results_data:
                    ndcn_results_data_dict[list(dd.keys())[0]] = list(dd.values())[0]
            
            
            
        
            data = []
            data_ndcn = []

            for bound_t_context in [0.0, 0.25, 0.5, 0.75]:
                # for bound_t_context in [0.0, 0.25, 0.5]:
                test_N = 225
                x_dim = 1  
                test_num_trials = 20
                add_str_topo = 'all'
                fname = 'results/saved_test_results_%s_MLlossFalse_deterTrue_uncerTrue_%s_%s_x%s_numgraph1000_timestep100_epoch30_bound_t_context%s_seed1_num_nodes%s_with_2nd_phase.pkl' % (
                            model_name, dynamics, add_str_topo+topo,  x_dim, bound_t_context, test_N)

                #fname = 'results/saved_test_results_GNDP_OneForAll_MLlossFalse_deterTrue_uncerTrue_%s_%s_x1_numgraph1000_timestep100_epoch60_bound_t_context%s_seed1_num_nodes225.pkl' % (
                #    dynamics, topo, bound_t_context)
                with open(fname, 'rb') as f:
                    saved_results_data = pickle.load(f)
                print('hhah')

                for idx in range(len(saved_results_data['test_id']) - test_num_trials, len(saved_results_data['test_id'])):
                    test_id = saved_results_data['test_id'][idx]
                    if show_run_no_indexs_per_dynamic_topo_dict is not None:
                        test_ids = show_run_no_indexs_per_dynamic_topo_dict[(dynamics, topo)]['trial_no']
                    if test_id in test_ids:
                        observations = saved_results_data['observations'][idx]
                        groundtruth = saved_results_data['groundtruth'][idx]
                        #groundtruth_sum = saved_results_data['groundtruth_sum'][idx].view(-1)
                        predictions = saved_results_data['predictions'][idx]
                        #predictions_sum = saved_results_data['predictions_sum'][idx].view(-1, len(groundtruth_sum))
                        #nl1_error = saved_results_data['nl1_error'][idx]
                        #l2_error = saved_results_data['l2_error'][idx]
                        
                        groundtruth_sum = torch.sum(groundtruth, dim=0).sum(-1).view(-1)
                        predictions_sum = torch.sum(predictions['mean'], dim=1).sum(-1)

                        ndcn_predictions_sum = torch.sum(
                            ndcn_results_data_dict[(dynamics, topo, bound_t_context, test_id)]['pred_y'].view(
                                len(groundtruth_sum), -1), dim=-1)

                        x_time = np.linspace(0, 1, len(groundtruth_sum))
                        for iidx in range(len(x_time)):
                            if bound_t_context == 0:
                                data.append([x_time[iidx], groundtruth_sum[iidx], 0, 'Groundtruth', dynamics, topo])
                            for j in range(len(predictions_sum)):
                                # data.append([x_time[iidx], predictions_sum[j][iidx], 'GraphNDP-%s-%s' % (bound_t_context,j), dynamics, topo])
                                if bound_t_context == 0:
                                    data.append(
                                        [x_time[iidx], predictions_sum[j][iidx], j + 1,
                                         'GNND s.t. $t_{obs}=%s$' % (bound_t_context),
                                         dynamics,
                                         topo])
                                else:
                                    data.append(
                                        [x_time[iidx], predictions_sum[j][iidx], j + 1,
                                         'GNND s.t. $t_{obs}<=%s$' % (bound_t_context),
                                         dynamics,
                                         topo])
                            if bound_t_context == 0:
                                data_ndcn.append([x_time[iidx], ndcn_predictions_sum[iidx], 0,
                                                  'NDCN s.t. $t_{obs}=%s$' % bound_t_context, dynamics, topo])
                            else:
                                data_ndcn.append([x_time[iidx], ndcn_predictions_sum[iidx], 0,
                                                  'NDCN s.t. $t_{obs}<=%s$' % bound_t_context, dynamics, topo])

            df = pd.DataFrame(data, columns=['time', 'sum of states', 'sampled_z_i', 'Methods', 'Dynamics types',
                                             'Topology types'], dtype=float)
            df_ndcn = pd.DataFrame(data_ndcn,
                                   columns=['time', 'sum of states', 'sampled_z_i', 'Methods', 'Dynamics types',
                                            'Topology types'], dtype=float)

            plt.figure(figsize=figsize)
            
            #print(df)

            y_min = min(df['sum of states'].to_numpy())
            y_max = max(df['sum of states'].to_numpy())

            idx = -1
            for bound_t_context in [0.0, 0.25, 0.5, 0.75]:
                idx += 1
                if bound_t_context == 0:
                    df_i = df[df['Methods'] == 'GNND s.t. $t_{obs}=%s$' % (bound_t_context)]
                else:
                    df_i = df[df['Methods'] == 'GNND s.t. $t_{obs}<=%s$' % (bound_t_context)]

                plt.plot(np.mean(df_i['time'].to_numpy().reshape(-1, 20), axis=-1),
                         np.mean(df_i['sum of states'].to_numpy().reshape(-1, 20), axis=-1),
                         '--', c=color_list[idx],
                         marker='o',
                         markerfacecolor='w',
                         markersize=10, markevery=10, alpha=0.5, linewidth=2)

                plt.fill_between(np.mean(df_i['time'].to_numpy().reshape(-1, 20), axis=-1),
                                 np.mean(df_i['sum of states'].to_numpy().reshape(-1, 20), axis=-1) \
                                 + 1.96 * np.std(df_i['sum of states'].to_numpy().reshape(-1, 20), axis=-1),
                                 np.mean(df_i['sum of states'].to_numpy().reshape(-1, 20), axis=-1) \
                                 - 1.96 * np.std(df_i['sum of states'].to_numpy().reshape(-1, 20), axis=-1),
                                 facecolor=color_list[idx], alpha=0.2)

            idx = -1
            for bound_t_context in [0.0, 0.25, 0.5, 0.75]:
                idx += 1
                if bound_t_context == 0:
                    df_ndcn_i = df_ndcn[df_ndcn['Methods'] == 'NDCN s.t. $t_{obs}=%s$' % (bound_t_context)]
                else:
                    df_ndcn_i = df_ndcn[df_ndcn['Methods'] == 'NDCN s.t. $t_{obs}<=%s$' % (bound_t_context)]
                plt.plot(df_ndcn_i['time'].to_numpy(),
                         df_ndcn_i['sum of states'].to_numpy(),
                         ':', c=color_list[idx],
                         marker='s',
                         markerfacecolor='w',
                         markersize=10, markevery=10, alpha=0.5, linewidth=2)

            df_gt = df[df['Methods'] == 'Groundtruth']
            plt.plot(df_gt['time'].to_numpy(),
                     df_gt['sum of states'].to_numpy(),
                     'k-', markersize=10, markevery=10, alpha=1, linewidth=1)

            plt.xlabel('Time')
            plt.ylabel('Total states')

            # if 'vary_dynamics' in dynamics and 'power_law' in topo:
            plt.ylim(y_min - (y_max - y_min) / 10, y_max + (y_max - y_min) / 10)

            plt.tight_layout()

            if save_fig:
                plt.savefig('results/EXP1_sum_states_%s_%s_%s.png' % (dynamics, topo, test_ids))
            else:
                plt.show()
            plt.close()


def plot_state_on_one_node_compare(test_ids, save_fig=False, dynamics_topo_list=[],
                                   show_run_no_indexs_per_dynamic_topo_dict=None):
    import pickle
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd

    figsize = (4, 3.5)

    sns.set(context='notebook', style='white', font_scale=2)
    color_palette = sns.color_palette("pastel")
    # color_palette.insert(0,(0,0,0))
    # sns.color_palette(color_palette)


    for dynamics,topo in dynamics_topo_list:
            #### load ndcn results
            ndcn_results_data_dict = {}
            fname = 'compared_methods/ndcn_all_%s_on_%s_ndcn_norm_adj.pickle'%(dynamics, topo)
            with open(fname, 'rb') as f:
                ndcn_results_data = pickle.load(f)
                for dd in ndcn_results_data:
                    ndcn_results_data_dict[list(dd.keys())[0]] = list(dd.values())[0]
        
        
        
            data = []
            data_std_ub = []
            data_std_lb = []
            data_observations = []

            data_ndcn = []

            for bound_t_context in [0.0, 0.25, 0.5, 0.75]:
                # for bound_t_context in [0.0, 0.25, 0.5]:
                test_N = 225
                x_dim = 1  
                test_num_trials = 20
                add_str_topo = 'all'
                fname = 'results/saved_test_results_%s_MLlossFalse_deterTrue_uncerTrue_%s_%s_x%s_numgraph1000_timestep100_epoch30_bound_t_context%s_seed1_num_nodes%s_with_2nd_phase.pkl' % (
                            model_name, dynamics, add_str_topo+topo,  x_dim, bound_t_context, test_N)
                            
                #fname = 'results/saved_test_results_GNDP_OneForAll_MLlossFalse_deterTrue_uncerTrue_%s_%s_x1_numgraph1000_timestep100_epoch60_bound_t_context%s_seed1_num_nodes225.pkl' % (
                #    dynamics, topo, bound_t_context)
                with open(fname, 'rb') as f:
                    saved_results_data = pickle.load(f)
                print('hhah')

                for idx in range(len(saved_results_data['test_id']) - test_num_trials, len(saved_results_data['test_id'])):
                    test_id = saved_results_data['test_id'][idx]
                    if show_run_no_indexs_per_dynamic_topo_dict is not None:
                        test_ids = show_run_no_indexs_per_dynamic_topo_dict[(dynamics, topo)]['trial_no']
                    if test_id in test_ids:
                        observations = saved_results_data['observations'][idx]
                        groundtruth = saved_results_data['groundtruth'][idx]
                        #groundtruth_sum = saved_results_data['groundtruth_sum'][idx].view(-1)
                        predictions = saved_results_data['predictions'][idx]
                        
                        groundtruth_sum = torch.sum(groundtruth, dim=0).sum(-1).view(-1)
                        predictions_sum = torch.sum(predictions['mean'], dim=1).sum(-1)
                        
                        #predictions_sum = saved_results_data['predictions_sum'][idx].view(-1, len(groundtruth_sum))
                        #nl1_error = saved_results_data['nl1_error'][idx]
                        #l2_error = saved_results_data['l2_error'][idx]

                        node_no = show_run_no_indexs_per_dynamic_topo_dict[(dynamics, topo)]['node_no'][0]

                        ndcn_predictions = ndcn_results_data_dict[(dynamics, topo, bound_t_context, test_id)][
                            'pred_y'].view(
                            len(groundtruth_sum), -1).t()

                        x_time = np.linspace(0, 0.99, len(groundtruth[node_no]))
                        for iidx in range(len(x_time)):
                            if bound_t_context == 0:
                                data.append(
                                    [x_time[iidx], groundtruth[node_no][iidx], 0, 'Groundtruth', dynamics, topo])

                            for j in range(len(predictions['mean'])):
                                # data.append([x_time[iidx], predictions_sum[j][iidx], 'GraphNDP-%s-%s' % (bound_t_context,j), dynamics, topo])
                                data.append(
                                    [x_time[iidx], predictions['mean'][j][node_no][iidx], j + 1,
                                     'GraphNDP-%s' % (bound_t_context),
                                     dynamics,
                                     topo])
                                data_std_ub.append(
                                    [x_time[iidx],
                                     predictions['mean'][j][node_no][iidx] + predictions['std'][j][node_no][iidx],
                                     j + 1,
                                     'GraphNDP-%s' % (bound_t_context),
                                     dynamics,
                                     topo])
                                data_std_lb.append(
                                    [x_time[iidx],
                                     predictions['mean'][j][node_no][iidx] - predictions['std'][j][node_no][iidx],
                                     j + 1,
                                     'GraphNDP-%s' % (bound_t_context),
                                     dynamics,
                                     topo])

                            data_ndcn.append(
                                [x_time[iidx], ndcn_predictions[node_no][iidx], 0, 'NDCN-%s' % bound_t_context,
                                 dynamics, topo])

                        for obs_idx in range(len(observations['t'])):
                            obs_t = observations['t'][obs_idx]
                            #print('observations[\'mask\'][obs_idx][node_no]=',observations['mask'][obs_idx][node_no])
                            if observations['mask'][obs_idx][node_no][0] == 1.:
                                data_observations.append(
                                    [obs_t, observations['x_self'][obs_idx][node_no,0].view(-1), 0,
                                     'Observations-%s' % (bound_t_context), dynamics,
                                     topo])

            df = pd.DataFrame(data, columns=['time', 'state on node', 'sampled_z_i', 'Methods', 'Dynamics types',
                                             'Topology types'], dtype=float)
            df_ub = pd.DataFrame(data_std_ub,
                                 columns=['time', 'state on node', 'sampled_z_i', 'Methods', 'Dynamics types',
                                          'Topology types'], dtype=float)
            df_lb = pd.DataFrame(data_std_lb,
                                 columns=['time', 'state on node', 'sampled_z_i', 'Methods', 'Dynamics types',
                                          'Topology types'], dtype=float)
            df_obs = pd.DataFrame(data_observations,
                                  columns=['time', 'state on node', 'sampled_z_i', 'Methods', 'Dynamics types',
                                           'Topology types'], dtype=float)

            df_ndcn = pd.DataFrame(data_ndcn,
                                   columns=['time', 'state on node', 'sampled_z_i', 'Methods', 'Dynamics types',
                                            'Topology types'], dtype=float)

            y_min = min(df['state on node'].to_numpy())
            y_max = max(df['state on node'].to_numpy())

            bound_t_context_list = [0.0, 0.25, 0.5, 0.75]
            for bound_t_context_idx in range(len(bound_t_context_list)):

                plt.figure(figsize=figsize)

                df_i = df[df['Methods'] == 'GraphNDP-%s' % (bound_t_context_list[bound_t_context_idx])]
                df_ub_i = df_ub[df_ub['Methods'] == 'GraphNDP-%s' % (bound_t_context_list[bound_t_context_idx])]
                df_lb_i = df_lb[df_lb['Methods'] == 'GraphNDP-%s' % (bound_t_context_list[bound_t_context_idx])]
                for j in range(20):
                    df_i_j = df_i[df_i['sampled_z_i'] == j + 1]
                    df_ub_i_j = df_ub_i[df_ub_i['sampled_z_i'] == j + 1]
                    df_lb_i_j = df_lb_i[df_lb_i['sampled_z_i'] == j + 1]
                    plt.fill_between(df_ub_i_j['time'], df_lb_i_j['state on node'], df_ub_i_j['state on node'],
                                     alpha=0.02, color=color_list[bound_t_context_idx])

                    plt.plot(df_i_j['time'], df_i_j['state on node'],
                             '--', c=color_list[bound_t_context_idx],
                             marker='o',
                             markerfacecolor='w',
                             markersize=10, markevery=15, alpha=0.5, linewidth=1)

                df_ndcn_i = df_ndcn[df_ndcn['Methods'] == 'NDCN-%s' % (bound_t_context_list[bound_t_context_idx])]
                plt.plot(df_ndcn_i['time'].to_numpy(),
                         df_ndcn_i['state on node'].to_numpy(),
                         '--', c=color_list[bound_t_context_idx],
                         marker='s',
                         markerfacecolor='w',
                         markersize=10, markevery=15, alpha=0.5, linewidth=2)

                df_gt = df[df['Methods'] == 'Groundtruth']
                plt.plot(df_gt['time'].to_numpy(),
                         df_gt['state on node'].to_numpy(),
                         'k-', markersize=10, markevery=15, alpha=1, linewidth=1)

                df_obs_i = df_obs[df_obs['Methods'] == 'Observations-%s' % (bound_t_context_list[bound_t_context_idx])]
                print('df_obs_i[\'time\']=',df_obs_i['time'])
                plt.scatter(df_obs_i['time'].to_numpy(),
                            df_obs_i['state on node'].to_numpy(),
                            marker='o', c='k', s=50, alpha=1, zorder=3)

                plt.xlabel('Time')
                plt.ylabel('States')

                plt.ylim(y_min - (y_max - y_min) / 10, y_max + (y_max - y_min) / 10)

                plt.tight_layout()
                plt.subplots_adjust(left=0.25, bottom=0.25, right=0.9, top=0.9)

                if save_fig:
                    plt.savefig('results/EXP1_states_on_node_%s_%s_%s_node%s_%s.png' % (
                        dynamics, topo, test_ids, node_no, bound_t_context_list[bound_t_context_idx]))
                    # plt.savefig('results/states_on_node_%s_%s_%s_node%s_%s.png' % (dynamics, topo, test_ids, node_no, bound_t_context_list[bound_t_context_idx]))
                else:
                    plt.show()
                plt.close()


def plot_opinion_dynamics(dynamics_list, topo_list, test_ids, bound_t_context_list, save_fig=False, test_N=-1,
                          test_num_trials=10, x_dim=1):
    import pickle
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd

    figsize = (12, 10)

    sns.set(context='notebook', style='whitegrid', font_scale=2)
    color_palette = sns.color_palette("pastel")
    # color_palette.insert(0,(0,0,0))
    # sns.color_palette(color_palette)

    for dynamics in dynamics_list:
        for topo in topo_list:
            data = []
            data_std_ub = []
            data_std_lb = []
            data_observations = []
            for bound_t_context in bound_t_context_list:
                # fname = 'results/saved_test_results_GNDP_MLlossFalse_deterTrue_uncerTrue_%s_%s_x1_numgraph1000_timestep100_epoch20_bound_t_context%s_seed1_num_nodes40.pkl' % (
                #     dynamics + dynamics, topo + topo, bound_t_context)

                # fname = 'results/saved_test_results_GNDP_MLlossFalse_deterTrue_uncerTrue_%s_%s_x1_numgraph1000_timestep100_epoch30_bound_t_context%s_seed1_num_nodes20.pkl' % (
                #     dynamics + dynamics, topo + topo, bound_t_context)
                # fname = 'results/saved_test_results_GNDP_MLlossFalse_deterTrue_uncerTrue_%s_%s_x1_numgraph1000_timestep100_epoch30_bound_t_context%s_seed1_num_nodes40.pkl' % (
                #     dynamics + dynamics, topo + topo, bound_t_context)
                # fname = 'results/saved_test_results_GNDP_MLlossFalse_deterTrue_uncerTrue_%s_%s_x1_numgraph1000_timestep100_epoch30_bound_t_context%s_seed1_num_nodes50.pkl' % (
                #     dynamics + dynamics, topo + topo, bound_t_context)
                # fname = 'results/saved_test_results_GNDP_MLlossFalse_deterTrue_uncerTrue_%s_%s_x1_numgraph1000_timestep100_epoch30_bound_t_context%s_seed1_num_nodes-1.pkl' % (
                #     dynamics + dynamics, topo + topo, bound_t_context)

                # fname = 'results/saved_test_results_GNDP_MLlossFalse_deterTrue_uncerTrue_%s_%s_x1_numgraph100_timestep100_epoch30_bound_t_context%s_seed1_num_nodes40.pkl' % (
                #     dynamics + dynamics, topo + topo, bound_t_context)

                # fname = 'results/saved_test_results_GNDP_MLlossFalse_deterTrue_uncerTrue_%s_%s_x1_numgraph1000_timestep100_epoch50_bound_t_context%s_seed1_num_nodes-1.pkl' % (
                #     dynamics + dynamics, topo + topo, bound_t_context)
                # fname = 'results/saved_test_results_GNDP_MLlossFalse_deterTrue_uncerTrue_%s_%s_x1_numgraph1000_timestep100_epoch50_bound_t_context%s_seed1_num_nodes%s.pkl' % (
                #     dynamics + dynamics, topo + topo, bound_t_context, test_N)
                # fname = 'results/saved_test_results_GNDP_MLlossFalse_deterTrue_uncerTrue_%s_%s_x1_numgraph400_timestep100_epoch20_bound_t_context%s_seed1_num_nodes%s.pkl' % (
                #     dynamics + dynamics, topo + topo, bound_t_context, test_N)
                # fname = 'results/saved_test_results_GNDP_MLlossFalse_deterTrue_uncerTrue_%s_%s_x1_numgraph1000_timestep100_epoch30_bound_t_context%s_seed1_num_nodes%s.pkl' % (
                #     dynamics + dynamics, topo + topo, bound_t_context, test_N)
                # fname = 'results/saved_test_results_GNDP_MLlossFalse_deterTrue_uncerTrue_%s_%s_x1_numgraph5000_timestep100_epoch20_bound_t_context%s_seed1_num_nodes%s.pkl' % (
                #     dynamics + dynamics, topo + topo, bound_t_context, test_N)
                fname = 'results/saved_test_results_GNDP_MLlossFalse_deterTrue_uncerTrue_%s_%s_x%s_numgraph1000_timestep100_epoch30_bound_t_context%s_seed1_num_nodes%s.pkl' % (
                    dynamics, topo, x_dim, bound_t_context, test_N)

                with open(fname, 'rb') as f:
                    saved_results_data = pickle.load(f)
                print('hhah')

                for idx in range(len(saved_results_data['test_id']) - test_num_trials,
                                 len(saved_results_data['test_id'])):
                    test_id = saved_results_data['test_id'][idx]
                    if test_id in test_ids:
                        observations = saved_results_data['observations'][idx]
                        groundtruth = saved_results_data['groundtruth'][idx]
                        groundtruth_sum = saved_results_data['groundtruth_sum'][idx].view(-1)
                        predictions = saved_results_data['predictions'][idx]
                        predictions_sum = saved_results_data['predictions_sum'][idx].view(-1, len(groundtruth_sum))
                        nl1_error = saved_results_data['nl1_error'][idx]
                        l2_error = saved_results_data['l2_error'][idx]

                        for node_no in range(len(groundtruth)):
                            x_time = np.linspace(0, 0.99, len(groundtruth[node_no]))
                            for iidx in range(len(x_time)):
                                data.append(
                                    [x_time[iidx], node_no, groundtruth[node_no][iidx], 0, 'Groundtruth', dynamics,
                                     topo])

                                for j in range(len(predictions['mean'])):
                                    # data.append([x_time[iidx], predictions_sum[j][iidx], 'GraphNDP-%s-%s' % (bound_t_context,j), dynamics, topo])
                                    data.append(
                                        [x_time[iidx], node_no,
                                         predictions['mean'][j][node_no][iidx], j + 1,
                                         'GraphNDP-%s' % (bound_t_context),
                                         dynamics,
                                         topo])
                                    data_std_ub.append(
                                        [x_time[iidx], node_no,
                                         predictions['mean'][j][node_no][iidx] + predictions['std'][j][node_no][iidx],
                                         j + 1,
                                         'GraphNDP-%s' % (bound_t_context),
                                         dynamics,
                                         topo])
                                    data_std_lb.append(
                                        [x_time[iidx], node_no,
                                         predictions['mean'][j][node_no][iidx] - predictions['std'][j][node_no][iidx],
                                         j + 1,
                                         'GraphNDP-%s' % (bound_t_context),
                                         dynamics,
                                         topo])

                            for obs_idx in range(len(observations['t'])):
                                obs_t = observations['t'][obs_idx]
                                if observations['mask'][obs_idx][node_no][0] == 1.:
                                    data_observations.append(
                                        [obs_t, node_no, observations['x_self'][obs_idx][node_no].view(-1), 0,
                                         'Observations-%s' % (bound_t_context), dynamics,
                                         topo])

            df = pd.DataFrame(data,
                              columns=['time', 'node_no', 'state on node', 'sampled_z_i', 'Methods', 'Dynamics types',
                                       'Topology types'], dtype=float)
            df_ub = pd.DataFrame(data_std_ub,
                                 columns=['time', 'node_no', 'state on node', 'sampled_z_i', 'Methods',
                                          'Dynamics types',
                                          'Topology types'], dtype=float)
            df_lb = pd.DataFrame(data_std_lb,
                                 columns=['time', 'node_no', 'state on node', 'sampled_z_i', 'Methods',
                                          'Dynamics types',
                                          'Topology types'], dtype=float)
            df_obs = pd.DataFrame(data_observations,
                                  columns=['time', 'node_no', 'state on node', 'sampled_z_i', 'Methods',
                                           'Dynamics types',
                                           'Topology types'], dtype=float)

            for bound_t_context_idx in range(len(bound_t_context_list)):
                plt.figure(figsize=figsize)
                for node_no_i in range(len(groundtruth)):
                    legend_flag = False
                    if node_no_i == 0:
                        legend_flag = True
                    df_ub_i = df_ub[df_ub['Methods'] == 'GraphNDP-%s' % (bound_t_context_list[bound_t_context_idx])]
                    df_ub_i = df_ub_i[df_ub_i['node_no'] == node_no_i]
                    df_lb_i = df_lb[df_lb['Methods'] == 'GraphNDP-%s' % (bound_t_context_list[bound_t_context_idx])]
                    df_lb_i = df_lb_i[df_lb_i['node_no'] == node_no_i]

                    df_i = df[df['node_no'] == node_no_i]
                    df_obs_i = df_obs[df_obs['node_no'] == node_no_i]

                    for j in range(20):
                        df_ub_i_j = df_ub_i[df_ub_i['sampled_z_i'] == j + 1]
                        df_lb_i_j = df_lb_i[df_lb_i['sampled_z_i'] == j + 1]
                        plt.fill_between(df_ub_i_j['time'], df_lb_i_j['state on node'], df_ub_i_j['state on node'],
                                         alpha=0.005, color=sns.color_palette("pastel")[bound_t_context_idx])

                    sns.lineplot(
                        data=df_i[df_i['Methods'] == 'GraphNDP-%s' % (bound_t_context_list[bound_t_context_idx])],
                        x='time', y='state on node', hue='Methods',
                        units='sampled_z_i', estimator=None,
                        alpha=0.2,
                        lw=0.5,
                        style='Methods',
                        palette=sns.color_palette("pastel")[bound_t_context_idx:bound_t_context_idx + 1],
                        legend=legend_flag
                    )
                    sns.lineplot(data=df_i[df_i['Methods'] == 'Groundtruth'], x='time', y='state on node',
                                 hue='Methods',
                                 units='sampled_z_i', estimator=None,
                                 alpha=0.5,
                                 lw=1,
                                 # style='Methods',
                                 palette=sns.color_palette([(0, 0, 0)]),
                                 legend=legend_flag)

                    ddf_i = df_obs_i[
                        df_obs_i['Methods'] == 'Observations-%s' % (bound_t_context_list[bound_t_context_idx])]
                    ddf_i['Methods'] = 'Observations'
                    sns.scatterplot(data=ddf_i,
                                    x='time', y='state on node', hue='Methods', palette=sns.color_palette([(0, 0, 0)]),
                                    alpha=0.5,
                                    size='Methods',
                                    sizes=(50, 60),
                                    legend=legend_flag
                                    )

                if save_fig:
                    plt.savefig('results/states_on_nodes_%s_%s_%s_%s.png' % (
                        dynamics, topo, test_ids, bound_t_context_list[bound_t_context_idx]))
                    # plt.savefig('results/states_on_node_%s_%s_%s_node%s_%s.png' % (dynamics, topo, test_ids, node_no, bound_t_context_list[bound_t_context_idx]))
                else:
                    plt.show()
                plt.close()


def plot_opinion_dynamics_2topic(dynamics_list, topo_list, test_ids, bound_t_context_list, save_fig=False, test_N=-1,
                                 test_num_trials=10, x_dim=1, draw_2d=True, ndcn_flag=False):
    import pickle
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd
    
    if ndcn_flag:
        #### load ndcn results
        fname = 'compared_methods/ndcn_all_opinion_dynamics_Baumann2021_2topic_on_small_world_ndcn_norm_adj.pickle'
        with open(fname, 'rb') as f:
            ndcn_results_data = pickle.load(f)
            ndcn_results_data_dict = {}
            for dd in ndcn_results_data:
                ndcn_results_data_dict[list(dd.keys())[0]] = list(dd.values())[0]

    figsize = (4, 3.5)

    sns.set(context='notebook', style='whitegrid', font_scale=2)
    color_palette = sns.color_palette("pastel")

    for dynamics in dynamics_list:
        for topo in topo_list:
            data = []
            data_std_ub = []
            data_std_lb = []
            data_observations = []

            data_ndcn = []
            for bound_t_context in bound_t_context_list:
                
                fname = 'results/saved_test_results_%s_MLlossFalse_deterTrue_uncerTrue_%s_%s_x%s_numgraph1000_timestep100_epoch30_bound_t_context%s_seed1_num_nodes%s_with_2nd_phase.pkl' % (
                            model_name, dynamics, topo,  x_dim, bound_t_context, test_N)

                #fname = 'results/saved_test_results_%s_MLlossFalse_deterTrue_uncerTrue_%s_%s_x%s_numgraph1000_timestep100_epoch60_bound_t_context%s_seed1_num_nodes%s.pkl' % (
                #    model_name, dynamics, topo, x_dim, bound_t_context, test_N)

                with open(fname, 'rb') as f:
                    saved_results_data = pickle.load(f)
                print('hhah')

                for idx in range(len(saved_results_data['test_id']) - test_num_trials,
                                 len(saved_results_data['test_id'])):
                    test_id = saved_results_data['test_id'][idx]
                    if test_id in test_ids:
                        observations = saved_results_data['observations'][idx]
                        groundtruth = saved_results_data['groundtruth'][idx]
                        #groundtruth_sum = saved_results_data['groundtruth_sum'][idx].view(-1)
                        predictions = saved_results_data['predictions'][idx]
                        #predictions_sum = saved_results_data['predictions_sum'][idx].view(-1, len(groundtruth_sum))
                        #nl1_error = saved_results_data['nl1_error'][idx]
                        #l2_error = saved_results_data['l2_error'][idx]
                        
                        if ndcn_flag:
                            ndcn_predictions = ndcn_results_data_dict[(dynamics, topo, bound_t_context, test_id)][
                            'pred_y'].cpu().numpy()  # t, n, d

                        for node_no in range(len(groundtruth)):
                            x_time = np.linspace(0, 0.99, len(groundtruth[node_no]))
                            for iidx in range(len(x_time)):
                                if ndcn_flag:
                                    data_ndcn.append(
                                        [x_time[iidx], node_no, ndcn_predictions[iidx][node_no][0],
                                         ndcn_predictions[iidx][node_no][1], 0, 'NDCN-%s' % (bound_t_context), dynamics,
                                         topo])

                                if bound_t_context == 0:
                                    data.append(
                                        [x_time[iidx], node_no, groundtruth[node_no][iidx][0],
                                         groundtruth[node_no][iidx][1], 0, 'Groundtruth', dynamics,
                                         topo])

                                for j in range(len(predictions['mean'])):
                                    # data.append([x_time[iidx], predictions_sum[j][iidx], 'GraphNDP-%s-%s' % (bound_t_context,j), dynamics, topo])
                                    data.append(
                                        [x_time[iidx], node_no,
                                         predictions['mean'][j][node_no][iidx][0],
                                         predictions['mean'][j][node_no][iidx][1],
                                         j + 1,
                                         'GraphNDP-%s' % (bound_t_context),
                                         dynamics,
                                         topo])
                                    data_std_ub.append(
                                        [x_time[iidx], node_no,
                                         predictions['mean'][j][node_no][iidx][0] +
                                         predictions['std'][j][node_no][iidx][0],
                                         predictions['mean'][j][node_no][iidx][1] +
                                         predictions['std'][j][node_no][iidx][1],
                                         j + 1,
                                         'GraphNDP-%s' % (bound_t_context),
                                         dynamics,
                                         topo])
                                    data_std_lb.append(
                                        [x_time[iidx], node_no,
                                         predictions['mean'][j][node_no][iidx][0] -
                                         predictions['std'][j][node_no][iidx][0],
                                         predictions['mean'][j][node_no][iidx][1] -
                                         predictions['std'][j][node_no][iidx][1],
                                         j + 1,
                                         'GraphNDP-%s' % (bound_t_context),
                                         dynamics,
                                         topo])

                            #for obs_idx in range(len(observations['t'])):
                            #    obs_t = observations['t'][obs_idx]
                            #    if observations['mask'][obs_idx][node_no] == 1.:
                            #        data_observations.append(
                            #            [obs_t, node_no, observations['x_self'][obs_idx][node_no][0].view(-1),
                            #             observations['x_self'][obs_idx][node_no][1].view(-1), 0,
                            #             'Observations-%s' % (bound_t_context), dynamics,
                            #             topo])

            df = pd.DataFrame(data,
                              columns=['time', 'node_no', 'state1 on node', 'state2 on node', 'sampled_z_i', 'Methods',
                                       'Dynamics types',
                                       'Topology types'], dtype=float)
            df_ub = pd.DataFrame(data_std_ub,
                                 columns=['time', 'node_no', 'state1 on node', 'state2 on node', 'sampled_z_i',
                                          'Methods', 'Dynamics types',
                                          'Topology types'], dtype=float)
            df_lb = pd.DataFrame(data_std_lb,
                                 columns=['time', 'node_no', 'state1 on node', 'state2 on node', 'sampled_z_i',
                                          'Methods', 'Dynamics types',
                                          'Topology types'], dtype=float)
            df_obs = pd.DataFrame(data_observations,
                                  columns=['time', 'node_no', 'state1 on node', 'state2 on node', 'sampled_z_i',
                                           'Methods', 'Dynamics types',
                                           'Topology types'], dtype=float)

            df_ndcn = pd.DataFrame(data_ndcn,
                                   columns=['time', 'node_no', 'state1 on node', 'state2 on node', 'sampled_z_i',
                                            'Methods', 'Dynamics types',
                                            'Topology types'], dtype=float)
            if draw_2d:

                def draw_2d(ddf, x_0dim_all_t=None, x_1dim_all_t=None,
                            x_0dim_std_all_t=None, x_1dim_std_all_t=None,
                            str_add="", plot_flag=True, color=None):
                    # x_0dim_all_t # [n, t]
                    # x_1dim_all_t # [n, t]

                    if ddf is not None:
                        x_0dim_all_t, x_1dim_all_t = ddf['state1 on node'].to_numpy().reshape(-1, groundtruth.size(1)), \
                                                     ddf[
                                                         'state2 on node'].to_numpy().reshape(-1, groundtruth.size(1))

                    pearsonr_list = []

                    figsize = (17.5, 6)

                    sns.set(context='notebook', style='ticks', font_scale=2, palette="pastel")

                    fig, axes = plt.subplots(4, 5, figsize=figsize)
                    axes_list = []
                    for ii in range(5):
                        ax1 = plt.subplot2grid((4, 5), (0, ii), colspan=1, rowspan=2)
                        ax2 = plt.subplot2grid((4, 5), (2, ii), colspan=1, rowspan=1)
                        ax3 = plt.subplot2grid((4, 5), (3, ii), colspan=1, rowspan=1)
                        axes_list.append([ax1, ax2, ax3])
                    ax_idx = -1
                    x_time = np.linspace(0, 0.99, len(groundtruth[0]))
                    for t_idx in range(groundtruth.size(1)):
                        ### Node states on the 1st dim
                        if t_idx % 25 == 0 or t_idx == groundtruth.size(1) - 1:
                            ax_idx += 1

                            x_0dim = x_0dim_all_t[:, t_idx]
                            x_1dim = x_1dim_all_t[:, t_idx]

                            # groundtruth
                            axes_list[ax_idx][0].scatter(x_0dim,
                                                         x_1dim,
                                                         color=color,
                                                         edgecolor="white",
                                                         alpha=0.8,
                                                         )
                            if x_0dim_std_all_t is not None and x_1dim_std_all_t is not None:
                                x_0dim_std = x_0dim_std_all_t[:, t_idx]
                                x_1dim_std = x_1dim_std_all_t[:, t_idx]
                                axes_list[ax_idx][0].errorbar(x_0dim, x_1dim,
                                                              yerr=1.96 * x_1dim_std,
                                                              xerr=1.96 * x_0dim_std,
                                                              ecolor=color,
                                                              ls='none',
                                                              alpha=0.2)

                            pearsonr = scipy.stats.pearsonr(
                                x_0dim,
                                x_1dim
                            )[0]
                            pearsonr_list.append(pearsonr)
                            # axes_list[ax_idx][0].text(-15, 17, "$pearsonr=%.2f$" % pearsonr)
                            axes_list[ax_idx][0].set_xlim(-15, 15)
                            axes_list[ax_idx][0].set_ylim(-15, 15)
                            axes_list[ax_idx][0].set_xlabel('$X_{1:N,0}$')
                            axes_list[ax_idx][0].set_ylabel('$X_{1:N,1}$')

                            sns.distplot(x_0dim,
                                         kde=True, bins=10, hist_kws={'color': color},
                                         ax=axes_list[ax_idx][1], color=color)
                            axes_list[ax_idx][1].set_xlim(-15, 15)
                            # axes_list[ax_idx][1].set_xlabel('$X_{1:N,0}$')
                            axes_list[ax_idx][1].set_xlabel('')
                            axes_list[ax_idx][1].set_ylabel('')
                            axes_list[ax_idx][1].set_yticks([])
                            # axes_list[ax_idx][1].axis('off')
                            sns.despine(top=True, right=True, left=True, bottom=False, ax=axes_list[ax_idx][1])

                            sns.distplot(x_1dim,
                                         kde=True, bins=10, hist_kws={'color': color},
                                         ax=axes_list[ax_idx][2],
                                         color=color)
                            axes_list[ax_idx][2].set_xlim(-15, 15)
                            # axes_list[ax_idx][2].set_xlabel('$X_{1:N,1}$')
                            axes_list[ax_idx][2].set_xlabel('')
                            axes_list[ax_idx][2].set_ylabel('')
                            axes_list[ax_idx][2].set_yticks([])
                            sns.despine(top=True, right=True, left=True, bottom=False, ax=axes_list[ax_idx][2])

                            print('=====================================')
                            print('GT t=%s' % x_time[t_idx])
                            print(pearsonr)
                            print(np.std(x_0dim),
                                  np.std(x_1dim))
                            # print(scipy.stats.spearmanr(New_X[t, :, 0].reshape(-1), New_X[t, :, 1].reshape(-1)))
                            # print(scipy.stats.kendalltau(New_X[t, :, 0].reshape(-1), New_X[t, :, 1].reshape(-1)))

                    plt.tight_layout()
                    if plot_flag:
                        if save_fig:
                            plt.savefig('results/EXP_2_states_on_nodes_%s_%s_%s_%s_t=all.png' % (
                                dynamics, topo, test_ids, str_add))
                            # plt.savefig('results/states_on_node_%s_%s_%s_node%s_%s.png' % (dynamics, topo, test_ids, node_no, bound_t_context_list[bound_t_context_idx]))
                        else:
                            plt.show()
                    plt.close()

                    return pearsonr_list

                ########### ground truth
                df_gt = df[df['Methods'] == 'Groundtruth']

                pearsonr_gt = draw_2d(df_gt, str_add='gt', color=(0.1, 0.1, 0.1))

                #####
                pearsonr_ndcn = []
                pearsonr_ours = []
                for bound_t_context_idx in range(len(bound_t_context_list)):
                    
                    if ndcn_flag:
                        ### ndcn
                        pearsonr_ndcn_i = draw_2d(
                            df_ndcn[df_ndcn['Methods'] == 'NDCN-%s' % (bound_t_context_list[bound_t_context_idx])],
                            str_add='ndcn_bound_context_t=%s' % bound_t_context_list[bound_t_context_idx],
                            color=color_list[bound_t_context_idx])
                        pearsonr_ndcn.append(pearsonr_ndcn_i)

                    ###
                    df_i = df[df['Methods'] == 'GraphNDP-%s' % (bound_t_context_list[bound_t_context_idx])]

                    x_0_ = []
                    x_1_ = []
                    pearsonr_ours_i = []
                    for sampled_z_idx in range(20):
                        df_i_z_i = df_i[df_i['sampled_z_i'] == sampled_z_idx + 1]
                        x_z_i_0 = df_i_z_i['state1 on node'].to_numpy().reshape(-1, groundtruth.size(1), 1)  # [n, t, 1]
                        x_z_i_1 = df_i_z_i['state2 on node'].to_numpy().reshape(-1, groundtruth.size(1), 1)  # [n, t, 1]
                        x_0_.append(x_z_i_0)
                        x_1_.append(x_z_i_1)

                        pearsonr_ours_i_z_i = draw_2d(
                            None,
                            x_z_i_0.reshape(-1, groundtruth.size(1)),
                            x_z_i_1.reshape(-1, groundtruth.size(1)),
                            str_add='ours_z_i',
                            plot_flag=False,
                            color=color_list[bound_t_context_idx]
                        )
                        pearsonr_ours_i.append(pearsonr_ours_i_z_i)
                    x_0_ = np.concatenate(x_0_, axis=-1)  # [n, t, 20]
                    x_1_ = np.concatenate(x_1_, axis=-1)  # [n, t, 20]

                    pearsonr_ours.append(pearsonr_ours_i)

                    pred_mean_x_0 = np.mean(x_0_, axis=-1)  # [n, t]
                    pred_std_x_0 = np.std(x_0_, axis=-1)  # [n, t]
                    pred_mean_x_1 = np.mean(x_1_, axis=-1)  # [n, t]
                    pred_std_x_1 = np.std(x_1_, axis=-1)  # [n, t]

                    draw_2d(
                        None,
                        pred_mean_x_0,
                        pred_mean_x_1,
                        pred_std_x_0,
                        pred_std_x_1,
                        str_add='ours_bound_context_t=%s' % bound_t_context_list[bound_t_context_idx],
                        color=color_list[bound_t_context_idx])

                ####
                figsize = (6, 6)
                sns.set(context='notebook', style='ticks', font_scale=2, palette="pastel")
                plt.figure(figsize=figsize)

                idx = -1
                for bound_t_context in [0.0, 0.25, 0.5, 0.75]:
                    idx += 1

                    plt.plot([0, 0.25, 0.5, 0.75, 1],
                             np.mean(pearsonr_ours[idx], axis=0),
                             '--', c=color_list[idx],
                             marker='o',
                             markerfacecolor='w',
                             markersize=10, markevery=1, alpha=0.5, linewidth=2)

                    plt.fill_between([0, 0.25, 0.5, 0.75, 1],
                                     np.mean(pearsonr_ours[idx], axis=0) \
                                     + 1.96 * np.std(pearsonr_ours[idx], axis=0),
                                     np.mean(pearsonr_ours[idx], axis=0) \
                                     - 1.96 * np.std(pearsonr_ours[idx], axis=0),
                                     facecolor=color_list[idx], alpha=0.2)
                
                if ndcn_flag:
                    idx = -1
                    for bound_t_context in [0.0, 0.25, 0.5, 0.75]:
                        idx += 1
    
                        plt.plot([0, 0.25, 0.5, 0.75, 1],
                                 pearsonr_ndcn[idx],
                                 ':', c=color_list[idx],
                                 marker='s',
                                 markerfacecolor='w',
                                 markersize=10, markevery=1, alpha=0.5, linewidth=2)

                plt.plot([0, 0.25, 0.5, 0.75, 1],
                         pearsonr_gt,
                         'k-', markersize=10, markevery=1, alpha=1, linewidth=1)
                plt.xlabel('Time')
                plt.ylabel('Pearsonr')
                plt.tight_layout()
                if save_fig:
                    plt.savefig('results/EXP_2_pearsonr_%s_%s_%s_t=all.png' % (
                        dynamics, topo, test_ids))
                    # plt.savefig('results/states_on_node_%s_%s_%s_node%s_%s.png' % (dynamics, topo, test_ids, node_no, bound_t_context_list[bound_t_context_idx]))
                else:
                    plt.show()
                plt.close()

                # errorbar(x, y, err, 'both', 'o', 'linewidth', 1.5)


def plot_epidemic_4dim(dynamics_list, topo_list, test_ids, bound_t_context_list, save_fig=False, test_N=-1,
                       test_num_trials=10, x_dim=1, draw_2d=True, ndcn_flag=False, show_each_node=False):
    import pickle
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd


    # figsize = (14, 10)
    figsize = (6, 4)

    sns.set(context='notebook', style='ticks', font_scale=2)
    color_palette = sns.color_palette("pastel")

    for dynamics in dynamics_list:
        for topo in topo_list:
        
            if ndcn_flag:
                #### load ndcn results
                fname = 'compared_methods/ndcn_all_%s_on_%s_ndcn_norm_adj.pickle'%(dynamics, topo)
                with open(fname, 'rb') as f:
                    ndcn_results_data = pickle.load(f)
                    ndcn_results_data_dict = {}
                    for dd in ndcn_results_data:
                        ndcn_results_data_dict[list(dd.keys())[0]] = list(dd.values())[0]
            
            if 'Coupled' in dynamics:
                state_label = ['SS', 'IS', 'SI', 'II']
            elif 'SI_' in dynamics or 'SIS_' in dynamics:
                state_label = ['Susceptible', 'Infected']
            elif 'SIR_' in dynamics:
                state_label = ['Susceptible', 'Infected', 'Removed']
            elif 'SEIS_' in dynamics:
                state_label = ['Susceptible', 'Infected', 'Exposed']
            else:
                state_label = ['Susceptible', 'Infected', 'Removed', 'Exposed']

                        
            
            data_ndcn = []

            data = []
            data_sum = []

            for bound_t_context_idx in range(len(bound_t_context_list)):
                bound_t_context = bound_t_context_list[bound_t_context_idx]
                
                fname = 'results/saved_test_results_%s_MLlossFalse_deterTrue_uncerTrue_%s_%s_x%s_numgraph1000_timestep100_epoch30_bound_t_context%s_seed1_num_nodes%s_with_2nd_phase.pkl' % (
                            model_name, dynamics, topo,  x_dim, bound_t_context, test_N)



                #fname = 'results/saved_test_results_%s_MLlossFalse_deterTrue_uncerTrue_%s_%s_x%s_numgraph1000_timestep100_epoch60_bound_t_context%s_seed1_num_nodes%s.pkl' % (
                #    model_name, dynamics, topo, x_dim, bound_t_context, test_N)

                with open(fname, 'rb') as f:
                    saved_results_data = pickle.load(f)
                print('hhah')

                #with open(fname_wo_softmax, 'rb') as f:
                #    saved_results_data_wo_softmax = pickle.load(f)

                for idx in range(len(saved_results_data['test_id']) - test_num_trials,
                                 len(saved_results_data['test_id'])):
                    test_id = saved_results_data['test_id'][idx]
                    if test_id in test_ids:
                        observations = saved_results_data['observations'][idx]
                        groundtruth = saved_results_data['groundtruth'][idx]
                        #groundtruth_sum = saved_results_data['groundtruth_sum'][idx].view(-1)
                        predictions = saved_results_data['predictions'][idx]
                        #predictions_sum = saved_results_data['predictions_sum'][idx].view(-1, len(groundtruth_sum))
                        #nl1_error = saved_results_data['nl1_error'][idx]
                        #l2_error = saved_results_data['l2_error'][idx]
                        
                        if ndcn_flag:
                            ndcn_predictions = ndcn_results_data_dict[(dynamics, topo, bound_t_context, test_id)][
                                'pred_y'].cpu().numpy()  # t, n, d

                        # saved_results_data_wo_softmax
                        #predictions_wo_softmax = saved_results_data_wo_softmax['predictions'][idx]

                        for node_no in range(len(groundtruth)):
                            x_time = np.linspace(0, 0.99, len(groundtruth[node_no]))
                            for iidx in range(len(x_time)):
                                # ground truth
                                if bound_t_context == 0:
                                    for dim_i in range(x_dim):
                                        data.append(
                                            [x_time[iidx], node_no,
                                             groundtruth[node_no][iidx][dim_i],
                                             state_label[dim_i],
                                             0, 'Groundtruth', dynamics,
                                             topo])
                                             
                                      

                                    data_sum.append(
                                        [x_time[iidx], node_no,
                                         torch.sum(groundtruth[node_no][iidx][0:x_dim]),
                                         'Sum',
                                         0, 'Groundtruth', dynamics,
                                         topo])

                                if ndcn_flag:
                                    ## ndcn
                                    for dim_i in range(x_dim):
                                        data_ndcn.append(
                                            [x_time[iidx], node_no,
                                             ndcn_predictions[iidx][node_no][dim_i],
                                             state_label[dim_i],
                                             0, 'NDCN-%s'% (bound_t_context), dynamics,
                                             topo])
                                    
                                    data_sum.append(
                                        [x_time[iidx], node_no,
                                         np.sum(ndcn_predictions[iidx][node_no][0:x_dim]),
                                         'Sum',
                                         0, 'NDCN-%s'% (bound_t_context), dynamics,
                                         topo])

                                for j in range(len(predictions['mean'])):
                                    # data.append([x_time[iidx], predictions_sum[j][iidx], 'GraphNDP-%s-%s' % (bound_t_context,j), dynamics, topo])
                                    for dim_i in range(x_dim):
                                        data.append(
                                            [x_time[iidx], node_no,
                                             predictions['mean'][j][node_no][iidx][dim_i],
                                             state_label[dim_i],
                                             j + 1,
                                             'GraphNDP-%s' % (bound_t_context),
                                             dynamics,
                                             topo])

                                    """
                                    data.append(
                                        [x_time[iidx], node_no,
                                         predictions_wo_softmax['mean'][j][node_no][iidx][0],
                                         'Susceptible',
                                         j + 1,
                                         'GraphNDP-wo-softmax-%s' % (bound_t_context),
                                         dynamics,
                                         topo])
                                    data.append(
                                        [x_time[iidx], node_no,
                                         predictions_wo_softmax['mean'][j][node_no][iidx][1],
                                         'Infected',
                                         j + 1,
                                         'GraphNDP-wo-softmax-%s' % (bound_t_context),
                                         dynamics,
                                         topo])
                                    data.append(
                                        [x_time[iidx], node_no,
                                         predictions_wo_softmax['mean'][j][node_no][iidx][2],
                                         'Removed',
                                         j + 1,
                                         'GraphNDP-wo-softmax-%s' % (bound_t_context),
                                         dynamics,
                                         topo])
                                    data.append(
                                        [x_time[iidx], node_no,
                                         predictions_wo_softmax['mean'][j][node_no][iidx][3],
                                         'Exposed',
                                         j + 1,
                                         'GraphNDP-wo-softmax-%s' % (bound_t_context),
                                         dynamics,
                                         topo])
                                    """     



                                    data_sum.append(
                                        [x_time[iidx], node_no,
                                         torch.sum(predictions['mean'][j][node_no][iidx][0:x_dim]),
                                         'Sum',
                                         j + 1,
                                         'GraphNDP-%s' % (bound_t_context),
                                         dynamics,
                                         topo])

            df = pd.DataFrame(data,
                                  columns=['time', 'node_no',
                                           'state',
                                           'State types',
                                           'sampled_z_i', 'Methods',
                                           'Dynamics types',
                                           'Topology types'], dtype=float)
            df_sum = pd.DataFrame(data_sum,
                                      columns=['time', 'node_no',
                                               'state',
                                               'State types',
                                               'sampled_z_i', 'Methods',
                                               'Dynamics types',
                                               'Topology types'], dtype=float)
            if ndcn_flag:
                df_ndcn = pd.DataFrame(data_ndcn,
                                      columns=['time', 'node_no',
                                               'state',
                                               'State types',
                                               'sampled_z_i', 'Methods',
                                               'Dynamics types',
                                               'Topology types'], dtype=float)

            # each states
            for bound_t_context_idx in range(len(bound_t_context_list)):
                bound_t_context = bound_t_context_list[bound_t_context_idx]
                for dim_i in range(x_dim):
                
                    plt.figure(figsize=figsize)
    
                    x_time = np.linspace(0, 0.99, len(groundtruth[0]))
    
                    df_gt = df[df['Methods'] == 'Groundtruth']
                    
                    
                    state_color = {'Susceptible': 'g', 'Infected': 'r', 'Removed': (0.2, 0.2, 0.2), 'Exposed': 'b',
                                  'SS': 'g', 'IS': 'b', 'SI': 'c', 'II': 'r',
                                  }
                    
                    New_X_i = df_gt[df_gt['State types'] == state_label[dim_i]]['state'].to_numpy().reshape(test_N, 100)
                    plt.plot(x_time, np.mean(New_X_i, axis=0), c=state_color[state_label[dim_i]], linewidth=3)
                    # plt.legend(['GT(Susceptible)', 'GT(Infected)', 'GT(Removed)', 'GT(Exposed)'])
                    # plt.title(str(function_.__name__))
                    
                    if ndcn_flag:
                        # ndcn
                        df_ndcn_i = df_ndcn[df_ndcn['Methods'] == 'NDCN-%s' % (bound_t_context)]
                        
                        New_X_i = df_ndcn_i[df_ndcn_i['State types'] == state_label[dim_i]]['state'].to_numpy().reshape(test_N, 100)
                        plt.plot(x_time, np.mean(New_X_i, axis=0), c=state_color[state_label[dim_i]], 
                                 markersize=10,
                                 markevery=25,
                                 alpha=0.7, linewidth=3, ls=':', markerfacecolor='none',marker='s')
    
                    # ours
                    
                    df_ours = df[df['Methods'] == 'GraphNDP-%s' % (bound_t_context)]
                    
                    
                    New_X_i = df_ours[df_ours['State types'] == state_label[dim_i]]['state'].to_numpy().reshape(test_N, 100, 20)
        
                    plt.plot(x_time, np.mean(np.mean(New_X_i, axis=0), axis=-1), ls='--', c=state_color[state_label[dim_i]], marker='o', markerfacecolor='none',
                             markersize=10,
                             markevery=20,
                             alpha=0.7, linewidth=3)
                        
                    plt.fill_between(x_time,
                                         np.mean(np.mean(New_X_i, axis=0), axis=-1) + 1.96 * np.std(np.mean(New_X_i, axis=0),
                                                                                                    axis=-1),
                                         np.mean(np.mean(New_X_i, axis=0), axis=-1) - 1.96 * np.std(np.mean(New_X_i, axis=0),
                                                                                                    axis=-1),
                                         facecolor=state_color[state_label[dim_i]], alpha=0.2)
    
                    # plt.legend(['Groundtruth(S)', 'Groundtruth(I)', 'Groundtruth(R)', 'Groundtruth(E)',
                    #             'GraphNDP-%s(S)' % (bound_t_context),
                    #             'GraphNDP-%s(I)' % (bound_t_context),
                    #             'GraphNDP-%s(R)' % (bound_t_context),
                    #             'GraphNDP-%s(E)' % (bound_t_context)])
    
                    plt.ylabel('Avg. states')
                    plt.xlabel('Time')
    
                    plt.ylim(-0.2, 1.2)
    
                    plt.tight_layout()
    
                    if save_fig:
                        plt.savefig('results/EXP3_epidemic_4dim_states_%s_%s_%s_%s_dim%s.png' % (
                            dynamics, topo, test_ids, bound_t_context, dim_i))
                    else:
                        plt.show()
                    plt.close()


            ### sum of state
            figsize = (9, 6)
            plt.figure(figsize=figsize)

            x_time = np.linspace(0, 0.99, len(groundtruth[0]))

            df_gt = df[df['Methods'] == 'Groundtruth']
            New_X_sum = 0
            for dim_i in range(x_dim):
              New_X_sum += df_gt[df_gt['State types'] == state_label[dim_i]]['state'].to_numpy().reshape(test_N, 100)
            
            plt.plot(x_time, np.mean(New_X_sum, axis=0), c='k', linewidth=2)

            for bound_t_context_idx in range(len(bound_t_context_list)):
                bound_t_context = bound_t_context_list[bound_t_context_idx]

                # ndcn
                if ndcn_flag:
                    df_ndcn_i = df_ndcn[df_ndcn['Methods'] == 'NDCN-%s' % (bound_t_context)]
                    New_X_sum = 0
                    for dim_i in range(x_dim):
                        New_X_sum += df_ndcn_i[df_ndcn_i['State types'] == state_label[dim_i]]['state'].to_numpy().reshape(test_N, 100)
                        
                    plt.plot(x_time, np.mean(New_X_sum, axis=0), ls=':', c=color_list[bound_t_context_idx], marker='s', markerfacecolor='w',
                              markersize=10,
                              markevery=25,
                              alpha=0.5, linewidth=2)
                
                """
                df_ours_wo_softmax = df[df['Methods'] == 'GraphNDP-wo-softmax-%s' % (bound_t_context)]
                New_X_S = df_ours_wo_softmax[df_ours_wo_softmax['State types'] == 'Susceptible']['state'].to_numpy().reshape(test_N, 100, 20)
                New_X_I = df_ours_wo_softmax[df_ours_wo_softmax['State types'] == 'Infected']['state'].to_numpy().reshape(test_N, 100, 20)
                New_X_R = df_ours_wo_softmax[df_ours_wo_softmax['State types'] == 'Removed']['state'].to_numpy().reshape(test_N, 100, 20)
                New_X_E = df_ours_wo_softmax[df_ours_wo_softmax['State types'] == 'Exposed']['state'].to_numpy().reshape(test_N, 100, 20)

                plt.plot(x_time, np.mean(np.mean(New_X_S + New_X_I + New_X_R + New_X_E, axis=0), axis=-1), ls='--',
                         c=color_list[bound_t_context_idx], marker='x', markerfacecolor='w',
                         markersize=10,
                         markevery=10,
                         alpha=0.5, linewidth=2)

                plt.fill_between(x_time,
                                 np.mean(np.mean(New_X_S + New_X_I + New_X_R + New_X_E, axis=0),
                                         axis=-1) + 1.96 * np.std(
                                     np.mean(New_X_S + New_X_I + New_X_R + New_X_E, axis=0),
                                     axis=-1),
                                 np.mean(np.mean(New_X_S + New_X_I + New_X_R + New_X_E, axis=0),
                                         axis=-1) - 1.96 * np.std(
                                     np.mean(New_X_S + New_X_I + New_X_R + New_X_E, axis=0),
                                     axis=-1),
                                 facecolor=color_list[bound_t_context_idx], alpha=0.2)
                """

                # ours
                df_ours = df[df['Methods'] == 'GraphNDP-%s' % (bound_t_context)]
                New_X_sum = 0
                for dim_i in range(x_dim):
                    New_X_sum += df_ours[df_ours['State types'] == state_label[dim_i]]['state'].to_numpy().reshape(test_N, 100, 20)
               
                plt.plot(x_time, np.mean(np.mean(New_X_sum, axis=0), axis=-1), ls='--',c=color_list[bound_t_context_idx], marker='o', markerfacecolor='w',
                         markersize=10,
                         markevery=20,
                         alpha=0.5, linewidth=2)

                plt.fill_between(x_time,
                                 np.mean(np.mean(New_X_sum, axis=0), axis=-1) + 1.96 * np.std(np.mean(New_X_sum, axis=0),
                                                                                            axis=-1),
                                 np.mean(np.mean(New_X_sum, axis=0), axis=-1) - 1.96 * np.std(np.mean(New_X_sum, axis=0),
                                                                                            axis=-1),
                                 facecolor=color_list[bound_t_context_idx], alpha=0.2)

                # plt.legend(['Groundtruth(S)', 'Groundtruth(I)', 'Groundtruth(R)', 'Groundtruth(E)',
                #             'GraphNDP-%s(S)' % (bound_t_context),
                #             'GraphNDP-%s(I)' % (bound_t_context),
                #             'GraphNDP-%s(R)' % (bound_t_context),
                #             'GraphNDP-%s(E)' % (bound_t_context)])

            plt.ylabel('Avg. prob. sum')
            plt.xlabel('Time')

            plt.tight_layout()

            if save_fig:
                    plt.savefig('results/EXP3_epidemic_4dim_states_%s_%s_%s_sum_states.png' % (
                        dynamics, topo, test_ids))
            else:
                    plt.show()
            plt.close()

            # show_each_node
            if show_each_node:
              for bound_t_context_idx in range(len(bound_t_context_list)):
                  bound_t_context = bound_t_context_list[bound_t_context_idx]
                  for dim_i in range(x_dim):
                      if dim_i == 1:
                          figsize = (6, 12)
                          plt.figure(figsize=figsize)
  
                          x_time = np.linspace(0, 0.99, len(groundtruth[0]))
  
                          # groundtruth
                          df_gt = df[df['Methods'] == 'Groundtruth']
  
                          state_color = {'Susceptible': 'g', 'Infected': 'r', 'Removed': (0.2, 0.2, 0.2), 'Exposed': 'b'}
  
                          New_X_i = df_gt[df_gt['State types'] == state_label[dim_i]]['state'].to_numpy().reshape(test_N, 100)
                          
                          print('groundtruth=',New_X_i)
                          
                          normalized_x_i = np.max(New_X_i, axis=-1)
                          normalized_x_i = np.ones_like(normalized_x_i)
                          
                          shift_y = 0
                          for i in range(test_N):
                              if i < 52:
                                  plt.plot(x_time,
                                           np.ones_like(x_time) * shift_y, 'k',
                                           alpha=0.5, linewidth=0.5)
                                  plt.fill_between(x_time,
                                                   np.ones_like(x_time) * shift_y,
                                                   New_X_i[i] / normalized_x_i[i] + shift_y,
                                                   color='g', alpha=0.8)
                                  plt.plot(x_time, New_X_i[i] / normalized_x_i[i] + shift_y, 'k',
                                           alpha=0.5,
                                           linewidth=0.5)
      
                                  shift_y += 0
  
                          # ours
                          df_ours = df[df['Methods'] == 'GraphNDP-%s' % (bound_t_context)]
  
                          New_X_i = df_ours[df_ours['State types'] == state_label[dim_i]]['state'].to_numpy().reshape(
                              test_N, 100, 20)
                              
                          print('predictions=', np.mean(New_X_i, axis=-1))
                          
                          #if bound_t_context == 0.75:
                          #    exit(1)
  
                          shift_y = 0
                          for i in range(test_N):
                              if i < 52:
                                  plt.plot(x_time,
                                           np.ones_like(x_time) * shift_y, 'k',
                                           alpha=0.5, linewidth=0.5)
                                  plt.fill_between(x_time,
                                                   np.ones_like(x_time) * shift_y,
                                                   np.mean(New_X_i[i], axis=-1) / normalized_x_i[i] + shift_y,
                                                   color='r', alpha=0.8)
                                  plt.plot(x_time, np.mean(New_X_i[i], axis=-1) / normalized_x_i[i] + shift_y, 'k',
                                           alpha=0.5,
                                           linewidth=0.5)
      
                                  shift_y += 0
  
                          plt.ylabel('Infected')
                          plt.xlabel('Time')
  
                          # plt.ylim(-0.2, 1.2)
  
                          plt.tight_layout()
  
                          if save_fig:
                              plt.savefig('results/EXP3_epidemic_each_node_%s_%s_%s_%s_dim%s.png' % (
                                  dynamics, topo, test_ids, bound_t_context, dim_i))
                          else:
                              plt.show()
                          plt.close()

            #
            # sns.lineplot(data)

            # sns.lineplot(data=df_gt, x='time', y='state',
            #              hue='State types',
            #              # units='sampled_z_i', estimator=None,
            #              alpha=1.,
            #              lw=1,
            #              style='State types',
            #              palette=sns.color_palette(['k', 'k', 'k', 'k']),
            #              legend='brief'
            #              )
            # for bound_t_context_idx in range(len(bound_t_context_list)):
            #     # plt.figure(figsize=figsize)
            #     # for sampled_z_i in range(1, 21):
            #     #     if sampled_z_i == 1:
            #     #         legend = 'brief'
            #     #     else:
            #     #         legend = False
            #     #     df_i = df[df['sampled_z_i'] == sampled_z_i]
            #     #     sns.lineplot(
            #     #         data=df_i[df_i['Methods'] == 'GraphNDP-%s' % (bound_t_context_list[bound_t_context_idx])],
            #     #         x='time', y='state', hue='State types',
            #     #         #units='sampled_z_i', estimator=None,
            #     #         alpha=0.5,
            #     #         lw=1,
            #     #         style='State types',
            #     #         palette=sns.color_palette("pastel")[bound_t_context_idx:bound_t_context_idx + 1] * 4,
            #     #         legend=legend
            #     #     )
            #
            #     sns.lineplot(
            #         data=df[df['Methods'] == 'GraphNDP-%s' % (bound_t_context_list[bound_t_context_idx])],
            #         x='time', y='state', hue='State types',
            #         # units='sampled_z_i', estimator=None,
            #         estimator='mean', ci=95,
            #         alpha=0.5,
            #         lw=1,
            #         style='State types',
            #         palette=sns.color_palette("pastel")[bound_t_context_idx:bound_t_context_idx + 1] * 4,
            #     )
            #
            # if save_fig:
            #     plt.savefig('results/states_on_nodes_%s_%s_%s_%s_draw_2D%s.png' % (
            #         dynamics, topo, test_ids, bound_t_context_list[bound_t_context_idx], str(draw_2d)))
            #     # plt.savefig('results/states_on_node_%s_%s_%s_node%s_%s.png' % (dynamics, topo, test_ids, node_no, bound_t_context_list[bound_t_context_idx]))
            # else:
            #     plt.show()
            # plt.close()
            #
            # #plot sum
            # sns.lineplot(data=df_sum, x='time', y='state',
            #              hue='Methods',
            #              # units='sampled_z_i', estimator=None,
            #              alpha=0.6,
            #              lw=1,
            #              style='Methods',
            #              palette=sns.color_palette([(0, 0, 0)] + sns.color_palette("pastel")[:4]),
            #              )
            #
            # sns.lineplot(data=df_sum[df_sum['Methods'] == 'Groundtruth'],
            #              x='time', y='state', hue='Methods',
            #              # units='sampled_z_i', estimator=None,
            #              alpha=0.6,
            #              lw=1,
            #              # style='Methods',
            #              palette=sns.color_palette([(0, 0, 0)]),
            #              legend=False)
            # plt.ylim(0.8, 1.2)
            #
            # if save_fig:
            #     plt.savefig('results/sum_states_on_nodes_%s_%s_%s_%s_draw_2D%s.png' % (
            #         dynamics, topo, test_ids, bound_t_context_list[bound_t_context_idx], str(draw_2d)))
            #     # plt.savefig('results/states_on_node_%s_%s_%s_node%s_%s.png' % (dynamics, topo, test_ids, node_no, bound_t_context_list[bound_t_context_idx]))
            # else:
            #     plt.show()
            # plt.close()


def display_3_dynamics_on_5_topo():
    save_fig = True
    show_run_no_indexs_per_dynamic_topo_dict = None
    
    dynamics_topo_list = [('heat_diffusion_dynamics', 'grid'),
                                   ('mutualistic_interaction_dynamics', 'grid'),
                                   ('gene_regulatory_dynamics', 'grid'),
                                    ('combination_dynamics_vary_coeff', 'grid'),
                                   # ('vary_dynamics_with_vary_type_and_coeff', 'grid'),
                                    ('heat_diffusion_dynamics', 'power_law'),
                                   ('mutualistic_interaction_dynamics', 'power_law'),
                                   ('gene_regulatory_dynamics', 'power_law'),
                                    ('combination_dynamics_vary_coeff', 'power_law'),
                                   # ('vary_dynamics_with_vary_type_and_coeff', 'power_law'),
                                   ('heat_diffusion_dynamics', 'random'),
                                   ('mutualistic_interaction_dynamics', 'random'),
                                   ('gene_regulatory_dynamics', 'random'),
                                    ('combination_dynamics_vary_coeff', 'random'),
                                   # ('vary_dynamics_with_vary_type_and_coeff', 'random'),
                                   ('heat_diffusion_dynamics', 'small_world'),
                                   ('mutualistic_interaction_dynamics', 'small_world'),
                                   ('gene_regulatory_dynamics', 'small_world'),
                                    ('combination_dynamics_vary_coeff', 'small_world'),
                                   # ('vary_dynamics_with_vary_type_and_coeff', 'small_world'),
                                   ('heat_diffusion_dynamics', 'community'),
                                   ('mutualistic_interaction_dynamics', 'community'),
                                   ('gene_regulatory_dynamics', 'community'),
                                    ('combination_dynamics_vary_coeff', 'community'),
                                   # ('vary_dynamics_with_vary_type_and_coeff', 'community'),
                                   ('opinion_dynamics_Baumann2021_2topic', 'small_world'),
                                   ('SI_Individual_dynamics', 'power_law'),
                                   ('SIS_Individual_dynamics', 'power_law'),
                                   ('SIR_Individual_dynamics', 'power_law'),
                                   ('SEIS_Individual_dynamics', 'power_law'),
                                   ('SEIR_Individual_dynamics', 'power_law'),
            ]
    # dynamics_topo_list = [ ('opinion_dynamics_Baumann2021_2topic', 'small_world'),
    #                                #('SI_Individual_dynamics', 'power_law'),
    #                                #('SIS_Individual_dynamics', 'power_law'),
    #                                #('SIR_Individual_dynamics', 'power_law'),
    #                                #('SEIS_Individual_dynamics', 'power_law'),
    #                                #('SEIR_Individual_dynamics', 'power_law'),
    #         ]
    #

    #bound_t_context_list = [0.0, 0.25, 0.5, 0.75]
    bound_t_context_list = [0.0, 0.25, 0.5, 0.75]

    #bound_t_context_list = [0.5]
    
    test_N = 225

    test_num_trials = 20

    
    #plot_violinplot_new(save_fig, add_str="3dynamics5topo_together_plot_vary_coeff_vary_type_and_coeff", plot_type='box',
    #                     dynamics_topo_list=dynamics_topo_list, bound_t_context_list=bound_t_context_list,
    #                     test_N=test_N, test_num_trials=test_num_trials, ndcn_flag=True)
    #exit(1)

    # for train_topo in ['grid',
    #                    'power_law',
    #                    'random',
    #                    'small_world',
    #                    'community']:
    #     topo_list_but_one = topo_list.copy()
    #     topo_list_but_one.remove(train_topo)
    #     print(topo_list_but_one)
    #     plot_violinplot(save_fig, add_str="3dynamics5topo_onedynamics_onetopo_difftopo_trainon%s" % train_topo,
    #                     plot_type='violin',
    #                     dynamics_list=dynamics_list, topo_list=topo_list_but_one,
    #                     bound_t_context_list=bound_t_context_list,
    #                     test_N=test_N, test_num_trials=test_num_trials,
    #                     exp_type='3dynamics5topo_onedynamics_onetopo_difftopo',
    #                     ndcn_flag=True,
    #                     train_topo=train_topo)

    # for ii in range(10):
    #     print(ii)
    #     plot_sum_state_compare([ii], save_fig)
    # exit(1)
    
    show_run_no_indexs_dict = {('heat_diffusion_dynamics', 'grid'): {'trial_no': [0], 'node_no': [79]},
                               ('heat_diffusion_dynamics', 'power_law'): {'trial_no': [5], 'node_no': [104]},  # ok
                               ('heat_diffusion_dynamics', 'random'): {'trial_no': [7], 'node_no': [100]},
                               ('heat_diffusion_dynamics', 'small_world'): {'trial_no': [4], 'node_no': [0]},
                               ('heat_diffusion_dynamics', 'community'): {'trial_no': [5], 'node_no': [178]},  # ok 160
                               ('mutualistic_interaction_dynamics', 'grid'): {'trial_no': [4], 'node_no': [0]},
                               ('mutualistic_interaction_dynamics', 'power_law'): {'trial_no': [1], 'node_no': [25]},
                               # ok
                               ('mutualistic_interaction_dynamics', 'random'): {'trial_no': [4], 'node_no': [0]},
                               ('mutualistic_interaction_dynamics', 'small_world'): {'trial_no': [1], 'node_no': [0]},
                               ('mutualistic_interaction_dynamics', 'community'): {'trial_no': [5], 'node_no': [101]},
                               # ok
                               ('gene_regulatory_dynamics', 'grid'): {'trial_no': [1], 'node_no': [0]},
                               ('gene_regulatory_dynamics', 'power_law'): {'trial_no': [1], 'node_no': [167]},  # ok 35
                               ('gene_regulatory_dynamics', 'random'): {'trial_no': [1], 'node_no': [0]},
                               ('gene_regulatory_dynamics', 'small_world'): {'trial_no': [1], 'node_no': [0]},
                               ('gene_regulatory_dynamics', 'community'): {'trial_no': [5], 'node_no': [223]},  # ok

                               ('combination_dynamics_vary_coeff', 'grid'): {'trial_no': [0], 'node_no': [0]},
                               ('combination_dynamics_vary_coeff', 'power_law'): {'trial_no': [0],
                                                                                  'node_no': [167]},  # ok 35
                               ('combination_dynamics_vary_coeff', 'random'): {'trial_no': [1],
                                                                               'node_no': [0]},
                               ('combination_dynamics_vary_coeff', 'small_world'): {'trial_no': [0],
                                                                                    'node_no': [0]},
                               ('combination_dynamics_vary_coeff', 'community'): {'trial_no': [1],
                                                                                  'node_no': [223]},  # ok

                               ('vary_dynamics_with_vary_type_and_coeff', 'grid'): {'trial_no': [0], 'node_no': [0]},
                               ('vary_dynamics_with_vary_type_and_coeff', 'power_law'): {'trial_no': [0],
                                                                                         'node_no': [167]},  # ok 35
                               ('vary_dynamics_with_vary_type_and_coeff', 'random'): {'trial_no': [0],
                                                                                      'node_no': [0]},
                               ('vary_dynamics_with_vary_type_and_coeff', 'small_world'): {'trial_no': [0],
                                                                                           'node_no': [0]},
                               ('vary_dynamics_with_vary_type_and_coeff', 'community'): {'trial_no': [0],
                                                                                         'node_no': [223]},  # ok

                               }
    #plot_sum_state_compare([0], save_fig, dynamics_topo_list, show_run_no_indexs_dict)
    plot_state_on_one_node_compare([0], save_fig, dynamics_topo_list, show_run_no_indexs_dict)
    


def display_opinion_dynamics():
    save_fig = True

    dynamics_list = [
        'opinion_dynamics'
    ]
    topo_list = [
        'full_connected'
    ]
    bound_t_context_list = [0.0, 0.25, 0.5, 0.75]

    test_N = 40
    # test_N = -1

    test_num_trials = 50
    # test_num_trials = 10

    plot_violinplot(save_fig, add_str="one_dynamics_one_topo", plot_type='box',
                    dynamics_list=dynamics_list, topo_list=topo_list, bound_t_context_list=bound_t_context_list,
                    test_N=test_N, test_num_trials=test_num_trials, exp_type='opinion_dynamics_onetopo')
    plot_opinion_dynamics(dynamics_list, topo_list, [44], bound_t_context_list, save_fig=save_fig, test_N=test_N,
                          test_num_trials=test_num_trials)  # [42]


def display_opinion_dynamics_Baumann2021():
    save_fig = True

    dynamics_list = [
        'opinion_dynamics_Baumann2021'
    ]
    topo_list = [
        'community'
    ]
    # bound_t_context_list = [0.0, 0.25, 0.5, 0.75]
    bound_t_context_list = [0.0, 0.2, 0.4]

    test_N = 200
    # test_N = 250
    # test_N = -1

    test_num_trials = 50
    # test_num_trials = 10

    plot_violinplot(save_fig, add_str="one_dynamics_one_topo", plot_type='violin',
                    dynamics_list=dynamics_list, topo_list=topo_list, bound_t_context_list=bound_t_context_list,
                    test_N=test_N, test_num_trials=test_num_trials, exp_type='opinion_dynamics_onetopo')
    plot_opinion_dynamics(dynamics_list, topo_list, [2], bound_t_context_list, save_fig=save_fig, test_N=test_N,
                          test_num_trials=test_num_trials)  # [0]


def display_opinion_dynamics_Baumann2021_2topic():
    save_fig = True

    dynamics_list = [
        'opinion_dynamics_Baumann2021_2topic'
    ]
    topo_list = [
        # 'community',
        'small_world'
    ]
    bound_t_context_list = [0.0, 0.25, 0.5, 0.75]
    # bound_t_context_list = [0.0, 0.2, 0.4]
    # bound_t_context_list = [0.0, 0.2]

    test_N = 200
    # test_N = 250
    # test_N = -1

    # test_num_trials = 500
    # test_num_trials = 50
    test_num_trials = 100
    #
    # plot_violinplot(save_fig, add_str="one_dynamics_one_topo", plot_type='box',
    #                 dynamics_list=dynamics_list, topo_list=topo_list, bound_t_context_list=bound_t_context_list,
    #                 test_N=test_N, test_num_trials=test_num_trials, exp_type='opinion_dynamics_onetopo', x_dim=2,
    #                 ndcn_flag=True)
    plot_violinplot(False, add_str="one_dynamics_one_topo", plot_type='box',
                    dynamics_list=dynamics_list, topo_list=topo_list, bound_t_context_list=bound_t_context_list,
                    test_N=test_N, test_num_trials=test_num_trials, exp_type='opinion_dynamics_onetopo', x_dim=2,
                   ndcn_flag=True)
    # for ii in [0, 1, 2, 3, 4]:
    #      plot_opinion_dynamics_2topic(dynamics_list, topo_list, [ii], bound_t_context_list, save_fig=save_fig,
    #                                   test_N=test_N,
    #                                   test_num_trials=test_num_trials, x_dim=2, draw_2d=True, ndcn_flag=True)  # [0]

    ###
    """
    data_metric_on_cases_NDCN = {
            0:{'MAE':[0.348, 0.080,0.066,0.059], 'Kendalltau':[-0.456,0.898,0.905,0.891]},
            1:{'MAE':[1.031,0.873,0.832,0.798], 'Kendalltau':[0.422,0.499,0.588,0.594]},
            2:{'MAE':[1.089,0.929,0.928,0.849], 'Kendalltau':[0.280,0.612,0.609,0.672]},
            3:{'MAE':[2.677,2.357, 2.354,2.356], 'Kendalltau':[-0.096, 0.619, 0.615, 0.617]},
            4:{'MAE':[2.256,2.250,2.204,1.182], 'Kendalltau':[0.156,0.115,0.132,0.684]},
        }

    data_metric_on_cases_Ours = {
            0:{'MAE':[0.423,0.272,0.243,0.057], 'Kendalltau':[0.466,0.467,0.490,0.877]},
            1:{'MAE':[0.615,0.811,0.684,0.496], 'Kendalltau':[0.752,0.537,0.661,0.768]},
            2:{'MAE':[1.270,1.082,1.082,0.405], 'Kendalltau':[0.062,0.167,0.167,0.797]},
            3:{'MAE':[2.199, 2.318, 2.318, 2.318], 'Kendalltau':[0.378, 0.450, 0.450, 0.450]},
            4:{'MAE':[2.122,1.888,1.294,0.517], 'Kendalltau':[0.146,0.642,0.809,0.903]},
        }
    for ii in [0, 1, 2, 3, 4]:
        figsize = (6, 6)

        sns.set(context='notebook', style='ticks', font_scale=2, palette="pastel")

        fig = plt.figure(figsize=figsize)
        ax1 = fig.add_subplot(111)

        ax1.plot(["$0$", "$(0,0.25]$", "$(0,0.5]$", "$(0,0.75]$"],
                     data_metric_on_cases_NDCN[ii]['MAE'],
                     '--', c='b',
                     marker='s',
                     markerfacecolor='w',
                     markersize=10, markevery=1, alpha=1, linewidth=2,
                 label='NDCN')
        ax1.plot(["$0$", "$(0,0.25]$", "$(0,0.5]$", "$(0,0.75]$"],
                 data_metric_on_cases_Ours[ii]['MAE'],
                 '-', c='b',
                 marker='o',
                 markerfacecolor='w',
                 markersize=10, markevery=1, alpha=1, linewidth=2,
                 label='GMNND')

        ax1.set_xlabel('$t_{obs}$')
        ax1.set_ylabel('MAE')
        # ax1.set_ylim(0, 3)

        ax2 = ax1.twinx()

        ax2.plot(["$0$", "$(0,0.25]$", "$(0,0.5]$", "$(0,0.75]$"],
                     data_metric_on_cases_NDCN[ii]['Kendalltau'],
                     '--', c='r',
                     marker='s',
                     markerfacecolor='w',
                     markersize=10, markevery=1, alpha=1, linewidth=2,
                 label='NDCN')
        ax2.plot(["$0$", "$(0,0.25]$", "$(0,0.5]$", "$(0,0.75]$"],
                 data_metric_on_cases_Ours[ii]['Kendalltau'],
                 '-', c='r',
                 marker='o',
                 markerfacecolor='w',
                 markersize=10, markevery=1, alpha=1, linewidth=2,
                 label='GMNND')

        ax2.set_ylabel('Kendalltau')
        ax2.set_ylim(-1,1)

        # ax1.legend(loc='upper left')
        # ax2.legend(loc='lower right')
        ax1.set_xticklabels(["$0$", "$(0,0.25]$", "$(0,0.5]$", "$(0,0.75]$"], rotation=90)

        plt.tight_layout()


        plt.savefig('results/EXP2_test_metric_for_case%s.png'%ii)


    """



def display_all_epidemic():
    save_fig = True

    dynamics_list = ['SI_Individual_dynamics', 'SIS_Individual_dynamics', 'SIR_Individual_dynamics',
                     'SEIS_Individual_dynamics', 'SEIR_Individual_dynamics']
    topo_list = [
        'power_law'
    ]
    bound_t_context_list = [0.0, 0.25, 0.5, 0.75]

    test_N = 200

    test_num_trials = 100
    #
    # plot_violinplot(save_fig, add_str="5dynamics1topo_onedynamics_onetopo_all_epidemic", plot_type='box',
    #                 dynamics_list=dynamics_list, topo_list=topo_list, bound_t_context_list=bound_t_context_list,
    #                 test_N=test_N, test_num_trials=test_num_trials,
    #                 exp_type='5dynamics1topo_onedynamics_onetopo_all_epidemic', x_dim=4,
    #                 ndcn_flag=True)
    
    #plot_violinplot(save_fig, add_str="5dynamics1topo_onedynamics_onetopo_all_epidemic", plot_type='box',
    #                 dynamics_list=dynamics_list, topo_list=topo_list, bound_t_context_list=bound_t_context_list,
    #                 test_N=test_N, test_num_trials=test_num_trials,
    #                 exp_type='5dynamics1topo_onedynamics_onetopo_all_epidemic', x_dim=4,
    #                 ndcn_flag=True)
    #return


    dynamics_list = ['SI_Individual_dynamics']
    for ii in [0]:
        plot_epidemic_4dim(dynamics_list, topo_list, [ii], bound_t_context_list, save_fig=save_fig, test_N=test_N,
                           test_num_trials=test_num_trials, x_dim=2, draw_2d=True, ndcn_flag=True)  # [0]

    dynamics_list = ['SIS_Individual_dynamics']
    for ii in [0]:
        plot_epidemic_4dim(dynamics_list, topo_list, [ii], bound_t_context_list, save_fig=save_fig, test_N=test_N,
                           test_num_trials=test_num_trials, x_dim=2, draw_2d=True, ndcn_flag=True)  # [0]
    dynamics_list = ['SEIS_Individual_dynamics']
    for ii in [0]:
        plot_epidemic_4dim(dynamics_list, topo_list, [ii], bound_t_context_list, save_fig=save_fig, test_N=test_N,
                           test_num_trials=test_num_trials, x_dim=3, draw_2d=True, ndcn_flag=True)  # [0]
    dynamics_list = ['SIR_Individual_dynamics']
    for ii in [0]:
        plot_epidemic_4dim(dynamics_list, topo_list, [ii], bound_t_context_list, save_fig=save_fig, test_N=test_N,
                           test_num_trials=test_num_trials, x_dim=3, draw_2d=True, ndcn_flag=True)  # [0]
    for ii in [1]:
        plot_epidemic_4dim(dynamics_list, topo_list, [ii], bound_t_context_list, save_fig=save_fig, test_N=test_N,
                           test_num_trials=test_num_trials, x_dim=3, draw_2d=True, ndcn_flag=True)  # [0]
    for ii in [2]:
        plot_epidemic_4dim(dynamics_list, topo_list, [ii], bound_t_context_list, save_fig=save_fig, test_N=test_N,
                           test_num_trials=test_num_trials, x_dim=3, draw_2d=True, ndcn_flag=True)  # [0]
    for ii in [3]:
        plot_epidemic_4dim(dynamics_list, topo_list, [ii], bound_t_context_list, save_fig=save_fig, test_N=test_N,
                           test_num_trials=test_num_trials, x_dim=3, draw_2d=True, ndcn_flag=True)  # [0]
    
    
    dynamics_list = ['SEIR_Individual_dynamics']
    for ii in [0,1,2]:
        plot_epidemic_4dim(dynamics_list, topo_list, [ii], bound_t_context_list, save_fig=save_fig, test_N=test_N,
                           test_num_trials=test_num_trials, x_dim=4, draw_2d=True, ndcn_flag=True)  # [0]
    """
    dynamics_list = ['SEIR_Individual_dynamics']
    for ii in [0]:
        plot_epidemic_4dim(dynamics_list, topo_list, [ii], bound_t_context_list, save_fig=save_fig, test_N=test_N,
                           test_num_trials=test_num_trials, x_dim=4, draw_2d=True, ndcn_flag=True)  # [0]
    """

def display_coupled_SIS_epidemic():
    save_fig = True

    dynamics_list = ['Coupled_Epidemic_dynamics', ]
    topo_list = [
        'power_law'
    ]
    bound_t_context_list = [0.0, 0.25, 0.5, 0.75]

    test_N = 200

    test_num_trials = 100

    plot_violinplot(save_fig, add_str="Coupled_Epidemic", plot_type='box',
                    dynamics_list=dynamics_list, topo_list=topo_list, bound_t_context_list=bound_t_context_list,
                    test_N=test_N, test_num_trials=test_num_trials,
                    exp_type='Coupled_Epidemic', x_dim=4,
                    ndcn_flag=True)

    for ii in [0, 1, 2, 3]:
    #for ii in [0]:
        plot_epidemic_4dim(dynamics_list, topo_list, [ii], bound_t_context_list, save_fig=save_fig, test_N=test_N,
                           test_num_trials=test_num_trials, x_dim=4, draw_2d=True, ndcn_flag=True, show_each_node=True)  # [0]
                           


def display_SIR_meta_epidemic():
    save_fig = True

    dynamics_list = ['SIR_meta_pop_dynamics', ]
    topo_list = [
        'directed_full_connected'
    ]
    bound_t_context_list = [0.0, 0.25, 0.5, 0.75]

    test_N = 52

    test_num_trials = 100

    plot_violinplot(save_fig, add_str="SIR_meta", plot_type='box',
                    dynamics_list=dynamics_list, topo_list=topo_list, bound_t_context_list=bound_t_context_list,
                    test_N=test_N, test_num_trials=test_num_trials,
                    exp_type='SIR_meta', x_dim=3,
                    ndcn_flag=False)

    for ii in [0, 1, 2, 3]:
        plot_epidemic_4dim(dynamics_list, topo_list, [ii], bound_t_context_list, save_fig=save_fig, test_N=test_N,
                           test_num_trials=test_num_trials, x_dim=3, draw_2d=True, ndcn_flag=False, show_each_node=True)  # [0]
                           


def display_real_epidemic():
    save_fig = True

    dynamics = 'real_data_spain_covid19_cases'
    topo = 'directed_full_connected'
    x_dim = 3

    bound_t_context_list = [0.0, 0.25, 0.5, 0.75]

    test_N = 52

    test_num_trials = 4

    cases_raw = pickle.load(open(
        r"data/DynamicsData/Spain_Covid19/case-timeseries.data",
        'rb'))

    network_raw = pickle.load(open(
        r"data/DynamicsData/Spain_Covid19/province_mobility.data",
        'rb'))
    network_raw = network_raw['all']
    network_raw = network_raw - np.diag(np.diag(network_raw))

    population_raw = pickle.load(open(
        r"data/DynamicsData/Spain_Covid19/population.data",
        'rb'))

    for ii in range(test_num_trials):
        

        sns.set(context='notebook', style='ticks', font_scale=2)
        color_palette = sns.color_palette("pastel")

        state_label = ['Susceptible', 'Infected', 'Removed']

        data = []

        for bound_t_context in bound_t_context_list:

            fname = 'results/saved_test_results_%s_MLlossFalse_deterTrue_uncerTrue_%s_%s_x%s_numgraph1000_timestep100_epoch30_bound_t_context%s_seed1_num_nodes%s_with_2nd_phase.pkl' % (
                model_name, dynamics, topo, x_dim, bound_t_context, test_N)

            with open(fname, 'rb') as f:
                saved_results_data = pickle.load(f)

                idx = ii

                if True:
                        observations = saved_results_data['observations'][idx]
                        groundtruth = saved_results_data['groundtruth'][idx]
                        predictions = saved_results_data['predictions'][idx]

                        for node_no in range(len(groundtruth)):
                            x_time = np.linspace(0, 0.99, len(groundtruth[node_no]))
                            for iidx in range(len(x_time)):
                                # ground truth
                                if bound_t_context == 0:
                                    for dim_i in range(x_dim):
                                        data.append(
                                            [x_time[iidx], node_no,
                                             groundtruth[node_no][iidx][dim_i],
                                             state_label[dim_i],
                                             0, 'Groundtruth', dynamics,
                                             topo])

                                for j in range(len(predictions['mean'])):
                                    # data.append([x_time[iidx], predictions_sum[j][iidx], 'GraphNDP-%s-%s' % (bound_t_context,j), dynamics, topo])
                                    for dim_i in range(x_dim):
                                        data.append(
                                            [x_time[iidx], node_no,
                                             predictions['mean'][j][node_no][iidx][dim_i],
                                             state_label[dim_i],
                                             j + 1,
                                             'GraphNDP-%s' % (bound_t_context),
                                             dynamics,
                                             topo])
        if True:
            df = pd.DataFrame(data,
                              columns=['time', 'node_no',
                                       'state',
                                       'State types',
                                       'sampled_z_i', 'Methods',
                                       'Dynamics types',
                                       'Topology types'], dtype=float)

            # show I
            for bound_t_context_idx in range(len(bound_t_context_list)):
                bound_t_context = bound_t_context_list[bound_t_context_idx]
                for dim_i in range(x_dim):
                    if dim_i == 1:
                        figsize = (6, 12)
                    
                        plt.figure(figsize=figsize)

                        x_time = np.linspace(0, 0.99, len(groundtruth[0]))

                        

                        # ours
                        df_ours = df[df['Methods'] == 'GraphNDP-%s' % (bound_t_context)]

                        New_X_i = df_ours[df_ours['State types'] == state_label[dim_i]]['state'].to_numpy().reshape(
                            test_N, 100, 20)
                            
                        print(np.mean(New_X_i, axis=-1))
                        
                        normalized_x_i = []

                        shift_y = 0
                        for i in range(test_N):
                        
                            normalized_x_i.append(np.max(New_X_i[i]))
                        
                            plt.plot(x_time,
                                     np.ones_like(x_time) * shift_y, 'k',
                                     alpha=0.5, linewidth=0.5)
                            plt.fill_between(x_time,
                                             np.ones_like(x_time) * shift_y,
                                             np.mean(New_X_i, axis=-1)[i] / normalized_x_i[i] + shift_y,
                                             color='r', alpha=0.7)
                            plt.plot(x_time, np.mean(New_X_i, axis=-1)[i] / normalized_x_i[i] + shift_y, 'k',
                                     alpha=0.5,
                                     linewidth=0.5)

                            shift_y += 1
                            
                        # groundtruth
                        df_gt = df[df['Methods'] == 'Groundtruth']

                        state_color = {'Susceptible': 'g', 'Infected': 'r', 'Removed': (0.2, 0.2, 0.2), 'Exposed': 'b'}

                        New_X_i = df_gt[df_gt['State types'] == state_label[dim_i]]['state'].to_numpy().reshape(test_N, 100)       
                        
                        
                        shift_y = 0
                        for i in range(test_N):

                            
                            
                        
                            plt.plot(x_time,
                                     np.ones_like(x_time) * shift_y, 'k',
                                     alpha=0.5, linewidth=0.5)
                            plt.fill_between(x_time,
                                             np.ones_like(x_time) * shift_y,
                                             New_X_i[i] / normalized_x_i[i] + shift_y,
                                             color='g', alpha=0.7)
                            plt.plot(x_time, New_X_i[i] / normalized_x_i[i] + shift_y, 'k',
                                     alpha=0.5,
                                     linewidth=0.5)

                            shift_y += 1

                        plt.ylabel('Infected')
                        plt.xlabel('Time')

                        # plt.ylim(-0.2, 1.2)

                        plt.tight_layout()

                        if save_fig:
                            plt.savefig('results/EXP4_real_epidemic_%s_%s_%s_%s_dim%s.png' % (
                                dynamics, topo, ii, bound_t_context, dim_i))
                        else:
                            plt.show()
                        plt.close()
                        
            # each states
            for bound_t_context_idx in range(len(bound_t_context_list)):
                bound_t_context = bound_t_context_list[bound_t_context_idx]
                for dim_i in range(x_dim):
                    figsize = (6,4)
                
                    plt.figure(figsize=figsize)
    
                    x_time = np.linspace(0, 0.99, len(groundtruth[0]))
    
                    df_gt = df[df['Methods'] == 'Groundtruth']
                    
                    
                    state_color = {'Susceptible': 'g', 'Infected': 'r', 'Removed': (0.2, 0.2, 0.2), 'Exposed': 'b'}
                    
                    New_X_i = df_gt[df_gt['State types'] == state_label[dim_i]]['state'].to_numpy().reshape(test_N, 100)
                    plt.plot(x_time, np.mean(New_X_i, axis=0), c=state_color[state_label[dim_i]], linewidth=3)
                    # plt.legend(['GT(Susceptible)', 'GT(Infected)', 'GT(Removed)', 'GT(Exposed)'])
                    # plt.title(str(function_.__name__))
                    
    
                    # ours
                    
                    df_ours = df[df['Methods'] == 'GraphNDP-%s' % (bound_t_context)]
                    
                    
                    New_X_i = df_ours[df_ours['State types'] == state_label[dim_i]]['state'].to_numpy().reshape(test_N, 100, 20)
        
                    plt.plot(x_time, np.mean(np.mean(New_X_i, axis=0), axis=-1), ls='--', c=state_color[state_label[dim_i]], marker='o', markerfacecolor='none',
                             markersize=10,
                             markevery=20,
                             alpha=0.7, linewidth=3)
                        
                    plt.fill_between(x_time,
                                         np.mean(np.mean(New_X_i, axis=0), axis=-1) + 1.96 * np.std(np.mean(New_X_i, axis=0),
                                                                                                    axis=-1),
                                         np.mean(np.mean(New_X_i, axis=0), axis=-1) - 1.96 * np.std(np.mean(New_X_i, axis=0),
                                                                                                    axis=-1),
                                         facecolor=state_color[state_label[dim_i]], alpha=0.2)
    
                    # plt.legend(['Groundtruth(S)', 'Groundtruth(I)', 'Groundtruth(R)', 'Groundtruth(E)',
                    #             'GraphNDP-%s(S)' % (bound_t_context),
                    #             'GraphNDP-%s(I)' % (bound_t_context),
                    #             'GraphNDP-%s(R)' % (bound_t_context),
                    #             'GraphNDP-%s(E)' % (bound_t_context)])
    
                    plt.ylabel('Avg. states')
                    plt.xlabel('Time')
    
                    plt.ylim(-0.2, 1.2)
    
                    plt.tight_layout()
    
                    if save_fig:
                        plt.savefig('results/EXP4_real_epidemic_3dim_states_%s_%s_%s_%s_dim%s.png' % (
                            dynamics, topo, ii, bound_t_context, dim_i))
                    else:
                        plt.show()
                    plt.close()


if __name__ == '__main__':

    model_name = 'GNDP_OneForAll'
    #display_3_dynamics_on_5_topo()
    #display_opinion_dynamics_Baumann2021_2topic()
    # display_all_epidemic()
    #display_coupled_SIS_epidemic()
    display_SIR_meta_epidemic()
    display_real_epidemic()
