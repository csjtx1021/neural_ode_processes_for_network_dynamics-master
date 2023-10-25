import os
import pickle

import numpy as np
import random
import scipy.integrate as spi
from scipy.sparse import *
import itertools

import networkx as nx
import matplotlib.pyplot as plt
import torch
from torch_scatter import scatter_sum, scatter_mean

from moviepy.editor import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage

#########===================================================
##
##             build network topology
##
#########===================================================
from plots import plot_functions


def grid_8_neighbor_graph(N):
    """
    Build discrete grid graph, each node has 8 neighbors
    :param n:  sqrt of the number of nodes
    :return:  A, the adjacency matrix
    """
    N = int(N)
    n = int(N ** 2)
    dx = [-1, 0, 1, -1, 1, -1, 0, 1]
    dy = [-1, -1, -1, 0, 0, 1, 1, 1]
    A = torch.zeros(n, n)
    for x in range(N):
        for y in range(N):
            index = x * N + y
            for i in range(len(dx)):
                newx = x + dx[i]
                newy = y + dy[i]
                if N > newx >= 0 and N > newy >= 0:
                    index2 = newx * N + newy
                    A[index, index2] = 1
    return A.float()


def build_topology(N, topo_type, seed, **params):
    """
    :param N: #nodes
    :param topo_type: the type of topology
    :param seed: random seed
    :param params:
    :return: G
    """
    print("building network topology [%s] ..." % topo_type)
    if topo_type == 'grid':
        nn = int(np.ceil(np.sqrt(N)))  # grid-layout pixels :20
        A = grid_8_neighbor_graph(nn)
        G = nx.from_numpy_array(A.numpy())
    elif topo_type == 'random':
        if 'p' in params:
            p = params['p']
            print("setting p to %s ..." % p)
        else:
            print("setting default values [0.1] to p ...")
            p = 0.1
        G = nx.erdos_renyi_graph(N, p, seed=seed)
    elif topo_type == 'power_law':
        if 'm' in params:
            m = params['m']
            print("setting m to %s ..." % m)
        else:
            print("setting default values [5] to m ...")
            m = 5
        if N <= m:
            N = N + m
        G = nx.barabasi_albert_graph(N, m, seed=seed)
    elif topo_type == 'small_world':
        if 'k' in params:
            k = params['k']
            print("setting k to %s ..." % k)
        else:
            print("setting default values [5] to k ...")
            k = 5
        if 'p' in params:
            p = params['p']
            print("setting p to %s ..." % p)
        else:
            print("setting default values [0.5] to p ...")
            p = 0.5
        G = nx.newman_watts_strogatz_graph(N, k, p, seed=seed)
    elif topo_type == 'community':
        n1 = int(N / 3)
        n2 = int(N / 3)
        n3 = int(N / 4)
        n4 = N - n1 - n2 - n3

        if 'p_in' in params:
            p_in = params['p_in']
            print("setting p_in to %s ..." % p_in)
        else:
            print("setting default values [0.25] to p_in ...")
            p_in = 0.25
        if 'p_out' in params:
            p_out = params['p_out']
            print("setting p_out to %s ..." % p_out)
        else:
            print("setting default values [0.01] to p_out ...")
            p_out = 0.01
        G = nx.random_partition_graph([n1, n2, n3, n4], p_in, p_out, seed=seed)
    elif topo_type == 'full_connected':
        G = nx.complete_graph(N)
        G.add_edges_from([(i, i) for i in range(N)])  # add self_loop
    elif topo_type == 'directed_full_connected':
        G = nx.complete_graph(N, nx.DiGraph())
        # G.add_edges_from([(i, i) for i in range(N)])  # add self_loop
    else:
        print("ERROR topo_type [%s]" % topo_type)
        exit(1)
    return G


#########===================================================
##
##                 network dynamics
##
#########===================================================

def heat_diffusion_dynamics(X, sparse_A,
                            t_start=0, t_end=100, t_inc=1, diff_flag=False, **params):
    """

    dx_i(t)/dt = -k_{i}\sum_{j=1}^{n}A_{i,j}(x_i-x_j)

    governed by Newton’s law of cooling [1], which states
    that the rate of heat change of node i is proportional to
    the difference of the temperature between node i and its
    neighbors with heat capacity matrix A.

    [1] A v Luikov. 2012. Analytical heat diffusion theory. Elsevier.

    input:
        X: [N, d]
        sparse_A: [row, col], which means row -> col
        params['K'] : int
    return X: [steps, N, d]
    """
    N, x_dim = X.shape
    if 'K' in params:
        print("setting K to %s ..." % params['K'])
        K = np.ones(N) * params['K']
    else:
        print("setting default values [0.1] to K ...")
        K = np.ones(N) * 1.

    row, col = sparse_A

    # print(len(row),len(col),X.shape)

    def diff_heat(X, t):
        # dx_i/dt = k_{i,j} \sum_{j=1}^{n} A_{i,j}(x_j-x_i)
        X_j = X.reshape(-1, x_dim)[row]
        X_i = X.reshape(-1, x_dim)[col]

        # we do not know the scatter_sum in numpy package, so we use scatter_sum in torch instead.
        dX = K.reshape(-1, 1) * scatter_sum(torch.from_numpy(
            X_j - X_i
        ), torch.from_numpy(col).long(), dim=0, dim_size=X.shape[0]).numpy()
        # dX = K.reshape(-1, 1) * (A @ X.reshape(-1, x_dim) - np.sum(A, axis=1, keepdims=True) * X.reshape(-1, x_dim))
        return dX.reshape(-1)

    t_range = np.arange(t_start, t_end + t_inc, t_inc)
    New_X = spi.odeint(diff_heat, X.reshape(-1), t_range)

    if diff_flag:
        return New_X.reshape(len(t_range), N, x_dim), \
               np.gradient(New_X, axis=0, edge_order=1).reshape(len(t_range), N, x_dim), \
               t_range.reshape(-1, 1)
    else:
        return New_X.reshape(len(t_range), N, x_dim), t_range.reshape(-1, 1)


def mutualistic_interaction_dynamics(X, sparse_A,
                                     t_start=0, t_end=100, t_inc=1, diff_flag=False, **params):
    """

    dx_i(t)/dt = b_i+x_i(1 - x_i/k_i)(x_i/c_i - 1)+\sum_{j=1}^{n}A_{i,j}(x_i*x_j)/(d_i + e_i*x_i + h_j*x_j)

    The mutualistic differential equation systems [1] capture the abundance xifi(t) of species i,
    consisting of incoming migration term bi, logistic growth with population capacity ki [2] and
    Allee effect [3] with cold-start threshold ci , and mutualistic interaction term with interaction network A.

    [1] Jianxi Gao, Baruch Barzel, and Albert-László Barabási. 2016. Universal resilience patterns in complex networks.
    Nature 530, 7590 (2016), 307.
    [2] Chengxi Zang, Peng Cui, Christos Faloutsos, and Wenwu Zhu. 2018. On Power Law Growth of Social Networks.
    IEEE Transactions on Knowledge and Data Engineering 30, 9 (2018), 1727–1740
    [3] Warder Clyde Allee, Orlando Park, Alfred Edwards Emerson, Thomas Park, Karl Patterson Schmidt, et al. 1949.
    Principles of animal ecology. Technical Report. Saunders Company Philadelphia, Pennsylvania, USA.

    input:
        X: [N, d]
        sparse_A: [row, col], which means row -> col

    return X: [steps, N, d]
    """
    N, x_dim = X.shape
    if 'b' in params:
        print("setting b to %s ..." % params['b'])
        b = np.ones(N) * params['b']
    else:
        default_val = 1.
        print("setting default values [%s] to b ..." % default_val)
        b = np.ones(N) * default_val
    if 'c' in params:
        print("setting c to %s ..." % params['c'])
        c = np.ones(N) * params['c']
    else:
        default_val = 1.
        print("setting default values [%s] to c ..." % default_val)
        c = np.ones(N) * default_val
    if 'd' in params:
        print("setting d to %s ..." % params['d'])
        d = np.ones(N) * params['d']
    else:
        default_val = 5.
        print("setting default values [%s] to d ..." % default_val)
        d = np.ones(N) * default_val
    if 'e' in params:
        print("setting e to %s ..." % params['e'])
        e = np.ones(N) * params['e']
    else:
        default_val = 0.9
        print("setting default values [%s] to e ..." % default_val)
        e = np.ones(N) * default_val
    if 'h' in params:
        print("setting h to %s ..." % params['h'])
        h = np.ones(N) * params['h']
    else:
        default_val = 0.1
        print("setting default values [%s] to h ..." % default_val)
        h = np.ones(N) * default_val
    if 'k' in params:
        print("setting k to %s ..." % params['k'])
        k = np.ones(N) * params['k']
    else:
        default_val = 5.
        print("setting default values [%s] to k ..." % default_val)
        k = np.ones(N) * default_val

    row, col = sparse_A

    def diff_mutual(X, t):
        # dx_i(t)/dt = b_i+x_i(1 - x_i/k_i)(x_i/c_i - 1)+\sum_{j=1}^{n}A_{i,j}(x_i*x_j)/(d_i + e_i*x_i + h_j*x_j)
        X_j = X.reshape(-1, x_dim)[row]
        X_i = X.reshape(-1, x_dim)[col]

        # we do not know the scatter_sum in numpy package, so we use scatter_sum in torch instead.
        dX = b.reshape(-1, 1) + \
             X.reshape(-1, x_dim) * (1. - X.reshape(-1, x_dim) / k.reshape(-1, 1)) * (
                     X.reshape(-1, x_dim) / c.reshape(-1, 1) - 1) + \
             scatter_sum(torch.from_numpy(
                 (X_i * X_j) / (d.reshape(-1, 1)[col] + e.reshape(-1, 1)[col] * X_i + h.reshape(-1, 1)[row] * X_j)
             ), torch.from_numpy(col).long(), dim=0, dim_size=X.shape[0]).numpy()
        return dX.reshape(-1)

    t_range = np.arange(t_start, t_end + t_inc, t_inc)
    New_X = spi.odeint(diff_mutual, X.reshape(-1), t_range)

    if diff_flag:
        return New_X.reshape(len(t_range), N, x_dim), \
               np.gradient(New_X, axis=0, edge_order=1).reshape(len(t_range), N, x_dim), \
               t_range.reshape(-1, 1)
    else:
        return New_X.reshape(len(t_range), N, x_dim), t_range.reshape(-1, 1)


def gene_regulatory_dynamics(X, sparse_A,
                             t_start=0, t_end=100, t_inc=1, diff_flag=False, **params):
    """
    The gene regulatory dynamics governed by Michaelis-Menten equation

    dx_i(t)/dt = -b_i*x_i^f + \sum_{j=1}^{n}A_{i,j}*x_j^h/(x_j^h + 1)

    where the first term models degradation when f = 1 or dimerization when f = 2, and the second term
    captures genetic activation tuned by the Hill coefficient h [1, 2].

    [1] Uri Alon. 2006. An introduction to systems biology: design principles of biological circuits.
     Chapman and Hall/CRC.
    [2]  Jianxi Gao, Baruch Barzel, and Albert-László Barabási. 2016. Universal resilience patterns in
    complex networks. Nature 530, 7590 (2016), 307.

    input:
        X: [N, d]
        sparse_A: [row, col], which means row -> col

    return X: [steps, N, d]
    """
    N, x_dim = X.shape
    if 'b' in params:
        print("setting b to %s ..." % params['b'])
        b = np.ones(N) * params['b']
    else:
        default_val = 2.  # [0.5,2.]
        print("setting default values [%s] to b ..." % default_val)
        b = np.ones(N) * default_val
    if 'f' in params:
        print("setting f to %s ..." % params['f'])
        f = params['f']
    else:
        default_val = 1.
        print("setting default values [%s] to f ..." % default_val)
        f = default_val
    if 'h' in params:
        print("setting h to %s ..." % params['h'])
        h = params['h']
    else:
        default_val = 2.
        print("setting default values [%s] to h ..." % default_val)
        h = default_val

    row, col = sparse_A

    def diff_gene(X, t):
        # dx_i(t)/dt = -b_i*x_i^f + \sum_{j=1}^{n}A_{i,j}*x_j^h/(x_j^h + 1)
        X_j = X.reshape(-1, x_dim)[row]

        # we do not know the scatter_sum in numpy package, so we use scatter_sum in torch instead.
        dX = -b.reshape(-1, 1) * (X.reshape(-1, x_dim) ** f) + \
             scatter_sum(torch.from_numpy(
                 (X_j ** h) / (X_j ** h + 1)
             ), torch.from_numpy(col).long(), dim=0, dim_size=X.shape[0]).numpy()

        return dX.reshape(-1)

    t_range = np.arange(t_start, t_end + t_inc, t_inc)
    New_X = spi.odeint(diff_gene, X.reshape(-1), t_range)

    if diff_flag:
        return New_X.reshape(len(t_range), N, x_dim), \
               np.gradient(New_X, axis=0, edge_order=1).reshape(len(t_range), N, x_dim), \
               t_range.reshape(-1, 1)
    else:
        return New_X.reshape(len(t_range), N, x_dim), t_range.reshape(-1, 1)


def combination_dynamics(X, sparse_A,
                         t_start=0, t_end=100, t_inc=1, diff_flag=False, **params):
    N, x_dim = X.shape
    row, col = sparse_A
    # a dynamics
    if 'a_K' in params:
        print("setting a_K to %s ..." % params['a_K'])
        a_K = np.ones(N) * params['a_K']
    else:
        print("setting default values [0.1] to a_K ...")
        a_K = np.ones(N) * 1.
    # b dynamics
    if 'b_b' in params:
        print("setting b_b to %s ..." % params['b_b'])
        b_b = np.ones(N) * params['b_b']
    else:
        default_val = 1.
        print("setting default values [%s] to b_b ..." % default_val)
        b_b = np.ones(N) * default_val
    if 'b_c' in params:
        print("setting b_c to %s ..." % params['b_c'])
        b_c = np.ones(N) * params['b_c']
    else:
        default_val = 1.
        print("setting default values [%s] to b_c ..." % default_val)
        b_c = np.ones(N) * default_val
    if 'b_d' in params:
        print("setting b_d to %s ..." % params['b_d'])
        b_d = np.ones(N) * params['b_d']
    else:
        default_val = 5.
        print("setting default values [%s] to b_d ..." % default_val)
        b_d = np.ones(N) * default_val
    if 'b_e' in params:
        print("setting b_e to %s ..." % params['b_e'])
        b_e = np.ones(N) * params['b_e']
    else:
        default_val = 0.9
        print("setting default values [%s] to b_e ..." % default_val)
        b_e = np.ones(N) * default_val
    if 'b_h' in params:
        print("setting b_h to %s ..." % params['b_h'])
        b_h = np.ones(N) * params['b_h']
    else:
        default_val = 0.1
        print("setting default values [%s] to b_h ..." % default_val)
        b_h = np.ones(N) * default_val
    if 'b_k' in params:
        print("setting b_k to %s ..." % params['b_k'])
        b_k = np.ones(N) * params['b_k']
    else:
        default_val = 5.
        print("setting default values [%s] to b_k ..." % default_val)
        b_k = np.ones(N) * default_val

    # c dynamics
    if 'c_b' in params:
        print("setting c_b to %s ..." % params['c_b'])
        c_b = np.ones(N) * params['c_b']
    else:
        default_val = 2.  # [0.5,2.]
        print("setting default values [%s] to c_b ..." % default_val)
        c_b = np.ones(N) * default_val
    if 'c_f' in params:
        print("setting c_f to %s ..." % params['c_f'])
        c_f = params['c_f']
    else:
        default_val = 1.
        print("setting default values [%s] to c_f ..." % default_val)
        c_f = default_val
    if 'c_h' in params:
        print("setting c_h to %s ..." % params['c_h'])
        c_h = params['c_h']
    else:
        default_val = 2.
        print("setting default values [%s] to c_h ..." % default_val)
        c_h = default_val

    def diff_combination(X, t):
        # dx_i/dt = k_{i,j} \sum_{j=1}^{n} A_{i,j}(x_j-x_i)
        X_j = X.reshape(-1, x_dim)[row]
        X_i = X.reshape(-1, x_dim)[col]

        # we do not know the scatter_sum in numpy package, so we use scatter_sum in torch instead.
        # a dyanmics
        dX_a = a_K.reshape(-1, 1) * scatter_sum(torch.from_numpy(
            X_j - X_i
        ), torch.from_numpy(col).long(), dim=0, dim_size=X.shape[0]).numpy()

        # b dynamics
        dX_b = b_b.reshape(-1, 1) + \
               X.reshape(-1, x_dim) * (1. - X.reshape(-1, x_dim) / b_k.reshape(-1, 1)) * (
                       X.reshape(-1, x_dim) / b_c.reshape(-1, 1) - 1) + \
               scatter_sum(torch.from_numpy(
                   (X_i * X_j) / (
                           b_d.reshape(-1, 1)[col] + b_e.reshape(-1, 1)[col] * X_i + b_h.reshape(-1, 1)[row] * X_j)
               ), torch.from_numpy(col).long(), dim=0, dim_size=X.shape[0]).numpy()

        # c dynamics
        dX_c = -c_b.reshape(-1, 1) * (X.reshape(-1, x_dim) ** c_f) + \
               scatter_sum(torch.from_numpy(
                   (X_j ** c_h) / (X_j ** c_h + 1)
               ), torch.from_numpy(col).long(), dim=0, dim_size=X.shape[0]).numpy()

        dX = dX_a + dX_b + dX_c

        return dX.reshape(-1)

    t_range = np.arange(t_start, t_end + t_inc, t_inc)
    New_X = spi.odeint(diff_combination, X.reshape(-1), t_range)

    if diff_flag:
        return New_X.reshape(len(t_range), N, x_dim), \
               np.gradient(New_X, axis=0, edge_order=1).reshape(len(t_range), N, x_dim), \
               t_range.reshape(-1, 1)
    else:
        return New_X.reshape(len(t_range), N, x_dim), t_range.reshape(-1, 1)


def combination_dynamics_vary_coeff(X, sparse_A,
                                    t_start=0, t_end=100, t_inc=1, diff_flag=False, **params):
    N, x_dim = X.shape
    row, col = sparse_A
    # coeff
    if 'lam_1' in params:
        print("setting lam_1 to %s ..." % params['lam_1'])
        lam_1 = params['lam_1']
    else:
        print("setting default values [1/3] to lam_1 ...")
        lam_1 = 1. / 3.
    if 'lam_2' in params:
        print("setting lam_2 to %s ..." % params['lam_2'])
        lam_2 = params['lam_2']
    else:
        print("setting default values [1/3] to lam_2 ...")
        lam_2 = 1. / 3.
    if 'lam_3' in params:
        print("setting lam_3 to %s ..." % params['lam_3'])
        lam_3 = params['lam_3']
    else:
        print("setting default values [1/3] to lam_3 ...")
        lam_3 = 1. / 3.
    # a dynamics
    if 'a_K' in params:
        print("setting a_K to %s ..." % params['a_K'])
        a_K = np.ones(N) * params['a_K']
    else:
        print("setting default values [0.1] to a_K ...")
        a_K = np.ones(N) * 1.
    # b dynamics
    if 'b_b' in params:
        print("setting b_b to %s ..." % params['b_b'])
        b_b = np.ones(N) * params['b_b']
    else:
        default_val = 1.
        print("setting default values [%s] to b_b ..." % default_val)
        b_b = np.ones(N) * default_val
    if 'b_c' in params:
        print("setting b_c to %s ..." % params['b_c'])
        b_c = np.ones(N) * params['b_c']
    else:
        default_val = 1.
        print("setting default values [%s] to b_c ..." % default_val)
        b_c = np.ones(N) * default_val
    if 'b_d' in params:
        print("setting b_d to %s ..." % params['b_d'])
        b_d = np.ones(N) * params['b_d']
    else:
        default_val = 5.
        print("setting default values [%s] to b_d ..." % default_val)
        b_d = np.ones(N) * default_val
    if 'b_e' in params:
        print("setting b_e to %s ..." % params['b_e'])
        b_e = np.ones(N) * params['b_e']
    else:
        default_val = 0.9
        print("setting default values [%s] to b_e ..." % default_val)
        b_e = np.ones(N) * default_val
    if 'b_h' in params:
        print("setting b_h to %s ..." % params['b_h'])
        b_h = np.ones(N) * params['b_h']
    else:
        default_val = 0.1
        print("setting default values [%s] to b_h ..." % default_val)
        b_h = np.ones(N) * default_val
    if 'b_k' in params:
        print("setting b_k to %s ..." % params['b_k'])
        b_k = np.ones(N) * params['b_k']
    else:
        default_val = 5.
        print("setting default values [%s] to b_k ..." % default_val)
        b_k = np.ones(N) * default_val

    # c dynamics
    if 'c_b' in params:
        print("setting c_b to %s ..." % params['c_b'])
        c_b = np.ones(N) * params['c_b']
    else:
        default_val = 2.  # [0.5,2.]
        print("setting default values [%s] to c_b ..." % default_val)
        c_b = np.ones(N) * default_val
    if 'c_f' in params:
        print("setting c_f to %s ..." % params['c_f'])
        c_f = params['c_f']
    else:
        default_val = 1.
        print("setting default values [%s] to c_f ..." % default_val)
        c_f = default_val
    if 'c_h' in params:
        print("setting c_h to %s ..." % params['c_h'])
        c_h = params['c_h']
    else:
        default_val = 2.
        print("setting default values [%s] to c_h ..." % default_val)
        c_h = default_val

    def diff_combination(X, t):
        # dx_i/dt = k_{i,j} \sum_{j=1}^{n} A_{i,j}(x_j-x_i)
        X_j = X.reshape(-1, x_dim)[row]
        X_i = X.reshape(-1, x_dim)[col]

        # we do not know the scatter_sum in numpy package, so we use scatter_sum in torch instead.
        # a dyanmics
        dX_a = a_K.reshape(-1, 1) * scatter_sum(torch.from_numpy(
            X_j - X_i
        ), torch.from_numpy(col).long(), dim=0, dim_size=X.shape[0]).numpy()

        # b dynamics
        dX_b = b_b.reshape(-1, 1) + \
               X.reshape(-1, x_dim) * (1. - X.reshape(-1, x_dim) / b_k.reshape(-1, 1)) * (
                       X.reshape(-1, x_dim) / b_c.reshape(-1, 1) - 1) + \
               scatter_sum(torch.from_numpy(
                   (X_i * X_j) / (
                           b_d.reshape(-1, 1)[col] + b_e.reshape(-1, 1)[col] * X_i + b_h.reshape(-1, 1)[row] * X_j)
               ), torch.from_numpy(col).long(), dim=0, dim_size=X.shape[0]).numpy()

        # c dynamics
        dX_c = -c_b.reshape(-1, 1) * (X.reshape(-1, x_dim) ** c_f) + \
               scatter_sum(torch.from_numpy(
                   (X_j ** c_h) / (X_j ** c_h + 1)
               ), torch.from_numpy(col).long(), dim=0, dim_size=X.shape[0]).numpy()

        dX = lam_1 * dX_a + lam_2 * dX_b + lam_3 * dX_c

        return dX.reshape(-1)

    t_range = np.arange(t_start, t_end + t_inc, t_inc)
    New_X = spi.odeint(diff_combination, X.reshape(-1), t_range)

    if diff_flag:
        return New_X.reshape(len(t_range), N, x_dim), \
               np.gradient(New_X, axis=0, edge_order=1).reshape(len(t_range), N, x_dim), \
               t_range.reshape(-1, 1)
    else:
        return New_X.reshape(len(t_range), N, x_dim), t_range.reshape(-1, 1)


def vary_dynamics_with_vary_type_and_coeff(X, sparse_A,
                                           t_start=0, t_end=100, t_inc=1, diff_flag=False, **params):
    N, x_dim = X.shape
    row, col = sparse_A
    # choose_list
    if 'choose_1' in params:
        print("setting choose_1 to %s ..." % params['choose_1'])
        choose_1 = params['choose_1']
    else:
        print("setting default values [0] to choose_1 ...")
        choose_1 = 0
    if 'choose_2' in params:
        print("setting choose_2 to %s ..." % params['choose_2'])
        choose_2 = params['choose_2']
    else:
        print("setting default values [0] to choose_2 ...")
        choose_2 = 0

    # for first half
    # a dynamics
    if 'a_K_1' in params:
        print("setting a_K_1 to %s ..." % params['a_K_1'])
        a_K_1 = np.ones(N) * params['a_K_1']
    else:
        print("setting default values [0.1] to a_K_1 ...")
        a_K_1 = np.ones(N) * 1.
    # b dynamics
    if 'b_b_1' in params:
        print("setting b_b_1 to %s ..." % params['b_b_1'])
        b_b_1 = np.ones(N) * params['b_b_1']
    else:
        default_val = 1.
        print("setting default values [%s] to b_b_1 ..." % default_val)
        b_b_1 = np.ones(N) * default_val
    if 'b_c_1' in params:
        print("setting b_c_1 to %s ..." % params['b_c_1'])
        b_c_1 = np.ones(N) * params['b_c_1']
    else:
        default_val = 1.
        print("setting default values [%s] to b_c_1 ..." % default_val)
        b_c_1 = np.ones(N) * default_val
    if 'b_d_1' in params:
        print("setting b_d_1 to %s ..." % params['b_d_1'])
        b_d_1 = np.ones(N) * params['b_d_1']
    else:
        default_val = 5.
        print("setting default values [%s] to b_d_1 ..." % default_val)
        b_d_1 = np.ones(N) * default_val
    if 'b_e_1' in params:
        print("setting b_e_1 to %s ..." % params['b_e_1'])
        b_e_1 = np.ones(N) * params['b_e_1']
    else:
        default_val = 0.9
        print("setting default values [%s] to b_e_1 ..." % default_val)
        b_e_1 = np.ones(N) * default_val
    if 'b_h_1' in params:
        print("setting b_h_1 to %s ..." % params['b_h_1'])
        b_h_1 = np.ones(N) * params['b_h_1']
    else:
        default_val = 0.1
        print("setting default values [%s] to b_h_1 ..." % default_val)
        b_h_1 = np.ones(N) * default_val
    if 'b_k_1' in params:
        print("setting b_k to %s ..." % params['b_k_1'])
        b_k_1 = np.ones(N) * params['b_k_1']
    else:
        default_val = 5.
        print("setting default values [%s] to b_k_1 ..." % default_val)
        b_k_1 = np.ones(N) * default_val

    # c dynamics
    if 'c_b_1' in params:
        print("setting c_b to %s ..." % params['c_b_1'])
        c_b_1 = np.ones(N) * params['c_b_1']
    else:
        default_val = 2.  # [0.5,2.]
        print("setting default values [%s] to c_b_1 ..." % default_val)
        c_b_1 = np.ones(N) * default_val
    if 'c_f_1' in params:
        print("setting c_f_1 to %s ..." % params['c_f_1'])
        c_f_1 = params['c_f_1']
    else:
        default_val = 1.
        print("setting default values [%s] to c_f_1 ..." % default_val)
        c_f_1 = default_val
    if 'c_h_1' in params:
        print("setting c_h to %s ..." % params['c_h_1'])
        c_h_1 = params['c_h_1']
    else:
        default_val = 2.
        print("setting default values [%s] to c_h_1 ..." % default_val)
        c_h_1 = default_val

    # for second half
    # a dynamics
    if 'a_K_2' in params:
        print("setting a_K_2 to %s ..." % params['a_K_2'])
        a_K_2 = np.ones(N) * params['a_K_2']
    else:
        print("setting default values [0.1] to a_K_2 ...")
        a_K_2 = np.ones(N) * 1.
    # b dynamics
    if 'b_b_2' in params:
        print("setting b_b_2 to %s ..." % params['b_b_2'])
        b_b_2 = np.ones(N) * params['b_b_2']
    else:
        default_val = 1.
        print("setting default values [%s] to b_b_2 ..." % default_val)
        b_b_2 = np.ones(N) * default_val
    if 'b_c_2' in params:
        print("setting b_c_2 to %s ..." % params['b_c_2'])
        b_c_2 = np.ones(N) * params['b_c_2']
    else:
        default_val = 1.
        print("setting default values [%s] to b_c_2 ..." % default_val)
        b_c_2 = np.ones(N) * default_val
    if 'b_d_2' in params:
        print("setting b_d_2 to %s ..." % params['b_d_2'])
        b_d_2 = np.ones(N) * params['b_d_2']
    else:
        default_val = 5.
        print("setting default values [%s] to b_d_2 ..." % default_val)
        b_d_2 = np.ones(N) * default_val
    if 'b_e_2' in params:
        print("setting b_e_2 to %s ..." % params['b_e_2'])
        b_e_2 = np.ones(N) * params['b_e_2']
    else:
        default_val = 0.9
        print("setting default values [%s] to b_e_2 ..." % default_val)
        b_e_2 = np.ones(N) * default_val
    if 'b_h_2' in params:
        print("setting b_h_2 to %s ..." % params['b_h_2'])
        b_h_2 = np.ones(N) * params['b_h_2']
    else:
        default_val = 0.1
        print("setting default values [%s] to b_h_2 ..." % default_val)
        b_h_2 = np.ones(N) * default_val
    if 'b_k_2' in params:
        print("setting b_k_2 to %s ..." % params['b_k_2'])
        b_k_2 = np.ones(N) * params['b_k_2']
    else:
        default_val = 5.
        print("setting default values [%s] to b_k_2 ..." % default_val)
        b_k_2 = np.ones(N) * default_val

    # c dynamics
    if 'c_b_2' in params:
        print("setting c_b_2 to %s ..." % params['c_b_2'])
        c_b_2 = np.ones(N) * params['c_b_2']
    else:
        default_val = 2.  # [0.5,2.]
        print("setting default values [%s] to c_b_2 ..." % default_val)
        c_b_2 = np.ones(N) * default_val
    if 'c_f_2' in params:
        print("setting c_f_2 to %s ..." % params['c_f_2'])
        c_f_2 = params['c_f_2']
    else:
        default_val = 1.
        print("setting default values [%s] to c_f_2 ..." % default_val)
        c_f_2 = default_val
    if 'c_h_2' in params:
        print("setting c_h_2 to %s ..." % params['c_h_2'])
        c_h_2 = params['c_h_2']
    else:
        default_val = 2.
        print("setting default values [%s] to c_h_2 ..." % default_val)
        c_h_2 = default_val

    def diff_combination(X, t):
        # dx_i/dt = k_{i,j} \sum_{j=1}^{n} A_{i,j}(x_j-x_i)
        X_j = X.reshape(-1, x_dim)[row]
        X_i = X.reshape(-1, x_dim)[col]

        if t < 0.5:
            choose_dynamics = choose_1
            a_K = a_K_1
            b_b = b_b_1
            b_c = b_c_1
            b_d = b_d_1
            b_e = b_e_1
            b_h = b_h_1
            b_k = b_k_1
            c_b = c_b_1
            c_f = c_f_1
            c_h = c_h_1
        else:
            choose_dynamics = choose_2
            a_K = a_K_2
            b_b = b_b_2
            b_c = b_c_2
            b_d = b_d_2
            b_e = b_e_2
            b_h = b_h_2
            b_k = b_k_2
            c_b = c_b_2
            c_f = c_f_2
            c_h = c_h_2

        if choose_dynamics == 0:
            # we do not know the scatter_sum in numpy package, so we use scatter_sum in torch instead.
            # a dyanmics
            dX = a_K.reshape(-1, 1) * scatter_sum(torch.from_numpy(
                X_j - X_i
            ), torch.from_numpy(col).long(), dim=0, dim_size=X.shape[0]).numpy()
        elif choose_dynamics == 1:
            # b dynamics
            dX = b_b.reshape(-1, 1) + \
                 X.reshape(-1, x_dim) * (1. - X.reshape(-1, x_dim) / b_k.reshape(-1, 1)) * (
                         X.reshape(-1, x_dim) / b_c.reshape(-1, 1) - 1) + \
                 scatter_sum(torch.from_numpy(
                     (X_i * X_j) / (b_d.reshape(-1, 1)[col] + b_e.reshape(-1, 1)[col] * X_i + b_h.reshape(-1, 1)[
                         row] * X_j)
                 ), torch.from_numpy(col).long(), dim=0, dim_size=X.shape[0]).numpy()
        elif choose_dynamics == 2:
            # c dynamics
            dX = -c_b.reshape(-1, 1) * (X.reshape(-1, x_dim) ** c_f) + \
                 scatter_sum(torch.from_numpy(
                     (X_j ** c_h) / (X_j ** c_h + 1)
                 ), torch.from_numpy(col).long(), dim=0, dim_size=X.shape[0]).numpy()
        else:
            print('Wrong choose_dynamics [%s]' % choose_dynamics)

        dX = dX

        return dX.reshape(-1)

    t_range = np.arange(t_start, t_end + t_inc, t_inc)
    New_X = spi.odeint(diff_combination, X.reshape(-1), t_range)

    if diff_flag:
        return New_X.reshape(len(t_range), N, x_dim), \
               np.gradient(New_X, axis=0, edge_order=1).reshape(len(t_range), N, x_dim), \
               t_range.reshape(-1, 1)
    else:
        return New_X.reshape(len(t_range), N, x_dim), t_range.reshape(-1, 1)


def opinion_dynamics(X, sparse_A,
                     t_start=0, t_end=100, t_inc=1, diff_flag=False, **params):
    N, x_dim = X.shape

    if 'a' in params:
        print('setting a to %s ...' % params['a'])
        a = params['a']
    else:
        default_val = 30.  # [30, 60]
        print('setting default values [%s] to a ...' % default_val)
        a = default_val
    if 'b' in params:
        print('setting b to %s ...' % params['b'])
        b = params['b']
    else:
        default_val = 3.  # [3, 6]
        print('setting default values [%s] to b ...' % default_val)
        b = default_val
    if 'c' in params:
        print('setting c to %s ...' % params['c'])
        c = params['c']
    else:
        default_val = 0.7  # [0.1, 0.9]
        print('setting default values [%s] to c ...' % default_val)
        c = default_val

    row, col = sparse_A  # sparse_A in opinion_dynamics should be a fully-connected graph.

    def diff_opinion(X, t):
        # dx_i/dt = \frac{1}{n} \sum_{j=1}^{n} \phi(||x_j-x_i||)(x_j-x_i)
        # \phi(r) := 1     0 <= r < 1/sqrt(2);
        #            0.1   1/sqrt(2) <= r < 1;
        #            0     1 <= r.
        X_j = X.reshape(-1, x_dim)[row]
        X_i = X.reshape(-1, x_dim)[col]

        def phi(r):
            new_r = np.zeros_like(r)
            new_r[(r >= 0) & (r < c)] = a
            new_r[(r >= c) & (r < 1)] = b
            new_r[r >= 1] = 0.
            return new_r

        dX = phi(np.linalg.norm(x=X_j - X_i, ord=2, axis=1, keepdims=True)) * (X_j - X_i)

        dX = (1 / N) * np.sum(dX.reshape(N, N, x_dim), axis=0)

        return dX.reshape(-1)

    t_range = np.arange(t_start, t_end + t_inc, t_inc)
    New_X = spi.odeint(diff_opinion, X.reshape(-1), t_range)

    if diff_flag:
        return New_X.reshape(len(t_range), N, x_dim), \
               np.gradient(New_X, axis=0, edge_order=1).reshape(len(t_range), N, x_dim), \
               t_range.reshape(-1, 1)
    else:
        return New_X.reshape(len(t_range), N, x_dim), t_range.reshape(-1, 1)


def opinion_dynamics_Baumann2021(X, sparse_A,
                                 t_start=0, t_end=100, t_inc=1, diff_flag=False, **params):
    N, x_dim = X.shape

    if 'a' in params:
        print('setting a to %s ...' % params['a'])
        a = params['a']
    else:
        default_val = 0.5  # [0.01, 0.5]
        print('setting default values [%s] to a ...' % default_val)
        a = default_val
    if 'k' in params:
        print('setting k to %s ...' % params['k'])
        k = params['k']
    else:
        default_val = 1.  # [0.5, 1.5]
        print('setting default values [%s] to k ...' % default_val)
        k = default_val
    if 'c' in params:
        print('setting c to %s ...' % params['c'])
        c = params['c']
    else:
        default_val = 1.  # [1., 3.]
        print('setting default values [%s] to c ...' % default_val)
        c = default_val
    row, col = sparse_A  # sparse_A in opinion_dynamics should be a fully-connected graph.

    def diff_opinion(X, t):
        # dx_i/dt = -x_i \sum_{j=1}^{n} A_i_j*tanh(a*x_j)
        X_j = X.reshape(-1, x_dim)[row]
        # X_i = X.reshape(-1, x_dim)[col]

        # we do not know the scatter_sum in numpy package, so we use scatter_sum in torch instead.
        dX = -c * X.reshape(-1, x_dim) + k * scatter_sum(torch.from_numpy(np.tanh(a * X_j)),
                                                         torch.from_numpy(col).long(), dim=0,
                                                         dim_size=X.shape[0]).numpy()

        return dX.reshape(-1)

    t_range = np.arange(t_start, t_end + t_inc, t_inc)
    New_X = spi.odeint(diff_opinion, X.reshape(-1), t_range)

    if diff_flag:
        return New_X.reshape(len(t_range), N, x_dim), \
               np.gradient(New_X, axis=0, edge_order=1).reshape(len(t_range), N, x_dim), \
               t_range.reshape(-1, 1)
    else:
        return New_X.reshape(len(t_range), N, x_dim), t_range.reshape(-1, 1)


def opinion_dynamics_Baumann2021_2topic(X, sparse_A,
                                        t_start=0, t_end=100, t_inc=1, diff_flag=False, **params):
    N, x_dim = X.shape

    if 'a1' in params:
        print('setting a1 to %s ...' % params['a1'])
        a1 = params['a1']
    else:
        default_val = 0.05  # [0.05, 3.]
        print('setting default values [%s] to a1 ...' % default_val)
        a1 = default_val
    if 'a2' in params:
        print('setting a2 to %s ...' % params['a2'])
        a2 = params['a2']
    else:
        default_val = 3.  # [0.05, 3.]
        print('setting default values [%s] to a2 ...' % default_val)
        a2 = default_val
    if 'k' in params:
        print('setting k to %s ...' % params['k'])
        k = params['k']
    else:
        default_val = 3  # [1, 3]
        print('setting default values [%s] to k ...' % default_val)
        k = default_val
    if 'c' in params:
        print('setting c to %s ...' % params['c'])
        c = params['c']
    else:
        default_val = 1.  # [1., 3.]
        print('setting default values [%s] to c ...' % default_val)
        c = default_val
    if 'd' in params:
        print('setting d to %s ...' % params['d'])
        d = params['d']
    else:
        # default_val = 3 * np.pi / 4.  # [0, pi/2]
        default_val = np.pi / 2.  # [0, pi/2]
        print('setting default values [%s] to d ...' % default_val)
        d = default_val

    row, col = sparse_A  # sparse_A in opinion_dynamics should be a fully-connected graph.

    def diff_opinion(X, t):
        # dx_i/dt = -x_i \sum_{j=1}^{n} A_i_j*tanh(a*x_j)
        X_j = X.reshape(-1, x_dim)[row]
        # X_i = X.reshape(-1, x_dim)[col]
        # we do not know the scatter_sum in numpy package, so we use scatter_sum in torch instead.
        dX_col1 = -c * X.reshape(-1, x_dim)[:, 0] + k * scatter_sum(torch.from_numpy(
            np.tanh(a1 * (X_j[:, 0] + np.cos(d) * X_j[:, 1]))
        ),
            torch.from_numpy(col).long(), dim=0,
            dim_size=X.reshape(-1, x_dim).shape[0]).numpy()

        dX_col2 = -c * X.reshape(-1, x_dim)[:, 1] + k * scatter_sum(torch.from_numpy(
            np.tanh(a2 * (X_j[:, 1] + np.cos(d) * X_j[:, 0]))
        ),
            torch.from_numpy(col).long(), dim=0,
            dim_size=X.reshape(-1, x_dim).shape[0]).numpy()

        dX = np.concatenate([dX_col1.reshape(-1, 1), dX_col2.reshape(-1, 1)], axis=-1)

        return dX.reshape(-1)

    t_range = np.arange(t_start, t_end + t_inc, t_inc)
    New_X = spi.odeint(diff_opinion, X.reshape(-1), t_range)

    if diff_flag:
        return New_X.reshape(len(t_range), N, x_dim), \
               np.gradient(New_X, axis=0, edge_order=1).reshape(len(t_range), N, x_dim), \
               t_range.reshape(-1, 1)
    else:
        return New_X.reshape(len(t_range), N, x_dim), t_range.reshape(-1, 1)


# 1st order
def predator_swarm_dynamics(X, sparse_A,
                            t_start=0, t_end=100, t_inc=1, diff_flag=False, **params):
    N, x_dim = X.shape

    if 'a' in params:
        print('setting a to %s ...' % params['a'])
        a = params['a']
    else:
        default_val = 1.5  # [0.5, 1.5]
        print('setting default values [%s] to a ...' % default_val)
        a = default_val
    if 'b' in params:
        print('setting b to %s ...' % params['b'])
        b = params['b']
    else:
        default_val = -3.  # [-1, -3]
        print('setting default values [%s] to b ...' % default_val)
        b = default_val
    if 'c' in params:
        print('setting c to %s ...' % params['c'])
        c = params['c']
    else:
        default_val = -3  # [-1, -3]
        print('setting default values [%s] to c ...' % default_val)
        c = default_val
    if 'd' in params:
        print('setting d to %s ...' % params['d'])
        d = params['d']
    else:
        default_val = -3  # [-1, -3]
        print('setting default values [%s] to d ...' % default_val)
        d = default_val
    if 'e' in params:
        print('setting e to %s ...' % params['e'])
        e = params['e']
    else:
        default_val = 4.  # [2, 4]
        print('setting default values [%s] to e ...' % default_val)
        e = default_val
    if 'f' in params:
        print('setting f to %s ...' % params['f'])
        f = params['f']
    else:
        default_val = 3.  # [1, 3]
        print('setting default values [%s] to f ...' % default_val)
        f = default_val

    row, col = sparse_A  # sparse_A in opinion_dynamics should be a fully-connected graph.

    def diff_predator_swarm(X, t):
        # dx_i/dt = \frac{1}{n} \sum_{j=1}^{n} \phi(||x_j-x_i||)(x_j-x_i)
        # \phi(r) := 1     0 <= r < 1/sqrt(2);
        #            0.1   1/sqrt(2) <= r < 1;
        #            0     1 <= r.
        X_j = X.reshape(-1, x_dim)[row]
        X_i = X.reshape(-1, x_dim)[col]

        def phi(i, j, r):
            if r == 0.:
                r = 1e-16
            if i > 0 and j > 0:
                new_r = a - np.power(r, b)  # 1,1
            elif i > 0 and j == 0:
                new_r = c * np.power(r, d)  # 1,2
            elif i == 0 and j > 0:
                new_r = e * np.power(r, f)  # 2,1
            else:
                new_r = np.zeros_like(r)  # 2,2

            return new_r

        diff_between_X_j_X_i = (X_j - X_i).reshape(N, N, x_dim)
        dX = np.zeros_like(diff_between_X_j_X_i)
        for i in range(N):
            for j in range(N):
                r = np.linalg.norm(x=diff_between_X_j_X_i[j, i, :], ord=2)
                dX[j, i, :] = phi(i, j, r) * diff_between_X_j_X_i[j, i, :]

        dX = (1 / N) * np.sum(dX.reshape(N, N, x_dim), axis=0)

        return dX.reshape(-1)

    t_range = np.arange(t_start, t_end + t_inc, t_inc)
    New_X = spi.odeint(diff_predator_swarm, X.reshape(-1), t_range)

    if diff_flag:
        return New_X.reshape(len(t_range), N, x_dim), \
               np.gradient(New_X, axis=0, edge_order=1).reshape(len(t_range), N, x_dim), \
               t_range.reshape(-1, 1)
    else:
        return New_X.reshape(len(t_range), N, x_dim), t_range.reshape(-1, 1)


# ---------------------------------------------------------
#
#             SI, SIS, SIR, SEIS, SEIR
#
# ---------------------------------------------------------
def SI_Individual_dynamics(X, sparse_A,
                           t_start=0, t_end=100, t_inc=1, diff_flag=False, **params):
    """
    :param X:
    :param sparse_A:
    :param t_start:
    :param t_end:
    :param t_inc:
    :param diff_flag:
    :param params:
    :return:

    Implemented according to [Mina Youssef, Caterina Scoglio. An individual-based approach to SIR epidemics in contact
    networks. Journal of Theoretical Biology 283 (2011) 136–144.]

    """
    if 'b' in params:
        print('setting b to %s ...' % params['b'])
        b = params['b']
    else:
        # default_val = 1.5 / 3.2  # for 2009 Hong Kong H1N1 Influenza Pandemic
        # default_val = 2.0 / 3.   # for 2010 Taiwan Seasonal Influenza
        # default_val = 7.75 / 5.  #  for 2010 Taiwan Varicella
        # default_val = 2.16 / 7.  # Mumps
        default_val = 0.5 / 7.  # [0.5 / 7., 8 / 2.]
        print('setting default values [%s] to b ...' % default_val)
        b = default_val

    N, x_dim = X.shape
    row, col = sparse_A  # sparse_A in opinion_dynamics should be a fully-connected graph.

    def diff_SI_Individual(X, t):
        # dS_i/dt = -b * \sum_{j=1}^{n} A_{i,j} * I_i * S_i
        # dI_i/dt = b * \sum_{j=1}^{n} A_{i,j} * I_i * S_i - r * I_i
        # dR_i/dt = r * I_i

        X_self = X.reshape(-1, x_dim)
        X_j = X_self[row]

        # we do not know the scatter_sum in numpy package, so we use scatter_sum in torch instead.
        sum_j = scatter_sum(torch.from_numpy(X_j[:, 1].reshape(-1, 1)), torch.from_numpy(col).long(), dim=0,
                            dim_size=X_self.shape[0]).numpy()

        dS = -b * sum_j.reshape(-1, 1) * X_self[:, 0].reshape(-1, 1)
        dI = b * sum_j.reshape(-1, 1) * X_self[:, 0].reshape(-1, 1)
        # dR = np.zeros_like(dI)
        # dE = np.zeros_like(dI)

        dX = np.concatenate([dS, dI], axis=-1)

        return dX.reshape(-1) * 60  # mul 60 is to scale the time

    t_range = np.arange(t_start, t_end + t_inc, t_inc)
    New_X = spi.odeint(diff_SI_Individual, X.reshape(-1), t_range)

    if diff_flag:
        return New_X.reshape(len(t_range), N, x_dim), \
               np.gradient(New_X, axis=0, edge_order=1).reshape(len(t_range), N, x_dim), \
               t_range.reshape(-1, 1)
    else:
        return New_X.reshape(len(t_range), N, x_dim), t_range.reshape(-1, 1)


def SIS_Individual_dynamics(X, sparse_A,
                            t_start=0, t_end=100, t_inc=1, diff_flag=False, **params):
    """
    :param X:
    :param sparse_A:
    :param t_start:
    :param t_end:
    :param t_inc:
    :param diff_flag:
    :param params:
    :return:

    Implemented according to [Mina Youssef, Caterina Scoglio. An individual-based approach to SIR epidemics in contact
    networks. Journal of Theoretical Biology 283 (2011) 136–144.]

    """
    if 'b' in params:
        print('setting b to %s ...' % params['b'])
        b = params['b']
    else:
        # default_val = 1.5 / 3.2  # for 2009 Hong Kong H1N1 Influenza Pandemic
        # default_val = 2.0 / 3.   # for 2010 Taiwan Seasonal Influenza
        # default_val = 7.75 / 5.  #  for 2010 Taiwan Varicella
        # default_val = 2.16 / 7.  # Mumps
        default_val = 0.5 / 7.  # [0.5 / 7., 8 / 2.]
        print('setting default values [%s] to b ...' % default_val)
        b = default_val
    if 'r' in params:
        print('setting r to %s ...' % params['r'])
        r = params['r']
    else:
        # default_val = 1 / 3.2  # for 2009 Hong Kong H1N1 Influenza Pandemic
        # default_val = 1 / 3.  # for 2010 Taiwan Seasonal Influenza
        # default_val = 1 / 5.  # for 2010 Taiwan Varicella
        # default_val = 1 / 7.  # Mumps
        default_val = 1 / 2.  # [1/7, 1/2]
        print('setting default values [%s] to r ...' % default_val)
        r = default_val

    N, x_dim = X.shape
    row, col = sparse_A  # sparse_A in opinion_dynamics should be a fully-connected graph.

    def diff_SIS_Individual(X, t):
        # dS_i/dt = -b * \sum_{j=1}^{n} A_{i,j} * I_i * S_i
        # dI_i/dt = b * \sum_{j=1}^{n} A_{i,j} * I_i * S_i - r * I_i
        # dR_i/dt = r * I_i

        X_self = X.reshape(-1, x_dim)
        X_j = X_self[row]

        # we do not know the scatter_sum in numpy package, so we use scatter_sum in torch instead.
        sum_j = scatter_sum(torch.from_numpy(X_j[:, 1].reshape(-1, 1)), torch.from_numpy(col).long(), dim=0,
                            dim_size=X_self.shape[0]).numpy()

        dS = -b * sum_j.reshape(-1, 1) * X_self[:, 0].reshape(-1, 1) + r * X_self[:, 1].reshape(-1, 1)
        dI = b * sum_j.reshape(-1, 1) * X_self[:, 0].reshape(-1, 1) - r * X_self[:, 1].reshape(-1, 1)
        # dR = np.zeros_like(dI)
        # dE = np.zeros_like(dI)

        dX = np.concatenate([dS, dI], axis=-1)

        return dX.reshape(-1) * 60  # mul 60 is to scale the time

    t_range = np.arange(t_start, t_end + t_inc, t_inc)
    New_X = spi.odeint(diff_SIS_Individual, X.reshape(-1), t_range)

    if diff_flag:
        return New_X.reshape(len(t_range), N, x_dim), \
               np.gradient(New_X, axis=0, edge_order=1).reshape(len(t_range), N, x_dim), \
               t_range.reshape(-1, 1)
    else:
        return New_X.reshape(len(t_range), N, x_dim), t_range.reshape(-1, 1)


def SIR_Individual_dynamics(X, sparse_A,
                            t_start=0, t_end=100, t_inc=1, diff_flag=False, **params):
    """
    :param X:
    :param sparse_A:
    :param t_start:
    :param t_end:
    :param t_inc:
    :param diff_flag:
    :param params:
    :return:

    Implemented according to [Mina Youssef, Caterina Scoglio. An individual-based approach to SIR epidemics in contact
    networks. Journal of Theoretical Biology 283 (2011) 136–144.]

    """
    if 'b' in params:
        print('setting b to %s ...' % params['b'])
        b = params['b']
    else:
        # default_val = 1.5 / 3.2  # for 2009 Hong Kong H1N1 Influenza Pandemic
        # default_val = 2.0 / 3.   # for 2010 Taiwan Seasonal Influenza
        # default_val = 7.75 / 5.  #  for 2010 Taiwan Varicella
        # default_val = 2.16 / 7.  # Mumps
        default_val = 8 / 2.  # [0.5 / 7., 8 / 2.]
        print('setting default values [%s] to b ...' % default_val)
        b = default_val
    if 'r' in params:
        print('setting r to %s ...' % params['r'])
        r = params['r']
    else:
        # default_val = 1 / 3.2  # for 2009 Hong Kong H1N1 Influenza Pandemic
        # default_val = 1 / 3.  # for 2010 Taiwan Seasonal Influenza
        # default_val = 1 / 5.  # for 2010 Taiwan Varicella
        # default_val = 1 / 7.  # Mumps
        default_val = 1 / 2.  # [1/7, 1/2]
        print('setting default values [%s] to r ...' % default_val)
        r = default_val

    N, x_dim = X.shape
    row, col = sparse_A  # sparse_A in opinion_dynamics should be a fully-connected graph.

    def diff_SIR_Individual(X, t):
        # dS_i/dt = -b * \sum_{j=1}^{n} A_{i,j} * I_i * S_i
        # dI_i/dt = b * \sum_{j=1}^{n} A_{i,j} * I_i * S_i - r * I_i
        # dR_i/dt = r * I_i

        X_self = X.reshape(-1, x_dim)
        X_j = X_self[row]

        # we do not know the scatter_sum in numpy package, so we use scatter_sum in torch instead.
        sum_j = scatter_sum(torch.from_numpy(X_j[:, 1].reshape(-1, 1)), torch.from_numpy(col).long(), dim=0,
                            dim_size=X_self.shape[0]).numpy()

        dS = -b * sum_j.reshape(-1, 1) * X_self[:, 0].reshape(-1, 1)
        dI = b * sum_j.reshape(-1, 1) * X_self[:, 0].reshape(-1, 1) - r * X_self[:, 1].reshape(-1, 1)
        dR = r * X_self[:, 1].reshape(-1, 1)
        # dE = np.zeros_like(dI)

        dX = np.concatenate([dS, dI, dR], axis=-1)

        return dX.reshape(-1) * 60  # mul 60 is to scale the time

    t_range = np.arange(t_start, t_end + t_inc, t_inc)
    New_X = spi.odeint(diff_SIR_Individual, X.reshape(-1), t_range)

    if diff_flag:
        return New_X.reshape(len(t_range), N, x_dim), \
               np.gradient(New_X, axis=0, edge_order=1).reshape(len(t_range), N, x_dim), \
               t_range.reshape(-1, 1)
    else:
        return New_X.reshape(len(t_range), N, x_dim), t_range.reshape(-1, 1)


def SEIS_Individual_dynamics(X, sparse_A,
                             t_start=0, t_end=100, t_inc=1, diff_flag=False, **params):
    """
    :param X:
    :param sparse_A:
    :param t_start:
    :param t_end:
    :param t_inc:
    :param diff_flag:
    :param params:
    :return:

    Implemented according to [Mina Youssef, Caterina Scoglio. An individual-based approach to SIR epidemics in contact
    networks. Journal of Theoretical Biology 283 (2011) 136–144.]

    """
    if 'b' in params:
        print('setting b to %s ...' % params['b'])
        b = params['b']
    else:
        # default_val = 1.5 / 3.2  # for 2009 Hong Kong H1N1 Influenza Pandemic
        # default_val = 2.0 / 3.   # for 2010 Taiwan Seasonal Influenza
        # default_val = 7.75 / 5.  #  for 2010 Taiwan Varicella
        # default_val = 2.16 / 7.  # Mumps
        default_val = 0.5 / 7.  # [0.5 / 7., 8 / 2.]
        print('setting default values [%s] to b ...' % default_val)
        b = default_val
    if 'c' in params:
        print('setting c to %s ...' % params['c'])
        c = params['c']
    else:
        # default_val = 1 / 14.  # Mumps
        default_val = 1 / 14.  # [1/14, 1/2]
        print('setting default values [%s] to c ...' % default_val)
        c = default_val
    if 'r' in params:
        print('setting r to %s ...' % params['r'])
        r = params['r']
    else:
        # default_val = 1 / 3.2  # for 2009 Hong Kong H1N1 Influenza Pandemic
        # default_val = 1 / 3.  # for 2010 Taiwan Seasonal Influenza
        # default_val = 1 / 5.  # for 2010 Taiwan Varicella
        # default_val = 1 / 7.  # Mumps
        default_val = 1 / 7.  # [1/7, 1/2]
        print('setting default values [%s] to r ...' % default_val)
        r = default_val

    N, x_dim = X.shape
    row, col = sparse_A  # sparse_A in opinion_dynamics should be a fully-connected graph.

    def diff_SEIS_Individual(X, t):
        # dS_i/dt = -b * \sum_{j=1}^{n} A_{i,j} * I_i * S_i
        # dI_i/dt = b * \sum_{j=1}^{n} A_{i,j} * I_i * S_i - r * I_i
        # dR_i/dt = r * I_i

        X_self = X.reshape(-1, x_dim)
        X_j = X_self[row]

        # we do not know the scatter_sum in numpy package, so we use scatter_sum in torch instead.
        sum_j = scatter_sum(torch.from_numpy(X_j[:, 1].reshape(-1, 1)), torch.from_numpy(col).long(), dim=0,
                            dim_size=X_self.shape[0]).numpy()

        dS = -b * sum_j.reshape(-1, 1) * X_self[:, 0].reshape(-1, 1) + r * X_self[:, 1].reshape(-1, 1)
        dI = c * X_self[:, -1].reshape(-1, 1) - r * X_self[:, 1].reshape(-1, 1)
        dE = b * sum_j.reshape(-1, 1) * X_self[:, 0].reshape(-1, 1) - c * X_self[:, -1].reshape(-1, 1)
        # dR = np.zeros_like(dI)

        dX = np.concatenate([dS, dI, dE], axis=-1)

        return dX.reshape(-1) * 60  # mul 60 is to scale the time

    t_range = np.arange(t_start, t_end + t_inc, t_inc)
    New_X = spi.odeint(diff_SEIS_Individual, X.reshape(-1), t_range)

    if diff_flag:
        return New_X.reshape(len(t_range), N, x_dim), \
               np.gradient(New_X, axis=0, edge_order=1).reshape(len(t_range), N, x_dim), \
               t_range.reshape(-1, 1)
    else:
        return New_X.reshape(len(t_range), N, x_dim), t_range.reshape(-1, 1)


def SEIR_Individual_dynamics(X, sparse_A,
                             t_start=0, t_end=100, t_inc=1, diff_flag=False, **params):
    """
    :param X:
    :param sparse_A:
    :param t_start:
    :param t_end:
    :param t_inc:
    :param diff_flag:
    :param params:
    :return:

    Implemented according to [Mina Youssef, Caterina Scoglio. An individual-based approach to SIR epidemics in contact
    networks. Journal of Theoretical Biology 283 (2011) 136–144.]

    """
    if 'b' in params:
        print('setting b to %s ...' % params['b'])
        b = params['b']
    else:
        # default_val = 1.5 / 3.2  # for 2009 Hong Kong H1N1 Influenza Pandemic
        # default_val = 2.0 / 3.   # for 2010 Taiwan Seasonal Influenza
        # default_val = 7.75 / 5.  #  for 2010 Taiwan Varicella
        # default_val = 2.16 / 7.  # Mumps
        default_val = 0.5 / 7.  # [0.5 / 7., 8 / 2.]
        print('setting default values [%s] to b ...' % default_val)
        b = default_val
    if 'c' in params:
        print('setting c to %s ...' % params['c'])
        c = params['c']
    else:
        # default_val = 1 / 14.  # Mumps
        default_val = 1 / 14  # [1/14, 1/7]
        print('setting default values [%s] to c ...' % default_val)
        c = default_val
    if 'r' in params:
        print('setting r to %s ...' % params['r'])
        r = params['r']
    else:
        # default_val = 1 / 3.2  # for 2009 Hong Kong H1N1 Influenza Pandemic
        # default_val = 1 / 3.  # for 2010 Taiwan Seasonal Influenza
        # default_val = 1 / 5.  # for 2010 Taiwan Varicella
        # default_val = 1 / 7.  # Mumps
        default_val = 1 / 7.  # [1/7, 1/2]
        print('setting default values [%s] to r ...' % default_val)
        r = default_val

    N, x_dim = X.shape
    row, col = sparse_A  # sparse_A in opinion_dynamics should be a fully-connected graph.

    def diff_SEIR_Individual(X, t):
        # dS_i/dt = -b * \sum_{j=1}^{n} A_{i,j} * I_i * S_i
        # dI_i/dt = b * \sum_{j=1}^{n} A_{i,j} * I_i * S_i - r * I_i
        # dR_i/dt = r * I_i

        X_self = X.reshape(-1, x_dim)
        X_j = X_self[row]

        # we do not know the scatter_sum in numpy package, so we use scatter_sum in torch instead.
        sum_j = scatter_sum(torch.from_numpy(X_j[:, 1].reshape(-1, 1)), torch.from_numpy(col).long(), dim=0,
                            dim_size=X_self.shape[0]).numpy()

        dS = -b * sum_j.reshape(-1, 1) * X_self[:, 0].reshape(-1, 1)
        dI = c * X_self[:, -1].reshape(-1, 1) - r * X_self[:, 1].reshape(-1, 1)
        dR = r * X_self[:, 1].reshape(-1, 1)
        dE = b * sum_j.reshape(-1, 1) * X_self[:, 0].reshape(-1, 1) - c * X_self[:, -1].reshape(-1, 1)

        dX = np.concatenate([dS, dI, dR, dE], axis=-1)

        return dX.reshape(-1) * 60  # mul 60 is to scale the time

    t_range = np.arange(t_start, t_end + t_inc, t_inc)
    New_X = spi.odeint(diff_SEIR_Individual, X.reshape(-1), t_range)

    if diff_flag:
        return New_X.reshape(len(t_range), N, x_dim), \
               np.gradient(New_X, axis=0, edge_order=1).reshape(len(t_range), N, x_dim), \
               t_range.reshape(-1, 1)
    else:
        return New_X.reshape(len(t_range), N, x_dim), t_range.reshape(-1, 1)


def Coupled_Epidemic_dynamics(X, sparse_A,
                              t_start=0, t_end=100, t_inc=1, diff_flag=False, **params):
    """
    :param X:
    :param sparse_A:
    :param t_start:
    :param t_end:
    :param t_inc:
    :param diff_flag:
    :param params:
    :return:

    Implemented according to [Mina Youssef, Caterina Scoglio. An individual-based approach to SIR epidemics in contact
    networks. Journal of Theoretical Biology 283 (2011) 136–144.]

    """
    if 'b1' in params:
        print('setting b1 to %s ...' % params['b1'])
        b1 = params['b1']
    else:
        default_val = 0.1  # [0.02, 0.2]
        print('setting default values [%s] to b1 ...' % default_val)
        b1 = default_val
    if 'b2' in params:
        print('setting b2 to %s ...' % params['b2'])
        b2 = params['b2']
    else:
        default_val = 0.1  # [0.02, 0.2]
        print('setting default values [%s] to b2 ...' % default_val)
        b2 = default_val
    if 'r1' in params:
        print('setting r1 to %s ...' % params['r1'])
        r1 = params['r1']
    else:
        default_val = 0.2  # [0.1, 0.4]
        print('setting default values [%s] to r1 ...' % default_val)
        r1 = default_val
    if 'r2' in params:
        print('setting r2 to %s ...' % params['r2'])
        r2 = params['r2']
    else:
        default_val = 0.2  # [0.1, 0.4]
        print('setting default values [%s] to r2 ...' % default_val)
        r2 = default_val
    if 'c' in params:
        print('setting c to %s ...' % params['c'])
        c = params['c']
    else:

        default_val = 1  # {0.1, 1, 10}
        print('setting default values [%s] to c ...' % default_val)
        c = default_val

    N, x_dim = X.shape
    row, col = sparse_A  # sparse_A in opinion_dynamics should be a fully-connected graph.

    def diff_Coupled_SIS_Individual(X, t):
        # dS_i/dt = -b * \sum_{j=1}^{n} A_{i,j} * I_i * S_i
        # dI_i/dt = b * \sum_{j=1}^{n} A_{i,j} * I_i * S_i - r * I_i
        # dR_i/dt = r * I_i

        X_self = X.reshape(-1, x_dim)
        X_j = X_self[row]

        # we do not know the scatter_sum in numpy package, so we use scatter_sum in torch instead.
        sum_j = scatter_sum(torch.from_numpy(X_j.reshape(-1, x_dim)), torch.from_numpy(col).long(), dim=0,
                            dim_size=X_self.shape[0]).numpy()

        dSS = -b1 * X_self[:, 0].reshape(-1, 1) * (sum_j[:, 1].reshape(-1, 1) + sum_j[:, 3].reshape(-1, 1)) \
              - b2 * X_self[:, 0].reshape(-1, 1) * (sum_j[:, 2].reshape(-1, 1) + sum_j[:, 3].reshape(-1, 1)) \
              + r1 * X_self[:, 1].reshape(-1, 1) + r2 * X_self[:, 2].reshape(-1, 1)
        dIS = b1 * X_self[:, 0].reshape(-1, 1) * (sum_j[:, 1].reshape(-1, 1) + sum_j[:, 3].reshape(-1, 1)) \
              - c * b2 * X_self[:, 1].reshape(-1, 1) * (sum_j[:, 2].reshape(-1, 1) + sum_j[:, 3].reshape(-1, 1)) \
              + r2 * X_self[:, 3].reshape(-1, 1) - r1 * X_self[:, 1].reshape(-1, 1)

        dSI = b2 * X_self[:, 0].reshape(-1, 1) * (sum_j[:, 2].reshape(-1, 1) + sum_j[:, 3].reshape(-1, 1)) \
              - c * b1 * X_self[:, 2].reshape(-1, 1) * (sum_j[:, 1].reshape(-1, 1) + sum_j[:, 3].reshape(-1, 1)) \
              + r1 * X_self[:, 3].reshape(-1, 1) - r2 * X_self[:, 2].reshape(-1, 1)
        dII = c * b2 * X_self[:, 1].reshape(-1, 1) * (sum_j[:, 2].reshape(-1, 1) + sum_j[:, 3].reshape(-1, 1)) \
              + c * b1 * X_self[:, 2].reshape(-1, 1) * (sum_j[:, 1].reshape(-1, 1) + sum_j[:, 3].reshape(-1, 1)) \
              - r1 * X_self[:, 3].reshape(-1, 1) - r2 * X_self[:, 3].reshape(-1, 1)

        dX = np.concatenate([dSS, dIS, dSI, dII], axis=-1)

        return dX.reshape(-1)

    t_range = np.arange(t_start, t_end + t_inc, t_inc)
    New_X = spi.odeint(diff_Coupled_SIS_Individual, X.reshape(-1), t_range)

    if diff_flag:
        return New_X.reshape(len(t_range), N, x_dim), \
               np.gradient(New_X, axis=0, edge_order=1).reshape(len(t_range), N, x_dim), \
               t_range.reshape(-1, 1)
    else:
        return New_X.reshape(len(t_range), N, x_dim), t_range.reshape(-1, 1)


def SIR_meta_pop_dynamics(X, sparse_A,
                          t_start=0, t_end=100, t_inc=1, diff_flag=False, **params):
    """
    :param X:
    :param sparse_A:
    :param t_start:
    :param t_end:
    :param t_inc:
    :param diff_flag:
    :param params:
    :return:

    Implemented according to [Mina Youssef, Caterina Scoglio. An individual-based approach to SIR epidemics in contact
    networks. Journal of Theoretical Biology 283 (2011) 136–144.]

    """

    if 'b' in params:
        print('setting b to %s ...' % params['b'])
        b = params['b']
    else:
        # default_val = 1.5 / 3.2  # for 2009 Hong Kong H1N1 Influenza Pandemic
        # default_val = 2.0 / 3.   # for 2010 Taiwan Seasonal Influenza
        # default_val = 7.75 / 5.  #  for 2010 Taiwan Varicella
        # default_val = 2.16 / 7.  # Mumps
        # default_val = 8 / 2.  # [0.5 / 7., 8 / 2.]
        # default_val = 2.5 / 7.5
        default_val = 2.5 / 7.5
        print('setting default values [%s] to b ...' % default_val)
        b = default_val
    if 'r' in params:
        print('setting r to %s ...' % params['r'])
        r = params['r']
    else:
        # default_val = 1 / 3.2  # for 2009 Hong Kong H1N1 Influenza Pandemic
        # default_val = 1 / 3.  # for 2010 Taiwan Seasonal Influenza
        # default_val = 1 / 5.  # for 2010 Taiwan Varicella
        # default_val = 1 / 7.  # Mumps
        # default_val = 1 / 2.  # [1/7, 1/2]
        default_val = 1 / 7.5
        print('setting default values [%s] to r ...' % default_val)
        r = default_val

    N, x_dim = X.shape
    row, col = sparse_A  # sparse_A in opinion_dynamics should be a fully-connected graph.

    if params['edge_weights'] is not None:
        edge_weights = params['edge_weights']
    else:
        print('Error: need [edge_weights] as input while computing!')
        exit(1)

    def diff_SIR_Meta_Pop(X, t):
        # dS_i/dt = -b * \sum_{j=1}^{n} A_{i,j} * I_i * S_i
        # dI_i/dt = b * \sum_{j=1}^{n} A_{i,j} * I_i * S_i - r * I_i
        # dR_i/dt = r * I_i

        X_self = X.reshape(-1, x_dim)
        X_j = X_self[row]

        X_j = 1 - np.exp(-b * X_j)

        X_j = X_j * edge_weights.reshape(-1, 1)

        # we do not know the scatter_sum in numpy package, so we use scatter_sum in torch instead.
        sum_j = 1 - np.exp(-b * X_self[:, 1].reshape(-1, 1)) + scatter_sum(torch.from_numpy(X_j[:, 1].reshape(-1, 1)),
                                                                           torch.from_numpy(col).long(), dim=0,
                                                                           dim_size=X_self.shape[0]).numpy()

        dS = -sum_j.reshape(-1, 1) * X_self[:, 0].reshape(-1, 1)
        dI = sum_j.reshape(-1, 1) * X_self[:, 0].reshape(-1, 1) - r * X_self[:, 1].reshape(-1, 1)
        dR = r * X_self[:, 1].reshape(-1, 1)

        dX = np.concatenate([dS, dI, dR], axis=-1)

        return dX.reshape(-1) * 100

    t_range = np.arange(t_start, t_end + t_inc, t_inc)
    New_X = spi.odeint(diff_SIR_Meta_Pop, X.reshape(-1), t_range)

    if diff_flag:
        return New_X.reshape(len(t_range), N, x_dim), \
               np.gradient(New_X, axis=0, edge_order=1).reshape(len(t_range), N, x_dim), \
               t_range.reshape(-1, 1)
    else:
        return New_X.reshape(len(t_range), N, x_dim), t_range.reshape(-1, 1)


def sim_epidemic_dynamics(X, sparse_A,
                          t_start=0, t_end=100, t_inc=1, diff_flag=False, **params):
    """
    :param X:
    :param sparse_A:
    :param t_start:
    :param t_end:
    :param t_inc:
    :param diff_flag:
    :param params:
    :return:

    Implemented according to [Mina Youssef, Caterina Scoglio. An individual-based approach to SIR epidemics in contact
    networks. Journal of Theoretical Biology 283 (2011) 136–144.]

    """

    if 'a' in params:
        print('setting a to %s ...' % params['a'])
        a = params['a']
    else:
        default_val = 0.074
        print('setting default values [%s] to a ...' % default_val)
        a = default_val
    if 'b' in params:
        print('setting b to %s ...' % params['b'])
        b = params['b']
    else:
        default_val = 7.130
        print('setting default values [%s] to b ...' % default_val)
        b = default_val

    N, x_dim = X.shape
    row, col = sparse_A  # sparse_A in opinion_dynamics should be a fully-connected graph.

    if params['edge_weights'] is not None:
        edge_weights = params['edge_weights']
    else:
        print('Error: need [edge_weights] as input while computing!')
        exit(1)

    def diff_sim_epidemic(X, t):
        # dx_i/dt = a * x_i + b * \sum_{j=1}^{n} A_{i,j} * \frac{1}{1 + \exp{-(x_j - x_i)}}

        X_self = X.reshape(-1, x_dim)
        X_j = X_self[row]
        X_i = X_self[col]

        X_j = 1. / (1. + np.exp(X_i - X_j))

        X_j = X_j * edge_weights.reshape(-1, 1)

        # we do not know the scatter_sum in numpy package, so we use scatter_sum in torch instead.
        sum_j = scatter_sum(torch.from_numpy(X_j.reshape(-1, 1)), torch.from_numpy(col).long(), dim=0,
                            dim_size=X_self.shape[0]).numpy()

        dX = a * X_self + b * sum_j.reshape(-1, 1)

        return dX.reshape(-1)

    #t_range = np.arange(t_start, t_end + t_inc, t_inc)
    t_range = np.arange(t_start, t_end, t_inc)
    New_X = spi.odeint(diff_sim_epidemic, X.reshape(-1), t_range)

    if diff_flag:
        return New_X.reshape(len(t_range), N, x_dim), \
               np.gradient(New_X, axis=0, edge_order=1).reshape(len(t_range), N, x_dim), \
               t_range.reshape(-1, 1)
    else:
        return New_X.reshape(len(t_range), N, x_dim), t_range.reshape(-1, 1)



# brain dynamics
def brain_FitzHugh_Nagumo_dynamics(X, sparse_A,
                            t_start=0, t_end=100, t_inc=1, diff_flag=False, **params):
    N, x_dim = X.shape

    if 'a' in params:
        print('setting a to %s ...' % params['a'])
        a = params['a']
    else:
        default_val = 0.28  #[0.2,0.3]
        print('setting default values [%s] to a ...' % default_val)
        a = default_val
    if 'b' in params:
        print('setting b to %s ...' % params['b'])
        b = params['b']
    else:
        default_val = 0.5  #[0.4,0.6]
        print('setting default values [%s] to b ...' % default_val)
        b = default_val
    if 'c' in params:
        print('setting c to %s ...' % params['c'])
        c = params['c']
    else:
        default_val = -0.04  #[-0.02, -0.06]
        print('setting default values [%s] to c ...' % default_val)
        c = default_val
    if 'd' in params:
        print('setting d to %s ...' % params['d'])
        d = params['d']
    else:
        default_val = 1.  #[0.8,1.2]
        print('setting default values [%s] to d ...' % default_val)
        d = default_val

    row, col = sparse_A  # sparse_A in opinion_dynamics should be a fully-connected graph.

    def diff_brain_FitzHugh_Nagumo(X, t):

        X_j = X.reshape(-1, x_dim)[row]
        X_i = X.reshape(-1, x_dim)[col]

        in_deg = scatter_sum(torch.from_numpy(np.ones_like(col)), torch.from_numpy(col).long(), dim=0, dim_size=X.reshape(-1, x_dim).shape[0]).numpy()

        aa = scatter_sum(torch.from_numpy((X_j[:,0]-X_i[:,0])/in_deg[col]), torch.from_numpy(col).long(), dim=0, dim_size=X.reshape(-1, x_dim).shape[0]).numpy()

        dX_0 = X.reshape(-1, x_dim)[:, 0] - X.reshape(-1, x_dim)[:, 0]**3 - X.reshape(-1, x_dim)[:, 1] - d * aa

        dX_1 = a + b * X.reshape(-1, x_dim)[:, 0] + c * X.reshape(-1, x_dim)[:, 1]

        dX = np.concatenate([dX_0.reshape(-1, 1), dX_1.reshape(-1, 1)], axis=-1)

        return dX.reshape(-1) * 30

    t_range = np.arange(t_start, t_end + t_inc, t_inc)
    New_X = spi.odeint(diff_brain_FitzHugh_Nagumo, X.reshape(-1), t_range)

    if diff_flag:
        return New_X.reshape(len(t_range), N, x_dim), \
               np.gradient(New_X, axis=0, edge_order=1).reshape(len(t_range), N, x_dim), \
               t_range.reshape(-1, 1)
    else:
        return New_X.reshape(len(t_range), N, x_dim), t_range.reshape(-1, 1)


# 2nd order
def phototaxis_dynamics(X, sparse_A,
                        t_start=0, t_end=100, t_inc=1, diff_flag=False, **params):
    N, x_dim = X.shape

    if 'I0' in params:
        print('setting I0 to %s ...' % params['I0'])
        I0 = params['I0']
    else:
        default_val = 0.1  # [0.01, 1.]
        # default_val = 1.  # [0.5, 1.5]
        print('setting default values [%s] to I0 ...' % default_val)
        I0 = default_val
    if 'V' in params:
        print('setting V to %s ...' % params['V'])
        V = params['V']
    else:
        default_val = np.array([60., 0]).reshape(1, 2)  # [20, 100]
        print('setting default values [%s] to V ...' % default_val)
        V = default_val
    if 'b' in params:
        print('setting b to %s ...' % params['b'])
        b = params['b']
    else:
        # default_val = np.array([60., 0]).reshape(1, 2)  # [-1, -3]
        default_val = -0.25  # [-0.4, -0.1]
        print('setting default values [%s] to b ...' % default_val)
        b = default_val
    if 'ec' in params:
        print('setting ec to %s ...' % params['ec'])
        ec = params['ec']
    else:
        # default_val = np.array([60., 0]).reshape(1, 2)  # [-1, -3]
        default_val = 0.3 # [0.1, 0.5]
        print('setting default values [%s] to ec ...' % default_val)
        ec = default_val

    row, col = sparse_A  # sparse_A in opinion_dynamics should be a fully-connected graph.

    def diff_phototaxis(X, t):
        # dx_i/dt = \frac{1}{n} \sum_{j=1}^{n} \phi(||x_j-x_i||)(x_j-x_i)
        # \phi(r) := 1     0 <= r < 1/sqrt(2);
        #            0.1   1/sqrt(2) <= r < 1;
        #            0     1 <= r.
        X_j = X.reshape(-1, x_dim)[row][:, :2]
        X_i = X.reshape(-1, x_dim)[col][:, :2]

        X_j_1st_diff = X.reshape(-1, x_dim)[row][:, 2:4]
        X_i_1st_diff = X.reshape(-1, x_dim)[col][:, 2:4]

        X_e = X.reshape(-1, x_dim)[:, 4:]

        def phi(r):
            return (1. + r ** 2.) ** (b)

        def e_func(e, e_c):
            e = e.reshape(-1)
            res_e = np.zeros_like(e)
            for ii in range(e.shape[0]):
                if e[ii] <= e_c:
                    res_e[ii] = 1.
                elif e_c < e[ii] <= 2. * e_c:
                    res_e[ii] = 1. / 2. * (np.cos((np.pi / e_c) * (e[ii] - e_c)) + 1.)
                    # res_e[ii] = e[ii]
                else:
                    res_e[ii] = 0.
            return res_e.reshape(-1, 1)

        X_j_minus_X_i = (X_j - X_i).reshape(-1, 2)
        r = np.linalg.norm(x=X_j_minus_X_i, ord=2, axis=-1, keepdims=True)
        diff_X_j_minus_diff_X_i = (X_j_1st_diff - X_i_1st_diff).reshape(-1, 2)
        aa = scatter_mean(torch.from_numpy(
            phi(r) * diff_X_j_minus_diff_X_i
        ),
            torch.from_numpy(col).long(), dim=0,
            dim_size=X.reshape(-1, x_dim).shape[0]).numpy()

        X_e_j_minus_X_e_i = (X_e[row] - X_e[col]).reshape(-1, 1)
        bb = scatter_mean(torch.from_numpy(
            phi(r) * X_e_j_minus_X_e_i
        ),
            torch.from_numpy(col).long(), dim=0,
            dim_size=X.reshape(-1, x_dim).shape[0]).numpy()

        X_1st_diff = X.reshape(-1, x_dim)[:, 2:4]
        X_2nd_diff = I0 * (V - X_1st_diff) * (1 - e_func(X_e, ec)) + 100. * aa
        dX_e = I0 * e_func(X_e, 2 * ec) + 100. * bb

        dX = np.concatenate([X_1st_diff.reshape(-1, 2), X_2nd_diff.reshape(-1, 2), dX_e.reshape(-1, 1)], axis=-1)

        return dX.reshape(-1) * 0.5

    t_range = np.arange(t_start, t_end + t_inc, t_inc)
    New_X = spi.odeint(diff_phototaxis, X.reshape(-1), t_range)

    if diff_flag:
        return New_X.reshape(len(t_range), N, x_dim), \
               np.gradient(New_X, axis=0, edge_order=1).reshape(len(t_range), N, x_dim), \
               t_range.reshape(-1, 1)
    else:
        return New_X.reshape(len(t_range), N, x_dim), t_range.reshape(-1, 1)


class NetDynamic:
    def __init__(self, func):
        if func == 'heat_diffusion_dynamics':
            self.Func = heat_diffusion_dynamics
        elif func == 'mutualistic_interaction_dynamics':
            self.Func = mutualistic_interaction_dynamics
        elif func == 'gene_regulatory_dynamics':
            self.Func = gene_regulatory_dynamics
        elif func == 'combination_dynamics':
            self.Func = combination_dynamics
        elif func == 'combination_dynamics_vary_coeff':
            self.Func = combination_dynamics_vary_coeff
        elif func == 'vary_dynamics_with_vary_type_and_coeff':
            self.Func = vary_dynamics_with_vary_type_and_coeff
        elif func == 'opinion_dynamics':
            self.Func = opinion_dynamics
        elif func == 'opinion_dynamics_Baumann2021':
            self.Func = opinion_dynamics_Baumann2021
        elif func == 'opinion_dynamics_Baumann2021_2topic':
            self.Func = opinion_dynamics_Baumann2021_2topic
        elif func == 'SI_Individual_dynamics':
            self.Func = SI_Individual_dynamics
        elif func == 'SIS_Individual_dynamics':
            self.Func = SIS_Individual_dynamics
        elif func == 'SIR_Individual_dynamics':
            self.Func = SIR_Individual_dynamics
        elif func == 'SEIS_Individual_dynamics':
            self.Func = SEIS_Individual_dynamics
        elif func == 'SEIR_Individual_dynamics':
            self.Func = SEIR_Individual_dynamics
        elif func == 'SIR_meta_pop_dynamics':
            self.Func = SIR_meta_pop_dynamics
        elif func == 'Coupled_Epidemic_dynamics':
            self.Func = Coupled_Epidemic_dynamics
        elif func == 'sim_epidemic_dynamics':
            self.Func = sim_epidemic_dynamics
        elif func == 'predator_swarm_dynamics':
            self.Func = predator_swarm_dynamics
        elif func == 'brain_FitzHugh_Nagumo_dynamics':
            self.Func = brain_FitzHugh_Nagumo_dynamics
        elif func == 'phototaxis_dynamics':
            self.Func = phototaxis_dynamics
        else:
            print('Error: func name [%s] is not recognized' % func)
            exit(1)

    def get_observations(self, X0, A, t_start, t_end, t_inc, diff_flag, **params):
        X = self.Func(X0, A, t_start=t_start, t_end=t_end, t_inc=t_inc, diff_flag=diff_flag, **params)
        return X


#########===================================================
##
##                   Datasets
##
#########===================================================
class dynamics_dataset:
    def __init__(self,
                 dynamics_name,
                 topo_type,
                 topo_fixed=False,
                 num_graphs_samples=50000,
                 time_steps=128,
                 t_inc=0.01,
                 x_dim=1,
                 make_test_set=False
                 ):

        self.state_ub = 1.
        self.state_lb = -1.
        self.dynamics_name = dynamics_name
        self.topo_type = topo_type
        self.topo_fixed = topo_fixed
        self.time_steps = time_steps
        self.x_dim = x_dim

        dynamics_topo_names = None

        if topo_type == 'all':
            topo_types = ['grid', 'power_law', 'random', 'small_world', 'community']
        else:
            topo_types = [topo_type]

        if dynamics_name == 'all':
            dynamics_names = ['heat_diffusion_dynamics', 'mutualistic_interaction_dynamics', 'gene_regulatory_dynamics']
        elif dynamics_name == '2nd_phase':
            dynamics_topo_names = [('heat_diffusion_dynamics', 'grid'),
                                   ('mutualistic_interaction_dynamics', 'grid'),
                                   ('gene_regulatory_dynamics', 'grid'),
                                   # ('combination_dynamics_vary_coeff', 'grid'),
                                   # ('vary_dynamics_with_vary_type_and_coeff', 'grid'),
                                   ('heat_diffusion_dynamics', 'power_law'),
                                   ('mutualistic_interaction_dynamics', 'power_law'),
                                   ('gene_regulatory_dynamics', 'power_law'),
                                   # ('combination_dynamics_vary_coeff', 'power_law'),
                                   # ('vary_dynamics_with_vary_type_and_coeff', 'power_law'),
                                   ('heat_diffusion_dynamics', 'random'),
                                   ('mutualistic_interaction_dynamics', 'random'),
                                   ('gene_regulatory_dynamics', 'random'),
                                   # ('combination_dynamics_vary_coeff', 'random'),
                                   # ('vary_dynamics_with_vary_type_and_coeff', 'random'),
                                   ('heat_diffusion_dynamics', 'small_world'),
                                   ('mutualistic_interaction_dynamics', 'small_world'),
                                   ('gene_regulatory_dynamics', 'small_world'),
                                   # ('combination_dynamics_vary_coeff', 'small_world'),
                                   # ('vary_dynamics_with_vary_type_and_coeff', 'small_world'),
                                   ('heat_diffusion_dynamics', 'community'),
                                   ('mutualistic_interaction_dynamics', 'community'),
                                   ('gene_regulatory_dynamics', 'community'),
                                   # ('combination_dynamics_vary_coeff', 'community'),
                                   # ('vary_dynamics_with_vary_type_and_coeff', 'community'),
                                   ('opinion_dynamics_Baumann2021_2topic', 'small_world'),
                                   ('SI_Individual_dynamics', 'power_law'),
                                   ('SIS_Individual_dynamics', 'power_law'),
                                   ('SIR_Individual_dynamics', 'power_law'),
                                   ('SEIS_Individual_dynamics', 'power_law'),
                                   ('SEIR_Individual_dynamics', 'power_law'),
                                   ('Coupled_Epidemic_dynamics', 'power_law'),
                                   ('SIR_meta_pop_dynamics', 'directed_full_connected'),
                                   #('RealEpidemicData_mix', 'power_law'),
                                   # ('RealEpidemicData_124', 'power_law'),
                                   # ('RealEpidemicData_134', 'power_law'),
                                   # ('RealEpidemicData_234', 'power_law'),
                                   ]
        elif dynamics_name == '2nd_phase_2':
            dynamics_topo_names = [('heat_diffusion_dynamics', 'grid'),
                                   ('mutualistic_interaction_dynamics', 'grid'),
                                   # ('gene_regulatory_dynamics', 'grid'),
                                   # ('combination_dynamics_vary_coeff', 'grid'),
                                   # ('vary_dynamics_with_vary_type_and_coeff', 'grid'),
                                   ('heat_diffusion_dynamics', 'power_law'),
                                   ('mutualistic_interaction_dynamics', 'power_law'),
                                   # ('gene_regulatory_dynamics', 'power_law'),
                                   # ('combination_dynamics_vary_coeff', 'power_law'),
                                   # ('vary_dynamics_with_vary_type_and_coeff', 'power_law'),
                                   ('heat_diffusion_dynamics', 'random'),
                                   ('mutualistic_interaction_dynamics', 'random'),
                                   # ('gene_regulatory_dynamics', 'random'),
                                   # ('combination_dynamics_vary_coeff', 'random'),
                                   # ('vary_dynamics_with_vary_type_and_coeff', 'random'),
                                   ('heat_diffusion_dynamics', 'small_world'),
                                   ('mutualistic_interaction_dynamics', 'small_world'),
                                   # ('gene_regulatory_dynamics', 'small_world'),
                                   # ('combination_dynamics_vary_coeff', 'small_world'),
                                   # ('vary_dynamics_with_vary_type_and_coeff', 'small_world'),
                                   ('heat_diffusion_dynamics', 'community'),
                                   ('mutualistic_interaction_dynamics', 'community'),
                                   # ('gene_regulatory_dynamics', 'community'),
                                   # ('combination_dynamics_vary_coeff', 'community'),
                                   # ('vary_dynamics_with_vary_type_and_coeff', 'community'),
                                   # ('opinion_dynamics_Baumann2021_2topic', 'small_world'),
                                   # ('SI_Individual_dynamics', 'power_law'),
                                   # ('SIS_Individual_dynamics', 'power_law'),
                                   # ('SIR_Individual_dynamics', 'power_law'),
                                   # ('SEIS_Individual_dynamics', 'power_law'),
                                   # ('SEIR_Individual_dynamics', 'power_law'),
                                   ]
        elif dynamics_name == '2nd_phase_3':
            dynamics_topo_names = [('heat_diffusion_dynamics', 'grid'),
                                   ('mutualistic_interaction_dynamics', 'grid'),
                                   ('gene_regulatory_dynamics', 'grid'),
                                   # ('combination_dynamics_vary_coeff', 'grid'),
                                   # ('vary_dynamics_with_vary_type_and_coeff', 'grid'),
                                   ('heat_diffusion_dynamics', 'power_law'),
                                   ('mutualistic_interaction_dynamics', 'power_law'),
                                   ('gene_regulatory_dynamics', 'power_law'),
                                   # ('combination_dynamics_vary_coeff', 'power_law'),
                                   # ('vary_dynamics_with_vary_type_and_coeff', 'power_law'),
                                   ('heat_diffusion_dynamics', 'random'),
                                   ('mutualistic_interaction_dynamics', 'random'),
                                   ('gene_regulatory_dynamics', 'random'),
                                   # ('combination_dynamics_vary_coeff', 'random'),
                                   # ('vary_dynamics_with_vary_type_and_coeff', 'random'),
                                   ('heat_diffusion_dynamics', 'small_world'),
                                   ('mutualistic_interaction_dynamics', 'small_world'),
                                   ('gene_regulatory_dynamics', 'small_world'),
                                   # ('combination_dynamics_vary_coeff', 'small_world'),
                                   # ('vary_dynamics_with_vary_type_and_coeff', 'small_world'),
                                   ('heat_diffusion_dynamics', 'community'),
                                   ('mutualistic_interaction_dynamics', 'community'),
                                   ('gene_regulatory_dynamics', 'community'),
                                   # ('combination_dynamics_vary_coeff', 'community'),
                                   # ('vary_dynamics_with_vary_type_and_coeff', 'community'),
                                   # ('opinion_dynamics_Baumann2021_2topic', 'small_world'),
                                   # ('SI_Individual_dynamics', 'power_law'),
                                   # ('SIS_Individual_dynamics', 'power_law'),
                                   # ('SIR_Individual_dynamics', 'power_law'),
                                   # ('SEIS_Individual_dynamics', 'power_law'),
                                   # ('SEIR_Individual_dynamics', 'power_law'),
                                   ]
        elif dynamics_name == '2nd_phase_4':
            dynamics_topo_names = [('heat_diffusion_dynamics', 'grid'),
                                   ('mutualistic_interaction_dynamics', 'grid'),
                                   ('gene_regulatory_dynamics', 'grid'),
                                   # ('combination_dynamics_vary_coeff', 'grid'),
                                   # ('vary_dynamics_with_vary_type_and_coeff', 'grid'),
                                   ('heat_diffusion_dynamics', 'power_law'),
                                   ('mutualistic_interaction_dynamics', 'power_law'),
                                   ('gene_regulatory_dynamics', 'power_law'),
                                   # ('combination_dynamics_vary_coeff', 'power_law'),
                                   # ('vary_dynamics_with_vary_type_and_coeff', 'power_law'),
                                   ('heat_diffusion_dynamics', 'random'),
                                   ('mutualistic_interaction_dynamics', 'random'),
                                   ('gene_regulatory_dynamics', 'random'),
                                   # ('combination_dynamics_vary_coeff', 'random'),
                                   # ('vary_dynamics_with_vary_type_and_coeff', 'random'),
                                   ('heat_diffusion_dynamics', 'small_world'),
                                   ('mutualistic_interaction_dynamics', 'small_world'),
                                   ('gene_regulatory_dynamics', 'small_world'),
                                   # ('combination_dynamics_vary_coeff', 'small_world'),
                                   # ('vary_dynamics_with_vary_type_and_coeff', 'small_world'),
                                   ('heat_diffusion_dynamics', 'community'),
                                   ('mutualistic_interaction_dynamics', 'community'),
                                   ('gene_regulatory_dynamics', 'community'),
                                   # ('combination_dynamics_vary_coeff', 'community'),
                                   # ('vary_dynamics_with_vary_type_and_coeff', 'community'),
                                   ('opinion_dynamics_Baumann2021_2topic', 'small_world'),
                                   # ('SI_Individual_dynamics', 'power_law'),
                                   # ('SIS_Individual_dynamics', 'power_law'),
                                   # ('SIR_Individual_dynamics', 'power_law'),
                                   # ('SEIS_Individual_dynamics', 'power_law'),
                                   # ('SEIR_Individual_dynamics', 'power_law'),
                                   ]
        elif dynamics_name == '2nd_phase_5':
            dynamics_topo_names = [('heat_diffusion_dynamics', 'grid'),
                                   ('mutualistic_interaction_dynamics', 'grid'),
                                   ('gene_regulatory_dynamics', 'grid'),
                                   # ('combination_dynamics_vary_coeff', 'grid'),
                                   # ('vary_dynamics_with_vary_type_and_coeff', 'grid'),
                                   ('heat_diffusion_dynamics', 'power_law'),
                                   ('mutualistic_interaction_dynamics', 'power_law'),
                                   ('gene_regulatory_dynamics', 'power_law'),
                                   # ('combination_dynamics_vary_coeff', 'power_law'),
                                   # ('vary_dynamics_with_vary_type_and_coeff', 'power_law'),
                                   ('heat_diffusion_dynamics', 'random'),
                                   ('mutualistic_interaction_dynamics', 'random'),
                                   ('gene_regulatory_dynamics', 'random'),
                                   # ('combination_dynamics_vary_coeff', 'random'),
                                   # ('vary_dynamics_with_vary_type_and_coeff', 'random'),
                                   ('heat_diffusion_dynamics', 'small_world'),
                                   ('mutualistic_interaction_dynamics', 'small_world'),
                                   ('gene_regulatory_dynamics', 'small_world'),
                                   # ('combination_dynamics_vary_coeff', 'small_world'),
                                   # ('vary_dynamics_with_vary_type_and_coeff', 'small_world'),
                                   ('heat_diffusion_dynamics', 'community'),
                                   ('mutualistic_interaction_dynamics', 'community'),
                                   ('gene_regulatory_dynamics', 'community'),
                                   # ('combination_dynamics_vary_coeff', 'community'),
                                   # ('vary_dynamics_with_vary_type_and_coeff', 'community'),
                                   ('opinion_dynamics_Baumann2021_2topic', 'small_world'),
                                   ('SI_Individual_dynamics', 'power_law'),
                                   # ('SIS_Individual_dynamics', 'power_law'),
                                   # ('SIR_Individual_dynamics', 'power_law'),
                                   # ('SEIS_Individual_dynamics', 'power_law'),
                                   # ('SEIR_Individual_dynamics', 'power_law'),
                                   ]
        elif dynamics_name == '2nd_phase_6':
            dynamics_topo_names = [('heat_diffusion_dynamics', 'grid'),
                                   ('mutualistic_interaction_dynamics', 'grid'),
                                   ('gene_regulatory_dynamics', 'grid'),
                                   # ('combination_dynamics_vary_coeff', 'grid'),
                                   # ('vary_dynamics_with_vary_type_and_coeff', 'grid'),
                                   ('heat_diffusion_dynamics', 'power_law'),
                                   ('mutualistic_interaction_dynamics', 'power_law'),
                                   ('gene_regulatory_dynamics', 'power_law'),
                                   # ('combination_dynamics_vary_coeff', 'power_law'),
                                   # ('vary_dynamics_with_vary_type_and_coeff', 'power_law'),
                                   ('heat_diffusion_dynamics', 'random'),
                                   ('mutualistic_interaction_dynamics', 'random'),
                                   ('gene_regulatory_dynamics', 'random'),
                                   # ('combination_dynamics_vary_coeff', 'random'),
                                   # ('vary_dynamics_with_vary_type_and_coeff', 'random'),
                                   ('heat_diffusion_dynamics', 'small_world'),
                                   ('mutualistic_interaction_dynamics', 'small_world'),
                                   ('gene_regulatory_dynamics', 'small_world'),
                                   # ('combination_dynamics_vary_coeff', 'small_world'),
                                   # ('vary_dynamics_with_vary_type_and_coeff', 'small_world'),
                                   ('heat_diffusion_dynamics', 'community'),
                                   ('mutualistic_interaction_dynamics', 'community'),
                                   ('gene_regulatory_dynamics', 'community'),
                                   # ('combination_dynamics_vary_coeff', 'community'),
                                   # ('vary_dynamics_with_vary_type_and_coeff', 'community'),
                                   ('opinion_dynamics_Baumann2021_2topic', 'small_world'),
                                   ('SI_Individual_dynamics', 'power_law'),
                                   ('SIS_Individual_dynamics', 'power_law'),
                                   # ('SIR_Individual_dynamics', 'power_law'),
                                   # ('SEIS_Individual_dynamics', 'power_law'),
                                   # ('SEIR_Individual_dynamics', 'power_law'),
                                   ]
        elif dynamics_name == '2nd_phase_7':
            dynamics_topo_names = [('heat_diffusion_dynamics', 'grid'),
                                   ('mutualistic_interaction_dynamics', 'grid'),
                                   ('gene_regulatory_dynamics', 'grid'),
                                   # ('combination_dynamics_vary_coeff', 'grid'),
                                   # ('vary_dynamics_with_vary_type_and_coeff', 'grid'),
                                   ('heat_diffusion_dynamics', 'power_law'),
                                   ('mutualistic_interaction_dynamics', 'power_law'),
                                   ('gene_regulatory_dynamics', 'power_law'),
                                   # ('combination_dynamics_vary_coeff', 'power_law'),
                                   # ('vary_dynamics_with_vary_type_and_coeff', 'power_law'),
                                   ('heat_diffusion_dynamics', 'random'),
                                   ('mutualistic_interaction_dynamics', 'random'),
                                   ('gene_regulatory_dynamics', 'random'),
                                   # ('combination_dynamics_vary_coeff', 'random'),
                                   # ('vary_dynamics_with_vary_type_and_coeff', 'random'),
                                   ('heat_diffusion_dynamics', 'small_world'),
                                   ('mutualistic_interaction_dynamics', 'small_world'),
                                   ('gene_regulatory_dynamics', 'small_world'),
                                   # ('combination_dynamics_vary_coeff', 'small_world'),
                                   # ('vary_dynamics_with_vary_type_and_coeff', 'small_world'),
                                   ('heat_diffusion_dynamics', 'community'),
                                   ('mutualistic_interaction_dynamics', 'community'),
                                   ('gene_regulatory_dynamics', 'community'),
                                   # ('combination_dynamics_vary_coeff', 'community'),
                                   # ('vary_dynamics_with_vary_type_and_coeff', 'community'),
                                   ('opinion_dynamics_Baumann2021_2topic', 'small_world'),
                                   ('SI_Individual_dynamics', 'power_law'),
                                   ('SIS_Individual_dynamics', 'power_law'),
                                   ('SIR_Individual_dynamics', 'power_law'),
                                   # ('SEIS_Individual_dynamics', 'power_law'),
                                   # ('SEIR_Individual_dynamics', 'power_law'),
                                   ]
        elif dynamics_name == '2nd_phase_8':
            dynamics_topo_names = [('heat_diffusion_dynamics', 'grid'),
                                   ('mutualistic_interaction_dynamics', 'grid'),
                                   ('gene_regulatory_dynamics', 'grid'),
                                   # ('combination_dynamics_vary_coeff', 'grid'),
                                   # ('vary_dynamics_with_vary_type_and_coeff', 'grid'),
                                   ('heat_diffusion_dynamics', 'power_law'),
                                   ('mutualistic_interaction_dynamics', 'power_law'),
                                   ('gene_regulatory_dynamics', 'power_law'),
                                   # ('combination_dynamics_vary_coeff', 'power_law'),
                                   # ('vary_dynamics_with_vary_type_and_coeff', 'power_law'),
                                   ('heat_diffusion_dynamics', 'random'),
                                   ('mutualistic_interaction_dynamics', 'random'),
                                   ('gene_regulatory_dynamics', 'random'),
                                   # ('combination_dynamics_vary_coeff', 'random'),
                                   # ('vary_dynamics_with_vary_type_and_coeff', 'random'),
                                   ('heat_diffusion_dynamics', 'small_world'),
                                   ('mutualistic_interaction_dynamics', 'small_world'),
                                   ('gene_regulatory_dynamics', 'small_world'),
                                   # ('combination_dynamics_vary_coeff', 'small_world'),
                                   # ('vary_dynamics_with_vary_type_and_coeff', 'small_world'),
                                   ('heat_diffusion_dynamics', 'community'),
                                   ('mutualistic_interaction_dynamics', 'community'),
                                   ('gene_regulatory_dynamics', 'community'),
                                   # ('combination_dynamics_vary_coeff', 'community'),
                                   # ('vary_dynamics_with_vary_type_and_coeff', 'community'),
                                   ('opinion_dynamics_Baumann2021_2topic', 'small_world'),
                                   ('SI_Individual_dynamics', 'power_law'),
                                   ('SIS_Individual_dynamics', 'power_law'),
                                   ('SIR_Individual_dynamics', 'power_law'),
                                   ('SEIS_Individual_dynamics', 'power_law'),
                                   # ('SEIR_Individual_dynamics', 'power_law'),
                                   ]

        elif dynamics_name == 'all_1dim_dynamics':
            dynamics_topo_names = [('heat_diffusion_dynamics', 'grid'),
                                   ('mutualistic_interaction_dynamics', 'grid'),
                                   ('gene_regulatory_dynamics', 'grid'),
                                   ('combination_dynamics_vary_coeff', 'grid'),
                                   ('vary_dynamics_with_vary_type_and_coeff', 'grid'),
                                   ('heat_diffusion_dynamics', 'power_law'),
                                   ('mutualistic_interaction_dynamics', 'power_law'),
                                   ('gene_regulatory_dynamics', 'power_law'),
                                   ('combination_dynamics_vary_coeff', 'power_law'),
                                   ('vary_dynamics_with_vary_type_and_coeff', 'power_law'),
                                   ('heat_diffusion_dynamics', 'random'),
                                   ('mutualistic_interaction_dynamics', 'random'),
                                   ('gene_regulatory_dynamics', 'random'),
                                   ('combination_dynamics_vary_coeff', 'random'),
                                   ('vary_dynamics_with_vary_type_and_coeff', 'random'),
                                   ('heat_diffusion_dynamics', 'small_world'),
                                   ('mutualistic_interaction_dynamics', 'small_world'),
                                   ('gene_regulatory_dynamics', 'small_world'),
                                   ('combination_dynamics_vary_coeff', 'small_world'),
                                   ('vary_dynamics_with_vary_type_and_coeff', 'small_world'),
                                   ('heat_diffusion_dynamics', 'community'),
                                   ('mutualistic_interaction_dynamics', 'community'),
                                   ('gene_regulatory_dynamics', 'community'),
                                   ('combination_dynamics_vary_coeff', 'community'),
                                   ('vary_dynamics_with_vary_type_and_coeff', 'community')]
        elif dynamics_name == 'all_opinion':
            dynamics_topo_names = [('opinion_dynamics_Baumann2021_2topic', 'small_world')]
        elif dynamics_name == 'all_epidemic':
            dynamics_topo_names = [('SI_Individual_dynamics', 'power_law'),
                                   ('SIS_Individual_dynamics', 'power_law'),
                                   ('SIR_Individual_dynamics', 'power_law'),
                                   ('SEIS_Individual_dynamics', 'power_law'),
                                   ('SEIR_Individual_dynamics', 'power_law'), ]
        elif dynamics_name == 'One_For_All':
            dynamics_topo_names = [('heat_diffusion_dynamics', 'grid'),
                                   ('mutualistic_interaction_dynamics', 'grid'),
                                   ('gene_regulatory_dynamics', 'grid'),
                                   ('combination_dynamics_vary_coeff', 'grid'),
                                   ('vary_dynamics_with_vary_type_and_coeff', 'grid'),
                                   ('heat_diffusion_dynamics', 'power_law'),
                                   ('mutualistic_interaction_dynamics', 'power_law'),
                                   ('gene_regulatory_dynamics', 'power_law'),
                                   ('combination_dynamics_vary_coeff', 'power_law'),
                                   ('vary_dynamics_with_vary_type_and_coeff', 'power_law'),
                                   ('heat_diffusion_dynamics', 'random'),
                                   ('mutualistic_interaction_dynamics', 'random'),
                                   ('gene_regulatory_dynamics', 'random'),
                                   ('combination_dynamics_vary_coeff', 'random'),
                                   ('vary_dynamics_with_vary_type_and_coeff', 'random'),
                                   ('heat_diffusion_dynamics', 'small_world'),
                                   ('mutualistic_interaction_dynamics', 'small_world'),
                                   ('gene_regulatory_dynamics', 'small_world'),
                                   ('combination_dynamics_vary_coeff', 'small_world'),
                                   ('vary_dynamics_with_vary_type_and_coeff', 'small_world'),
                                   ('heat_diffusion_dynamics', 'community'),
                                   ('mutualistic_interaction_dynamics', 'community'),
                                   ('gene_regulatory_dynamics', 'community'),
                                   ('combination_dynamics_vary_coeff', 'community'),
                                   ('vary_dynamics_with_vary_type_and_coeff', 'community'),
                                   ('opinion_dynamics_Baumann2021_2topic', 'small_world'),
                                   ('SI_Individual_dynamics', 'power_law'),
                                   ('SIS_Individual_dynamics', 'power_law'),
                                   ('SIR_Individual_dynamics', 'power_law'),
                                   ('SEIS_Individual_dynamics', 'power_law'),
                                   ('SEIR_Individual_dynamics', 'power_law'),
                                   ]
        else:
            dynamics_names = [dynamics_name]

        if dynamics_topo_names is None:
            dynamics_topo_names = []
            for i in dynamics_names:
                for j in topo_types:
                    dynamics_topo_names.append((i, j))

        if not make_test_set:
            self.data = None
            for dynamics_name_i, topo_type_i in dynamics_topo_names:
                if 'RealEpidemicData' in dynamics_name_i:
                    #num_graphs_samples = 60
                    num_graphs_samples = 1000
                    x_dim = 1
                elif 'sim_epidemic' in dynamics_name_i:
                    num_graphs_samples = 1000
                    x_dim = 1
                elif 'SEIR' in dynamics_name_i or 'Coupled' in dynamics_name_i:
                    num_graphs_samples = 1000
                    x_dim = 4
                elif 'SIR' in dynamics_name_i or 'SEIS' in dynamics_name_i:
                    num_graphs_samples = 1000
                    x_dim = 3
                elif 'SI' in dynamics_name_i or 'SIS' in dynamics_name_i:
                    num_graphs_samples = 1000
                    x_dim = 2
                elif 'opinion' in dynamics_name_i:
                    x_dim = 2
                    num_graphs_samples = 5000
                else:
                    num_graphs_samples = 300
                    x_dim = 1

                fname_i = 'data/DynamicsData/saved_dynamics_%s_topo_%s_dataset_x%s_numgraph%s_timestep%s.pickle' % (
                    dynamics_name_i, topo_type_i, x_dim, num_graphs_samples, time_steps)
                if os.path.isfile(fname_i):
                    print('file exists --[%s, %s]-- loading ...' % (dynamics_name_i, topo_type_i))
                    with open(fname_i, 'rb') as f:
                        data_i = pickle.load(f)
                        # add 'name' key and value
                        for dd in data_i['tasks']:
                            dd['name'] = '%s_%s' % (dynamics_name_i, topo_type_i)

                        if self.data is None:
                            self.data = data_i
                        else:
                            self.data['tasks'].extend(data_i['tasks'])
                    print('--ok')
                else:
                    print('file not found, generating ...')
                    data_i = self.query(seed=0, num_graph=num_graphs_samples, max_time_points=time_steps,
                                        dynamics_name=dynamics_name_i, topo_type=topo_type_i, t_inc=t_inc)
                    f = open(fname_i, 'wb')
                    pickle.dump(data_i, f)
                    f.close()
                    if self.data is None:
                        self.data = data_i
                    else:
                        self.data['tasks'].extend(data_i['tasks'])

        print('done.')

    def get_topo(self, topo_type, N, seed):
        if topo_type == 'grid':
            G = build_topology(N, topo_type, seed)
        elif topo_type == 'random':
            # p = 0.01 + np.random.rand() * (0.5 - 0.01)
            p = 0.1
            G = build_topology(N, topo_type, seed, p=p)
        elif topo_type == 'power_law':
            # m = int(1. + np.random.rand() * (5. - 1.))
            m = 5
            G = build_topology(N, topo_type, seed, m=m)
        elif topo_type == 'small_world':
            # k = int(1. + np.random.rand() * (9. - 1.))
            # p = 0.01 + np.random.rand() * (0.5 - 0.01)
            k = 5
            p = 0.5
            G = build_topology(N, topo_type, seed, k=k, p=p)
        elif topo_type == 'community':
            # p_in = 0.1 + np.random.rand() * (0.5 - 0.1)
            # p_out = 0.001 + np.random.rand() * (0.1 - 0.001)
            p_in = .25
            p_out = .01
            G = build_topology(N, topo_type, seed, p_in=p_in, p_out=p_out)
        elif topo_type == 'full_connected':
            G = build_topology(N, topo_type, seed)
        elif topo_type == 'directed_full_connected':
            G = nx.complete_graph(N, nx.DiGraph())
            # G.add_edges_from([(i, i) for i in range(N)])  # add self_loop
        else:
            print("ERROR topo_type [%s]" % topo_type)
            exit(1)
        return G

    def query(self,
              seed=0,
              num_graph=8,
              max_time_points=50,
              query_all_node=False,
              query_all_t=False,
              t_inc=0.01,
              **kwargs):

        if seed >= 0:
            np.random.seed(seed)
        else:
            seed = None

        if 'topo_type' in kwargs:
            topo_type = kwargs['topo_type']
        else:
            topo_type = self.topo_type

        if 'dynamics_name' in kwargs:
            dynamics_name = kwargs['dynamics_name']
        else:
            dynamics_name = self.dynamics_name

        # if 'N' in kwargs:
        #     N = kwargs['N']
        # else:
        #     N = np.random.randint(10, 51)  ##  [10, 50]
        #     # N = 50
        # G = self.get_topo(N, seed)

        params_dynamics = []

        tasks = {'tasks': []}
        for i in range(num_graph):
            ATask = {'points': [],
                     'name': dynamics_name + '_' + topo_type}
            # if not self.topo_fixed:
            #     if 'N' in kwargs:
            #         N = kwargs['N']
            #     else:
            #         N = np.random.randint(10, 51)  ##  [10, 50]
            #         # N = 50
            #     G = self.get_topo(N, seed)
            if 'N' in kwargs:
                N = kwargs['N']
            else:
                # N = np.random.randint(10, 51)  ##  [10, 50]
                # N = 400
                if topo_type == 'full_connected':
                    #N = np.random.randint(20, 31)  ##  [20, 30]
                    N = np.random.randint(20, 51)  ##  [20, 50]
                elif topo_type == 'directed_full_connected':
                    N = 52
                else:
                    N = np.random.randint(100, 201)  ##  [100, 200]
                # N = np.random.randint(10, 101)  ##  [10, 100]
            G = self.get_topo(topo_type, N, seed)

            if 'SIR_meta_pop' in dynamics_name:
                G = nx.DiGraph(G)

            N = G.number_of_nodes()
            A = np.array(nx.adjacency_matrix(G).todense(), dtype=np.float64)
            sparse_A = coo_matrix(A)
            row, col = sparse_A.row, sparse_A.col

            t_start = 0
            t_inc = t_inc
            t_end = t_start + (max_time_points - 1) * t_inc
            #t_end = 1
            #t_inc = (t_end-t_start) / (max_time_points)
            

            print("building network dynamics [%s] ..." % dynamics_name)
            # Initial Value
            # N = 20
            # x0 = torch.zeros(N, N)
            # x0[int(0.05 * N):int(0.25 * N),
            # int(0.05 * N):int(0.25 * N)] = 25  # x0[1:5, 1:5] = 25  for N = 20 or n= 400 case
            # x0[int(0.45 * N):int(0.75 * N),
            # int(0.45 * N):int(0.75 * N)] = 20  # x0[9:15, 9:15] = 20 for N = 20 or n= 400 case
            # x0[int(0.05 * N):int(0.25 * N),
            # int(0.35 * N):int(0.65 * N)] = 17  # x0[1:5, 7:13] = 17 for N = 20 or n= 400 case
            # X = x0.view(-1, self.x_dim).float()
            # N = 400

            edge_weights = None

            if dynamics_name == "heat_diffusion_dynamics":
                x_dim = 1
                if 'make_test_set' in kwargs:
                    # Initial Value
                    # sqrt_N = int(np.sqrt(N))
                    # X = np.zeros((sqrt_N, sqrt_N))
                    # X[int(0.05 * sqrt_N):int(0.25 * sqrt_N), int(0.05 * sqrt_N):int(0.25 * sqrt_N)] = 25.  # X[1:5, 1:5] = 25  for N = 225 or 400 case
                    # X[int(0.45 * sqrt_N):int(0.75 * sqrt_N), int(0.45 * sqrt_N):int(0.75 * sqrt_N)] = 20.  # X[9:15, 9:15] = 20 for N = 225 or 400 case
                    # X[int(0.05 * sqrt_N):int(0.25 * sqrt_N), int(0.35 * sqrt_N):int(0.65 * sqrt_N)] = 17.  # X[1:5, 7:13] = 17 for N = 225 or 400 case
                    # X = X.reshape(-1, 1)
                    X = 0. + np.random.rand(N, x_dim) * (25.)  # for heat_diffusion_dynamics
                else:
                    X = 0. + np.random.rand(N, x_dim) * (25.)  # for heat_diffusion_dynamics
                # X = -1. + np.random.rand(N, self.x_dim) * (1. + 1.)  # for heat_diffusion_dynamics

                # state and rel2 have the shape of [ # steps, # nodes ] if x_dim = 1
                # K = 0.01 + np.random.rand() * (5. - 0.01)
                # K = 0.01 + np.random.rand() * (0.1 - 0.01)

                # if 'make_test_set' in kwargs:
                #     K = 1.
                # else:
                #     K = 0.5 + np.random.rand() * (2. - 0.5)
                #     while np.abs(K - 1.) < 1e-3:
                #         K = 0.5 + np.random.rand() * (2. - 0.5)

                if 'make_test_set' in kwargs:
                    params_dynamics = np.loadtxt('data/DynamicsData/dynamics_%s_topo_%s_train_set_params.txt' % (
                        dynamics_name, topo_type))

                    params_dynamics = params_dynamics.reshape(-1, 1)

                    K = 0.5 + np.random.rand() * (2. - 0.5)

                    while np.sum(np.sum(np.abs(params_dynamics - np.array([K])), axis=1) < params_dynamics.shape[
                        -1] * 1e-3) > 0.:
                        K = 0.5 + np.random.rand() * (2. - 0.5)
                else:
                    K = 0.5 + np.random.rand() * (2. - 0.5)
                    params_dynamics.append([K])
                # K = 0.1 + np.random.rand() * (1. - 0.1)
                # K = 0.2 + np.random.rand() * (2. - 0.2)
                state, t_range = NetDynamic(dynamics_name).get_observations(X, [row, col], t_start,
                                                                            t_end, t_inc, False,
                                                                            K=K)

            elif dynamics_name == "mutualistic_interaction_dynamics":
                x_dim = 1
                if 'make_test_set' in kwargs:
                    # Initial Value
                    # sqrt_N = int(np.sqrt(N))
                    # X = np.zeros((sqrt_N, sqrt_N))
                    # X[int(0.05 * sqrt_N):int(0.25 * sqrt_N), int(0.05 * sqrt_N):int(0.25 * sqrt_N)] = 25.  # X[1:5, 1:5] = 25  for N = 225 or 400 case
                    # X[int(0.45 * sqrt_N):int(0.75 * sqrt_N), int(0.45 * sqrt_N):int(0.75 * sqrt_N)] = 20.  # X[9:15, 9:15] = 20 for N = 225 or 400 case
                    # X[int(0.05 * sqrt_N):int(0.25 * sqrt_N), int(0.35 * sqrt_N):int(0.65 * sqrt_N)] = 17.  # X[1:5, 7:13] = 17 for N = 225 or 400 case
                    # X = X.reshape(-1, 1)
                    X = np.random.rand(N, x_dim) * 25.  # for mutualistic_interaction_dynamics
                else:
                    X = np.random.rand(N, x_dim) * 25.  # for mutualistic_interaction_dynamics

                #
                # if 'make_test_set' in kwargs:
                #     # b = 0.1
                #     # c = 1.
                #     # d = 5.
                #     # e = 0.9
                #     # h = 0.1
                #     # k = 5.
                #     b = 0.05 + np.random.rand() * (0.15 - 0.05)
                #     c = 0.5 + np.random.rand() * (1.5 - 0.5)
                #     d = 4. + np.random.rand() * (6. - 4.)
                #     e = 0.8 + np.random.rand() * (1. - 0.8)
                #     h = 0.05 + np.random.rand() * (0.15 - 0.05)
                #     k = 4. + np.random.rand() * (6. - 4.)
                # else:
                #     b = 0.05 + np.random.rand() * (0.15 - 0.05)
                #     c = 0.5 + np.random.rand() * (1.5 - 0.5)
                #     d = 4. + np.random.rand() * (6. - 4.)
                #     e = 0.8 + np.random.rand() * (1. - 0.8)
                #     h = 0.05 + np.random.rand() * (0.15 - 0.05)
                #     k = 4. + np.random.rand() * (6. - 4.)
                #
                #     while np.abs(b - 0.1) < 1e-3 and np.abs(c - 1.) < 1e-3 and np.abs(d - 5.) < 1e-3 and np.abs(e - 0.9) < 1e-3 and np.abs(h - 0.1) < 1e-3 and np.abs(k - 5) < 1e-3:
                #         b = 0.05 + np.random.rand() * (0.15 - 0.05)
                #         c = 0.5 + np.random.rand() * (1.5 - 0.5)
                #         d = 4. + np.random.rand() * (6. - 4.)
                #         e = 0.8 + np.random.rand() * (1. - 0.8)
                #         h = 0.05 + np.random.rand() * (0.15 - 0.05)
                #         k = 4. + np.random.rand() * (6. - 4.)

                if 'make_test_set' in kwargs:
                    params_dynamics = np.loadtxt('data/DynamicsData/dynamics_%s_topo_%s_train_set_params.txt' % (
                        dynamics_name, topo_type))

                    params_dynamics = params_dynamics.reshape(-1, 6)

                    b = 0.05 + np.random.rand() * (0.15 - 0.05)
                    c = 0.5 + np.random.rand() * (1.5 - 0.5)
                    d = 4. + np.random.rand() * (6. - 4.)
                    e = 0.8 + np.random.rand() * (1. - 0.8)
                    h = 0.05 + np.random.rand() * (0.15 - 0.05)
                    k = 4. + np.random.rand() * (6. - 4.)

                    while np.sum(np.sum(np.abs(params_dynamics - np.array([b, c, d, e, h, k])), axis=1) <
                                 params_dynamics.shape[-1] * 1e-3) > 0.:
                        b = 0.05 + np.random.rand() * (0.15 - 0.05)
                        c = 0.5 + np.random.rand() * (1.5 - 0.5)
                        d = 4. + np.random.rand() * (6. - 4.)
                        e = 0.8 + np.random.rand() * (1. - 0.8)
                        h = 0.05 + np.random.rand() * (0.15 - 0.05)
                        k = 4. + np.random.rand() * (6. - 4.)
                else:

                    b = 0.05 + np.random.rand() * (0.15 - 0.05)
                    c = 0.5 + np.random.rand() * (1.5 - 0.5)
                    d = 4. + np.random.rand() * (6. - 4.)
                    e = 0.8 + np.random.rand() * (1. - 0.8)
                    h = 0.05 + np.random.rand() * (0.15 - 0.05)
                    k = 4. + np.random.rand() * (6. - 4.)

                    params_dynamics.append([b, c, d, e, h, k])

                state, t_range = NetDynamic(dynamics_name).get_observations(X, [row, col], t_start,
                                                                            t_end, t_inc, False,
                                                                            b=b, c=c, d=d, e=e, h=h, k=k)
            elif dynamics_name == "gene_regulatory_dynamics":
                x_dim = 1
                if 'make_test_set' in kwargs:
                    # Initial Value
                    # sqrt_N = int(np.sqrt(N))
                    # X = np.zeros((sqrt_N, sqrt_N))
                    # X[int(0.05 * sqrt_N):int(0.25 * sqrt_N), int(0.05 * sqrt_N):int(0.25 * sqrt_N)] = 25.  # X[1:5, 1:5] = 25  for N = 225 or 400 case
                    # X[int(0.45 * sqrt_N):int(0.75 * sqrt_N), int(0.45 * sqrt_N):int(0.75 * sqrt_N)] = 20.  # X[9:15, 9:15] = 20 for N = 225 or 400 case
                    # X[int(0.05 * sqrt_N):int(0.25 * sqrt_N), int(0.35 * sqrt_N):int(0.65 * sqrt_N)] = 17.  # X[1:5, 7:13] = 17 for N = 225 or 400 case
                    # X = X.reshape(-1, 1)
                    X = np.random.rand(N, x_dim) * 25.  # for gene_regulatory_dynamics
                else:
                    X = np.random.rand(N, x_dim) * 25.  # for gene_regulatory_dynamics

                # if 'make_test_set' in kwargs:
                #     # b = 1.
                #     # f = 1.
                #     # h = 2.
                #     b = 0.5 + np.random.rand() * (1.5 - 0.5)
                #     f = np.round(1. + np.random.rand() * (2. - 1.))
                #     h = 2.
                # else:
                #     b = 0.5 + np.random.rand() * (1.5 - 0.5)
                #     f = np.round(1. + np.random.rand() * (2. - 1.))
                #     while np.abs(b - 1.) < 1e-3 and np.abs(f - 2.) < 1e-3:
                #         b = 0.5 + np.random.rand() * (1.5 - 0.5)
                #         f = np.round(1. + np.random.rand() * (2. - 1.))
                #
                #     # f = 1.
                #     # h = np.round(1. + np.random.rand() * (2. - 1.))
                #     h = 2.

                if 'make_test_set' in kwargs:
                    params_dynamics = np.loadtxt('data/DynamicsData/dynamics_%s_topo_%s_train_set_params.txt' % (
                        dynamics_name, topo_type))

                    params_dynamics = params_dynamics.reshape(-1, 3)

                    b = 0.5 + np.random.rand() * (1.5 - 0.5)
                    f = np.round(1. + np.random.rand() * (2. - 1.))
                    h = 2.

                    while np.sum(np.sum(np.abs(params_dynamics - np.array([b, f, h])), axis=1) < params_dynamics.shape[
                        -1] * 1e-3) > 0.:
                        b = 0.5 + np.random.rand() * (1.5 - 0.5)
                        f = np.round(1. + np.random.rand() * (2. - 1.))
                        h = 2.

                else:
                    b = 0.5 + np.random.rand() * (1.5 - 0.5)
                    f = np.round(1. + np.random.rand() * (2. - 1.))
                    h = 2.

                    params_dynamics.append([b, f, h])

                state, t_range = NetDynamic(dynamics_name).get_observations(X, [row, col], t_start,
                                                                            t_end, t_inc, False,
                                                                            b=b, f=f, h=h)
            elif dynamics_name == 'combination_dynamics':
                x_dim = 1
                if 'make_test_set' in kwargs:
                    # Initial Value
                    X = np.random.rand(N, x_dim) * 25.  # for combination_dynamics
                else:
                    X = np.random.rand(N, x_dim) * 25.  # for combination_dynamics

                if 'make_test_set' in kwargs:
                    params_dynamics = np.loadtxt('data/DynamicsData/dynamics_%s_topo_%s_train_set_params.txt' % (
                        dynamics_name, topo_type))

                    params_dynamics = params_dynamics.reshape(-1, 10)

                    a_K = 0.5 + np.random.rand() * (2. - 0.5)

                    b_b = 0.05 + np.random.rand() * (0.15 - 0.05)
                    b_c = 0.5 + np.random.rand() * (1.5 - 0.5)
                    b_d = 4. + np.random.rand() * (6. - 4.)
                    b_e = 0.8 + np.random.rand() * (1. - 0.8)
                    b_h = 0.05 + np.random.rand() * (0.15 - 0.05)
                    b_k = 4. + np.random.rand() * (6. - 4.)

                    c_b = 0.5 + np.random.rand() * (1.5 - 0.5)
                    c_f = np.round(1. + np.random.rand() * (2. - 1.))
                    c_h = 2.

                    while np.sum(np.sum(np.abs(
                            params_dynamics - np.array([a_K, b_b, b_c, b_d, b_e, b_h, b_k, c_b, c_f, c_h])), axis=1) <
                                 params_dynamics.shape[
                                     -1] * 1e-3) > 0.:
                        a_K = 0.5 + np.random.rand() * (2. - 0.5)

                        b_b = 0.05 + np.random.rand() * (0.15 - 0.05)
                        b_c = 0.5 + np.random.rand() * (1.5 - 0.5)
                        b_d = 4. + np.random.rand() * (6. - 4.)
                        b_e = 0.8 + np.random.rand() * (1. - 0.8)
                        b_h = 0.05 + np.random.rand() * (0.15 - 0.05)
                        b_k = 4. + np.random.rand() * (6. - 4.)

                        c_b = 0.5 + np.random.rand() * (1.5 - 0.5)
                        c_f = np.round(1. + np.random.rand() * (2. - 1.))
                        c_h = 2.

                else:
                    a_K = 0.5 + np.random.rand() * (2. - 0.5)

                    b_b = 0.05 + np.random.rand() * (0.15 - 0.05)
                    b_c = 0.5 + np.random.rand() * (1.5 - 0.5)
                    b_d = 4. + np.random.rand() * (6. - 4.)
                    b_e = 0.8 + np.random.rand() * (1. - 0.8)
                    b_h = 0.05 + np.random.rand() * (0.15 - 0.05)
                    b_k = 4. + np.random.rand() * (6. - 4.)

                    c_b = 0.5 + np.random.rand() * (1.5 - 0.5)
                    c_f = np.round(1. + np.random.rand() * (2. - 1.))
                    c_h = 2.

                    params_dynamics.append([a_K, b_b, b_c, b_d, b_e, b_h, b_k, c_b, c_f, c_h])

                state, t_range = NetDynamic(dynamics_name).get_observations(X, [row, col], t_start,
                                                                            t_end, t_inc, False,
                                                                            a_K=a_K,
                                                                            b_b=b_b, b_c=b_c, b_d=b_d, b_e=b_e, b_h=b_h,
                                                                            b_k=b_k,
                                                                            c_b=c_b, c_f=c_f, c_h=c_h)
            elif dynamics_name == 'combination_dynamics_vary_coeff':
                x_dim = 1
                if 'make_test_set' in kwargs:
                    # Initial Value
                    X = np.random.rand(N, x_dim) * 25.  # for combination_dynamics
                else:
                    X = np.random.rand(N, x_dim) * 25.  # for combination_dynamics

                if 'make_test_set' in kwargs:
                    params_dynamics = np.loadtxt('data/DynamicsData/dynamics_%s_topo_%s_train_set_params.txt' % (
                        dynamics_name, topo_type))

                    params_dynamics = params_dynamics.reshape(-1, 13)

                    lam_1 = 0. + np.random.rand() * (1. - 0.)
                    lam_2 = 0. + np.random.rand() * (1. - 0.)
                    lam_3 = 0. + np.random.rand() * (1. - 0.)
                    lam_sum = lam_1 + lam_2 + lam_3
                    lam_1 = lam_1 / lam_sum
                    lam_2 = lam_2 / lam_sum
                    lam_3 = lam_3 / lam_sum

                    a_K = 0.5 + np.random.rand() * (2. - 0.5)

                    b_b = 0.05 + np.random.rand() * (0.15 - 0.05)
                    b_c = 0.5 + np.random.rand() * (1.5 - 0.5)
                    b_d = 4. + np.random.rand() * (6. - 4.)
                    b_e = 0.8 + np.random.rand() * (1. - 0.8)
                    b_h = 0.05 + np.random.rand() * (0.15 - 0.05)
                    b_k = 4. + np.random.rand() * (6. - 4.)

                    c_b = 0.5 + np.random.rand() * (1.5 - 0.5)
                    c_f = np.round(1. + np.random.rand() * (2. - 1.))
                    c_h = 2.

                    while np.sum(np.sum(np.abs(params_dynamics - np.array(
                            [lam_1, lam_2, lam_3, a_K, b_b, b_c, b_d, b_e, b_h, b_k, c_b, c_f, c_h])), axis=1) <
                                 params_dynamics.shape[
                                     -1] * 1e-3) > 0.:
                        lam_1 = 0. + np.random.rand() * (1. - 0.)
                        lam_2 = 0. + np.random.rand() * (1. - 0.)
                        lam_3 = 0. + np.random.rand() * (1. - 0.)
                        lam_sum = lam_1 + lam_2 + lam_3
                        lam_1 = lam_1 / lam_sum
                        lam_2 = lam_2 / lam_sum
                        lam_3 = lam_3 / lam_sum

                        a_K = 0.5 + np.random.rand() * (2. - 0.5)

                        b_b = 0.05 + np.random.rand() * (0.15 - 0.05)
                        b_c = 0.5 + np.random.rand() * (1.5 - 0.5)
                        b_d = 4. + np.random.rand() * (6. - 4.)
                        b_e = 0.8 + np.random.rand() * (1. - 0.8)
                        b_h = 0.05 + np.random.rand() * (0.15 - 0.05)
                        b_k = 4. + np.random.rand() * (6. - 4.)

                        c_b = 0.5 + np.random.rand() * (1.5 - 0.5)
                        c_f = np.round(1. + np.random.rand() * (2. - 1.))
                        c_h = 2.

                else:
                    lam_1 = 0. + np.random.rand() * (1. - 0.)
                    lam_2 = 0. + np.random.rand() * (1. - 0.)
                    lam_3 = 0. + np.random.rand() * (1. - 0.)
                    lam_sum = lam_1 + lam_2 + lam_3
                    lam_1 = lam_1 / lam_sum
                    lam_2 = lam_2 / lam_sum
                    lam_3 = lam_3 / lam_sum

                    a_K = 0.5 + np.random.rand() * (2. - 0.5)

                    b_b = 0.05 + np.random.rand() * (0.15 - 0.05)
                    b_c = 0.5 + np.random.rand() * (1.5 - 0.5)
                    b_d = 4. + np.random.rand() * (6. - 4.)
                    b_e = 0.8 + np.random.rand() * (1. - 0.8)
                    b_h = 0.05 + np.random.rand() * (0.15 - 0.05)
                    b_k = 4. + np.random.rand() * (6. - 4.)

                    c_b = 0.5 + np.random.rand() * (1.5 - 0.5)
                    c_f = np.round(1. + np.random.rand() * (2. - 1.))
                    c_h = 2.

                    params_dynamics.append([lam_1, lam_2, lam_3, a_K, b_b, b_c, b_d, b_e, b_h, b_k, c_b, c_f, c_h])

                print('comb weights: ', lam_1, lam_2, lam_3)
                state, t_range = NetDynamic(dynamics_name).get_observations(X, [row, col], t_start,
                                                                            t_end, t_inc, False,
                                                                            lam_1=lam_1, lam_2=lam_2, lam_3=lam_3,
                                                                            a_K=a_K,
                                                                            b_b=b_b, b_c=b_c, b_d=b_d, b_e=b_e, b_h=b_h,
                                                                            b_k=b_k,
                                                                            c_b=c_b, c_f=c_f, c_h=c_h)
            elif dynamics_name == 'vary_dynamics_with_vary_type_and_coeff':
                x_dim = 1
                if 'make_test_set' in kwargs:
                    # Initial Value
                    X = np.random.rand(N, x_dim) * 25.  # for vary_dynamics_with_vary_type_and_coeff
                else:
                    X = np.random.rand(N, x_dim) * 25.  # for vary_dynamics_with_vary_type_and_coeff

                if 'make_test_set' in kwargs:
                    params_dynamics = np.loadtxt('data/DynamicsData/dynamics_%s_topo_%s_train_set_params.txt' % (
                        dynamics_name, topo_type))

                    params_dynamics = params_dynamics.reshape(-1, 22)

                    choose_list = np.random.choice([0, 1, 2], 2, replace=True)
                    choose_1 = choose_list[0]
                    choose_2 = choose_list[1]

                    a_K_1 = 0.5 + np.random.rand() * (2. - 0.5)

                    b_b_1 = 0.05 + np.random.rand() * (0.15 - 0.05)
                    b_c_1 = 0.5 + np.random.rand() * (1.5 - 0.5)
                    b_d_1 = 4. + np.random.rand() * (6. - 4.)
                    b_e_1 = 0.8 + np.random.rand() * (1. - 0.8)
                    b_h_1 = 0.05 + np.random.rand() * (0.15 - 0.05)
                    b_k_1 = 4. + np.random.rand() * (6. - 4.)

                    c_b_1 = 0.5 + np.random.rand() * (1.5 - 0.5)
                    c_f_1 = np.round(1. + np.random.rand() * (2. - 1.))
                    c_h_1 = 2.

                    a_K_2 = 0.5 + np.random.rand() * (2. - 0.5)

                    b_b_2 = 0.05 + np.random.rand() * (0.15 - 0.05)
                    b_c_2 = 0.5 + np.random.rand() * (1.5 - 0.5)
                    b_d_2 = 4. + np.random.rand() * (6. - 4.)
                    b_e_2 = 0.8 + np.random.rand() * (1. - 0.8)
                    b_h_2 = 0.05 + np.random.rand() * (0.15 - 0.05)
                    b_k_2 = 4. + np.random.rand() * (6. - 4.)

                    c_b_2 = 0.5 + np.random.rand() * (1.5 - 0.5)
                    c_f_2 = np.round(1. + np.random.rand() * (2. - 1.))
                    c_h_2 = 2.

                    while np.sum(np.sum(np.abs(params_dynamics - np.array(
                            [choose_1, choose_2, a_K_1, b_b_1, b_c_1, b_d_1, b_e_1, b_h_1, b_k_1, c_b_1, c_f_1, c_h_1,
                             a_K_2, b_b_2, b_c_2, b_d_2, b_e_2, b_h_2, b_k_2, c_b_2, c_f_2, c_h_2])), axis=1) <
                                 params_dynamics.shape[
                                     -1] * 1e-3) > 0.:
                        choose_list = np.random.choice([0, 1, 2], 2, replace=True)
                        choose_1 = choose_list[0]
                        choose_2 = choose_list[1]

                        a_K_1 = 0.5 + np.random.rand() * (2. - 0.5)

                        b_b_1 = 0.05 + np.random.rand() * (0.15 - 0.05)
                        b_c_1 = 0.5 + np.random.rand() * (1.5 - 0.5)
                        b_d_1 = 4. + np.random.rand() * (6. - 4.)
                        b_e_1 = 0.8 + np.random.rand() * (1. - 0.8)
                        b_h_1 = 0.05 + np.random.rand() * (0.15 - 0.05)
                        b_k_1 = 4. + np.random.rand() * (6. - 4.)

                        c_b_1 = 0.5 + np.random.rand() * (1.5 - 0.5)
                        c_f_1 = np.round(1. + np.random.rand() * (2. - 1.))
                        c_h_1 = 2.

                        a_K_2 = 0.5 + np.random.rand() * (2. - 0.5)

                        b_b_2 = 0.05 + np.random.rand() * (0.15 - 0.05)
                        b_c_2 = 0.5 + np.random.rand() * (1.5 - 0.5)
                        b_d_2 = 4. + np.random.rand() * (6. - 4.)
                        b_e_2 = 0.8 + np.random.rand() * (1. - 0.8)
                        b_h_2 = 0.05 + np.random.rand() * (0.15 - 0.05)
                        b_k_2 = 4. + np.random.rand() * (6. - 4.)

                        c_b_2 = 0.5 + np.random.rand() * (1.5 - 0.5)
                        c_f_2 = np.round(1. + np.random.rand() * (2. - 1.))
                        c_h_2 = 2.

                else:
                    choose_list = np.random.choice([0, 1, 2], 2, replace=True)
                    choose_1 = choose_list[0]
                    choose_2 = choose_list[1]

                    a_K_1 = 0.5 + np.random.rand() * (2. - 0.5)

                    b_b_1 = 0.05 + np.random.rand() * (0.15 - 0.05)
                    b_c_1 = 0.5 + np.random.rand() * (1.5 - 0.5)
                    b_d_1 = 4. + np.random.rand() * (6. - 4.)
                    b_e_1 = 0.8 + np.random.rand() * (1. - 0.8)
                    b_h_1 = 0.05 + np.random.rand() * (0.15 - 0.05)
                    b_k_1 = 4. + np.random.rand() * (6. - 4.)

                    c_b_1 = 0.5 + np.random.rand() * (1.5 - 0.5)
                    c_f_1 = np.round(1. + np.random.rand() * (2. - 1.))
                    c_h_1 = 2.

                    a_K_2 = 0.5 + np.random.rand() * (2. - 0.5)

                    b_b_2 = 0.05 + np.random.rand() * (0.15 - 0.05)
                    b_c_2 = 0.5 + np.random.rand() * (1.5 - 0.5)
                    b_d_2 = 4. + np.random.rand() * (6. - 4.)
                    b_e_2 = 0.8 + np.random.rand() * (1. - 0.8)
                    b_h_2 = 0.05 + np.random.rand() * (0.15 - 0.05)
                    b_k_2 = 4. + np.random.rand() * (6. - 4.)

                    c_b_2 = 0.5 + np.random.rand() * (1.5 - 0.5)
                    c_f_2 = np.round(1. + np.random.rand() * (2. - 1.))
                    c_h_2 = 2.

                    params_dynamics.append(
                        [choose_1, choose_2, a_K_1, b_b_1, b_c_1, b_d_1, b_e_1, b_h_1, b_k_1, c_b_1, c_f_1, c_h_1,
                         a_K_2, b_b_2, b_c_2, b_d_2, b_e_2, b_h_2, b_k_2, c_b_2, c_f_2, c_h_2, ])

                state, t_range = NetDynamic(dynamics_name).get_observations(X, [row, col], t_start,
                                                                            t_end, t_inc, False,
                                                                            choose_1=choose_1, choose_2=choose_2,
                                                                            a_K_1=a_K_1,
                                                                            b_b_1=b_b_1, b_c_1=b_c_1, b_d_1=b_d_1,
                                                                            b_e_1=b_e_1, b_h_1=b_h_1,
                                                                            b_k_1=b_k_1,
                                                                            c_b_1=c_b_1, c_f_1=c_f_1, c_h_1=c_h_1,
                                                                            a_K_2=a_K_2,
                                                                            b_b_2=b_b_2, b_c_2=b_c_2, b_d_2=b_d_2,
                                                                            b_e_2=b_e_2, b_h_2=b_h_2,
                                                                            b_k_2=b_k_2,
                                                                            c_b_2=c_b_2, c_f_2=c_f_2, c_h_2=c_h_2
                                                                            )

            elif dynamics_name == "opinion_dynamics":
                x_dim = 1
                if 'make_test_set' in kwargs:
                    # Initial Value
                    # sqrt_N = int(np.sqrt(N))
                    # X = np.zeros((sqrt_N, sqrt_N))
                    # X[int(0.05 * sqrt_N):int(0.25 * sqrt_N), int(0.05 * sqrt_N):int(0.25 * sqrt_N)] = 25.  # X[1:5, 1:5] = 25  for N = 225 or 400 case
                    # X[int(0.45 * sqrt_N):int(0.75 * sqrt_N), int(0.45 * sqrt_N):int(0.75 * sqrt_N)] = 20.  # X[9:15, 9:15] = 20 for N = 225 or 400 case
                    # X[int(0.05 * sqrt_N):int(0.25 * sqrt_N), int(0.35 * sqrt_N):int(0.65 * sqrt_N)] = 17.  # X[1:5, 7:13] = 17 for N = 225 or 400 case
                    # X = X.reshape(-1, 1)
                    X = np.random.rand(N, x_dim) * 10.  # for opinion_dynamics
                else:
                    X = np.random.rand(N, x_dim) * 10.  # for opinion_dynamics

                if 'make_test_set' in kwargs:
                    params_dynamics = np.loadtxt('data/DynamicsData/dynamics_%s_topo_%s_train_set_params.txt' % (
                        dynamics_name, topo_type))

                    params_dynamics = params_dynamics.reshape(-1, 3)

                    # a = 30 + np.random.rand() * (60 - 30)
                    # b = 3 + np.random.rand() * (6 - 3)
                    # c = 0.5 + np.random.rand() * (2. - 0.5)

                    a = 20 + np.random.rand() * (40 - 20)
                    b = 2 + np.random.rand() * (4 - 2)
                    # c = 1. / np.sqrt(2)
                    c = 0.5 + np.random.rand() * (1. - 0.5)

                    while np.sum(np.sum(np.abs(params_dynamics - np.array([a, b, c])), axis=1) < params_dynamics.shape[
                        -1] * 1e-3) > 0.:
                        # a = 30 + np.random.rand() * (60 - 30)
                        # b = 3 + np.random.rand() * (6 - 3)
                        # c = 0.5 + np.random.rand() * (2. - 0.5)

                        a = 20 + np.random.rand() * (40 - 20)
                        b = 2 + np.random.rand() * (4 - 2)
                        # c = 1. / np.sqrt(2)
                        c = 0.5 + np.random.rand() * (1. - 0.5)

                else:
                    # a = 30 + np.random.rand() * (60 - 30)
                    # b = 3 + np.random.rand() * (6 - 3)
                    # c = 0.5 + np.random.rand() * (2. - 0.5)

                    a = 20 + np.random.rand() * (40 - 20)
                    b = 2 + np.random.rand() * (4 - 2)
                    # c = 1. / np.sqrt(2)
                    c = 0.5 + np.random.rand() * (1. - 0.5)

                    params_dynamics.append([a, b, c])

                state, t_range = NetDynamic(dynamics_name).get_observations(X, [row, col], t_start,
                                                                            t_end, t_inc, False,
                                                                            a=a, b=b, c=c)
            elif dynamics_name == "opinion_dynamics_Baumann2021":
                x_dim = 1
                if 'make_test_set' in kwargs:
                    # Initial Value
                    # sqrt_N = int(np.sqrt(N))
                    # X = np.zeros((sqrt_N, sqrt_N))
                    # X[int(0.05 * sqrt_N):int(0.25 * sqrt_N), int(0.05 * sqrt_N):int(0.25 * sqrt_N)] = 25.  # X[1:5, 1:5] = 25  for N = 225 or 400 case
                    # X[int(0.45 * sqrt_N):int(0.75 * sqrt_N), int(0.45 * sqrt_N):int(0.75 * sqrt_N)] = 20.  # X[9:15, 9:15] = 20 for N = 225 or 400 case
                    # X[int(0.05 * sqrt_N):int(0.25 * sqrt_N), int(0.35 * sqrt_N):int(0.65 * sqrt_N)] = 17.  # X[1:5, 7:13] = 17 for N = 225 or 400 case
                    # X = X.reshape(-1, 1)
                    X = -12.5 + np.random.rand(N, x_dim) * (12.5 + 12.5)  # for opinion_dynamics
                else:
                    X = -12.5 + np.random.rand(N, x_dim) * (12.5 + 12.5)  # for opinion_dynamics

                if 'make_test_set' in kwargs:
                    params_dynamics = np.loadtxt('data/DynamicsData/dynamics_%s_topo_%s_train_set_params.txt' % (
                        dynamics_name, topo_type))

                    params_dynamics = params_dynamics.reshape(-1, 3)

                    a = 0.01 + np.random.rand() * (0.5 - 0.01)
                    k = 0.5 + np.random.rand() * (1.5 - 0.5)
                    c = 1. + np.random.rand() * (3. - 1.)

                    while np.sum(np.sum(np.abs(params_dynamics - np.array([a, k, c])), axis=1) < params_dynamics.shape[
                        -1] * 1e-3) > 0.:
                        a = 0.01 + np.random.rand() * (0.5 - 0.01)
                        k = 0.5 + np.random.rand() * (1.5 - 0.5)
                        c = 1. + np.random.rand() * (3. - 1.)

                else:
                    a = 0.01 + np.random.rand() * (0.5 - 0.01)
                    k = 0.5 + np.random.rand() * (1.5 - 0.5)
                    c = 1. + np.random.rand() * (3. - 1.)

                    params_dynamics.append([a, k, c])

                state, t_range = NetDynamic(dynamics_name).get_observations(X, [row, col], t_start,
                                                                            t_end, t_inc, False,
                                                                            a=a, k=k, c=c)
            elif dynamics_name == "opinion_dynamics_Baumann2021_2topic":
                x_dim = 2
                if 'make_test_set' in kwargs:
                    # Initial Value
                    # sqrt_N = int(np.sqrt(N))
                    # X = np.zeros((sqrt_N, sqrt_N))
                    # X[int(0.05 * sqrt_N):int(0.25 * sqrt_N), int(0.05 * sqrt_N):int(0.25 * sqrt_N)] = 25.  # X[1:5, 1:5] = 25  for N = 225 or 400 case
                    # X[int(0.45 * sqrt_N):int(0.75 * sqrt_N), int(0.45 * sqrt_N):int(0.75 * sqrt_N)] = 20.  # X[9:15, 9:15] = 20 for N = 225 or 400 case
                    # X[int(0.05 * sqrt_N):int(0.25 * sqrt_N), int(0.35 * sqrt_N):int(0.65 * sqrt_N)] = 17.  # X[1:5, 7:13] = 17 for N = 225 or 400 case
                    # X = X.reshape(-1, 1)
                    # X = -5. + np.random.rand(N, self.x_dim) * (5 + 5)  # [0, 10]
                    X = np.random.randn(N, x_dim)
                else:
                    X = np.random.randn(N, x_dim)

                if 'make_test_set' in kwargs:
                    if 'fixed_param_case1' in kwargs:
                        # a1 = 0.01
                        # a2 = 0.01
                        # k = 3.
                        # c = 1.
                        # d = np.pi / 2.
                        a1 = 0.015
                        a2 = 0.015
                        k = 3.
                        c = 1.
                        d = np.pi / 2.0
                    elif 'fixed_param_case2' in kwargs:
                        # a1 = 0.01
                        # a2 = 0.5
                        # k = 3.
                        # c = 1.
                        # d = np.pi / 2.
                        a1 = 0.015
                        a2 = 0.495
                        k = 3.
                        c = 1.
                        d = np.pi / 2.0
                    elif 'fixed_param_case3' in kwargs:
                        # a1 = 0.5
                        # a2 = 0.01
                        # k = 3.
                        # c = 1.
                        # d = np.pi / 2.
                        a1 = 0.495
                        a2 = 0.015
                        k = 3.
                        c = 1.
                        d = np.pi / 2.0
                    elif 'fixed_param_case4' in kwargs:
                        # a1 = 0.5
                        # a2 = 0.5
                        # k = 3.
                        # c = 1.
                        # d = np.pi / 2.
                        a1 = 0.495
                        a2 = 0.495
                        k = 3.
                        c = 1.
                        d = np.pi / 2.0
                    elif 'fixed_param_case5' in kwargs:
                        # a1 = 0.5
                        # a2 = 0.5
                        # k = 3.
                        # c = 1.
                        # d = np.pi / 3.
                        a1 = 0.495
                        a2 = 0.495
                        k = 3.
                        c = 1.
                        d = np.pi / 3.
                    else:
                        params_dynamics = np.loadtxt('data/DynamicsData/dynamics_%s_topo_%s_train_set_params.txt' % (
                            dynamics_name, topo_type))

                        params_dynamics = params_dynamics.reshape(-1, 5)

                        a1 = 0.01 + np.random.rand() * (0.5 - 0.01)
                        a2 = 0.01 + np.random.rand() * (0.5 - 0.01)
                        k = 3.
                        c = 1.
                        # d = np.pi / 3. + np.random.rand() * (np.pi / 2. - np.pi / 3.)
                        d = np.pi / 3. + np.round(np.random.rand()) * (np.pi / 2. - np.pi / 3.)

                        while np.sum(np.sum(np.abs(params_dynamics - np.array([a1, a2, k, c, d])), axis=1) <
                                     params_dynamics.shape[
                                         -1] * 1e-3) > 0.:
                            a1 = 0.01 + np.random.rand() * (0.5 - 0.01)
                            a2 = 0.01 + np.random.rand() * (0.5 - 0.01)
                            k = 3.
                            c = 1.
                            # d = np.pi / 3. + np.random.rand() * (np.pi / 2. - np.pi / 3.)
                            d = np.pi / 3. + np.round(np.random.rand()) * (np.pi / 2. - np.pi / 3.)

                else:

                    a1_list = np.linspace(0.01, 0.5, 50).tolist()
                    a2_list = np.linspace(0.01, 0.5, 50).tolist()
                    k = 3.
                    c = 1.
                    # d_list = np.linspace(np.pi / 3., np.pi / 2., 20).tolist()
                    d_list = [np.pi / 3., np.pi / 2.]

                    count_ = 0
                    for item in itertools.product(*[a1_list, a2_list, d_list]):
                        count_ += 1
                        if count_ == i + 1:
                            # print(item)
                            a1, a2, d = item
                            break

                    # a1 = 0.01 + np.random.rand() * (0.5 - 0.01)
                    # a2 = 0.01 + np.random.rand() * (0.5 - 0.01)
                    # k = 2. + np.random.rand() * (3. - 2.)
                    # c = 1.
                    # d = np.pi / 3. + np.random.rand() * (np.pi / 2. - np.pi / 3.)

                    params_dynamics.append([a1, a2, k, c, d])

                state, t_range = NetDynamic(dynamics_name).get_observations(X, [row, col], t_start,
                                                                            t_end, t_inc, False,
                                                                            a1=a1, a2=a2, k=k, c=c, d=d)

            elif dynamics_name == "SI_Individual_dynamics":
                x_dim = 2
                if 'make_test_set' in kwargs:
                    X_I = 1e-6 + np.random.rand(N, 1) * (1e-3 - 1e-6)  # [0.5, 1]
                    X_S = 1. - X_I  # [0., 0.2]
                    # X_R = np.zeros((N, 1))
                    # X_E = np.zeros((N, 1))

                    X = np.concatenate([X_S, X_I], axis=-1)
                else:
                    X_I = 1e-6 + np.random.rand(N, 1) * (1e-3 - 1e-6)  # [0.5, 1]
                    X_S = 1. - X_I  # [0., 0.2]
                    # X_R = np.zeros((N, 1))
                    # X_E = np.zeros((N, 1))

                    X = np.concatenate([X_S, X_I], axis=-1)

                if 'make_test_set' in kwargs:

                    params_dynamics = np.loadtxt('data/DynamicsData/dynamics_%s_topo_%s_train_set_params.txt' % (
                        dynamics_name, topo_type))

                    params_dynamics = params_dynamics.reshape(-1, 1)

                    b = 0.02 + np.random.rand() * (0.2 - 0.02)

                    while np.sum(np.sum(np.abs(params_dynamics - np.array([b])), axis=1) < params_dynamics.shape[
                        -1] * 1e-5) > 0.:
                        b = 0.02 + np.random.rand() * (0.2 - 0.02)

                else:

                    b_list = np.linspace(0.02, 0.2, 1000).tolist()

                    count_ = 0
                    for item in range(len(b_list)):
                        count_ += 1
                        if count_ == i + 1:
                            # print(item)
                            b = b_list[item]
                            break

                    # a1 = 0.01 + np.random.rand() * (0.5 - 0.01)
                    # a2 = 0.01 + np.random.rand() * (0.5 - 0.01)
                    # k = 2. + np.random.rand() * (3. - 2.)
                    # c = 1.
                    # d = np.pi / 3. + np.random.rand() * (np.pi / 2. - np.pi / 3.)

                    params_dynamics.append([b])

                state, t_range = NetDynamic(dynamics_name).get_observations(X, [row, col], t_start,
                                                                            t_end, t_inc, False,
                                                                            b=b)

            elif dynamics_name == "SIS_Individual_dynamics":
                x_dim = 2
                if 'make_test_set' in kwargs:
                    X_I = 1e-6 + np.random.rand(N, 1) * (1e-3 - 1e-6)  # [0.5, 1]
                    X_S = 1. - X_I  # [0., 0.2]
                    # X_R = np.zeros((N, 1))
                    # X_E = np.zeros((N, 1))

                    X = np.concatenate([X_S, X_I], axis=-1)
                else:
                    X_I = 1e-6 + np.random.rand(N, 1) * (1e-3 - 1e-6)  # [0.5, 1]
                    X_S = 1. - X_I  # [0., 0.2]
                    # X_R = np.zeros((N, 1))
                    # X_E = np.zeros((N, 1))

                    X = np.concatenate([X_S, X_I], axis=-1)

                if 'make_test_set' in kwargs:

                    params_dynamics = np.loadtxt('data/DynamicsData/dynamics_%s_topo_%s_train_set_params.txt' % (
                        dynamics_name, topo_type))

                    params_dynamics = params_dynamics.reshape(-1, 2)

                    b = 0.02 + np.random.rand() * (0.2 - 0.02)
                    r = 0.1 + np.random.rand() * (0.4 - 0.1)

                    while np.sum(np.sum(np.abs(params_dynamics - np.array([b, r])), axis=1) < params_dynamics.shape[
                        -1] * 1e-3) > 0.:
                        b = 0.02 + np.random.rand() * (0.2 - 0.02)
                        r = 0.1 + np.random.rand() * (0.4 - 0.1)

                else:

                    b_list = np.linspace(0.02, 0.2, 40).tolist()
                    r_list = np.linspace(0.1, 0.4, 25).tolist()

                    count_ = 0
                    for item in itertools.product(*[b_list, r_list]):
                        count_ += 1
                        if count_ == i + 1:
                            # print(item)
                            b, r = item
                            break

                    # a1 = 0.01 + np.random.rand() * (0.5 - 0.01)
                    # a2 = 0.01 + np.random.rand() * (0.5 - 0.01)
                    # k = 2. + np.random.rand() * (3. - 2.)
                    # c = 1.
                    # d = np.pi / 3. + np.random.rand() * (np.pi / 2. - np.pi / 3.)

                    params_dynamics.append([b, r])

                state, t_range = NetDynamic(dynamics_name).get_observations(X, [row, col], t_start,
                                                                            t_end, t_inc, False,
                                                                            b=b, r=r)
            elif dynamics_name == "SIR_Individual_dynamics":
                x_dim = 3
                if 'make_test_set' in kwargs:
                    X_I = 1e-6 + np.random.rand(N, 1) * (1e-3 - 1e-6)  # [0.5, 1]
                    X_S = 1. - X_I  # [0., 0.2]
                    X_R = np.zeros((N, 1))
                    # X_E = np.zeros((N, 1))

                    X = np.concatenate([X_S, X_I, X_R], axis=-1)
                else:
                    X_I = 1e-6 + np.random.rand(N, 1) * (1e-3 - 1e-6)  # [0.5, 1]
                    X_S = 1. - X_I  # [0., 0.2]
                    X_R = np.zeros((N, 1))
                    # X_E = np.zeros((N, 1))

                    X = np.concatenate([X_S, X_I, X_R], axis=-1)

                if 'make_test_set' in kwargs:

                    if 'fixed_param_case1' in kwargs:
                        # for 2009 Hong Kong H1N1 Influenza Pandemic
                        b = 1.5 / 3.2
                        r = 1 / 3.2
                    elif 'fixed_param_case2' in kwargs:
                        # for 2010 Taiwan Seasonal Influenza
                        b = 2.0 / 3.
                        r = 1. / 3.
                    elif 'fixed_param_case3' in kwargs:
                        # for 2010 Taiwan Varicella
                        b = 7.75 / 5.
                        r = 1 / 5.
                    elif 'fixed_param_case4' in kwargs:
                        # for 2020 Spain Covid-19
                        # A. Aleta and Y. Moreno, "Evaluation of the potential incidence of Covid-19 and effectiveness
                        # of containment measures in Spain: a data-driven approach," BMC Med. 18, 157 (2020)
                        b = 2.5 / 7.5
                        r = 1 / 7.5
                    else:

                        params_dynamics = np.loadtxt('data/DynamicsData/dynamics_%s_topo_%s_train_set_params.txt' % (
                            dynamics_name, topo_type))

                        params_dynamics = params_dynamics.reshape(-1, 2)

                        b = 0.02 + np.random.rand() * (2 - 0.02)
                        r = 0.1 + np.random.rand() * (0.4 - 0.1)

                        while np.sum(np.sum(np.abs(params_dynamics - np.array([b, r])), axis=1) < params_dynamics.shape[
                            -1] * 1e-3) > 0.:
                            b = 0.02 + np.random.rand() * (2 - 0.02)
                            r = 0.1 + np.random.rand() * (0.4 - 0.1)

                else:

                    b_list = np.linspace(0.02, 2., 40).tolist()
                    r_list = np.linspace(0.1, 0.4, 25).tolist()

                    count_ = 0
                    for item in itertools.product(*[b_list, r_list]):
                        count_ += 1
                        if count_ == i + 1:
                            # print(item)
                            b, r = item
                            break

                    # a1 = 0.01 + np.random.rand() * (0.5 - 0.01)
                    # a2 = 0.01 + np.random.rand() * (0.5 - 0.01)
                    # k = 2. + np.random.rand() * (3. - 2.)
                    # c = 1.
                    # d = np.pi / 3. + np.random.rand() * (np.pi / 2. - np.pi / 3.)

                    params_dynamics.append([b, r])

                state, t_range = NetDynamic(dynamics_name).get_observations(X, [row, col], t_start,
                                                                            t_end, t_inc, False,
                                                                            b=b, r=r)
            elif dynamics_name == "SEIS_Individual_dynamics":
                x_dim = 3
                if 'make_test_set' in kwargs:
                    X_I = 1e-6 + np.random.rand(N, 1) * (1e-3 - 1e-6)  # [0.5, 1]
                    X_S = 1. - X_I  # [0., 0.2]
                    # X_R = np.zeros((N, 1))
                    X_E = np.zeros((N, 1))

                    X = np.concatenate([X_S, X_I, X_E], axis=-1)
                else:
                    X_I = 1e-6 + np.random.rand(N, 1) * (1e-3 - 1e-6)  # [0.5, 1]
                    X_S = 1. - X_I  # [0., 0.2]
                    # X_R = np.zeros((N, 1))
                    X_E = np.zeros((N, 1))

                    X = np.concatenate([X_S, X_I, X_E], axis=-1)

                if 'make_test_set' in kwargs:

                    params_dynamics = np.loadtxt('data/DynamicsData/dynamics_%s_topo_%s_train_set_params.txt' % (
                        dynamics_name, topo_type))

                    params_dynamics = params_dynamics.reshape(-1, 3)

                    b = 0.3 + np.random.rand() * (2. - 0.3)
                    r = 0.1 + np.random.rand() * (0.4 - 0.1)
                    c = 0.05 + np.random.rand() * (0.1 - 0.05)

                    while np.sum(np.sum(np.abs(params_dynamics - np.array([b, r, c])), axis=1) < params_dynamics.shape[
                        -1] * 1e-3) > 0.:
                        b = 0.3 + np.random.rand() * (2. - 0.3)
                        r = 0.1 + np.random.rand() * (0.4 - 0.1)
                        c = 0.05 + np.random.rand() * (0.1 - 0.05)

                else:

                    b_list = np.linspace(0.3, 2., 10).tolist()
                    r_list = np.linspace(0.1, 0.4, 10).tolist()
                    c_list = np.linspace(0.05, 0.1, 10).tolist()

                    count_ = 0
                    for item in itertools.product(*[b_list, r_list, c_list]):
                        count_ += 1
                        if count_ == i + 1:
                            # print(item)
                            b, r, c = item
                            break

                    # a1 = 0.01 + np.random.rand() * (0.5 - 0.01)
                    # a2 = 0.01 + np.random.rand() * (0.5 - 0.01)
                    # k = 2. + np.random.rand() * (3. - 2.)
                    # c = 1.
                    # d = np.pi / 3. + np.random.rand() * (np.pi / 2. - np.pi / 3.)

                    params_dynamics.append([b, r, c])

                state, t_range = NetDynamic(dynamics_name).get_observations(X, [row, col], t_start,
                                                                            t_end, t_inc, False,
                                                                            b=b, r=r, c=c)
            elif dynamics_name == "SEIR_Individual_dynamics":
                x_dim = 4
                if 'make_test_set' in kwargs:
                    X_I = 1e-6 + np.random.rand(N, 1) * (1e-3 - 1e-6)  # [0.5, 1]
                    X_S = 1. - X_I  # [0., 0.2]
                    X_R = np.zeros((N, 1))
                    X_E = np.zeros((N, 1))

                    X = np.concatenate([X_S, X_I, X_R, X_E], axis=-1)
                else:
                    X_I = 1e-6 + np.random.rand(N, 1) * (1e-3 - 1e-6)  # [0.5, 1]
                    X_S = 1. - X_I  # [0., 0.2]
                    X_R = np.zeros((N, 1))
                    X_E = np.zeros((N, 1))

                    X = np.concatenate([X_S, X_I, X_R, X_E], axis=-1)

                if 'make_test_set' in kwargs:
                    if 'fixed_param_case1' in kwargs:
                        # for Mumps
                        b = 2.16 / 7.
                        r = 1 / 7.
                        c = 1 / 14.
                    else:
                        params_dynamics = np.loadtxt('data/DynamicsData/dynamics_%s_topo_%s_train_set_params.txt' % (
                            dynamics_name, topo_type))

                        params_dynamics = params_dynamics.reshape(-1, 3)

                        b = 0.3 + np.random.rand() * (2. - 0.3)
                        r = 0.1 + np.random.rand() * (0.4 - 0.1)
                        c = 0.05 + np.random.rand() * (0.1 - 0.05)

                        while np.sum(
                                np.sum(np.abs(params_dynamics - np.array([b, r, c])), axis=1) < params_dynamics.shape[
                                    -1] * 1e-3) > 0.:
                            b = 0.3 + np.random.rand() * (2. - 0.3)
                            r = 0.1 + np.random.rand() * (0.4 - 0.1)
                            c = 0.05 + np.random.rand() * (0.1 - 0.05)

                else:

                    b_list = np.linspace(0.3, 2., 10).tolist()
                    r_list = np.linspace(0.1, 0.4, 10).tolist()
                    c_list = np.linspace(0.05, 0.1, 10).tolist()

                    count_ = 0
                    for item in itertools.product(*[b_list, r_list, c_list]):
                        count_ += 1
                        if count_ == i + 1:
                            # print(item)
                            b, r, c = item
                            break

                    # a1 = 0.01 + np.random.rand() * (0.5 - 0.01)
                    # a2 = 0.01 + np.random.rand() * (0.5 - 0.01)
                    # k = 2. + np.random.rand() * (3. - 2.)
                    # c = 1.
                    # d = np.pi / 3. + np.random.rand() * (np.pi / 2. - np.pi / 3.)

                    params_dynamics.append([b, r, c])

                state, t_range = NetDynamic(dynamics_name).get_observations(X, [row, col], t_start,
                                                                            t_end, t_inc, False,
                                                                            b=b, r=r, c=c)

            elif dynamics_name == "SIR_meta_pop_dynamics":
                x_dim = 3
                """
                cases_raw = pickle.load(open(
                    r"data/DynamicsData/Spain_Covid19/case-timeseries.data",
                    'rb'))
                population_raw = pickle.load(open(
                    r"data/DynamicsData/Spain_Covid19/population.data",
                    'rb'))
                population_raw = population_raw * 1e-2  # for scale

                # for 2020 Spain Covid-19
                # A. Aleta and Y. Moreno, "Evaluation of the potential incidence of Covid-19 and effectiveness
                # of containment measures in Spain: a data-driven approach," BMC Med. 18, 157 (2020)

                b = 2.5 / 7.5
                r = 1 / 7.5

                S_raw = np.zeros_like(cases_raw)
                I_raw = np.zeros_like(cases_raw)
                R_raw = np.zeros_like(cases_raw)

                S_raw[0] = population_raw - cases_raw[0]
                I_raw[0] = cases_raw[0]
                for t in range(cases_raw.shape[0]):
                    if t > 0:
                        S_raw[t] = S_raw[t - 1] - cases_raw[t]
                        I_raw[t] = I_raw[t - 1] + cases_raw[t] - r * I_raw[t - 1]
                        R_raw[t] = R_raw[t - 1] + r * I_raw[t - 1]

                S_normed = S_raw / population_raw.reshape(1, -1)
                I_normed = I_raw / population_raw.reshape(1, -1)
                R_normed = R_raw / population_raw.reshape(1, -1)
                """

                if 'make_test_set' in kwargs:
                    X_I = np.zeros(N)
                    init_N = np.random.choice(a=list(range(N)))
                    init_idxs = np.random.choice(a=list(range(N)), size=init_N, replace=False)
                    # init_idxs = list(range(N))
                    for init_idx in init_idxs:
                        # X_I[init_idx] = np.random.rand() * 2e-2
                        X_I[init_idx] = 1e-6 + np.random.rand() * (1e-3 - 1e-6)
                        # X_I[init_idx] = I_normed[50, init_idx]

                        # X_I[init_idx] = cases_raw[50][init_idx] / node_constants[init_idx]
                    X_I = X_I.reshape(-1, 1)

                    X_S = 1. - X_I  # [0., 0.2]
                    X_R = np.zeros((N, 1))
                    # X_E = np.zeros((N, 1))

                    X = np.concatenate([X_S, X_I, X_R], axis=-1)
                else:
                    X_I = np.zeros(N)
                    init_N = np.random.choice(a=list(range(N)))
                    init_idxs = np.random.choice(a=list(range(N)), size=init_N, replace=False)
                    # init_idxs = list(range(N))
                    for init_idx in init_idxs:
                        # X_I[init_idx] = np.random.rand() * 2e-2
                        X_I[init_idx] = 1e-6 + np.random.rand() * (1e-3 - 1e-6)
                        # X_I[init_idx] = cases_raw[50][init_idx] / population_raw[init_idx]
                        # X_I[init_idx] = I_normed[50, init_idx]

                        # X_I[init_idx] = cases_raw[50][init_idx] / node_constants[init_idx]
                    X_I = X_I.reshape(-1, 1)

                    X_S = 1. - X_I  # [0., 0.2]
                    X_R = np.zeros((N, 1))
                    # X_E = np.zeros((N, 1))

                    X = np.concatenate([X_S, X_I, X_R], axis=-1)

                if 'make_test_set' in kwargs:
                    if 'fixed_param_case1' in kwargs:
                        # for 2009 Hong Kong H1N1 Influenza Pandemic
                        b = 1.5 / 3.2
                        r = 1 / 3.2
                    elif 'fixed_param_case2' in kwargs:
                        # for 2010 Taiwan Seasonal Influenza
                        b = 2.0 / 3.
                        r = 1. / 3.
                    elif 'fixed_param_case3' in kwargs:
                        # for 2010 Taiwan Varicella
                        b = 7.75 / 5.
                        r = 1 / 5.
                    elif 'fixed_param_case4' in kwargs:
                        # for 2020 Spain Covid-19
                        # A. Aleta and Y. Moreno, "Evaluation of the potential incidence of Covid-19 and effectiveness
                        # of containment measures in Spain: a data-driven approach," BMC Med. 18, 157 (2020)

                        # b = 2.5 /7.5
                        # r = 1 / 7.5
                        b = 0.0085
                        r = 1 / 7.5
                    else:

                        params_dynamics = np.loadtxt('data/DynamicsData/dynamics_%s_topo_%s_train_set_params.txt' % (
                            dynamics_name, topo_type))

                        params_dynamics = params_dynamics.reshape(-1, 2)

                        b = 0.005 + np.random.rand() * (0.05 - 0.005)
                        r = 0.1 + np.random.rand() * (0.2 - 0.1)

                        while np.sum(np.sum(np.abs(params_dynamics - np.array([b, r])), axis=1) < params_dynamics.shape[
                            -1] * 1e-3) > 0.:
                            b = 0.005 + np.random.rand() * (0.05 - 0.005)
                            r = 0.1 + np.random.rand() * (0.2 - 0.1)

                else:

                    b_list = np.linspace(0.005, 0.05, 40).tolist()
                    r_list = np.linspace(0.1, 0.2, 25).tolist()

                    count_ = 0
                    for item in itertools.product(*[b_list, r_list]):
                        count_ += 1
                        if count_ == i + 1:
                            # print(item)
                            b, r = item
                            break

                    # a1 = 0.01 + np.random.rand() * (0.5 - 0.01)
                    # a2 = 0.01 + np.random.rand() * (0.5 - 0.01)
                    # k = 2. + np.random.rand() * (3. - 2.)
                    # c = 1.
                    # d = np.pi / 3. + np.random.rand() * (np.pi / 2. - np.pi / 3.)

                    params_dynamics.append([b, r])

                # load weights on edges
                network_raw = pickle.load(open(
                    r"data/DynamicsData/Spain_Covid19/province_mobility.data",
                    'rb'))
                network_raw = network_raw['all']

                # network_raw = np.zeros((N,N))
                # for iii in row:
                #    for jjj in col:
                #        network_raw[iii][jjj] = np.random.rand()

                network_raw = network_raw - np.diag(np.diag(network_raw))
                network_normed = network_raw / np.sum(network_raw, axis=1, keepdims=True)

                row = []
                col = []
                for ii in range(N):
                    for jj in range(N):
                        if network_raw[jj, ii] > 0:
                            row.append(ii)
                            col.append(jj)
                row = np.array(row)
                col = np.array(col)

                edge_weights = []
                for edge_idx in range(len(row)):
                    if row[edge_idx] == col[edge_idx]:
                        edge_weights.append(0)
                    else:
                        edge_weights.append(
                            (np.sum(col == col[edge_idx])) * network_normed[col[edge_idx], row[edge_idx]])
                        # edge_weights.append(network_normed[row[edge_idx], col[edge_idx]])
                edge_weights = np.array(edge_weights).reshape(-1, 1)

                state, t_range = NetDynamic(dynamics_name).get_observations(X, [row, col], t_start,
                                                                            t_end, t_inc, False,
                                                                            b=b, r=r, edge_weights=edge_weights)
            elif dynamics_name == 'real_data_spain_covid19_cases':
                x_dim = 3
                cases_raw = pickle.load(open(
                    r"data/DynamicsData/Spain_Covid19/case-timeseries.data",
                    'rb'))

                # load weights on edges
                network_raw = pickle.load(open(
                    r"data/DynamicsData/Spain_Covid19/province_mobility.data",
                    'rb'))
                network_raw = network_raw['all']
                network_raw = network_raw - np.diag(np.diag(network_raw))

                network_normed = network_raw / np.sum(network_raw, axis=1, keepdims=True)

                edge_weights = []
                for edge_idx in range(len(row)):
                    if row[edge_idx] == col[edge_idx]:
                        edge_weights.append(0)
                    else:
                        edge_weights.append((np.sum(col == col[edge_idx])) * network_normed[col[edge_idx], row[
                            edge_idx]])  # network_raw[i,j] means the average number of people that are traveling from node j to node i.

                edge_weights = np.array(edge_weights).reshape(-1, 1)

                population_raw = pickle.load(open(
                    r"data/DynamicsData/Spain_Covid19/population.data",
                    'rb'))
                population_raw = population_raw * 1e-2  # for scale

                # for 2020 Spain Covid-19
                # A. Aleta and Y. Moreno, "Evaluation of the potential incidence of Covid-19 and effectiveness
                # of containment measures in Spain: a data-driven approach," BMC Med. 18, 157 (2020)

                b = 2.5 / 7.5
                r = 1 / 7.5

                S_raw = np.zeros_like(cases_raw)
                I_raw = np.zeros_like(cases_raw)
                R_raw = np.zeros_like(cases_raw)

                S_raw[0] = population_raw - cases_raw[0]
                I_raw[0] = cases_raw[0]
                for t in range(cases_raw.shape[0]):
                    if t > 0:
                        S_raw[t] = S_raw[t - 1] - cases_raw[t]
                        I_raw[t] = I_raw[t - 1] + cases_raw[t] - r * I_raw[t - 1]
                        R_raw[t] = R_raw[t - 1] + r * I_raw[t - 1]

                S_normed = S_raw / population_raw.reshape(1, -1)
                I_normed = I_raw / population_raw.reshape(1, -1)
                R_normed = R_raw / population_raw.reshape(1, -1)

                if 'case1' in kwargs:
                    start_t, end_t = 50, 150
                elif 'case2' in kwargs:
                    start_t, end_t = 150, 250
                elif 'case3' in kwargs:
                    start_t, end_t = 250, 350
                elif 'case4' in kwargs:
                    start_t, end_t = 350, 450
                else:
                    print('ERROR No specific case!')
                    exit(1)

                S_normed = S_normed[start_t:end_t].reshape(-1, S_normed.shape[1], 1)
                I_normed = I_normed[start_t:end_t].reshape(-1, I_normed.shape[1], 1)
                R_normed = R_normed[start_t:end_t].reshape(-1, R_normed.shape[1], 1)

                X = np.concatenate([S_normed[0], I_normed[0], R_normed[0]], axis=-1)

                state = np.concatenate([S_normed, I_normed, R_normed], axis=-1)  # t, n, d
                t_range = np.linspace(0, 1, state.shape[0] + 1).reshape(-1, 1)  # t,1

            elif dynamics_name == "Coupled_Epidemic_dynamics":
                x_dim = 4
                if 'make_test_set' in kwargs:
                    X_IS = 1e-3 + np.random.rand(N, 1) * (1e-1 - 1e-3)  # [0.5, 1]
                    X_SI = 1e-3 + np.random.rand(N, 1) * (1e-1 - 1e-3)  # [0.5, 1]
                    X_SS = 1. - X_IS - X_SI  # [0., 0.2]
                    X_II = np.zeros((N, 1))

                    X = np.concatenate([X_SS, X_IS, X_SI, X_II], axis=-1)
                else:
                    X_IS = 1e-3 + np.random.rand(N, 1) * (1e-1 - 1e-3)  # [0.5, 1]
                    X_SI = 1e-3 + np.random.rand(N, 1) * (1e-1 - 1e-3)  # [0.5, 1]
                    X_SS = 1. - X_IS - X_SI  # [0., 0.2]
                    X_II = np.zeros((N, 1))

                    X = np.concatenate([X_SS, X_IS, X_SI, X_II], axis=-1)

                if 'make_test_set' in kwargs:
                    if 'fixed_param_case1' in kwargs:
                        b1 = 0.9
                        b2 = 0.6
                        r1 = 0.25
                        r2 = 0.25
                        c = 0.01
                    elif 'fixed_param_case2' in kwargs:
                        b1 = 0.9
                        b2 = 0.6
                        r1 = 0.25
                        r2 = 0.25
                        c = 5
                    else:
                        params_dynamics = np.loadtxt('data/DynamicsData/dynamics_%s_topo_%s_train_set_params.txt' % (
                            dynamics_name, topo_type))

                        params_dynamics = params_dynamics.reshape(-1, 5)

                        b1 = 0.5 + np.random.rand() * (1. - 0.5)
                        b2 = 0.5 + np.random.rand() * (1. - 0.5)
                        r1 = 0.1 + np.random.rand() * (0.4 - 0.1)
                        r2 = 0.1 + np.random.rand() * (0.4 - 0.1)
                        c = 0.01 + np.round(np.random.rand()) * (5 - 0.01)

                        while np.sum(
                                np.sum(np.abs(params_dynamics - np.array([b1, b2, r1, r2, c])), axis=1) <
                                params_dynamics.shape[
                                    -1] * 1e-3) > 0.:
                            b1 = 0.5 + np.random.rand() * (1. - 0.5)
                            b2 = 0.5 + np.random.rand() * (1. - 0.5)
                            r1 = 0.1 + np.random.rand() * (0.4 - 0.1)
                            r2 = 0.1 + np.random.rand() * (0.4 - 0.1)
                            c = 0.01 + np.round(np.random.rand()) * (5 - 0.01)

                else:

                    b1_list = np.linspace(0.5, 1., 5).tolist()
                    b2_list = np.linspace(0.5, 1., 5).tolist()
                    r1_list = np.linspace(0.1, 0.4, 5).tolist()
                    r2_list = np.linspace(0.1, 0.4, 5).tolist()
                    c_list = [0.01, 5]

                    count_ = 0
                    for item in itertools.product(*[b1_list, b2_list, r1_list, r2_list, c_list]):
                        count_ += 1
                        if count_ == i + 1:
                            # print(item)
                            b1, b2, r1, r2, c = item
                            break

                    params_dynamics.append([b1, b2, r1, r2, c])

                state, t_range = NetDynamic(dynamics_name).get_observations(X, [row, col], t_start,
                                                                            t_end, t_inc, False,
                                                                            b1=b1, b2=b2, r1=r1, r2=r2, c=c)

            elif dynamics_name == 'RealEpidemicData':
                x_dim = 1
                adj_timeseries_file_names = [
                    ('data/DynamicsData/RealEpidemicData/H1N1_adj.txt',
                     'data/DynamicsData/RealEpidemicData/H1N1_filter_total_cases_raw.txt'),
                    ('data/DynamicsData/RealEpidemicData/SARS_adj.txt',
                     'data/DynamicsData/RealEpidemicData/SARS_filter_total_cases_raw.txt'),
                    ('data/DynamicsData/RealEpidemicData/COVID_adj.txt',
                     'data/DynamicsData/RealEpidemicData/COVID_filter_total_cases_raw.txt'),
                ]
                
                delta_t = 10
                
                if 'make_test_set' in kwargs:
                    if 'case' in kwargs:
                        case_id = int(kwargs['case'])
                        disease_id = 0
                        start_t_ = case_id
                        delta_t = 2
                    """ 
                    if 'case1' in kwargs:
                        disease_id = 0
                        start_t_ = 0
                    elif 'case2' in kwargs:
                        disease_id = 0
                        start_t_ = delta_t
                    elif 'case3' in kwargs:
                        disease_id = 0
                        start_t_ = delta_t * 2
                    elif 'case4' in kwargs:
                        disease_id = 0
                        start_t_ = delta_t * 3
                    elif 'case5' in kwargs:
                        disease_id = 0
                        start_t_ = delta_t * 4
                    elif 'case6' in kwargs:
                        disease_id = 0
                        start_t_ = delta_t * 5
                    elif 'case7' in kwargs:
                        disease_id = 1
                        start_t_ = 0
                    elif 'case8' in kwargs:
                        disease_id = 1
                        start_t_ = delta_t
                    elif 'case9' in kwargs:
                        disease_id = 1
                        start_t_ = delta_t * 2
                    elif 'case10' in kwargs:
                        disease_id = 1
                        start_t_ = delta_t * 3
                    elif 'case11' in kwargs:
                        disease_id = 1
                        start_t_ = delta_t * 4
                    elif 'case12' in kwargs:
                        disease_id = 1
                        start_t_ = delta_t * 5
                    elif 'case13' in kwargs:
                        disease_id = 2
                        start_t_ = 0
                    elif 'case14' in kwargs:
                        disease_id = 2
                        start_t_ = delta_t
                    elif 'case15' in kwargs:
                        disease_id = 2
                        start_t_ = delta_t * 2
                    elif 'case16' in kwargs:
                        disease_id = 2
                        start_t_ = delta_t * 3
                    elif 'case17' in kwargs:
                        disease_id = 2
                        start_t_ = delta_t * 4
                    elif 'case18' in kwargs:
                        disease_id = 2
                        start_t_ = delta_t * 5
                        
                    else:
                        print('ERROR: unknown case')
                    """
                else:
                
                    disease_id = 0
                    start_t_ = i

                # N * N
                epidemic_data_adj = np.loadtxt(adj_timeseries_file_names[disease_id][0])
                # T, N
                total_cases_raw = np.loadtxt(adj_timeseries_file_names[disease_id][1])

                N = len(epidemic_data_adj)

                # make weights
                row = []
                col = []
                for from_node in range(N):
                    for to_node in range(N):
                        if epidemic_data_adj[to_node, from_node] > 0:
                            row.append(from_node)
                            col.append(to_node)
                row = np.array(row)
                col = np.array(col)

                edge_weights = []
                for edge_idx in range(len(row)):
                    if row[edge_idx] == col[edge_idx]:
                        edge_weights.append(0)
                    else:
                        edge_weights.append(epidemic_data_adj[col[edge_idx], row[
                            edge_idx]])  # epidemic_data_adj[i,j] means the average number of people that are traveling from node j to node i.
                edge_weights = np.array(edge_weights).reshape(-1, 1)

                # make cases
                end_t_ = start_t_ + delta_t
                total_cases_raw = total_cases_raw[start_t_:end_t_, :]
                # normalization
                total_cases_norm = (total_cases_raw - np.min(total_cases_raw)) / (
                            np.max(total_cases_raw) - np.min(total_cases_raw)) * 1000 # make same range of [0, 1000]

                state = total_cases_norm.reshape(total_cases_norm.shape[0], total_cases_norm.shape[1], x_dim)  # t, n, d

                X = state[0]

                t_range = np.linspace(0, 1, state.shape[0] + 1).reshape(-1, 1)  # t,1
            
            
            elif dynamics_name == 'sim_epidemic_dynamics':
                x_dim = 1
                adj_timeseries_file_names = [
                    ('data/DynamicsData/RealEpidemicData/H1N1_adj.txt',
                     'data/DynamicsData/RealEpidemicData/H1N1_filter_total_cases_raw.txt'),
                ]
                
                # N * N
                epidemic_data_adj = np.loadtxt(adj_timeseries_file_names[0][0])
                # T, N
                total_cases_raw = np.loadtxt(adj_timeseries_file_names[0][1])

                N = len(epidemic_data_adj)
                
                # make weights
                row = []
                col = []
                for from_node in range(N):
                    for to_node in range(N):
                        if epidemic_data_adj[to_node, from_node] > 0:
                            row.append(from_node)
                            col.append(to_node)
                row = np.array(row)
                col = np.array(col)

                edge_weights = []
                for edge_idx in range(len(row)):
                    if row[edge_idx] == col[edge_idx]:
                        edge_weights.append(0)
                    else:
                        edge_weights.append(epidemic_data_adj[col[edge_idx], row[
                            edge_idx]])  # epidemic_data_adj[i,j] means the average number of people that are traveling from node j to node i.
                edge_weights = np.array(edge_weights).reshape(-1, 1)
                
                
                
                if 'make_test_set' in kwargs:
                    if 'case_1' in kwargs:
                        case_id = int(kwargs['case_1'])
                        delta_t = case_id
                        
                        # make cases
                        start_t_ = 0
                        end_t_ = start_t_ + delta_t
                        total_cases_raw = total_cases_raw[start_t_:end_t_ + 1, :]
                        # normalization
                        total_cases_norm = (total_cases_raw - np.min(total_cases_raw)) / (
                                    np.max(total_cases_raw) - np.min(total_cases_raw)) * 1000 # make same range of [0, 1000]
        
                        state = total_cases_norm.reshape(total_cases_norm.shape[0], total_cases_norm.shape[1], x_dim)  # t, n, d
        
                        X = state[0]
        
                        t_range = np.linspace(0, 1, state.shape[0]).reshape(-1, 1)  # t,1
                        
                    elif 'case_2' in kwargs:
                        X = total_cases_raw[0].reshape(-1, x_dim)
                        
                        a = 1 + np.random.rand() * (10 - 1)
                        b = 10 + np.random.rand() * (100 - 10)
                        
                        
                        state, t_range = NetDynamic(dynamics_name).get_observations(X.reshape(-1, x_dim), [row, col], t_start,
                                                                                    t_end, t_inc, False,
                                                                                    a=a, b=b, edge_weights=edge_weights)
                        state = state / np.max(state) * 1000
                        
                    else:
                        print('ERROR: unknown case')
                        
                    # print(t_range)
                else:
                    
                    """
                    X = np.zeros(N)
                    init_N = np.random.choice(a=list(range(N)))
                    init_idxs = np.random.choice(a=list(range(N)), size=init_N, replace=False)
                    for init_idx in init_idxs:
                        X[init_idx] = 0. + np.random.rand() * (1e3 - 0.)
                    """
                    X = total_cases_raw[0].reshape(-1, x_dim)
                    
                    a_list = np.linspace(1, 10, 20).tolist()
                    b_list = np.linspace(10, 100, 50).tolist()
                    #t_inc_list = [1/1, 1/5, 1/10, 1/20]
                    
                    count_ = 0
                    for item in itertools.product(*[a_list, b_list]):
                        count_ += 1
                        if count_ == i + 1:
                            # print(item)
                            a, b = item
                            break
                            
                    #t_start = 0
                    #t_end = 1
                    print(t_start, t_inc, t_end)
                    
                    state, t_range = NetDynamic(dynamics_name).get_observations(X.reshape(-1, x_dim), [row, col], t_start,
                                                                                t_end, t_inc, False,
                                                                                a=a, b=b, edge_weights=edge_weights)
                    state = state / np.max(state) * 1000
                    
                                                                            
               
            elif dynamics_name == 'RealEpidemicData_mix':
                x_dim = 1
                adj_timeseries_file_names = [
                    ('data/DynamicsData/RealEpidemicData/H1N1_adj.txt',
                     'data/DynamicsData/RealEpidemicData/H1N1_filter_total_cases_raw.txt'),
                    ('data/DynamicsData/RealEpidemicData/SARS_adj.txt',
                     'data/DynamicsData/RealEpidemicData/SARS_filter_total_cases_raw.txt'),
                    ('data/DynamicsData/RealEpidemicData/COVID_adj.txt',
                     'data/DynamicsData/RealEpidemicData/COVID_filter_total_cases_raw.txt'),
                ]
                
                if 'make_test_set' in kwargs:
                    if 'case_1_h1n1' in kwargs:
                        case_id = int(kwargs['case_1_h1n1'])
                        disease_id = 0
                        start_t_ = case_id
                        delta_t = 5
                    elif 'case_2_h1n1' in kwargs:
                        case_id = int(kwargs['case_2_h1n1'])
                        disease_id = 0
                        start_t_ = case_id
                        delta_t = 9
                    elif 'case_3_h1n1' in kwargs:
                        case_id = int(kwargs['case_3_h1n1'])
                        disease_id = 0
                        start_t_ = case_id
                        delta_t = 10
                    elif 'case_4_h1n1' in kwargs:
                        case_id = int(kwargs['case_4_h1n1'])
                        disease_id = 0
                        start_t_ = case_id
                        delta_t = 19
                    elif 'case_1_sars' in kwargs:
                        case_id = int(kwargs['case_1_sars'])
                        disease_id = 1
                        start_t_ = case_id
                        delta_t = 5
                    elif 'case_2_sars' in kwargs:
                        case_id = int(kwargs['case_2_sars'])
                        disease_id = 1
                        start_t_ = case_id
                        delta_t = 9
                    elif 'case_3_sars' in kwargs:
                        case_id = int(kwargs['case_3_sars'])
                        disease_id = 1
                        start_t_ = case_id
                        delta_t = 10
                    elif 'case_4_sars' in kwargs:
                        case_id = int(kwargs['case_4_sars'])
                        disease_id = 1
                        start_t_ = case_id
                        delta_t = 19
                    elif 'case_1_covid' in kwargs:
                        case_id = int(kwargs['case_1_covid'])
                        disease_id = 2
                        start_t_ = case_id
                        delta_t = 5
                    elif 'case_2_covid' in kwargs:
                        case_id = int(kwargs['case_2_covid'])
                        disease_id = 2
                        start_t_ = case_id
                        delta_t = 9
                    elif 'case_3_covid' in kwargs:
                        case_id = int(kwargs['case_3_covid'])
                        disease_id = 2
                        start_t_ = case_id
                        delta_t = 10
                    elif 'case_4_covid' in kwargs:
                        case_id = int(kwargs['case_4_covid'])
                        disease_id = 2
                        start_t_ = case_id
                        delta_t = 19
                        
                    else:
                        print('ERROR: unknown case')
                else:
                
                    #disease_id_list = [0, 1]
                    disease_id_list = [0]
                    
                    start_t_list = np.linspace(0, 49, 50, dtype=np.int64).tolist()
                    delta_t_list = np.linspace(2, 21, 20, dtype=np.int64).tolist()
                    #delta_t_list = [2, 5, 11, 21]
                    
                    count_ = 0
                    for item in itertools.product(*[disease_id_list, start_t_list, delta_t_list]):
                        count_ += 1
                        if count_ == i + 1:
                            # print(item)
                            disease_id, start_t_, delta_t = item
                            break

                print('disease_id=%s, start_t_=%s, delta_t=%s'%(disease_id, start_t_, delta_t))
                # N * N
                epidemic_data_adj = np.loadtxt(adj_timeseries_file_names[disease_id][0])
                # T, N
                total_cases_raw = np.loadtxt(adj_timeseries_file_names[disease_id][1])

                N = len(epidemic_data_adj)

                # make weights
                row = []
                col = []
                for from_node in range(N):
                    for to_node in range(N):
                        if epidemic_data_adj[to_node, from_node] > 0:
                            row.append(from_node)
                            col.append(to_node)
                row = np.array(row)
                col = np.array(col)

                edge_weights = []
                for edge_idx in range(len(row)):
                    if row[edge_idx] == col[edge_idx]:
                        edge_weights.append(0)
                    else:
                        edge_weights.append(epidemic_data_adj[col[edge_idx], row[
                            edge_idx]])  # epidemic_data_adj[i,j] means the average number of people that are traveling from node j to node i.
                edge_weights = np.array(edge_weights).reshape(-1, 1)

                # make cases
                end_t_ = start_t_ + delta_t + 1
                total_cases_raw = total_cases_raw[start_t_:end_t_, :]
                # normalization
                total_cases_norm = (total_cases_raw - np.min(total_cases_raw)) / (
                            np.max(total_cases_raw) - np.min(total_cases_raw)) * 999 + 1 # make same range of [1, 1000]

                state = total_cases_norm.reshape(total_cases_norm.shape[0], total_cases_norm.shape[1], x_dim)  # t, n, d

                X = state[0]

                #t_range = np.linspace(0, 1, state.shape[0]).reshape(-1, 1)  # t, 1
                t_range = np.linspace(0, 1, 22)[:state.shape[0]].reshape(-1, 1)  # t, 1
                #print(t_range.shape, t_range)
                #print(start_t_, end_t_, state.shape, t_range.shape)

            elif dynamics_name == 'RealEpidemicData_12' or dynamics_name == 'RealEpidemicData_13' or \
                    dynamics_name == 'RealEpidemicData_23':
                x_dim = 1
                adj_timeseries_file_names = [
                    ('data/DynamicsData/RealEpidemicData/H1N1_adj.txt',
                     'data/DynamicsData/RealEpidemicData/H1N1_filter_total_cases_raw.txt'),
                    ('data/DynamicsData/RealEpidemicData/SARS_adj.txt',
                     'data/DynamicsData/RealEpidemicData/SARS_filter_total_cases_raw.txt'),
                    ('data/DynamicsData/RealEpidemicData/COVID_adj.txt',
                     'data/DynamicsData/RealEpidemicData/COVID_filter_total_cases_raw.txt'),
                    # ('data/DynamicsData/RealEpidemicData/Covid19_Spain_adj.txt',
                    # 'data/DynamicsData/RealEpidemicData/Covid19_Spain_total_cases_raw.txt'),
                ]

                if 'make_test_set' in kwargs:
                    if 'case1' in kwargs:
                        epidemic_data_adj = np.loadtxt(adj_timeseries_file_names[0][0])
                        total_cases_raw = np.loadtxt(adj_timeseries_file_names[0][1])
                    elif 'case2' in kwargs:
                        epidemic_data_adj = np.loadtxt(adj_timeseries_file_names[1][0])
                        total_cases_raw = np.loadtxt(adj_timeseries_file_names[1][1])
                    elif 'case3' in kwargs:
                        epidemic_data_adj = np.loadtxt(adj_timeseries_file_names[2][0])
                        total_cases_raw = np.loadtxt(adj_timeseries_file_names[2][1])
                    else:
                        print('ERROR: unknown case')
                else:

                    if '12' in dynamics_name:
                        adj_timeseries_file_names_ = adj_timeseries_file_names.pop(2)
                    elif '13' in dynamics_name:
                        adj_timeseries_file_names_ = adj_timeseries_file_names.pop(1)
                    elif '23' in dynamics_name:
                        adj_timeseries_file_names_ = adj_timeseries_file_names.pop(0)
                    # elif '234' in dynamics_name:
                    #    adj_timeseries_file_names_ = adj_timeseries_file_names.pop(0)
                    else:
                        print('ERROR: unknown dynamics_name [%s]' % dynamics_name)

                    epidemic_data_adj = np.loadtxt(adj_timeseries_file_names[i][0])
                    total_cases_raw = np.loadtxt(adj_timeseries_file_names[i][1])

                N = len(epidemic_data_adj)

                # make weights
                row = []
                col = []
                for from_node in range(N):
                    for to_node in range(N):
                        if epidemic_data_adj[to_node, from_node] > 0:
                            row.append(from_node)
                            col.append(to_node)
                row = np.array(row)
                col = np.array(col)

                edge_weights = []
                for edge_idx in range(len(row)):
                    if row[edge_idx] == col[edge_idx]:
                        edge_weights.append(0)
                    else:
                        edge_weights.append(epidemic_data_adj[col[edge_idx], row[
                            edge_idx]])  # epidemic_data_adj[i,j] means the average number of people that are traveling from node j to node i.
                edge_weights = np.array(edge_weights).reshape(-1, 1)

                # make cases
                # t, n
                # total_cases_norm = total_cases_raw / np.max(total_cases_raw, axis=0, keepdims=True)
                total_cases_norm = total_cases_raw / 1e4
                state = total_cases_norm.reshape(total_cases_norm.shape[0], total_cases_norm.shape[1], x_dim)  # t, n, d

                X = state[0]

                t_range = np.linspace(0, 1, state.shape[0] + 1).reshape(-1, 1)  # t,1
            
            elif dynamics_name == "predator_swarm_dynamics":
                x_dim = 2
                if 'make_test_set' in kwargs:
                    # Initial Value
                    # sqrt_N = int(np.sqrt(N))
                    # X = np.zeros((sqrt_N, sqrt_N))
                    # X[int(0.05 * sqrt_N):int(0.25 * sqrt_N), int(0.05 * sqrt_N):int(0.25 * sqrt_N)] = 25.  # X[1:5, 1:5] = 25  for N = 225 or 400 case
                    # X[int(0.45 * sqrt_N):int(0.75 * sqrt_N), int(0.45 * sqrt_N):int(0.75 * sqrt_N)] = 20.  # X[9:15, 9:15] = 20 for N = 225 or 400 case
                    # X[int(0.05 * sqrt_N):int(0.25 * sqrt_N), int(0.35 * sqrt_N):int(0.65 * sqrt_N)] = 17.  # X[1:5, 7:13] = 17 for N = 225 or 400 case
                    # X = X.reshape(-1, 1)
                    X = np.random.rand(N, x_dim)  # for predator_swarm_dynamics
                else:
                    X = np.random.rand(N, x_dim)  # for predator_swarm_dynamics

                if 'make_test_set' in kwargs:
                    params_dynamics = np.loadtxt('data/DynamicsData/dynamics_%s_topo_%s_train_set_params.txt' % (
                        dynamics_name, topo_type))

                    params_dynamics = params_dynamics.reshape(-1, 6)

                    a = 0.5 + np.random.rand() * (1.5 - 0.5)
                    b = -1 + np.random.rand() * (-3 + 1)
                    c = -1 + np.random.rand() * (-3 + 1)
                    d = -1 + np.random.rand() * (-3 + 1)
                    e = 2 + np.random.rand() * (4 - 2)
                    f = 1 + np.random.rand() * (3 - 1)

                    while np.sum(np.sum(np.abs(params_dynamics - np.array([a, b, c, d, e, f])), axis=1) <
                                 params_dynamics.shape[
                                     -1] * 1e-3) > 0.:
                        a = 0.5 + np.random.rand() * (1.5 - 0.5)
                        b = -1 + np.random.rand() * (-3 + 1)
                        c = -1 + np.random.rand() * (-3 + 1)
                        d = -1 + np.random.rand() * (-3 + 1)
                        e = 2 + np.random.rand() * (4 - 2)
                        f = 1 + np.random.rand() * (3 - 1)

                else:
                    a = 0.5 + np.random.rand() * (1.5 - 0.5)
                    b = -1 + np.random.rand() * (-3 + 1)
                    c = -1 + np.random.rand() * (-3 + 1)
                    d = -1 + np.random.rand() * (-3 + 1)
                    e = 2 + np.random.rand() * (4 - 2)
                    f = 1 + np.random.rand() * (3 - 1)

                    params_dynamics.append([a, b, c, d, e, f])

                state, t_range = NetDynamic(dynamics_name).get_observations(X, [row, col], t_start,
                                                                            t_end, t_inc, False,
                                                                            a=a, b=b, c=c, d=d, e=e, f=f)
            elif dynamics_name == "brain_FitzHugh_Nagumo_dynamics":
                x_dim = 2
                if 'make_test_set' in kwargs:
                    # Initial Value
                    # sqrt_N = int(np.sqrt(N))
                    # X = np.zeros((sqrt_N, sqrt_N))
                    # X[int(0.05 * sqrt_N):int(0.25 * sqrt_N), int(0.05 * sqrt_N):int(0.25 * sqrt_N)] = 25.  # X[1:5, 1:5] = 25  for N = 225 or 400 case
                    # X[int(0.45 * sqrt_N):int(0.75 * sqrt_N), int(0.45 * sqrt_N):int(0.75 * sqrt_N)] = 20.  # X[9:15, 9:15] = 20 for N = 225 or 400 case
                    # X[int(0.05 * sqrt_N):int(0.25 * sqrt_N), int(0.35 * sqrt_N):int(0.65 * sqrt_N)] = 17.  # X[1:5, 7:13] = 17 for N = 225 or 400 case
                    # X = X.reshape(-1, 1)
                    X = -1 + np.random.rand(N, x_dim) * 2  # for brain_FitzHugh_Nagumo_dynamics
                else:
                    X = -1 + np.random.rand(N, x_dim) * 2  # for brain_FitzHugh_Nagumo_dynamics

                if 'make_test_set' in kwargs:
                    params_dynamics = np.loadtxt('data/DynamicsData/dynamics_%s_topo_%s_train_set_params.txt' % (
                        dynamics_name, topo_type))

                    params_dynamics = params_dynamics.reshape(-1, 4)

                    a = 0.2 + np.random.rand() * (0.3 - 0.2)
                    b = 0.4 + np.random.rand() * (0.6 - 0.4)
                    c = -0.06 + np.random.rand() * (-0.02 + 0.06)
                    d = 0.8 + np.random.rand() * (1.2 - 0.8)
                    
                    while np.sum(np.sum(np.abs(params_dynamics - np.array([a, b, c, d])), axis=1) <
                                 params_dynamics.shape[
                                     -1] * 1e-3) > 0.:
                        a = 0.2 + np.random.rand() * (0.3 - 0.2)
                        b = 0.4 + np.random.rand() * (0.6 - 0.4)
                        c = -0.06 + np.random.rand() * (-0.02 + 0.06)
                        d = 0.8 + np.random.rand() * (1.2 - 0.8)

                else:
                    a = 0.2 + np.random.rand() * (0.3 - 0.2)
                    b = 0.4 + np.random.rand() * (0.6 - 0.4)
                    c = -0.06 + np.random.rand() * (-0.02 + 0.06)
                    d = 0.8 + np.random.rand() * (1.2 - 0.8)
                    

                    params_dynamics.append([a, b, c, d])

                state, t_range = NetDynamic(dynamics_name).get_observations(X, [row, col], t_start,
                                                                            t_end, t_inc, False,
                                                                            a=a, b=b, c=c, d=d)
            elif dynamics_name == "phototaxis_dynamics":
                x_dim = 5
                if 'make_test_set' in kwargs:
                    # Initial Value
                    # sqrt_N = int(np.sqrt(N))
                    # X = np.zeros((sqrt_N, sqrt_N))
                    # X[int(0.05 * sqrt_N):int(0.25 * sqrt_N), int(0.05 * sqrt_N):int(0.25 * sqrt_N)] = 25.  # X[1:5, 1:5] = 25  for N = 225 or 400 case
                    # X[int(0.45 * sqrt_N):int(0.75 * sqrt_N), int(0.45 * sqrt_N):int(0.75 * sqrt_N)] = 20.  # X[9:15, 9:15] = 20 for N = 225 or 400 case
                    # X[int(0.05 * sqrt_N):int(0.25 * sqrt_N), int(0.35 * sqrt_N):int(0.65 * sqrt_N)] = 17.  # X[1:5, 7:13] = 17 for N = 225 or 400 case
                    # X = X.reshape(-1, 1)
                    X = np.random.rand(N, x_dim) * 100.  # for phototaxis_dynamics
                    X[:, 4] = np.random.rand(N) * 0.001
                else:
                    X = np.random.rand(N, x_dim) * 100.  # for phototaxis_dynamics
                    X[:, 4] = np.random.rand(N) * 0.001

                if 'make_test_set' in kwargs:
                    params_dynamics = np.loadtxt('data/DynamicsData/dynamics_%s_topo_%s_train_set_params.txt' % (
                        dynamics_name, topo_type))

                    params_dynamics = params_dynamics.reshape(-1, 3)

                    I0 = 0.01 + np.random.rand() * (1. - 0.01)
                    b = -0.4 + np.random.rand() * (-0.1 + 0.4)
                    ec = 0.1 + np.random.rand() * (0.5 - 0.1)
                    
                    
                    while np.sum(np.sum(np.abs(params_dynamics - np.array([I0, b, ec])), axis=1) <
                                 params_dynamics.shape[
                                     -1] * 1e-3) > 0.:
                        I0 = 0.01 + np.random.rand() * (1. - 0.01)
                        b = -0.4 + np.random.rand() * (-0.1 + 0.4)
                        ec = 0.1 + np.random.rand() * (0.5 - 0.1)

                else:
                    I0 = 0.01 + np.random.rand() * (1. - 0.01)
                    b = -0.4 + np.random.rand() * (-0.1 + 0.4)
                    ec = 0.1 + np.random.rand() * (0.5 - 0.1)
                    

                    params_dynamics.append([I0, b, ec])

                state, t_range = NetDynamic(dynamics_name).get_observations(X, [row, col], t_start,
                                                                            t_end, t_inc, False,
                                                                            I0=I0, b=b, ec=ec)
                                                                            


            else:
                print("ERROR dynamic_name [%s]" % dynamics_name)
                exit(1)

            if 'obs_noise' in kwargs:
                ## add noise
                noise = kwargs['obs_noise']
                state_w_noise = state + noise * np.random.randn(state.shape[0], state.shape[1], state.shape[2])
                state_w_noise[0] = state[0] ## we do not add noise to the initial state
                

            if edge_weights is None:
                edge_weights = np.ones_like(row)
            else:
                edge_weights = edge_weights.reshape(-1)
            ATask['adj'] = np.array([row, col, edge_weights])
            ATask['X0'] = X.reshape(-1, x_dim)
            ATask['task_info'] = np.array([0] * N)

            # 返回第一个节点的动力学机制
            # x = np.linspace(t_start, t_start + max_num - 1, rel2.shape[0]).reshape(1, -1, 1)
            if 'RealEpidemicData' in dynamics_name:
                query_all_t = True
                query_all_node = True

            if query_all_t:
                choosed_t_list = list(range(state.shape[0]))
            else:
                # num_sampling_t = np.random.randint(4, state.shape[0])  ## sampling from [9, state.shape[0]-1]
                num_sampling_t = 60
                choosed_t_list = [0] + np.random.choice(a=list(range(1, state.shape[0])),
                                                        size=num_sampling_t,
                                                        replace=False).tolist()  ## sampling from [10, state.shape[0]]
            # choosed_t_list = list(range(state.shape[0]))  ############################################
            for choose in choosed_t_list:
                if query_all_node or choose == 0:  # initial state
                    sampled_idxs = np.array(list(range(N)))
                else:
                    num_sampling_points_per_time = np.random.randint(1, int(N / 2) + 1)  ## [1, N/2]
                    sampled_idxs = np.random.choice(a=list(range(N)), size=num_sampling_points_per_time, replace=False)

                if len(sampled_idxs) > 0:
                    
                    t = t_range[choose, :].reshape(-1)  # t
                    
                    x = state[choose, :].reshape(-1, x_dim)  # x
                    x_self = np.zeros_like(x)
                    x_self[sampled_idxs] = x[sampled_idxs]
                    
                    if 'obs_noise' in kwargs:
                        x_w_noise = state_w_noise[choose, :].reshape(-1, x_dim)  # x
                        x_self_w_noise = np.zeros_like(x_w_noise)
                        x_self_w_noise[sampled_idxs] = x_w_noise[sampled_idxs]
                    
                    mask = np.zeros(N)
                    mask[sampled_idxs] = 1.
                    if 'obs_noise' in kwargs:
                        point = {"t": t,  # [1,] NOTICE: the t of all points in a task should be different from each other
                                 "x_self": x_self,  # [N, x_dim]
                                 "x_self_w_noise": x_self_w_noise,  # [N, x_dim]
                                 "mask": mask,  # [N,]
                                 "point_info": np.zeros(N),  # [N,]
                                 "adj": np.array([row, col, edge_weights]),
                                 }
                    else:
                        point = {"t": t,  # [1,] NOTICE: the t of all points in a task should be different from each other
                                 "x_self": x_self,  # [N, x_dim]
                                 "mask": mask,  # [N,]
                                 "point_info": np.zeros(N),  # [N,]
                                 "adj": np.array([row, col, edge_weights]),
                                 }
                    ATask['points'].append(point)

            tasks['tasks'].append(ATask)

            print("%s/%s ..." % (i + 1, num_graph))

        if 'make_test_set' not in kwargs:
            params_dynamics = np.array(params_dynamics)
            np.savetxt(
                'data/DynamicsData/dynamics_%s_topo_%s_train_set_params.txt' % (dynamics_name, topo_type),
                params_dynamics)

        return tasks


#########===================================================
##
##                   display
##
#########===================================================

def display_3D(states_GT, states_mean, states_std, topo, t_start, t_end, t_inc, x_context,
               available_time_list=None, animation=False,
               name='test_heat_diffusion_dynamics', pos=None, max_value=None, min_value=None):
    """
    :param states:  tensor size: [#nodes, time_steps, dim]
    states_mean
    :param topo:  tensor [row, col], size: [2, #edges]
    :param t_start: int
    :param t_end: int
    :param t_inc: int
    :param name: str
    :return:
    """
    row, col = topo  # [row, col]

    edges_list = [(row[i], col[i]) for i in range(len(row))]
    G = nx.from_edgelist(edges_list)
    if max_value is None or min_value is None:
        X_max = torch.max(states_GT)
        X_min = torch.min(states_GT)
    else:
        X_max = max_value
        X_min = min_value

    # if pos is None:
    #     pos = nx.spring_layout(G)
    #     pos = nx.rescale_layout_dict(pos)

    if not animation:

        # if available_time_list is None:
        #     available_time_list = np.linspace(t_start, t_end, len(states))
        # num_plots = len(available_time_list)
        #
        # for t_draw in available_time_list:
        #     fig = plt.figure(figsize=(12, 12), facecolor=(1, 1, 1))
        #     nx.draw(G, pos=pos)  # 绘图函数
        #     cf = nx.draw_networkx_nodes(G,
        #                                 pos=pos,
        #                                 node_color=states[int(t_draw/t_inc)],
        #                                 cmap=plt.cm.get_cmap('RdYlBu_r'),
        #                                 vmin=X_min,
        #                                 vmax=X_max)
        #     plt.colorbar(cf, shrink=0.5)
        #
        #     plt.savefig('%s_t=%s.png' % (name,t_draw))
        #     plt.close()
        # plt.show()

        num_sampling, N, time_steps, x_dim = states_mean.size()
        n = int(np.ceil(np.sqrt(N)))
        x_time = torch.linspace(t_start, t_end, int((t_end - t_start + t_inc) / t_inc)).view(1, 1, -1, 1)

        fig, axes = plt.subplots(n, n, figsize=(40, 40))
        for i in range(N):
            row_idx = i // n
            col_idx = i % n
            for ddim in range(x_dim):
                for one_smpling in range(num_sampling):
                    plot_functions(axes[row_idx, col_idx],
                                   x_time,
                                   states_GT[i, :, ddim].view(1, 1, x_time.size(2), 1),
                                   x_time,
                                   states_GT[i, :, ddim].view(1, 1, x_time.size(2), 1),
                                   states_mean[one_smpling, i, :, ddim].view(1, 1, x_time.size(2), 1),
                                   states_std[one_smpling, i, :, ddim].view(1, 1, x_time.size(2), 1))

                for iii in range(len(x_context[0])):
                    if x_context[2][iii][i] == 1:
                        axes[row_idx, col_idx].scatter(x_context[0][iii].cpu(), x_context[1][iii][i, ddim].cpu(),
                                                       color='k')

        plt.savefig('results/%s.png' % (name))
        plt.close()


def display_diff(x_time, GT_sum, predictions_sum_diff, name):
    """
    :param x_time: size [timesteps,]
    :param GT_sum: size [timesteps, 1]
    :param predictions_sum_diff: size [num_sampling, timesteps, 1]
    :return:
    """
    num_sampling, time_steps, _ = predictions_sum_diff.size()
    assert time_steps == len(x_time)
    fig, axes = plt.subplots(1, 1)
    for one_smpling in range(num_sampling):
        plot_functions(axes,
                       x_time.view(1, 1, -1, 1),
                       GT_sum.view(1, 1, -1, 1),
                       x_time.view(1, 1, -1, 1),
                       GT_sum.view(1, 1, -1, 1),
                       predictions_sum_diff[one_smpling, :, :].view(1, 1, -1, 1),
                       torch.zeros_like(predictions_sum_diff[one_smpling, :, :].view(1, 1, -1, 1)))
    plt.savefig('results/%s.png' % (name))
    plt.close()


def display(rel, G, t_start, t_end, t_inc, name='test_heat_diffusion_dynamics'):
    t_start = 0

    pos = nx.spring_layout(G)
    pos = nx.rescale_layout_dict(pos)
    X_max = np.max(rel)
    X_min = np.min(rel)

    fig = plt.figure(figsize=(12, 12), facecolor=(1, 1, 1))

    def make_frame(t):
        fig.clear(True)

        global t_start

        X = rel[t_start]

        t_start += 1

        nx.draw(G, pos=pos)  # 绘图函数
        cf = nx.draw_networkx_nodes(G,
                                    pos=pos,
                                    node_color=X,
                                    cmap=plt.cm.get_cmap('RdYlBu_r'),
                                    vmin=X_min,
                                    vmax=X_max)
        node_labels = dict()
        for i in range(len(X)):
            node_labels[i] = "%.2f" % X[i]
        nx.draw_networkx_labels(G, pos, labels=node_labels)
        print(X)
        # plt.show()  # 展示图
        plt.colorbar(cf, shrink=0.5)
        # plt.title('heat_diffusion_dynamics')

        return mplfig_to_npimage(fig)

    total_duration = (t_end - t_start) / t_inc - 1  # total time
    fps = 10  # speed
    duration = total_duration / fps
    animation = VideoClip(make_frame, duration=duration)
    animation.write_gif("%s.gif" % name, fps=fps)

    plt.close()

    for i in range(rel.shape[1]):
        plt.plot(np.linspace(t_start, t_end, rel.shape[0]), rel[:, i])
    plt.savefig('%s_each_node_curve.png' % name)
    plt.close()

# # test
# if __name__ == '__main__':
#
#     from graph_neural_ODE_process_solution3 import make_batch, set_rand_seed
#
#     rseed = 666
#     set_rand_seed(rseed)
#
#     N = 225
#     # N = 400
#
#     saved_test_set = {}
#
#     for dynamics_name in [
#         'heat_diffusion_dynamics',
#         'mutualistic_interaction_dynamics',
#         'gene_regulatory_dynamics'
#     ]:
#         for topo_type in [
#             'grid',
#             'random',
#             'power_law',
#             'small_world',
#             'community'
#         ]:
#             time_steps = 100
#             x_dim = 1
#             num_graphs = 1
#
#             dataset = dynamics_dataset(dynamics_name, topo_type,
#                                        num_graphs_samples=num_graphs, time_steps=time_steps, x_dim=x_dim,
#                                        make_test_set=True)
#             test_time_steps = time_steps
#
#             for i in range(10):
#
#                 test_data = dataset.query(seed=rseed + i * 10, num_graph=1, max_time_points=test_time_steps,
#                                           query_all_node=True, query_all_t=True,
#                                           N=N,
#                                           make_test_set=True
#                                           )
#                 # print(test_data)
#
#                 for bound_t_context in [0., 0.25, 0.5, 0.75]:
#
#                     set_rand_seed(rseed + i * 100)
#
#                     gen_batch = make_batch(test_data, 1, is_shuffle=True, bound_t_context=bound_t_context, is_test=True,
#                                            is_shuffle_target=False)
#                     for batch_data in gen_batch:
#                         saved_test_set[(dynamics_name, topo_type, bound_t_context, i)] = batch_data
#
#             # save test data
#             import pickle
#
#             fname = 'data/DynamicsData/test_data_on_dynamics_%s_topo_%s_dataset_x%s_numgraph%s_timestep%s_seed%s_num_nodes=%s_split_train_and_test.pickle' % (
#                 dynamics_name, topo_type, dataset.x_dim, 1, time_steps, rseed,
#                 N)
#             f = open(fname, 'wb')
#             pickle.dump(saved_test_set, f)
#             f.close()

# # # 'heat_diffusion_dynamics'
# # # 'mutualistic_interaction_dynamics'
# # # 'gene_regulatory_dynamics'
# for dynamic_name in [
#                      'heat_diffusion_dynamics',
#                      # 'mutualistic_interaction_dynamics',
#                      # 'gene_regulatory_dynamics'
#                      ]:
#
#     # 'grid'
#     # 'random'
#     # 'power_law'
#     # 'small_world'
#     # 'community'
#     for topo_type in [
#         # 'grid',
#         # 'random',
#         'power_law',
#         # 'small_world',
#         # 'community'
#     ]:
#         ## display an instance
#         x_dim = 1
#         N = 100
#
#         G = build_topology(N, topo_type, 0)
#         N = G.number_of_nodes()
#         A = np.array(nx.adjacency_matrix(G).todense(), dtype=np.float64)
#         if dynamic_name == "heat_diffusion_dynamics":
#             # X = -100. + np.random.rand(N, x_dim) * (100. + 100.)  # for heat_diffusion_dynamics
#             # X = 0. + np.random.rand(N, x_dim) * (25.)  # for heat_diffusion_dynamics
#             X = np.zeros((N, x_dim))
#             X[0:10, 0] = 100.  # for heat_diffusion_dynamics
#             # X = np.random.rand(N, x_dim)   # for heat_diffusion_dynamics
#         elif dynamic_name == "mutualistic_interaction_dynamics":
#             X = np.random.rand(N, x_dim) * 25.  # for mutualistic_interaction_dynamics
#         elif dynamic_name == "gene_regulatory_dynamics":
#             X = np.random.rand(N, x_dim) * 25.  # for gene_regulatory_dynamics
#         else:
#             print("ERROR dynamic_name [%s]" % dynamic_name)
#             exit(1)
#
#         plt.matshow(A)
#         plt.savefig('topo_example_%s.png'%(topo_type))
#         plt.close()
#         # exit(1)
#
#         t_start = 0
#         t_inc = 0.01
#         t_steps = 100
#         t_end = (t_steps - 1) * t_inc
#
#         sparse_A = coo_matrix(A)
#
#         ND = NetDynamic(dynamic_name)
#         rel1, _ = ND.get_observations(X, [sparse_A.row, sparse_A.col], t_start, t_end, t_inc, False, K=0.5)
#         print(rel1.shape)
#
#         for ddim in range(x_dim):
#             t_start = 0
#             display(rel1[:, :, ddim], G, t_start, t_end, t_inc,
#                     name='test_%s_on_%s_state_%sth-dim' % (dynamic_name, topo_type, ddim))
#             # display_3D(rel1, [sparse_A.row, sparse_A.col], t_start, t_end, t_inc, available_time_list=[0,0.01,0.02,0.03,0.1], animation=False,
#             #            name='test_heat_diffusion_dynamics')
#
#             # rel1 = torch.from_numpy(rel1)
#             # display_3D(rel1, rel1, rel1, [sparse_A.row, sparse_A.col], t_start, t_end, t_inc,
#             #                available_time_list=None, animation=False,
#             #                name='results/test_%s'%dynamic_name, pos=None, max_value=None, min_value=None)
#
#         ## generate the dataset
#         # num_graphs = 2000
#         # time_steps = 50
#         # data = dynamics_dataset(dynamic_name, topo_type,
#         #                         num_graphs_samples=num_graphs, time_steps=time_steps, x_dim=x_dim)
#         # print(data)
