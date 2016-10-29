#!/usr/bin/env python
# -*- coding: utf-8 -*-

import scipy
import numpy as np
import networkx as nx
from networkx.exception import NetworkXError
from networkx.utils import not_implemented_for

"""
Returns the Biased PageRank of the nodes in the graph.
This code is based on networkx.pagerank.
"""

def biased_pagerank(G, num, alpha=0.05, personalization=None, max_iter=100, tol=1.0e-6, weight='weight', dangling=None):
    '''
    Parameters
    ----------
    G : graph
      A NetworkX graph.  Undirected graphs will be converted to a directed
      graph with two directed edges for each undirected edge.

    num : integer
      Label number of a seed-node.

    alpha : float, optional
      Damping parameter for biased PageRank, default=0.05.

    personalization: dict, optional
      The "personalization vector" consisting of a dictionary with a
      key for every graph node and nonzero personalization value for each node.
      By default, a uniform distribution is used.

    max_iter : integer, optional
      Maximum number of iterations in power method eigenvalue solver.

    tol : float, optional
      Error tolerance used to check convergence in power method solver.

    weight : key, optional
      Edge data key to use as weight.  If None weights are set to 1.

    dangling: dict, optional
      The outedges to be assigned to any "dangling" nodes, i.e., nodes without
      any outedges. The dict key is the node the outedge points to and the dict
      value is the weight of that outedge. By default, dangling nodes are given
      outedges according to the personalization vector (uniform if not
      specified). This must be selected to result in an irreducible transition
      matrix (see notes under google_matrix). It may be common to have the
      dangling dict to be the same as the personalization dict.

    Returns
    -------
    pagerank : dictionary
       Dictionary of nodes with PageRank as value.
    '''

    # Number of nodes
    N = len(G)
    if N == 0:
        return {}
    
    nodelist = G.nodes()
    
    # Adjacency matrix
    M = nx.to_scipy_sparse_matrix(G, nodelist=nodelist, weight=weight,
                                  dtype=float)

    S = scipy.array(M.sum(axis=1)).flatten()
    
    S[S != 0] = 1.0 / S[S != 0]
    Q = scipy.sparse.spdiags(S.T, 0, *M.shape, format='csr')
    
    M = Q * M

    # Initialize vector
    x = scipy.repeat(1.0 / N, N)
    
    # Personalization vector
    if personalization is None:
        p = scipy.repeat(1.0 / N, N)
    else:
        missing = set(nodelist) - set(personalization)
        if missing:
            raise NetworkXError('Personalization vector dictionary '
                                'must have a value for every node. '
                                'Missing nodes %s' % missing)
        p = scipy.array([personalization[n] for n in nodelist],
                        dtype=float)
        p = p / p.sum()
    
    # Dangling nodes
    if dangling is None:
        dangling_weights = p
    else:
        missing = set(nodelist) - set(dangling)
        if missing:
            raise NetworkXError('Dangling node dictionary '
                                'must have a value for every node. '
                                'Missing nodes %s' % missing)
        # Convert the dangling dictionary into an array in nodelist order
        dangling_weights = scipy.array([dangling[n] for n in nodelist],
                                       dtype=float)
        dangling_weights /= dangling_weights.sum()
    is_dangling = scipy.where(S == 0)[0]

    # power iteration: make up to max_iter iterations
    for _ in range(max_iter):
        xlast = x
        x = alpha * (x * M + sum(x[is_dangling]) * dangling_weights)
        
        x[num] += np.float64((1 - alpha) * p[num])
        
        # check convergence, l1 norm
        err = scipy.absolute(x - xlast).sum()
        if err < N * tol:
            print 'roop:', _ + 1
            return dict(zip(nodelist, map(float, x)))
    raise NetworkXError('pagerank_scipy: power iteration failed to converge '
                        'in %d iterations.' % max_iter)