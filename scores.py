import math

import networkx as nx
import numpy as np
import torch
# from numba import njit

from constants import N, INF, WORD_LEN
from utils import CounterexampleFoundException


def bfs(Gdeg, edgeListG):
    # simple breadth first search algorithm, from each vertex

    distMat1 = np.zeros((N, N))
    conn = True
    for s in range(N):
        visited = np.zeros(N, dtype=np.int8)

        # Create a queue for BFS. Queues are not suported with njit yet so do it manually
        myQueue = np.zeros(N, dtype=np.int8)
        dist = np.zeros(N, dtype=np.int8)
        startInd = 0
        endInd = 0

        # Mark the source node as visited and enqueue it
        myQueue[endInd] = s
        endInd += 1
        visited[s] = 1

        while endInd > startInd:
            pivot = myQueue[startInd]
            startInd += 1

            for i in range(Gdeg[pivot]):
                if visited[edgeListG[pivot][i]] == 0:
                    myQueue[endInd] = edgeListG[pivot][i]
                    dist[edgeListG[pivot][i]] = dist[pivot] + 1
                    endInd += 1
                    visited[edgeListG[pivot][i]] = 1
        if endInd < N:
            conn = False  # not connected

        for i in range(N):
            distMat1[s][i] = dist[i]

    return distMat1, conn


# jitted_bfs = njit()(bfs)


def conj23_score(states):
    """
    Reward function for Conjecture 2.3, using numba
    With n=30 it took a day to converge to the graph in figure 5, I don't think it will ever find the best graph
    (which I believe is when the neigbourhood of that almost middle vertex is one big clique).
    (This is not the best graph for all n, but seems to be for n=30)

    """
    rewards = []
    iterator = [states] if states.ndim == 1 else states
    for state in iterator:
        # construct the graph G
        adjMatG = np.zeros((N, N),
                           dtype=np.int8)  # adjacency matrix determined by the state
        edgeListG = np.zeros((N, N), dtype=np.int8)  # neighbor list
        Gdeg = np.zeros(N, dtype=np.int8)  # degree sequence
        count = 0
        for i in range(N):
            for j in range(i + 1, N):
                if state[count] == 1:
                    adjMatG[i][j] = 1
                    adjMatG[j][i] = 1
                    edgeListG[i][Gdeg[i]] = j
                    edgeListG[j][Gdeg[j]] = i
                    Gdeg[i] += 1
                    Gdeg[j] += 1
                count += 1

        distMat, conn = bfs(Gdeg, edgeListG)
        # G has to be connected
        if not conn:
            rewards.append(-INF)
            continue

        diam = np.amax(distMat)
        sumLengths = np.zeros(N, dtype=np.int8)
        sumLengths = np.sum(distMat, axis=0)
        evals = np.linalg.eigvalsh(distMat)
        evals = -np.sort(-evals)
        proximity = np.amin(sumLengths) / (N - 1.0)

        ans = -(proximity + evals[math.floor(2 * diam / 3) - 1])
        rewards.append(ans)
    return torch.FloatTensor(rewards)
# jitted_calcScore = njit()(calcScore)


def conj21_score(states: torch.tensor) -> torch.tensor:
    """Calculate reward for conjecture 2.1 from the original paper.
    This function is very slow, it can be massively sped up with numba
        but numba doesn't support networkx yet, which is very convenient to use here
    Args:
        states (torch.tensor): final state tensor of shape
            (N_CANDIDATES, OBSERVATION_LEN), in which first
            WORD_LEN symbols denote generated sequence
        multiprocess (bool): whether stat

    Returns:
        torch.tensor: rewards of shape (N_CANDIDATES,)

    Raises:
        CounterexampleFoundException: in case counterexample was found

    """

    # Example reward function, for Conjecture 2.1
    # Given a graph, it minimizes lambda_1 + mu.
    # Takes a few hours  (between 300 and 10000 iterations) to converge (loss < 0.01)
    # on my computer with these parameters if not using parallelization.
    # There is a lot of run-to-run variance.
    # Finds the counterexample some 30% (?) of the time with these parameters,
    # but you can run several instances in parallel.

    rewards = []

    # when multiprocessing.pool is used, we receive one candidate per call
    # thus if states has only one dimension - wrap it in list
    # for correct handling
    iterator = [states] if states.ndim == 1 else states
    for state in iterator:
        # Construct the graph
        G = nx.Graph()
        G.add_nodes_from(list(range(N)))
        count = 0
        for i in range(N):
            for j in range(i + 1, N):
                if state[count] == 1:
                    G.add_edge(i, j)
                count += 1

        # G is assumed to be connected in the conjecture.
        # If it isn't, return a very negative reward.
        if not (nx.is_connected(G)):
            rewards.append(-INF)
            continue

        # Calculate the eigenvalues of G
        evals = np.linalg.eigvalsh(nx.adjacency_matrix(G).todense())
        evalsRealAbs = np.zeros_like(evals)
        for i in range(len(evals)):
            evalsRealAbs[i] = abs(evals[i])
        lambda1 = max(evalsRealAbs)

        # Calculate the matching number of G
        maxMatch = nx.max_weight_matching(G)
        mu = len(maxMatch)

        # Calculate the reward. Since we want to minimize lambda_1 + mu,
        # we return the negative of this.
        # We add to this the conjectured best value.
        # This way if the reward is positive we know we have a counterexample.
        myScore = math.sqrt(N - 1) + 1 - lambda1 - mu
        if myScore > 0:
            raise CounterexampleFoundException(state, G, myScore)

        rewards.append(myScore)
    return torch.FloatTensor(rewards)


def sum_score(states: torch.tensor) -> torch.tensor:
    """Calculate score for given state.

    Args:
        states (torch.tensor): final state tensor of shape
            (N_CANDIDATES, OBSERVATION_LEN), in which first
            WORD_LEN symbols denote generated sequence

    Returns:
        torch.tensor: rewards of shape (N_CANDIDATES,)
    """
    return torch.sum(states[:, :WORD_LEN], dim=1)


def random_score(states: torch.tensor) -> torch.tensor:
    """Calculate score for given state.

    Args:
        states (torch.tensor): final state tensor of shape
            (N_CANDIDATES, OBSERVATION_LEN), in which first
            WORD_LEN symbols denote generated sequence

    Returns:
        torch.tensor: rewards of shape (N_CANDIDATES,)
    """
    return torch.randn(states.shape[0])