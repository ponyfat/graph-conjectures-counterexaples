import io
import os
import random

import networkx as nx
import numpy as np
import torch
from matplotlib import pyplot as plt

from constants import N


class CounterexampleFoundException(Exception):
    """Special exception raised when counterexaple has been found."""

    def __init__(self, state, graph, score):
        self.state = state
        self.graph = graph
        self.score = score


def seed_everything(seed=42):
    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def get_graph_image(states):
    """Create a pyplot plot and save to buffer."""
    imgs = []
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
        fig = plt.figure(figsize=(7, 7), dpi=64)
        io_buf = io.BytesIO()
        nx.draw_kamada_kawai(G)
        fig.savefig(io_buf, format='raw', dpi=64)
        io_buf.seek(0)
        img_arr = np.frombuffer(io_buf.getvalue(), dtype=np.uint8)
        img_arr = img_arr.reshape(*fig.canvas.get_width_height(), -1)[:, :, :3]
        io_buf.close()
        plt.close(fig)
        imgs.append(torch.from_numpy(img_arr).permute(2, 0, 1))
    return imgs
