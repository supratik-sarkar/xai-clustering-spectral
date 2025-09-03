import numpy as np
import networkx as nx
from sklearn.neighbors import kneighbors_graph
from typing import List, Tuple
from collections import defaultdict

def build_knn_graph(embeddings: np.ndarray, k: int = 15, mode="connectivity"):
    """
    Returns symmetric adjacency (scipy sparse) and NetworkX graph
    """
    A = kneighbors_graph(embeddings, n_neighbors=k, include_self=False, mode=mode)
    # symmetrize
    A = 0.5 * (A + A.T)
    G = nx.from_scipy_sparse_matrix(A)
    return A, G

def cluster_representatives(embeddings: np.ndarray, labels: np.ndarray) -> dict:
    """
    For each cluster label, return index of the point closest to cluster mean.
    """
    reps = {}
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        centroid = embeddings[idx].mean(axis=0)
        dists = np.linalg.norm(embeddings[idx] - centroid[None, :], axis=1)
        reps[c] = idx[int(np.argmin(dists))]
    return reps

def shortest_path_attribution(G: nx.Graph, source_idx: int, target_idx: int) -> Tuple[List[int], float]:
    """
    Returns the node path (list of node indices) for the shortest path between source_idx and target_idx,
    and a scalar attribution score equal to 1/(length+1) (shorter path -> higher attribution).
    """
    try:
        path = nx.shortest_path(G, source=source_idx, target=target_idx)
        score = 1.0 / (len(path) + 1e-6)
    except nx.NetworkXNoPath:
        path = []
        score = 0.0
    return path, score

def betweenness_centrality_scores(G: nx.Graph, k=None):
    """
    Approx betweenness centrality. If k is provided, approximate using k node samples.
    """
    return nx.betweenness_centrality(G, k=k, normalized=True)
