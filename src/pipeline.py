import argparse
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler

from data import load_mnist, pca_reduce
from xai_utils import build_knn_graph, cluster_representatives, shortest_path_attribution, betweenness_centrality_scores

import mlflow

def visualize_sample_images(images, indices, path, titles=None, ncols=8):
    n = len(indices)
    ncols = min(ncols, n)
    nrows = (n + ncols - 1) // ncols
    plt.figure(figsize=(1.6*ncols,1.6*nrows))
    for i, idx in enumerate(indices):
        plt.subplot(nrows, ncols, i+1)
        plt.imshow(images[idx], cmap="gray")
        plt.axis("off")
        if titles:
            plt.title(titles[i], fontsize=8)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def run_pipeline(out_dir="output", sample=2000, k=15, pca_dim=50, n_clusters=10, random_state=0):
    os.makedirs(out_dir, exist_ok=True)
    mlflow.set_experiment("xai-clustering-spectral")
    with mlflow.start_run():
        X, y, images = load_mnist(sample_size=sample, random_state=random_state)
        mlflow.log_param("sample", sample)
        mlflow.log_param("k", k)
        mlflow.log_param("pca_dim", pca_dim)
        mlflow.log_param("n_clusters", n_clusters)

        # Preprocess
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        Z, pca = pca_reduce(Xs, n_components=pca_dim, random_state=random_state)

        # Build kNN graph
        A, G = build_knn_graph(Z, k=k)
        mlflow.log_metric("nodes", G.number_of_nodes())
        mlflow.log_metric("edges", G.number_of_edges())

        # Spectral clustering (use precomputed affinity)
        # Use adjacency as affinity; convert to dense if small
        sc = SpectralClustering(n_clusters=n_clusters, affinity="precomputed", assign_labels="kmeans", random_state=random_state)
        labels_sc = sc.fit_predict(A)
        mlflow.log_metric("unique_clusters_sc", len(np.unique(labels_sc)))

        # KMeans baseline on PCA embedding
        km = KMeans(n_clusters=n_clusters, random_state=random_state)
        labels_km = km.fit_predict(Z)
        mlflow.log_metric("unique_clusters_km", len(np.unique(labels_km)))

        # Metrics
        ari = adjusted_rand_score(labels_km, labels_sc)
        nmi = normalized_mutual_info_score(labels_km, labels_sc)
        mlflow.log_metric("ari_km_vs_sc", float(ari))
        mlflow.log_metric("nmi_km_vs_sc", float(nmi))

        # Representatives per cluster
        reps = cluster_representatives(Z, labels_sc)

        # Compute betweenness centrality (approx)
        bc = betweenness_centrality_scores(G, k=max(32, int(0.01*len(Z))))
        # Save top-10 BC nodes visuals
        top_bc = sorted(bc.items(), key=lambda kv: kv[1], reverse=True)[:16]
        top_idxs = [int(k) for k,v in top_bc]
        visualize_sample_images(images, top_idxs, os.path.join(out_dir, "top_betweenness.png"),
                                titles=[f"idx:{i}\nbc:{bc[i]:.3f}" for i in top_idxs])

        # For a few sample nodes, compute shortest-path attribution to their cluster reps
        sample_nodes = list(np.random.RandomState(random_state).choice(len(Z), size=8, replace=False))
        sp_info = []
        for node in sample_nodes:
            cluster = int(labels_sc[node])
            rep_idx = int(reps[cluster])
            path, score = shortest_path_attribution(G, source_idx=node, target_idx=rep_idx)
            sp_info.append({"node": int(node), "cluster": cluster, "rep": rep_idx, "path_len": len(path), "score": float(score), "path": path})
            # Save path images
            if len(path) > 0:
                visualize_sample_images(images, path, os.path.join(out_dir, f"path_node_{node}.png"),
                                        titles=[f"i:{i}" for i in path], ncols=8)

        # Save results
        results = {
            "ari_km_vs_sc": float(ari),
            "nmi_km_vs_sc": float(nmi),
            "n_nodes": len(Z),
            "n_edges": G.number_of_edges(),
            "representatives": reps,
            "sample_shortest_paths": sp_info
        }
        with open(os.path.join(out_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=2)

        print("Done. Metrics:", results["ari_km_vs_sc"], results["nmi_km_vs_sc"])
        mlflow.log_artifact(os.path.join(out_dir, "top_betweenness.png"))
        for node in sample_nodes:
            p = os.path.join(out_dir, f"path_node_{node}.png")
            if os.path.exists(p):
                mlflow.log_artifact(p)
        mlflow.log_artifact(os.path.join(out_dir, "results.json"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="output")
    parser.add_argument("--sample", type=int, default=2000)
    parser.add_argument("--k", type=int, default=15)
    parser.add_argument("--pca_dim", type=int, default=50)
    parser.add_argument("--n_clusters", type=int, default=10)
    parser.add_argument("--random_state", type=int, default=0)
    args = parser.parse_args()
    run_pipeline(out_dir=args.out_dir, sample=args.sample, k=args.k, pca_dim=args.pca_dim, n_clusters=args.n_clusters, random_state=args.random_state)
