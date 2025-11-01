import numpy as np
from math import log

# ------------------------
# INTERNAL METRICS
# ------------------------

def silhouette_score(X, labels):
    """
    Compute silhouette score from scratch.
    X: array of shape (n_samples, n_features)
    labels: cluster assignments
    """
    from scipy.spatial.distance import cdist
    
    n = len(X)
    unique_labels = np.unique(labels)
    k = len(unique_labels)

    if k == 1 or k == n:  # trivial clustering
        return 0

    sil_scores = []
    for i in range(n):
        same_cluster = X[labels == labels[i]]
        other_clusters = [X[labels == l] for l in unique_labels if l != labels[i]]
        
        # a(i): mean intra-cluster distance
        a_i = np.mean(cdist([X[i]], same_cluster)[0]) if len(same_cluster) > 1 else 0
        
        # b(i): mean nearest-cluster distance
        b_i = min([np.mean(cdist([X[i]], cluster)[0]) for cluster in other_clusters])
        
        sil_scores.append((b_i - a_i) / max(a_i, b_i + 1e-10))
    
    return np.mean(sil_scores)


def davies_bouldin_score(X, labels):
    """
    Davies–Bouldin Index (lower is better).
    """
    from scipy.spatial.distance import cdist
    
    clusters = [X[labels == l] for l in np.unique(labels)]
    centroids = [np.mean(cluster, axis=0) for cluster in clusters]
    
    S = [np.mean(cdist(cluster, [centroid])) for cluster, centroid in zip(clusters, centroids)]
    
    n_clusters = len(clusters)
    db_index = 0
    for i in range(n_clusters):
        max_ratio = -np.inf
        for j in range(n_clusters):
            if i != j:
                M_ij = np.linalg.norm(centroids[i] - centroids[j])
                max_ratio = max(max_ratio, (S[i] + S[j]) / (M_ij + 1e-10))
        db_index += max_ratio
    return db_index / n_clusters


def calinski_harabasz_score(X, labels):
    """
    Calinski–Harabasz Index (higher is better).
    """
    n_samples, _ = X.shape
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    
    overall_mean = np.mean(X, axis=0)
    between_disp = 0
    within_disp = 0
    
    for l in unique_labels:
        cluster = X[labels == l]
        n_l = len(cluster)
        cluster_mean = np.mean(cluster, axis=0)
        between_disp += n_l * np.sum((cluster_mean - overall_mean) ** 2)
        within_disp += np.sum((cluster - cluster_mean) ** 2)
    
    return (between_disp / (n_clusters - 1)) / (within_disp / (n_samples - n_clusters))

# ------------------------
# EXTERNAL METRICS
# ------------------------

def adjusted_rand_index(labels_true, labels_pred):
    """
    Adjusted Rand Index (ARI).
    """
    from math import comb
    
    n = len(labels_true)
    contingency = {}
    
    for i in range(n):
        contingency.setdefault(labels_true[i], {}).setdefault(labels_pred[i], 0)
        contingency[labels_true[i]][labels_pred[i]] += 1
    
    sum_comb_c = sum(comb(sum(row.values()), 2) for row in contingency.values())
    sum_comb_k = sum(comb(sum(contingency[l][k] for l in contingency), 2) for k in set(labels_pred))
    sum_comb = sum(comb(n_ij, 2) for row in contingency.values() for n_ij in row.values())
    
    prod = (sum_comb_c * sum_comb_k) / comb(n, 2)
    mean = (sum_comb_c + sum_comb_k) / 2
    return (sum_comb - prod) / (mean - prod + 1e-10)


def normalized_mutual_info(labels_true, labels_pred):
    """
    Normalized Mutual Information (NMI).
    """
    from collections import Counter
    
    n = len(labels_true)
    counter_true = Counter(labels_true)
    counter_pred = Counter(labels_pred)
    
    contingency = {}
    for t, p in zip(labels_true, labels_pred):
        contingency.setdefault(t, {}).setdefault(p, 0)
        contingency[t][p] += 1
    
    MI = 0
    for t in contingency:
        for p in contingency[t]:
            n_ij = contingency[t][p]
            MI += (n_ij / n) * log((n * n_ij) / (counter_true[t] * counter_pred[p] + 1e-10) + 1e-10, 2)
    
    H_true = -sum((count/n) * log(count/n, 2) for count in counter_true.values())
    H_pred = -sum((count/n) * log(count/n, 2) for count in counter_pred.values())
    
    return MI / ((H_true + H_pred) / 2 + 1e-10)

if __name__ == '__main__':
    from sklearn.datasets import make_blobs
    from sklearn.cluster import KMeans

    
    # Create synthetic data
    X, y_true = make_blobs(n_samples=200, centers=3, random_state=42)

    # Cluster with KMeans
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(X)
    y_pred = kmeans.labels_

    # Internal metrics
    print("Silhouette Score:", silhouette_score(X, y_pred))
    print("Davies–Bouldin Index:", davies_bouldin_score(X, y_pred))
    print("Calinski–Harabasz Index:", calinski_harabasz_score(X, y_pred))

    # External metrics (need true labels)
    print("Adjusted Rand Index:", adjusted_rand_index(y_true, y_pred))
    print("Normalized Mutual Info:", normalized_mutual_info(y_true, y_pred))
