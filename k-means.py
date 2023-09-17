import numpy as np
import matplotlib.pyplot as plt

def k_means(X, K):
    # Initialize centroids randomly
    centroids = X[np.random.choice(X.shape[0], K, replace=False), :]
    old_centroids = np.zeros((K, X.shape[1]))
    clusters = np.zeros(X.shape[0])
    
    while np.linalg.norm(centroids - old_centroids) > 1e-6:
        # Assign data points to nearest centroids
        for i in range(X.shape[0]):
            distances = np.linalg.norm(X[i, :] - centroids, axis=1)
            clusters[i] = np.argmin(distances)
        
        # Update centroids
        old_centroids = centroids
        for k in range(K):
            centroids[k, :] = np.mean(X[clusters == k, :], axis=0)
    
    return clusters, centroids

# Generate some random data
np.random.seed(0)
X = np.random.randn(100, 2)

# Cluster the data into 3 clusters
clusters, centroids = k_means(X, 3)

# Plot the data points and centroids
plt.scatter(X[:, 0], X[:, 1], c=clusters)
plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=200, linewidth=3, color='r')
plt.show()
