import numpy as np

class KMeans:
    def __init__(self, k=10, n_iters=50, min_movement_threshold=1e-2):
        self.k = k
        self.n_iters = n_iters
        self.min_movement_threshold = min_movement_threshold
        self.centroids = []
        
    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum(np.square(x2 - x1)))
        
    def fit(self, X):
        i = 0
        centroid_moved_significantly = True
        indexes = np.random.choice([i for i in range(X.shape[0])], (self.k,), replace=False)
        centroids = X[indexes, :]
        
        while i < self.n_iters and  centroid_moved_significantly:
            distances = [[self.euclidean_distance(data_row, centroid_row) for centroid_row in centroids] for data_row in X]
            cluster_assignment = np.argmin(distances, axis=1)
            
            new_centroids = np.array([[0 for _ in range(X.shape[1])] for _ in range(self.k)])
            for index, assigned_cluster in enumerate(cluster_assignment):
                data_item = X[index]
                new_centroids[assigned_cluster] += data_item
            new_centroids = new_centroids / X.shape[0]
            
            movement = [self.euclidean_distance(old_centroid, new_centroid) for old_centroid, new_centroid in zip(centroids, new_centroids)]
            movement = np.array(movement)
            if np.all(movement < self.min_movement_threshold):
                centroid_moved_significantly = False
            
            centroids = new_centroids
            i += 1
            
        self.centroids = centroids
        
    def predict(self, x):
        distances = [self.euclidean_distance(x, cluster) for cluster in self.centroids]
        assigned_cluster = np.argmin(distances)
        return assigned_cluster
    
kmeans = KMeans(k=5, min_movement_threshold=1e-5)
kmeans.fit(np.array(
    [[i*j for i in range(5)] for j in range(100)]
))

print(kmeans.predict([100*i for i in range(5)]))
print(kmeans.centroids)