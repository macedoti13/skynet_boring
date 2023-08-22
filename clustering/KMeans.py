import numpy as np

class KMeans:
    """"""
    
    class Cluster:
        """"""

        def __init__(self, centroid_min, centroid_max, n_dims) -> None:
            """"""
            self.centroid = np.random.uniform(centroid_min, centroid_max, n_dims)
            self.points = []
            
        def recalculate_centroid(self):
            points_array = np.array(self.points) # Convert list of points to numpy array for computation
            self.centroid = np.mean(points_array, axis=0)
    
            
    def __init__(self, k: int, distance: str) -> None:
        """"""
        self.k = k
        self.distance = distance
        self.clusters = []
    
    
    def initialize_centroids(self, centroid_min: float, centroid_max: float, n_dims: int) -> None:
        """"""
        for _ in range(self.k):
            self.clusters.append(self.Cluster(centroid_min, centroid_max, n_dims))        
            
            
    def _calculate_distance(self, x: np.array, y: np.array, p: int) -> float:
        """"""
        total_sum = 0
        for i in range(len(x)):
            result = np.abs(x[i] - y[i])**p
            total_sum += result
            
        return np.power(total_sum, 1/p)
    
    
    def calculate_distance(self, x: np.array, y: np.array) -> float: 
        """distance between two vectors"""
        if self.distance == "euclidian":
            p = 2
            distance = self._calculate_distance(x, y, p)
        elif self.distance == "manhattan":
            p = 1
            distance = self._calculate_distance(x, y, p)
            
        return distance
    
    
    def _assign_point_to_cluster(self, x: np.array) -> None:
        """"""
        distances_to_clusters = []
        
        for cluster in self.clusters:
            distance = self.calculate_distance(x, cluster.centroid)
            distances_to_clusters.append(distance)
            
        best_cluster_index = distances_to_clusters.index(min(distances_to_clusters))
        
        self.clusters[best_cluster_index].points.append(x)
        
        
    def assign_points_to_clusters(self, X: np.array) -> None:
        """"""
        for point in X:
            self._assign_point_to_cluster(point)
            
            
    def update_centroids(self) -> None:
        """"""
        for cluster in self.clusters:
            cluster.recalculate_centroid()
            
            
    def fit(self, X: np.array, max_iters: int = 100, tolerance: float = 1e-4) -> None:
        """"""
        n_dims = X.shape[1]
        centroid_min = X.min()
        centroid_max = X.max()
        
        # initialize the centroids randomly 
        self.initialize_centroids(centroid_min, centroid_max, n_dims)
        
        for _ in range(max_iters):
            
            # save old centroids to check for movement
            old_centroids = [cluster.centroid.copy() for cluster in self.clusters]
        
            # assign points to clusters
            self.assign_points_to_clusters(X)
            
            # recalculate the centroids
            self.update_centroids()
            
            # check for convergence (centroids don't change anymore)
            movements = [self.calculate_distance(old, new.centroid) for old, new in zip(old_centroids, self.clusters)]
            if max(movements) < tolerance:
                break
            
            # clear points for next iteration
            for cluster in self.clusters:
                cluster.points = []
    
                
    def assign(self, X: np.array):
        """"""
        self.assign_points_to_clusters(X)
        