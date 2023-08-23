import numpy as np

class KMeans:
    
    def __init__(self, k: int, distance: str = "euclidian") -> None:
        """
        Initializes the KMeans clustering model.

        Args:
            k (int): The number of clusters to form.
            distance (str, optional): The distance metric to use. Options are "euclidian" and "manhattan". Defaults to "euclidian".
        """
        self.k = k
        self.distance = distance
        self.centroids = None
        
    def _random_initialize_centroids(self, X: np.array) -> None:
        """
        Randomly selects k data points from X as the initial centroids.

        Args:
            X (np.array): The input dataset.
        """
        random_indices = np.random.choice(X.shape[0], self.k, replace=False)  # Selects k unique indices 
        self.centroids = X[random_indices]  # Assign these points as centroids
        
    def _calculate_distance(self, x: np.array, y: np.array) -> float:
        """
        Calculates the distance between vectors x and y using the specified metric.

        Args:
            x (np.array): Vector or array of vectors.
            y (np.array): Single vector.

        Returns:
            float or np.array: Distance(s) between x and y.
        """
        if len(x.shape) == 1:  # If x is a single point
            if self.distance == "euclidian":
                return np.linalg.norm(x - y)
            elif self.distance == "manhattan":
                return np.sum(np.abs(x - y))
        else:  # If x contains multiple points
            if self.distance == "euclidian":
                return np.linalg.norm(x - y.reshape(1, -1), axis=1)
            elif self.distance == "manhattan":
                return np.sum(np.abs(x - y.reshape(1, -1)), axis=1)
        
    def fit(self, X: np.array, max_iters: int = 100, tolerance: float = 1e-4) -> None:
        """
        Fits the model to the data using the KMeans clustering algorithm.

        Args:
            X (np.array): Input dataset.
            max_iters (int, optional): Maximum number of iterations for the algorithm. Defaults to 100.
            tolerance (float, optional): Convergence threshold. Algorithm stops if the change is below this threshold. Defaults to 1e-4.
        """
        # Initialize the centroids
        self._random_initialize_centroids(X)
        
        for _ in range(max_iters):
            old_centroids = self.centroids.copy()  # Copy old centroids for comparison 
            
            # Compute distances and assign each data point to the nearest centroid
            distances = np.array([self._calculate_distance(X, centroid) for centroid in self.centroids])
            labels = np.argmin(distances, axis=0)
            
            # Recompute the centroids
            for i in range(self.k):
                self.centroids[i] = X[labels == i].mean(axis=0)
                
            # Check for convergence
            if np.all(np.abs(self.centroids - old_centroids) < tolerance):
                break
            
    def predict(self, X: np.array) -> np.array:
        """
        Assigns each data point in X to the nearest centroid.

        Args:
            X (np.array): Input dataset.

        Returns:
            np.array: Array of cluster labels.
        """
        distances = np.array([self._calculate_distance(X, centroid) for centroid in self.centroids])  # Compute distances for each data point
        return np.argmin(distances, axis=0)  # Return the closest centroid's label for each data point
