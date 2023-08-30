import numpy as np 
from .distances import get_distance

class DBSCAN:
    
    
    def __init__(self, max_distance: float, min_points: int, distance_metric: str = "euclidian") -> None:
        self.max_distance = max_distance
        self.min_points = min_points
        self.distance = get_distance(distance_metric)
        self.clusters = []
        
        
    def points_within_range(self, X: np.array, point: np.array) -> np.array:
        """
        Retrieve points from dataset `X` that are within `self.max_distance` from the analyzed `point`, excluding the `point` itself.

        Args:
            X (np.array): The dataset.
            point (np.array): The analyzed point.

        Returns:
            np.array: An array of points (inner arrays) from `X` that are within `self.max_distance` from the analyzed `point`.
        """     
        # filters the point from the dataset X
        mask = np.all(X != point, axis=1)
        filtered_dataset = X[mask]
        
        # calculates the distance between all points in X from point
        distances_to_point = self.distance(filtered_dataset, point)
        
        # filter only the points that are within self.max_distance from point
        within_range = filtered_dataset[distances_to_point <= self.max_distance]
        
        return within_range
        
        
    def expand_cluster(self, X: np.array, point: np.array, point_neighbours: np.array) -> np.array:
        """
        Expands a cluster from a given point. Given a point, we want to create a cluster containing all the points that are reachable from it. To know if a point is reachable from other point,
        it has to be within self.max_distance from this other point, or it has to be within self.max_distance from a core point that is reachable from this other point. Therefore, what this 
        function does is the following: Starting from point x, we create a cluster containing only this point x. Then, we go through every single neighbour n (n is within self.max_distance from x) 
        and check if it's a core point (there are at least self.min_points within self.max_distance from n). If it is, then all points that are reachable from this neighbour n are also reachable 
        from the original point x, therefore, they should belong to our cluster. So we take all this neighbour point neighbours (neighbours of n) and mark them as neighbour from x (because they 
        are reachable from x). Now, we just updated the list of neighbours from the orinal point x. In the end, we add the neighbours to our cluster. We do that until there are no more neighbours
        to look, returning the final cluster. 

        Args:
            X (np.array): The dataset. 
            point (np.array): The point we will start the cluster expansion from. All points reachable from this point will be on our cluster. 
            point_neighbours (np.array): Array of points that are directly reachable (within self.max_distance) from the original point.

        Returns:
            cluster (np.array): A numpy array of numpy arrays. Each inner array is a point that is reachable from the original point.
        """        
        # creates the cluster by creating an empty numpy array of shape (0 (lines), n_features (columns))
        n_features = point.shape[0] # point is of shape (x,) where x is the number of features
        cluster = np.empty((0, n_features))
        
        # Appends the initial point to the cluster
        cluster = np.vstack([cluster, point])
        
        # iterates through all the neighbors from the point, checking if they are a core point (then we add their neighbors to our cluster) or not (then they are a border point)
        for i, neighbour in enumerate(point_neighbours):
            neighbour_neighbours = self.points_within_range(X, neighbour)  # gives me a numpy array of numpy arrays

            # checks if the point is a core point (if it is, we want to add its neighbors to our cluster)
            if len(neighbour_neighbours) >= self.min_points:
                # if it is:
                # look at each neighbors neighbor
                for new_neighbour in neighbour_neighbours:
                    # if it's not already an original point neighbor and it's not yet in the cluster
                    # the same as "if new_neighbour not in point_neighbours and new_neighbour not in cluster:"
                    if not np.any(np.all(point_neighbours == new_neighbour, axis=1)) and not np.any(np.all(cluster == new_neighbour, axis=1)):
                        # add point to numpy array of the neighbors of the original point
                        point_neighbours = np.vstack([point_neighbours, new_neighbour])

            # if its not: add the point to my cluster (add the point as a border point)
            if not np.any(np.all(cluster == neighbour, axis=1)):
                cluster = np.vstack([cluster, neighbour])

        return cluster
