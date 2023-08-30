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
        
        
    def expand_cluster(self, point: np.array, point_neighbours: list):
        pass