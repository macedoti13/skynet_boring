import numpy as np

def _euclidian_distance(x: np.array, y: np.array) -> float:
    """returns the distance between two vectors"""
    return np.linalg.norm(x - y)

class DBSCAN:
    
    
    def __init__(self, max_radius: float, min_points: int, distance: str = "euclidian") -> None:
        self.max_radius = max_radius
        self.min_points = min_points
        if distance == "euclidian":
            self.distance = _euclidian_distance
    
    
    def _find_neighbours(self, X: np.array, x_i: int) -> list:
        """Receives the dataset and an index of this dataset (a point). Then returns a list with the indicies of points that are neighbours to x_i. 
        To be a neighbour to x_i, it needs to be within max_radius of it."""
        return [index for index, point in enumerate(X) if (_euclidian_distance(point, X[x_i]) <= self.max_radius) and (index != x_i)]
    
    
    
    
    