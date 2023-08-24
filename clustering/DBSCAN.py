import numpy as np
from collections import deque

def _euclidian_distance(x: np.array, y: np.array) -> float:
    """returns the euclidian distance between two vectors"""
    return np.linalg.norm(x - y)

class DBSCAN:
    
    
    def __init__(self, max_radius: float, min_points: int, distance: str = "euclidian") -> None:
        self.max_radius = max_radius
        self.min_points = min_points
        if distance == "euclidian":
            self.distance = _euclidian_distance
    
    
    def _find_neighbours(self, X: np.array, x_i: int) -> list:
        """
        Receives the dataset and an index of this dataset (a point). Then returns a list with the indicies of points that are neighbours to x_i. 
        To be a neighbour to x_i, it needs to be within max_radius of it.

        Args:
            X (np.array): _description_
            x_i (int): _description_

        Returns:
            list: _description_
        """
        return [index for index, point in enumerate(X) if (_euclidian_distance(point, X[x_i]) <= self.max_radius) and (index != x_i)]
    
    
    def _expand_cluster(self, X: np.array, x_i: int, cluster_label: int, visited_points: set, labels_list: list) -> None:
        """_summary_

        Args:
            X (np.array): _description_
            x_i (int): _description_
            cluster_label (int): _description_
            visited_points (set): _description_
            labels_list (list): _description_
        """
        points_to_visit = deque([x_i]) # deque to hold the indicies of the points to visit, starts with x_i. better because of performance
        
        # iterates while points_to_visit is not empty
        while len(points_to_visit) > 0:
            
            # removes the first point from the list 
            current_point_index = points_to_visit.popleft()
            
            # Assign the current point to the cluster (whether it's a core point or a border point)
            labels_list[current_point_index] = cluster_label
                
            # finds the indexes for the neighbours of this point
            neighbours = self._find_neighbours(X, current_point_index)
            
            # checks if point has enough neighbours
            if len(neighbours) >= self.min_points:
                
                # iterates through every neighbour
                for neighbour_index in neighbours:
                    
                    # if i haven't looked at this neighbour 
                    if neighbour_index not in visited_points:
                        
                        # mark neighbour as visited
                        visited_points.add(neighbour_index)
                        
                        # get this neighbour's neighbours
                        neighbours_of_neighbour = self._find_neighbours(X, neighbour_index)
                        
                        # checks if this neighbour is a core point
                        if len(neighbours_of_neighbour) >= self.min_points:
                            
                            # adds the neighbour's neighbours to the points to visit set
                            points_to_visit.extend(neighbours_of_neighbour)
                            
                    # If the neighbor hasn't been assigned to a cluster yet, assign it to the current cluster
                    if labels_list[neighbour_index] == -1:
                        labels_list[neighbour_index] = cluster_label

            
            
            
            
    