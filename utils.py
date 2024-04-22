import numpy as np

def euclidean_distance(point1, point2):
    """Calculates the Euclidean distance between two points represented as tuples (latitude, longitude)."""
    return np.linalg.norm(np.array(point1) - np.array(point2))  

def normalize_data(data):
    """Normalizes data to have zero mean and unit variance."""
    return (data - data.mean()) / data.std()