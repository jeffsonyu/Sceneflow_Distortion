import numpy as np
import os

def array2samples_distance(array1, array2):
    """
    arguments: 
        array1: the array, size: (num_point, num_feature)
        array2: the samples, size: (num_point, num_feature)
    returns:
        distances: each entry is the distance from a sample to array1 
    """
    num_point, num_features = array1.shape
    expanded_array1 = np.tile(array1, (num_point, 1))
    expanded_array2 = np.reshape(
            np.tile(np.expand_dims(array2, 1), 
                    (1, num_point, 1)),
            (-1, num_features))
    distances = np.linalg.norm(expanded_array1-expanded_array2, ord=2, axis=1, keepdims = True)
    distances = np.reshape(distances, (num_point, num_point))
    distances = np.min(distances, axis=1)
    distances = np.mean(distances)
    return distances

def chamfer_distance_numpy(array1, array2):
    num_point, num_features = array1.shape

    av_dist1 = array2samples_distance(array1, array2)
    av_dist2 = array2samples_distance(array2, array1)
    dist = av_dist1 + av_dist2
    return dist