from scipy.stats import qmc
import numpy as np

def initialize_lhs(lb, ub, num_points):
    # generate LHS points
    sample = qmc.LatinHypercube(d=lb.shape[0], seed=7).random(num_points)
    scaled = qmc.scale(sample, lb, ub)
    return scaled


def calculate_max_distance(points):
    max_distance = 0
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            distance = np.linalg.norm(points[i] - points[j])
            if distance > max_distance:
                max_distance = distance
    return max_distance