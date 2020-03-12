import numpy as np


def moment(region, m, n):
    height, width = region.shape
    j_coords, i_coords = np.mgrid[:height, :width]
    return (region * i_coords ** m * j_coords ** n).sum()


def central_moment(region, m, n, centroid):
    height, width = region.shape
    j_coords, i_coords = np.mgrid[:height, :width]
    return (region * (i_coords - centroid.i) ** m * (j_coords - centroid.j) ** n).sum()


def covariance_matrix(region, centroid):
    area = moment(region, 0, 0)
    sigma_ii = central_moment(region, 2, 0, centroid) / area
    sigma_jj = central_moment(region, 0, 2, centroid) / area
    sigma_ij = central_moment(region, 1, 1, centroid) / area
    return np.array([[sigma_ii, sigma_ij], [sigma_ij, sigma_jj]])
