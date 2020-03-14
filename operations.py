import numba
import numpy as np
from grid import Grid

@numba.njit
def getNp(points_position, grid_deltas, grid_bounds, grid_N):
    ndims = grid_N.shape[0]

    x = points_position[:, 0]
    dx = grid_deltas[0]
    xmin = grid_bounds[0][0]
    Nx = grid_N[0]
    npx = np.trunc((x - xmin)/ dx) - 1

    if (ndims > 1):
        y = points_position[:, 1]
        dy = grid_deltas[1]
        ymin = grid_bounds[1][0]
        npy = (np.trunc((y - ymin) / dy) - 1) * Nx
    else:
        npy = np.zeros(npx.shape)

    if (ndims > 2):
        Ny = grid_N[1]
        z = points_position[:, 2]
        dz = grid_deltas[2]
        zmin = grid_bounds[2][0]
        npz = (np.trunc((z - zmin) / dz) - 1) * Nx * Ny
    else:
        npz = np.zeros(npx.shape)

    np_ = npx + npy + npz
    return np_

if (__name__ == '__main__'):
    mat_point_position = np.array([(2, ), (3, ), (4, )])
    grid = Grid(bounds=np.array([(1, 10)]), deltas=np.array([1]), props={ 'mass': 1, 'momentum': 1 })
    print(getNp(mat_point_position, grid.deltas, grid.bounds, grid.N))

    mat_point_position = np.array([(2, 2 ), (3, 3 ), (4, 4 )])
    grid = Grid(bounds=np.array([(1, 10), (1, 10)]), deltas=np.array([1, 1]), props={ 'mass': 1, 'momentum': 1 })
    print(getNp(mat_point_position, grid.deltas, grid.bounds, grid.N))
