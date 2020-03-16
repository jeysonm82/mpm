import numba
import numpy as np
from grid import Grid
import common
from common import MODE_2D, MODE_1D

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

    np_arr = npx + npy + npz
    return np_arr

# TODO define if this is useful
def _get_rijk3d(npi, grid_N):
    nx, ny = grid_N[0], grid_N[1]
    nxny = nx *ny
    ri = min(nx -  (npi % nxny) % nx, 4)
    rj = min(ny - (npi % nxny)/nx , 4)
    rk = min(grid_N[2] - npi / nxny, 4)
    return nx, ny, ri, rj, rk


@numba.njit
def grid_support_indexes(np_arr, grid_N, grid_mode, arr_out):
    """Returns all indexes that forms the grid support according to reference node np_"""
    Nx = grid_N[0]
    ri = Nx - np_arr % Nx
    ri[ri > 4] = 4
    if (grid_mode == MODE_2D):
        Ny = grid_N[1]
        rj = Ny - np.trunc(np_arr / Nx) % Ny
        rj[rj > 4] = 4

    # For each np in np_arr
    for ele, np_ in enumerate(np_arr):
        r = 0
        for yi in range(rj[ele]):
            base = Nx * yi + np_
            index_row = np.arange(base, base + ri[ele])
            for c in range(index_row.shape[0]):
                arr_out[ele][r] = index_row[c]
                r += 1

@numba.njit
def snp(xp, xn, lp, L):
    #L is grid delta
    #lp
    d = (xp - xn)

    if d <= -L -lp:
        Snp = 0.0
    elif d <= -L + lp:
        Snp = ((L + lp + d) ** 2.0) / (4.0 * L * lp)
    elif d <= -lp:
        Snp = 1.0 + d/L
    elif d<= lp:
        Snp = 1.0-(d*d+lp*lp)/(2.0*L*lp)
    elif d<= L-lp:
        Snp = 1.0 - d/L
    elif d<= L + lp:
        Snp = ((L + lp - d) ** 2.0) / (4.0 * L * lp)
    else:
        Snp = 0.0

    return Snp

@numba.njit
def calc_Snp(xn, xp, lp, L):
    """Returns Grid shape function Snp value"""
    Snp = 1.0
    for i in range(xp.shape[0]):
        xin, xip = xn[i], xp[i]
        Snp *= snp(xip, xin, lp, L)
    return Snp

@numba.njit
def calc_Snp_arr(xn_arr, xp_arr, lp, L, arr_out):
    for p in range(xp_arr.shape[0]):
        xn = xn_arr[p]
        xp = xp_arr[p]
        arr_out[p] = calc_Snp(xn, xp, lp, L)

@numba.njit
def gnp(xp, xn, lp, L):
    d = (xp - xn)

    if d<=-L-lp: Gnp = 0.0
    elif d>L+lp: Gnp = 0.0
    elif d<=-L+lp: Gnp = (L+d+lp)/(2.0*L*lp)
    elif d<=-lp: Gnp = 1.0/L
    elif d<=lp: Gnp = -d/(L*lp)
    elif d<=L-lp: Gnp = -1.0/L
    else:        Gnp = (-L+d-lp)/(2.0*L*lp)
    return Gnp

@numba.njit
def calc_Gnp(xn, xp, lp, L, Gnp):
    for i in range(xp.shape[0]):
        xin, xip = xn[i], xp[i]
        Gnp[i]= gnp(xip, xin, lp, L) #TODO not sure about this. Gradient should be grad(f) = d(f)/dx + d(f)/dy + d(f)/dz
    return Gnp
