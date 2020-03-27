from .base import Step
from .operations import getNp, grid_support_indexes, calc_Snp_arr
from mpm.input import SimulationModel
import numpy as np

class ComputeInterpolationValues(Step):

    def execute(self, model: SimulationModel):
        xp_arr = model.material_points.props['position']
        grid = model.grid
        np_arr = getNp(xp_arr, grid.deltas, grid.bounds, grid.N)
        grid_support_node_indexes = np.zeros((xp_arr.shape[0], 16))
        grid_support_indexes(np_arr, grid.N, grid.mode, grid_support_node_indexes)
        # TODO
        xn_arr = np.zeros((1))
        snp_arr = np.zeros((1))
        #xn arr
        calc_Snp_arr(xn_arr, xp_arr, model.grid.L, model.material_points.lp, snp_arr)
