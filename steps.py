from abc import ABC, abstractmethod
import operations as op
from simulation import SimulationModel
import numpy as np

class Step(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def execute(self, model):
        pass

class DiscardPreviousGrid(Step):

    def execute(self, model: SimulationModel):
        model.grid.reset()

class ComputeInterpolationValues(Step):

    def execute(self, model: SimulationModel):
        xp_arr = model.material_points.props['position']
        grid = model.grid
        np_arr = op.getNp(xp_arr, grid.deltas, grid.bounds, grid.N)
        grid_support_node_indexes = np.zeros((xp_arr.shape[0], 16))
        op.grid_support_indexes(np_arr, grid.N, grid.mode, grid_support_node_indexes)
        xn_arr = np.zeros((1))
        snp_arr = np.zeros((1))
        #xn arr
        op.calc_Snp_arr(xn_arr, xp_arr, model.grid.L, model.material_points.lp, snp_arr)

class InitializeGridState(Step):

    def execute(self):
        pass

class UpdateStrainStressUSF(Step):

    def execute(self):
        pass

class ComputeInternalAndExternalForces(Step):

    def execute(self):
        pass

class ComputeRateOfMomentumAndUpdateNodes(Step):

    def execute(self):
        pass

class UpdateMaterialPoints(Step):

    def execute(self):
        pass

class UpdateStrainStressUSL(Step):

    def execute(self):
        pass


