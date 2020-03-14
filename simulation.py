import numpy as np
import numba
from abc import ABC, abstractmethod
from typing import List
import operations as op
from grid import Grid

USF = 'USF'
USL = 'USL'

class MaterialPoints:
    props = {}

    def __init__(self, points, props):
        self.num_points = points.shape[0]

        for prop in props:
            self.add_material_property(prop, float, props[prop])

        for i, point in enumerate(points):
            for prop in props:
                self.props[prop][i] = point[prop]

    def add_material_property(self, name, dtype, nelems=1):
        propShape = (self.num_points, nelems)
        self.props[name] = np.zeros(propShape, dtype=dtype)

class SimulationParams:
    delta_t = 0.1
    tf = 1.0
    stress_strategy = USF

class SimulationModel:
    grid = None
    material_points = None

    def __init__(self, grid: Grid, material_points):
        self.grid = grid
        self.material_points = material_points

class Simulation(ABC):
    steps = []
    sim_params: SimulationParams = None
    sim_model: SimulationModel = None
    t = 0

    def __init__(self):
        pass

    def run(self, simulation_params: SimulationParams, simulation_model: SimulationModel):
        self.initialize(simulation_params, simulation_model)

        while(self.t < self.sim_params.tf):
            for step in self.steps:
                step.execute(self)
            self.t += self.sim_params.delta_t

    def initialize(self, simulation_params: SimulationParams, simulation_model: SimulationModel):
        self.sim_params = simulation_params
        self.sim_model = simulation_model
        self.t = 0
        self.setup_simulation()

    def add_step(self, step):
        self.steps.append(step)

    @abstractmethod
    def setup_simulation(self):
        pass

class GMPMSimulation(Simulation):

    def setup_simulation(self):
        self.add_step(DiscardPreviousGrid())
        self.add_step(ComputeInterpolationValues())
        self.add_step(InitializeGridState())
        if (self.sim_params.stress_strategy == USF):
            self.add_step(UpdateStrainStressUSF)
        self.add_step(ComputeInterpolationValues())
        self.add_step(ComputeRateOfMomentumAndUpdateNodes())
        self.add_step(UpdateMaterialPoints())
        if (self.sim_params.stress_strategy == USL):
            self.add_step(UpdateStrainStressUSL)

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
        mat_point_position = model.material_points.props['position']
        grid = model.grid
        op.getNp(mat_point_position, grid.deltas, grid.bounds, grid.N)

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

if (__name__ == '__main__'):
    ####
    material_points_props = {
        'position': 1,
        'mass': 1,
        'velocity': 1,
        'strain': 1,
        'stress': 1,
        'body_masses': 1,
    }

    material_points_input = np.array([{
        'position': 2,
        'mass': 1,
        'velocity': 1,
        'strain': 1,
        'stress': 1,
        'body_masses': 1,
    }])
    grid_nodes_props = {
        'position': 1,
        'mass': 1,
        'momentum': 1,
        'internal_force': 1,
        'external_force': 1,
        'rate_of_momenta': 1
    }
    ###

    material_points = MaterialPoints(material_points_input, material_points_props)
    grid = Grid(bounds=np.array([(1, 10)]), deltas=np.array([1]), props=grid_nodes_props)
    model = SimulationModel(grid, material_points)
    params = SimulationParams()

    simulation = GMPMSimulation()
    # simulation.run(params, model)
