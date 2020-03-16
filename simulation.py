from abc import ABC, abstractmethod
from typing import List
import numpy as np
import numba
import common
import steps
from grid import Grid


class MaterialPoints:
    storage = {}
    lp = 0

    def __init__(self, points, spec):
        self.num_points = points.shape[0]

        for prop in spec.props:
            self.add_material_property_storage(prop, float, spec[prop])

        self.lp = spec.lp
        for i, point in enumerate(points):
            for prop in spec:
                self.storage[prop][i] = point[prop]

    def add_material_property_storage(self, name, dtype, nelems=1):
        propShape = (self.num_points, nelems)
        self.storage[name] = np.zeros(propShape, dtype=dtype)

class SimulationParams:
    delta_t = 0.1
    tf = 1.0
    stress_strategy = common.USF

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
        self.add_step(steps.DiscardPreviousGrid())
        self.add_step(steps.ComputeInterpolationValues())
        self.add_step(steps.InitializeGridState())
        if (self.sim_params.stress_strategy == common.USF):
            self.add_step(steps.UpdateStrainStressUSF)
        self.add_step(steps.ComputeInterpolationValues())
        self.add_step(steps.ComputeRateOfMomentumAndUpdateNodes())
        self.add_step(steps.UpdateMaterialPoints())
        if (self.sim_params.stress_strategy == common.USL):
            self.add_step(steps.UpdateStrainStressUSL)
