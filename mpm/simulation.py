from abc import ABC, abstractmethod
from typing import List
import numpy as np
import numba
import mpm.common.constants as c
import mpm.steps as steps
from mpm.grid import Grid
from mpm.input import SimulationModel,  MaterialPoints


class Simulation(ABC):
    steps = []
    simulation_model: SimulationModel = None
    t = 0

    def __init__(self):
        pass

    def run(self, simulation_model: SimulationModel):
        self.initialize(simulation_model)

        while(self.t < self.simulation_model.tf):
            for step in self.steps:
                step.execute(self)
            self.t += self.simulation_model.delta_t

    def initialize(self, simulation_model: SimulationModel):
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
        if (self.simulation_model.stress_strategy == c.USF):
            self.add_step(steps.UpdateStrainStressUSF)
        self.add_step(steps.ComputeInterpolationValues())
        self.add_step(steps.ComputeRateOfMomentumAndUpdateNodes())
        self.add_step(steps.UpdateMaterialPoints())
        if (self.simulation_model.stress_strategy == c.USL):
            self.add_step(steps.UpdateStrainStressUSL)
