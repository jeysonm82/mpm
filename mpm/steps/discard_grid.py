from .base import Step
from mpm.input import SimulationModel

class DiscardPreviousGrid(Step):

    def execute(self, model: SimulationModel):
        model.grid.reset()
