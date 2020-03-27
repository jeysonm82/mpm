from abc import ABC, abstractmethod

class Step(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def execute(self, model):
        pass

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
