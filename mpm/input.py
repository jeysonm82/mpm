import mpm.common.constants as c
from mpm.grid import Grid
import numpy as np

class MaterialPoints:
    storage = {}
    lp = 0

    def __init__(self, points, spec):
        self.num_points = points.shape[0]

        spec_props = spec['props']
        for prop in spec_props:
            self.add_material_property_storage(prop, float, spec_props[prop])

        self.lp = spec['lp']
        for i, point in enumerate(points):
            for prop in spec_props:
                self.storage[prop][i] = point[prop]

    def add_material_property_storage(self, name, dtype, nelems=1):
        propShape = (self.num_points, nelems)
        self.storage[name] = np.zeros(propShape, dtype=dtype)

class SimulationModel:
    delta_t = 0.1
    tf = 1.0
    stress_strategy = c.USF
    grid = None
    material_points = None

    def __init__(self, grid: Grid, material_points):
        self.grid = grid
        self.material_points = material_points
