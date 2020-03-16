import numpy as np
from grid import Grid
from simulation import SimulationModel, SimulationParams, GMPMSimulation, MaterialPoints

if (__name__ == '__main__'):
    ####
    material_points_spec = {
        'lp': 0.15,
        'props': {
            'position': 1,
            'mass': 1,
            'velocity': 1,
            'strain': 1,
            'stress': 1,
            'body_masses': 1,
         }
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

    material_points = MaterialPoints(material_points_input, material_points_spec)
    grid = Grid(bounds=np.array([(1, 10)]), deltas=np.array([1]), props=grid_nodes_props)
    model = SimulationModel(grid, material_points)
    params = SimulationParams()

    simulation = GMPMSimulation()
    # simulation.run(params, model)
