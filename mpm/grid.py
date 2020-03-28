import numpy as np
import matplotlib.pyplot as plt
import mpm.common.constants as c

class Grid:
    props = {}
    Nx = 0
    Ny = 0
    Nz = 0
    L = 0
    storage = {}

    def __init__(self, bounds, deltas, props):
        self.bounds = bounds
        self.deltas = deltas
        self.ndims = bounds.shape[0]
        if (self.ndims == 1):
            self.mode = c.MODE_1D
        elif (self.ndims == 2):
            self.mode = c.MODE_2D

        bounds_x = bounds[0]
        self.Nx = int(((bounds_x[1] - bounds_x[0]) + 1) / deltas[0])
        num_nodes = self.Nx
        node_dims = [self.Nx]

        if (self.ndims > 1):
            bounds_y = bounds[1]
            self.Ny = int(((bounds_y[1] - bounds_y[0]) + 1) / deltas[1])
            num_nodes *= self.Ny
            node_dims.append(self.Ny)

        if (self.ndims > 2):
            bounds_z = bounds[2]
            self.Nz = int(((bounds_z[1] - bounds_z[0]) + 1) / deltas[2])
            num_nodes *= self.Nz
            node_dims.append(self.Nz)

        num_nodes = int(num_nodes)
        self.N = np.array(node_dims, dtype=np.int)
        self.num_nodes = num_nodes
        self.L = deltas[0]

        for prop in props:
            self.add_node_property_storage(prop, float, props[prop])
        self._setup_position()

    def add_node_property_storage(self, name, dtype, nelems=1):
        propShape = (self.num_nodes, nelems)
        self.storage[name] = np.zeros(propShape, dtype=dtype)

    def reset(self):
        for propName in self.storage:
            self.storage[propName].fill(0)

    def _setup_position(self):
        dims_grid = []
        if (self.Nx):
            bounds_x = self.bounds[0]
            nx_grid = np.linspace(bounds_x[0], bounds_x[1], self.Nx)
            dims_grid.append(nx_grid)

        if (self.Ny):
            bounds_y = self.bounds[1]
            ny_grid = np.linspace(bounds_y[0], bounds_y[1], self.Ny)
            dims_grid.append(ny_grid)

        if (self.Nz):
            bounds_z = self.bounds[2]
            nz_grid = np.linspace(bounds_z[0], bounds_z[1], self.Nz)
            dims_grid.append(nz_grid)

        mgrid = np.meshgrid(*dims_grid)
        flat_mgrid = [mgrid_ele.reshape(mgrid_ele.size) for mgrid_ele in mgrid]
        # join x, y, z elements
        position = list(zip(*flat_mgrid))
        self.props['position'] = np.array(position)

if (__name__ == '__main__'):
    grid = Grid(bounds=np.array([(1, 10), (1, 10)]), deltas=np.array([0.5, 1]), props={ 'mass': 1, 'momentum': 1 })
    xticks = np.arange(grid.bounds[0][0], grid.bounds[0][1] + grid.deltas[0], grid.deltas[0])
    yticks = np.arange(grid.bounds[1][0], grid.bounds[1][1] + grid.deltas[1], grid.deltas[1])
    fig, ax = plt.subplots()
    # plt.scatter(x, y)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    plt.grid(True)
    plt.show()

