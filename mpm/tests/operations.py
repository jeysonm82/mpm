import unittest
import numpy as np
from mpm.grid import Grid
from mpm.steps.operations import getNp, grid_support_indexes

class TestOperations(unittest.TestCase):

    def test_getNp(self):
        arr_input = np.array([(2.3, 2.3), (1, 1)])
        grid = Grid(bounds=np.array([(0, 5), (0, 5)]), deltas=np.array([1, 1]), props={ 'mass': 1, 'momentum': 1 })
        arr_out = getNp(arr_input, grid.deltas, grid.bounds, grid.N)
        self.assertEqual(arr_out[0], 7)
        self.assertEqual(arr_out[1], 0)

    def test_grid_support_indexes(self):
        arr_input = np.array([(2.3, 2.3), (1, 1)])
        grid = Grid(bounds=np.array([(0, 5), (0, 5)]), deltas=np.array([1, 1]), props={ 'mass': 1, 'momentum': 1 })
        np_arr = getNp(arr_input, grid.deltas, grid.bounds, grid.N)
        arr_out = np.zeros((np_arr.shape[0], 16))
        grid_support_indexes(np_arr, grid.N, grid.mode, arr_out)
        self.assertEqual(arr_out[0][0], 7)
        self.assertEqual(arr_out[0][15], 28)
        self.assertEqual(arr_out[1][0], 0)
        self.assertEqual(arr_out[1][15], 21)

if __name__ == '__main__':
    unittest.main()
