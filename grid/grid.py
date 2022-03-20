#! usr/bin/env/python3

import numpy as np


def create_simplex_grid(dimension_n: int=3,  grid_ticks_n: int=3) -> np.ndarray:
    """
    Creates a grid of coordinates across a simplex space 

    dimension_n: number of dimensions of the simplex grid
    grid_ticks_n: number of points along each axis for defining the grid

    return: 2-D Numpy array with coordinates evenly spaced as a grid across the 
        simplex; array shape: (number of data points, 'dimension_n')
    """

    # the number of simplex axes with a degree of freedom is the number of 
    #   dimensions minus 1
    dimension_df = dimension_n - 1

    # define grid coordinates for "degrees of freedom" axes, then calculate 
    #   and add remaining dimension
    # NOTE:  this approach -- generating all combinations of grid points, then 
    #   eliminating the impossible ones -- is memory-inefficient
    grid_axes_ticks = [np.linspace(0, 1, grid_ticks_n) for _ in range(dimension_df)]
    # NOTE:  returns dense coordinates; might need to optimize with sparsity
    dim_df_coordinates = np.meshgrid(*grid_axes_ticks, sparse=False)
    last_dim_coordinates = 1 - np.stack(dim_df_coordinates).sum(axis=0)
    dim_df_coordinates.append(last_dim_coordinates)

    # remove all grid coordinates that are not defined on the simplex
    coordinates = np.stack(dim_df_coordinates)
    coordinates = (
        coordinates
        .reshape(1, -1, order='F')
        .reshape(-1, coordinates.shape[0]))
    simplex_mask = coordinates[:, :-1].sum(axis=1) <= 1
    coordinates = coordinates[simplex_mask, :]

    # coordinates should sum to 1 for each data point
    assert (coordinates[:, :-1].sum(axis=1) <= 1).all()
    assert np.allclose(coordinates.sum(axis=1), 1)

    return coordinates


if __name__ == "__main__":
    print(create_simplex_grid(3, 5))
