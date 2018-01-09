collocate
=========

[![Build Status](https://travis-ci.org/cistools/collocate.svg?branch=master)](https://travis-ci.org/cistools/collocate)

collocate un-structured xarray DataArray's (or Iris Cube's) in arbitrary physical dimensions:

    >>> import numpy as np
    >>> import xarray as xr
    >>> from collocate import collocate

    # Create an 'ungridded' DataArray
    >>> x_points = np.linspace(-10, 10, 5)
    >>> y_points = np.linspace(-5, 5, 3)
    >>> y, x = np.meshgrid(y_points, x_points)

    >>> da = xr.DataArray(np.reshape(np.arange(15), (5, 3)),
                          coords={'longitude': (['x', 'y'], x), 'latitude': (['x', 'y'], y)},
                          dims=['x', 'y'])
    >>> da
     <xarray.DataArray (x: 5, y: 3)>
    array([[ 0,  1,  2],
           [ 3,  4,  5],
           [ 6,  7,  8],
           [ 9, 10, 11],
           [12, 13, 14]])
    Coordinates:
        longitude  (x, y) float64 -10.0 -10.0 -10.0 -5.0 -5.0 -5.0 0.0 0.0 0.0 ...
        latitude   (x, y) float64 -5.0 0.0 5.0 -5.0 0.0 5.0 -5.0 0.0 5.0 -5.0 ...
    Dimensions without coordinates: x, y

    # Create some points to sample at
    >>> points = xr.DataArray(np.zeros(3), dims=['obs'], 
                              coords={'latitude': (['obs'], [0.5, 5.4, 12]),
                              'longitude': (['obs'], [-0.7, 0.2, 3])})
    >> points
    <xarray.DataArray (obs: 3)>
    array([ 0.,  0.,  0.])
    Coordinates:
        latitude   (obs) float64 0.5 5.4 12.0
        longitude  (obs) float64 -0.7 0.2 3.0
    Dimensions without coordinates: obs

    
    # Do the collocation to find the mean value of the data with 500km of each sample point
    >>> collocate(points, da, h_sep=500)
     
    <xarray.Dataset>
    Dimensions:    (obs: 3)
    Coordinates:
      * obs        (obs) int64 0 1 2
    Data variables:
        latitude   (obs) float64 0.5 5.4 12.0
        longitude  (obs) float64 -0.7 0.2 3.0
        var        (obs) float64 5.5 8.0 nan
    

**Warning** this is still a prototype and the API is likely to change!

Contact
-------

Duncan.Watson-Parris@physics.ox.ac.uk
