"""
Module for creating mock, dummies and fakes
"""
from nose.tools import raises
import numpy as np
import pandas as pd
import numpy.ma as ma
import xarray as xr


def make_dummy_time_series(len=10, as_data_frame=False):
    """
    Create a time series of ungridded data of length len, with a single lat/lon coordinate (65.2, -12.1)
    :param len: length of teh time series and associated data
    :return:
    """

    data = 15 + 8 * np.random.randn(1, 1, len)
    times = pd.date_range('2014-09-06', periods=len)
    lon = [[-12.1]]
    lat = [[65.2]]

    da = xr.DataArray(data, coords={'longitude': (['x', 'y'], lon), 'latitude': (['x', 'y'], lat), 'time': times})
    if as_data_frame:
        return da.to_dataframe('vals')
    else:
        return da


def make_dummy_sample_points(data=None, as_data_frame=False, **kwargs):
    # Find the length of the first array
    n_values = len(list(kwargs.values())[0])
    data = data if data is not None else np.empty((n_values,))
    da = xr.DataArray(data, dims=['obs'], coords={k: (['obs'], v) for k, v in kwargs.items()})
    if as_data_frame:
        return da.to_dataframe('vals')
    else:
        return da


def make_regular_2d_ungridded_data(lat_dim_length=5, lat_min=-10, lat_max=10, lon_dim_length=3, lon_min=-5, lon_max=5,
                                   data_offset=0, mask=None, as_data_frame=False):
    """
    Makes a well defined ungridded data object. If no arguments are supplied, it is of shape 5x3 with data as follows
        array([[1,2,3],
               [4,5,6],
               [7,8,9],
               [10,11,12],
               [13,14,15]])
        and coordinates in latitude:
        array([[-10,-10,-10],
               [-5,-5,-5],
               [0,0,0],
               [5,5,5],
               [10,10,10]])
        longitude:
        array([[-5,0,5],
               [-5,0,5],
               [-5,0,5],
               [-5,0,5],
               [-5,0,5]])

    They are different lengths to make it easier to distinguish. Note the latitude increases
    as you step through the array in order - so downwards as it's written above

    :param lat_dim_length: number of latitude coordinate values
    :param lat_min: minimum latitude coordinate value
    :param lat_max: maximum latitude coordinate value
    :param lon_dim_length: number of longitude coordinate values
    :param lon_min: minimum longitude coordinate value
    :param lon_max: maximum longitude coordinate value
    :param data_offset: value by which to increase data values
    :param ndarray mask: missing value mask
    :return: UngriddedData object as specified
    """

    x_points = np.linspace(lat_min, lat_max, lat_dim_length)
    y_points = np.linspace(lon_min, lon_max, lon_dim_length)
    y, x = np.meshgrid(y_points, x_points)

    data = np.reshape(np.arange(lat_dim_length * lon_dim_length) + data_offset + 1.0, (lat_dim_length, lon_dim_length))
    if mask is not None:
        data = np.ma.asarray(data)
        data.mask = mask

    da = xr.DataArray(data, coords={'longitude': (['x', 'y'], x), 'latitude': (['x', 'y'], y)},
                        dims=['x', 'y'])
    if as_data_frame:
        return da.to_dataframe('vals')
    else:
        return da


def make_regular_2d_ungridded_data_with_missing_values(**kwargs):
    """
        Makes a well defined ungridded data object of shape 5x3 with data as follows, in which M denotes a missing
        value:
        array([[1,2,3],
               [4,M,6],
               [7,8,M],
               [10,11,12],
               [M,14,15]])
        and coordinates in latitude:
        array([[-10,-10,-10],
               [-5,-5,-5],
               [0,0,0],
               [5,5,5],
               [10,10,10]])
        longitude:
        array([[-5,0,5],
               [-5,0,5],
               [-5,0,5],
               [-5,0,5],
               [-5,0,5]])

        They are different lengths to make it easier to distinguish. Note the latitude increases
        as you step through the array in order - so downwards as it's written above
    """
    mask = np.zeros((kwargs.get('lat_dim_length', 5), kwargs.get('lon_dim_length', 3)), dtype=bool)
    mask[[4, 8, 12]] = True

    return make_regular_2d_ungridded_data(mask=mask, **kwargs)


def make_regular_2d_with_time_ungridded_data(as_data_frame=False):
    """
        Makes a well defined ungridded data object of shape 5x3 with data as follows
        array([[1,2,3],
               [4,5,6],
               [7,8,9],
               [10,11,12],
               [13,14,15]])
        and coordinates in latitude:
        array([[-10,-10,-10],
               [-5,-5,-5],
               [0,0,0],
               [5,5,5],
               [10,10,10]])
        longitude:
        array([[-5,0,5],
               [-5,0,5],
               [-5,0,5],
               [-5,0,5],
               [-5,0,5]])
        time: np.array( [ 15 day increments from 27th August 1984 ] )
        They are different lengths to make it easier to distinguish. Note the latitude increases
        as you step through the array in order - so downwards as it's written above
    """

    x_points = np.arange(-10, 11, 5)
    y_points = np.arange(-5, 6, 5)
    y, x = np.meshgrid(y_points, x_points)

    times = np.reshape(np.asarray(pd.date_range('1984-08-27', periods=15)), (5, 3))

    data = np.reshape(np.arange(15) + 1.0, (5, 3))

    da = xr.DataArray(data, coords={'longitude': (['x', 'y'], x), 'latitude': (['x', 'y'], y),
                                      'time': (['x', 'y'], times)},
                        dims=['x', 'y'])
    if as_data_frame:
        return da.to_dataframe('vals')
    else:
        return da


def make_regular_4d_ungridded_data(as_data_frame=False):
    """
        Makes a well defined ungridded data object of shape 10x5 with data as follows

        data:
        [[  1.   2.   3.   4.   5.]
         [  6.   7.   8.   9.  10.]
         [ 11.  12.  13.  14.  15.]
         [ 16.  17.  18.  19.  20.]
         [ 21.  22.  23.  24.  25.]
         [ 26.  27.  28.  29.  30.]
         [ 31.  32.  33.  34.  35.]
         [ 36.  37.  38.  39.  40.]
         [ 41.  42.  43.  44.  45.]
         [ 46.  47.  48.  49.  50.]]

        latitude:
        [-10.  -5.   0.   5.  10.]

        longitude:
        [-5.  -2.5  0.   2.5  5. ]

        altitude:
        [[  0. ]
         [ 10. ]
         [ 20. ]
         [ 30. ]
         [ 40. ]
         [ 50. ]
         [ 60. ]
         [ 70. ]
         [ 80. ]
         [ 90. ]]

        pressure:
        [[  4.   4.   4.   4.   4.]
         [ 16.  16.  16.  16.  16.]
         [ 20.  20.  20.  20.  20.]
         [ 30.  30.  30.  30.  30.]
         [ 40.  40.  40.  40.  40.]
         [ 50.  50.  50.  50.  50.]
         [ 60.  60.  60.  60.  60.]
         [ 70.  70.  70.  70.  70.]
         [ 80.  80.  80.  80.  80.]
         [ 90.  90.  90.  90.  90.]]

        time:
        [1984-08-27 1984-08-28 1984-08-29 1984-08-30 1984-08-31]

        They are shaped to represent a typical lidar type satelite data set.
    """

    x_points = np.linspace(-10, 10, 5)
    y_points = np.linspace(-5, 5, 5)
    times = pd.date_range('1984-08-27', periods=5)

    alt = np.linspace(0, 90, 10)
    data = np.reshape(np.arange(50) + 1.0, (10, 5))

    _, pres = np.meshgrid(times, alt)
    pres[0, :] = 4
    pres[1, :] = 16

    # a = AuxCoord(a, standard_name='altitude', units='meters')
    # x = AuxCoord(x, standard_name='latitude', units='degrees')
    # y = AuxCoord(y, standard_name='longitude', units='degrees')
    # p = AuxCoord(p, standard_name='air_pressure', units='Pa')
    # t = AuxCoord(t, standard_name='time', units=cis_standard_time_unit)

    # return Cube(data, standard_name='rainfall_flux', long_name="TOTAL RAINFALL RATE: LS+CONV KG/M2/S",
    #             units="kg m-2 s-1", dim_coords_and_dims=[(DimCoord(range(10), var_name="z"), 0),
    #                                                      (DimCoord(range(5), var_name="t"), 1)],
    #             aux_coords_and_dims=[(x, (0, 1)), (y, (0, 1)), (t, (0, 1)), (a, (0, 1)), (p, (0, 1))])
    da = xr.DataArray(data, coords={'altitude': (['z'], alt),
                                      'latitude': (['t'], x_points),
                                      'longitude': (['t'], y_points),
                                      'air_pressure': (['z', 't'], pres),
                                      'time': (['t'], times)},
                        dims=['z', 't'])
    if as_data_frame:
        return da.to_dataframe('vals')
    else:
        return da