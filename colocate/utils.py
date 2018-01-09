"""
Iris wrapper and associated utilities
"""
from functools import wraps
from colocate import collocate


def cube_wrapper(xr_func):
    """
    Wrap a function which works on two xarray Datasets with an Cube->Dataset converter to allow calling with an
     two Cube objects. Takes advantage of the cube metadata to perform unification on the two cubes before converting.

    :param xr_func: A (collocation) function which takes two Datasets as its first arguments and returns a Dataset
    :return: A function which takes two Cube objects as its first arguments and returns a Cube object
    """
    from iris.util import unify_time_units
    import xarray as xr

    @wraps
    def cube_func(a, b, *args, **kwargs):

        # Unify the coordinate units
        for a_c in a.coords():
            for b_c in b.coords(standard_name=a_c.standard_name):
                a_c.convert_units(b_c.units)
        # Fix the longitude ranges
        if a.coords(standard_name='longitude'):
            lon_min = a.coord(standard_name='longitude').points.min()
            # First fix the sample points so that they all fall within the same 360 degree longitude range
            _fix_longitude_range(a, lon_min)
            # Then fix the data points so that they fall onto the same 360 degree longitude range as the sample points
            _fix_longitude_range(b, lon_min)

        unify_time_units([a, b])
        # Convert to xarray
        ds_a = xr.DataArray.from_iris(a)
        ds_b = xr.DataArray.from_iris(b)
        # Collocate
        ds = xr_func(ds_a, ds_b, *args, **kwargs)
        # Convert back and return
        res = ds.to_iris()

        return res

    return cube_func


iris_colocate = cube_wrapper(collocate)


def _fix_longitude_range(data_points, range_start):
    """Sets the longitude range of the data points to match that of the sample coordinates.

    :param float range_start: The longitude
    :param iris.cube.Cube data_points: Cube to fix
    """
    from iris.analysis.cartography import wrap_lons

    if data_points.coords('longitude', dim_coords=True) and (len(data_points.shape) > 1):
        # For multidimensional cubes we need to rotate the whole object to keep the points monotonic
        set_longitude_range(data_points, range_start)
    else:
        # But we can just wrap auxilliary longitude coordinates
        data_points.coord('longitude').points = wrap_lons(data_points.coord('longitude').points, range_start, 360)


def set_longitude_range(cube, range_start):
    """Rotates the longitude coordinate array and changes its values by
    360 as necessary to force the values to be within a 360 range starting
    at the specified value, i.e.,
    range_start <= longitude < range_start + 360

    The data array is rotated correspondingly around the dimension
    corresponding to the longitude coordinate.

    :param iris.cube.Cube: The Cube to fix
    :param float range_start: starting value of required longitude range
    """
    import numpy as np
    lon_coord = cube.coords(standard_name="longitude")
    if len(lon_coord) == 0:
        return
    lon_coord = lon_coord[0]
    lon_idx = cube.dim_coords.index(lon_coord)
    # Check if there are bounds which we will need to wrap as well
    roll_bounds = (lon_coord.bounds is not None) and (lon_coord.bounds.size != 0)
    idx1 = np.searchsorted(lon_coord.points, range_start)
    idx2 = np.searchsorted(lon_coord.points, range_start + 360.)
    shift = 0
    new_lon_points = None
    new_lon_bounds = None
    if 0 < idx1 < len(lon_coord.points):
        shift = -idx1
        lon_min = lon_coord.points[idx1]
        new_lon_points = np.roll(lon_coord.points, shift, 0)
        # Calculate which indices need 360 adding to them...
        indices_to_shift_value_of = new_lon_points < lon_min
        # ... then, add 360 to all those longitude values
        new_lon_points[indices_to_shift_value_of] += 360.0
        if roll_bounds:
            # If the coordinate has bounds then roll those as well
            new_lon_bounds = np.roll(lon_coord.bounds, shift, 0)
            # And shift all of the bounds (upper and lower) for those points which we had to shift. We can't do the
            # check independently because there may be cases where an upper or lower bound falls outside of the
            # 360 range, we leave those as they are to preserve monotonicity. See e.g.
            # test_set_longitude_bounds_wrap_at_360
            new_lon_bounds[indices_to_shift_value_of] += 360.0
    elif 0 < idx2 < len(lon_coord.points):
        shift = len(lon_coord.points) - idx2
        lon_max = lon_coord.points[idx2]
        new_lon_points = np.roll(lon_coord.points, shift, 0)
        indices_to_shift_value_of = new_lon_points >= lon_max
        new_lon_points[indices_to_shift_value_of] -= 360.0
        if roll_bounds:
            new_lon_bounds = np.roll(lon_coord.bounds, shift, 0)
            # See comment above re preserving monotinicity.
            new_lon_bounds[indices_to_shift_value_of] -= 360.0
    if shift != 0:
        # Ensure we also roll any auxilliary coordinates
        for aux_coord in cube.aux_coords:
            # Find all of the data dimensions which the auxiliary coordinate spans...
            dims = cube.coord_dims(aux_coord)
            # .. and check if longitude is one of those dimensions
            if lon_idx in dims:
                # Now roll the axis of the auxiliary coordinate which is associated with the longitude data
                # dimension: dims.index(lon_idx)
                new_points = np.roll(aux_coord.points, shift, dims.index(lon_idx))
                aux_coord.points = new_points
        # Now roll the data itcube
        new_data = np.roll(cube.data, shift, lon_idx)
        cube.data = new_data
        # Put the new coordinates back in their relevant places
        cube.dim_coords[lon_idx].points = new_lon_points
        if roll_bounds:
            cube.dim_coords[lon_idx].bounds = new_lon_bounds

