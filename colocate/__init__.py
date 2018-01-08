import logging

import numpy as np
from xarray import Dataset, DataArray

from colocate.sepconstraint import SepConstraint
from colocate.kernels import nn_horizontal_kdtree

__version__ = '0.0.1'


def collocate(sample, data, kernel=None, index=None, fill_value=None, missing_data_for_missing_sample=True,
              **kwargs):
    """
    This collocator takes a list of HyperPoints and a data object (currently either Ungridded
    data or a Cube) and returns one new LazyData object with the values as determined by the
    constraint and kernel objects. The metadata for the output LazyData object is copied from
    the input data object.

    :param UngriddedData or UngriddedCoordinates points: Object defining the sample points
    :param UngriddedData data: The source data to collocate from
    :param constraint: An instance of a Constraint subclass which takes a data object and
                       returns a subset of that data based on it's internal parameters
    :param kernel: An instance of a Kernel subclass which takes a number of points and returns
                   a single value
    :return UngriddedData or DataList: Depending on the input
    """
    index = index or SepConstraint(**kwargs)
    # We can have any kernel, default to moments

    if isinstance(data, Dataset):
        # Indexing (for SepConstraintKdTree) will only take place on the first iteration,
        # so we really can just call this method recursively if we've got a list of data.
        output = Dataset()
        for var in data.values():
            output.update(collocate(sample, var, kernel, index))
        return output

    index.index_data(data)

    logging.info("--> Collocating...")

    # Create output arrays.
    var_set_details = kernel.get_variable_details(data.name or 'var', data.attrs.get('long_name', ''),
                                                  data.attrs.get('standard_name', ''), data.attrs.get('units', None))

    sample_points_count = len(sample)
    # Create an empty masked array to store the collocated values. The elements will be unmasked by assignment.
    values = np.ma.masked_all((sample_points_count, len(var_set_details)))
    values.fill_value = fill_value

    logging.info("    {} sample points".format(sample_points_count))
    # Apply constraint and/or kernel to each sample point.

    # If we just want the nearest point in lat/lon we can shortcut the collocation
    if isinstance(kernel, nn_horizontal_kdtree):
        values[:, 0] = kernel.get_value(sample, data)
    else:
        for i, point, con_points in index.get_iterator(missing_data_for_missing_sample, data, sample):
            values[i, :] = kernel.get_value(point, con_points)

    return_data = Dataset()
    for idx, var_details in enumerate(var_set_details):
        new = DataArray(values[:, idx], coords=sample.coords,
                        attrs=dict(long_name=var_details[1], units=var_details[3]))
        return_data[var_details[0]] = new

    return return_data
