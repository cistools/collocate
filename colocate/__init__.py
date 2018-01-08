import logging

import numpy as np
from xarray import Dataset, DataArray

from colocate.sepconstraint import SepConstraint
from colocate.kernels import nn_horizontal_kdtree

__version__ = '0.0.1'


def collocate(sample, data, kernel=None, index=None, missing_data_for_missing_sample=True, **kwargs):
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
        for var in data.data_vars.values():
            output.update(collocate(sample, var, kernel, index))
        return output

    # Create a flattened dataframe (with a standard name) to work with the data
    flattened_data = data.to_dataframe('vals')
    flattened_sample = sample.to_dataframe('vals')

    index.index_data(flattened_data)

    logging.info("--> Collocating...")

    # Create output arrays.
    var_set_details = kernel.get_variable_details(data.name or 'var', data.attrs.get('long_name', ''),
                                                  data.attrs.get('standard_name', ''), data.attrs.get('units', None))

    # Create the output dataset
    result = flattened_sample.copy().drop('vals', axis=1)
    # Set the empty columns
    result = result.reindex(columns=result.columns.tolist() + [v[0] for v in var_set_details])

    # If we just want the nearest point in lat/lon we can shortcut the collocation
    if isinstance(kernel, nn_horizontal_kdtree):
        result.loc[:, var_set_details[0][0]] = kernel.get_value(flattened_sample, flattened_data)
    else:
        for i, point, con_points in index.get_iterator(missing_data_for_missing_sample, flattened_data, flattened_sample):
            result.loc[i, [v[0] for v in var_set_details]] = kernel.get_value(point, con_points.vals)

    # Convert the dataframe back to a dataset
    return_data = Dataset.from_dataframe(result)
    return return_data
