import logging
from xarray import Dataset
from collocate.sepconstraint import SepConstraint
from collocate.kernels import nn_horizontal_kdtree, mean

__version__ = '0.1.0'


def collocate(sample, data, kernel=None, constraint=None, missing_data_for_missing_sample=True, **kwargs):
    """
    Find all the data points which meet the specified criteria for each sample point and apply the kernel. The kernel
    may return multiple DataArrays but they will all have the same (possibly flattened) shape as the sample points.

    :param xr.DataArray points: The sample points
    :param xr.DataArray data: The data to collocate from
    :param Kernel kernel: An instance of a Kernel subclass which takes a number of points and returns one or more values
    :param SepConstraint constraint: An optional, pre-constructed constraint object
    :return xr.Dataset: With the same coordinates as the sample and one DataArray for each Kernel return value
    """
    constraint = constraint or SepConstraint(**kwargs)

    # We can have any kernel, default to moments
    kernel = kernel or mean()

    if isinstance(data, Dataset):
        # Indexing (for SepConstraint) will only take place on the first iteration,
        # so we really can just call this method recursively if we've got a list of data.
        output = Dataset()
        for var in data.data_vars.values():
            output.update(collocate(sample, var, kernel, constraint))
        return output

    # Create a flattened dataframe (with a standard name) to work with the data
    flattened_data = data.to_dataframe('vals')
    flattened_sample = sample.to_dataframe('vals')

    constraint.index_data(flattened_data)

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
        for i, point, con_points in constraint.get_iterator(missing_data_for_missing_sample, flattened_data, flattened_sample):
            result.loc[i, [v[0] for v in var_set_details]] = kernel.get_value(point, con_points)

    # Convert the dataframe back to a dataset
    return_data = Dataset.from_dataframe(result)
    return return_data
