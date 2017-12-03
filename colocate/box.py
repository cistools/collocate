import logging

import numpy as np

from colocate.col_framework import Kernel
from xarray import Dataset, DataArray
from colocate.col import get_kernel


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
    index = index or SepConstraintKdtree(**kwargs)
    # We can have any kernel, default to moments
    kernel = get_kernel(kernel)

    if isinstance(data, list):
        # Indexing and constraints (for SepConstraintKdTree) will only take place on the first iteration,
        # so we really can just call this method recursively if we've got a list of data.
        output = Dataset()
        for var in data:
            output.extend(collocate(sample, var, index, kernel))
        return output

    # Create index if constraint and/or kernel require one.
    coord_map = None
    index.index_data(sample, data, coord_map)

    logging.info("--> Collocating...")

    # Create output arrays.
    var_set_details = kernel.get_variable_details(data.var_name or 'var', data.long_name or '',
                                                  data.standard_name, data.units)

    sample_points_count = len(sample)
    # Create an empty masked array to store the collocated values. The elements will be unmasked by assignment.
    values = np.ma.masked_all((len(var_set_details), sample_points_count))
    values.fill_value = fill_value

    logging.info("    {} sample points".format(sample_points_count))
    # Apply constraint and/or kernel to each sample point.

    # If we just want the nearest point in lat/lon we can shortcut the collocation
    if isinstance(kernel, nn_horizontal_kdtree):
        values[0, :] = kernel.get_value(sample, data)
    else:
        for i, point, con_points in index.get_iterator(missing_data_for_missing_sample, data, sample):

            try:
                values[:, i] = kernel.get_value(point, con_points)
            except ValueError as e:
                pass

    return_data = Dataset()
    for idx, var_details in enumerate(var_set_details):
        new = DataArray(values[idx, :], coords=sample.coords,
                        attrs=dict(var_name=var_details[0], long_name=var_details[1], units=var_details[3]))
        return_data.append(new)

    return return_data


class SepConstraintKdtree:
    """A separation constraint that uses a k-D tree to optimise spatial constraining.
    If no horizontal separation parameter is supplied, this reduces to an exhaustive
    search using the other parameter(s).
    """

    def __init__(self, h_sep=None, a_sep=None, p_sep=None, t_sep=None):

        self.haversine_distance_kd_tree_index = False

        super(SepConstraintKdtree, self).__init__()

        self._index_cache = {}
        self.checks = []
        if h_sep is not None:
            self.h_sep = h_sep
            self.haversine_distance_kd_tree_index = None
        else:
            self.h_sep = None

        if a_sep is not None:
            self.a_sep = a_sep
            self.checks.append(self.alt_constraint)
        if p_sep is not None:
            try:
                self.p_sep = float(p_sep)
            except:
                raise ValueError('Separation Constraint p_sep must be a valid float')
            self.checks.append(self.pressure_constraint)
        if t_sep is not None:
            self.t_sep = t_sep
            self.checks.append(self.time_constraint)

    def time_constraint(self, points, ref_point):
        return np.nonzero(np.abs(points.time - ref_point.time) < self.t_sep)[0]

    def alt_constraint(self, points, ref_point):
        return np.nonzero(np.abs(points.altitude - ref_point.altitude) < self.a_sep)[0]

    def pressure_constraint(self, points, ref_point):
        greater_pressures = np.nonzero(((points.air_pressure / ref_point.air_pressure) < self.p_sep) &
                                       (points.air_pressure > ref_point.air_pressure))[0]
        lesser_pressures = np.nonzero(((ref_point.air_pressure / points.air_pressure) < self.p_sep) &
                                      (points.air_pressure <= ref_point.air_pressure))[0]
        return np.concatenate([lesser_pressures, greater_pressures])

    def constrain_points(self, ref_point, data):
        if self.haversine_distance_kd_tree_index and self.h_sep:
            point_indices = self._get_cached_indices(ref_point)
            if point_indices is None:
                point_indices = self.haversine_distance_kd_tree_index.find_points_within_distance(ref_point, self.h_sep)
                self._add_cached_indices(ref_point, point_indices)
            con_points = data.iloc[point_indices]
        else:
            con_points = data
        for check in self.checks:
            con_points = con_points.iloc[check(con_points, ref_point)]

        return con_points

    def _get_cached_indices(self, ref_point):
        # Don't use the value as a key (it's both irrelevant and un-hashable)
        return self._index_cache.get(tuple(ref_point[['latitude', 'longitude']].values), None)

    def _add_cached_indices(self, ref_point, indices):
        # Don't use the value as a key (it's both irrelevant and un-hashable)
        self._index_cache[tuple(ref_point[['latitude', 'longitude']].values)] = indices

    def get_iterator(self, missing_data_for_missing_sample, data_points, points):
        indices = False

        iterator = index_iterator_nditer(points, not missing_data_for_missing_sample)

        if self.haversine_distance_kd_tree_index and self.h_sep:
            indices = self.haversine_distance_kd_tree_index.find_points_within_distance_sample(points, self.h_sep)

        for i in iterator:
            p = points.iloc[0]
            if indices:
                # Note that data_points has to be a dataframe at this point because of the indexing
                d_points = data_points[indices[i]]
            else:
                d_points = data_points
            for check in self.checks:
                con_points_indices = check(d_points, p)
                d_points = d_points[con_points_indices]

            yield i, p, d_points


def index_iterator_nditer(points, include_masked=True):
    """Iterates over the indexes of a multi-dimensional array of a specified shape.
    The last index changes most rapidly.

    :param points: array to iterate over
    :param include_masked: iterate over masked elements
    :return: yields tuples of array indexes
    """

    num_cells = np.product(points.data.shape)
    cell_count = 0
    cell_total = 0

    it = np.nditer(points.data, flags=['multi_index'])
    while not it.finished:
        if include_masked or it[0] is not np.ma.masked:
            yield it.multi_index

        it.iternext()

        # Log progress periodically.
        if cell_count == 10000:
            cell_total += 1
            number_cells_processed = cell_total * 10000
            logging.info("    Processed %d points of %d (%d%%)", number_cells_processed, num_cells,
                         int(number_cells_processed * 100 / num_cells))
            cell_count = 0
        cell_count += 1


class nn_horizontal(Kernel):
    def get_value(self, point, data):
        """
            Collocation using nearest neighbours along the face of the earth where both points and
              data are a list of HyperPoints. The default point is the first point.
        """
        from colocate.kdtree import haversine
        iterator = data.iterrows()
        try:
            nearest_point = next(iterator)[1]
        except StopIteration:
            # No points to check
            raise ValueError
        for idx, data_point in iterator:
            if (haversine(point, nearest_point) > haversine(point, data_point)):
                nearest_point = data_point
        return nearest_point.vals


class nn_horizontal_kdtree(Kernel):
    def __init__(self):
        self.haversine_distance_kd_tree_index = None

    def get_value(self, points, data):
        """
        Collocation using nearest neighbours along the face of the earth using a k-D tree index.
        """
        nearest_index = self.haversine_distance_kd_tree_index.find_nearest_point(points)
        nearest_points = data.iloc[nearest_index]
        return nearest_points.vals.values


class nn_altitude(Kernel):
    def get_value(self, point, data):
        """
            Collocation using nearest neighbours in altitude, where both points and
              data are a list of HyperPoints. The default point is the first point.
        """
        iterator = data.iterrows()
        try:
            nearest_point = next(iterator)[1]
        except StopIteration:
            # No points to check
            raise ValueError
        for idx, data_point in iterator:
            if abs(point.altitude - nearest_point.altitude) > abs(point.altitude - data_point.altitude):
                nearest_point = data_point
        return nearest_point.vals


class nn_pressure(Kernel):


    def get_value(self, point, data):
        """
            Collocation using nearest neighbours in pressure, where both points and
              data are a list of HyperPoints. The default point is the first point.
        """

        def pres_sep(point1, point2):
            """
                Computes the pressure ratio between two points, this is always >= 1.
            """
            if point1.air_pressure > point2.air_pressure:
                return point1.air_pressure / point2.air_pressure
            else:
                return point2.air_pressure / point1.air_pressure

        iterator = data.iterrows()
        try:
            nearest_point = next(iterator)[1]
        except StopIteration:
            # No points to check
            raise ValueError
        for idx, data_point in iterator:
            if pres_sep(point, nearest_point) > pres_sep(point, data_point):
                nearest_point = data_point
        return nearest_point.vals


class nn_time(Kernel):
    def get_value(self, point, data):
        """
            Collocation using nearest neighbours in time, where both points and
              data are a list of HyperPoints. The default point is the first point.
        """
        iterator = data.iterrows()
        try:
            nearest_point = next(iterator)[1]
        except StopIteration:
            # No points to check
            raise ValueError
        for idx, data_point in iterator:
            if abs(point.time - nearest_point.time) > abs(point.time - data_point.time):
                nearest_point = data_point
        return nearest_point.vals
