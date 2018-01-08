import logging
import numpy as np


class SepConstraint:
    """A separation constraint that uses a k-D tree to optimise spatial constraining.
    If no horizontal separation parameter is supplied, this reduces to an exhaustive
    search using the other parameter(s).
    """

    def __init__(self, h_sep=None, a_sep=None, p_sep=None, t_sep=None):

        self.haversine_distance_kd_tree_index = False

        self._index_cache = {}
        self.constraints = []
        if h_sep is not None:
            self.h_sep = h_sep
            self.haversine_distance_kd_tree_index = None
        else:
            self.h_sep = None

        if a_sep is not None:
            self.a_sep = a_sep
            self.constraints.append(self.alt_constraint)
        if p_sep is not None:
            try:
                self.p_sep = float(p_sep)
            except:
                raise ValueError('Separation Constraint p_sep must be a valid float')
            self.constraints.append(self.pressure_constraint)
        if t_sep is not None:
            self.t_sep = t_sep
            self.constraints.append(self.time_constraint)

    def time_constraint(self, points, ref_point):
        indices = np.nonzero(np.abs(points.time - ref_point.time) < self.t_sep).values[0]
        dimension = points.coords['time'].dims[0]
        return points.sel(**{dimension: indices})

    def alt_constraint(self, points, ref_point):
        indices = np.nonzero(np.abs(points.altitude - ref_point.altitude) < self.a_sep).values[0]
        dimension = points.coords['altitude'].dims[0]
        return points.sel(**{dimension: indices})

    def pressure_constraint(self, points, ref_point):
        greater_pressures = np.nonzero(((points.air_pressure / ref_point.air_pressure.item()) < self.p_sep) &
                                       (points.air_pressure > ref_point.air_pressure.item())).values[0]
        lesser_pressures = np.nonzero(((ref_point.air_pressure.item() / points.air_pressure) < self.p_sep) &
                                      (points.air_pressure <= ref_point.air_pressure.item())).values[0]
        indices = np.concatenate([lesser_pressures, greater_pressures])
        dimension = points.coords['air_pressure'].dims[0]
        return points.sel(**{dimension: indices})

    def constrain_points(self, ref_point, data):
        if self.haversine_distance_kd_tree_index and self.h_sep:
            point_indices = self._get_cached_indices(ref_point)
            if point_indices is None:
                point_indices = self.haversine_distance_kd_tree_index.find_points_within_distance(ref_point, self.h_sep)
                self._add_cached_indices(ref_point, point_indices)
            dimension = data.coords['longitude'].dims[0]
            con_points = data.sel(**{dimension: point_indices})
        else:
            con_points = data
        for constrain in self.constraints:
            con_points = constrain(con_points, ref_point)

        return con_points

    def _get_cached_indices(self, ref_point):
        # Don't use the value as a key (it's both irrelevant and un-hashable)
        return self._index_cache.get((ref_point.latitude.item(), ref_point.longitude.item()), None)

    def _add_cached_indices(self, ref_point, indices):
        # Don't use the value as a key (it's both irrelevant and un-hashable)
        self._index_cache[(ref_point.latitude.item(), ref_point.longitude.item())] = indices

    def get_iterator(self, missing_data_for_missing_sample, data_points, points):
        indices = False

        iterator = index_iterator_nditer(points, not missing_data_for_missing_sample)

        if self.haversine_distance_kd_tree_index and self.h_sep:
            indices = self.haversine_distance_kd_tree_index.find_points_within_distance_sample(points, self.h_sep)

        for i in iterator:
            p = points[i]
            if indices:
                # Note that data_points has to be a dataframe at this point because of the indexing
                d_points = data_points[indices[i]]
            else:
                d_points = data_points
            for check in self.constraints:
                con_points_indices = check(d_points, p)
                d_points = d_points[con_points_indices]

            yield i, p, d_points

    def index_data(self, data, leafsize=10):
        """
        Creates the k-D tree index.

        :param DataArray data: points to index
        :param int leafsize: The leafsize to use when creating the tree
        """
        from colocate.haversinedistancekdtreeindex import HaversineDistanceKDTreeIndex
        from colocate.utils import get_lat_lon_names

        # lat_lon_points = data.to_dataframe(data.name or 'unknown').loc[:, get_lat_lon_names(data)]
        lat_lon_points =  np.column_stack((data.latitude.values.ravel(),
                                           data.longitude.values.ravel()))
        self.haversine_distance_kd_tree_index = HaversineDistanceKDTreeIndex(lat_lon_points, leafsize)


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


