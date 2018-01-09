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
        return np.nonzero(np.abs(points.time.values - ref_point.time.values) < self.t_sep)[0]

    def alt_constraint(self, points, ref_point):
        return np.nonzero(np.abs(points.altitude.values - ref_point.altitude.values) < self.a_sep)[0]

    def pressure_constraint(self, points, ref_point):
        greater_pressures = np.nonzero(((points.air_pressure.values / ref_point.air_pressure.values) < self.p_sep) &
                                       (points.air_pressure.values > ref_point.air_pressure.values))[0]
        lesser_pressures = np.nonzero(((ref_point.air_pressure.values / points.air_pressure.values) < self.p_sep) &
                                      (points.air_pressure.values <= ref_point.air_pressure.values))[0]
        return np.concatenate([lesser_pressures, greater_pressures])

    def constrain_points(self, ref_point, data):
        if hasattr(ref_point, 'indices'):
            # Note that data_points has to be a dataframe at this point because of the indexing
            con_points = data.iloc[ref_point.indices]
        else:
            con_points = data

        for check in self.constraints:
            con_points = con_points.iloc[check(con_points, ref_point)]

        return con_points

    def get_iterator(self, missing_data_for_missing_sample, data_points, points):
        cell_count = 0
        total_count = 0
        sample_points_count = len(points)

        if self.haversine_distance_kd_tree_index and self.h_sep:
            points['indices'] = self.haversine_distance_kd_tree_index.find_points_within_distance_sample(points, self.h_sep)

        for i, p in points.iterrows():

            # Log progress periodically.
            cell_count += 1
            if cell_count == 1000:
                total_count += cell_count
                cell_count = 0
                logging.info("    Processed {} points of {}".format(total_count, sample_points_count))

            # If missing_data_for_missing_sample
            if not (missing_data_for_missing_sample and (hasattr(p, 'vals') and np.isnan(p.vals))):
                d_points = self.constrain_points(p, data_points)

                yield i, p, d_points

    def index_data(self, data, leafsize=10):
        """
        Creates the k-D tree index.

        :param DataArray data: points to index
        :param int leafsize: The leafsize to use when creating the tree
        """
        from colocate.haversinedistancekdtreeindex import HaversineDistanceKDTreeIndex

        self.haversine_distance_kd_tree_index = HaversineDistanceKDTreeIndex(data[['latitude', 'longitude']], leafsize)
