import numpy as np

from colocate.kdtree import HaversineDistanceKDTree


def create_index(data, leafsize=10):
    """
    Creates the k-D tree index.

    :param data: list of HyperPoints to index
    """
    spatial_points = data[['latitude', 'longitude']]
    if hasattr(data, 'data'):
        mask = np.ma.getmask(data.data).ravel()
    else:
        mask = None
    return HaversineDistanceKDTree(spatial_points, mask=mask, leafsize=leafsize)


class HaversineDistanceKDTreeIndex(object):
    """k-D tree index that can be used to query using distance along the Earth's surface.
    """
    def __init__(self, data, leafsize=10):
        """
        Creates the k-D tree index.

        :param data: list of HyperPoints to index
        :param leafsize: The leafsize to use when creating the tree
        """
        self.index = create_index(data, leafsize=leafsize)

    def find_nearest_point(self, point):
        """Finds the indexed point nearest to a specified point.
        :param point: point for which the nearest point is required
        :return: index in data of closest point
        """
        query_pt = point[['latitude', 'longitude']]
        (distances, indices) = self.index.query(query_pt)
        return indices

    def find_points_within_distance(self, point, distance):
        """Finds the points within a specified distance of a specified point.
        :param point: reference point
        :param distance: distance in kilometres
        :return: list indices in data of points
        """
        query_pt = [[point.latitude, point.longitude]]
        return self.index.query_ball_point(query_pt, distance)[0]

    def find_points_within_distance_sample(self, sample, distance):
        """Finds the points within a specified distance of a specified point.
        :param sample: the sample points
        :param distance: distance in kilometres
        :return list of lists:
        For each element ``self.data[i]`` of this tree, ``results[i]`` is a
            list of the indices of its neighbors in ``other.data``.
        """
        return create_index(sample).query_ball_tree(self.index, distance)
