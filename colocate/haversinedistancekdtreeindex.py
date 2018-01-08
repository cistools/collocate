from colocate.kdtree import HaversineDistanceKDTree


class HaversineDistanceKDTreeIndex(object):
    """k-D tree index that can be used to query using distance along the Earth's surface.
    """
    def __init__(self, data, leafsize=10):
        """
        Creates the k-D tree index.

        :param data: list of HyperPoints to index
        :param leafsize: The leafsize to use when creating the tree
        """
        self.index = HaversineDistanceKDTree(data, leafsize=leafsize)

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
        query_pt = [[point.latitude.item(), point.longitude.item()]]
        return self.index.query_ball_point(query_pt, distance)[0]

    def find_points_within_distance_sample(self, sample, distance):
        """Finds the points within a specified distance of a specified point.
        :param sample: the sample points
        :param distance: distance in kilometres
        :return list of lists:
        For each element ``self.data[i]`` of this tree, ``results[i]`` is a
            list of the indices of its neighbors in ``other.data``.
        """
        return HaversineDistanceKDTree(sample).query_ball_tree(self.index, distance)
