import numpy as np
from numpy import mean as np_mean, std as np_std, min as np_min, max as np_max, sum as np_sum
from abc import ABCMeta, abstractmethod


class Kernel(object):
    """
    Class which provides a method for taking a number of points and returning one value. For example a nearest
    neighbour algorithm or sort algorithm or mean. This just defines the interface which the subclasses must implement.
    """
    __metaclass__ = ABCMeta

    #: The number of values the :meth:`.Kernel.get_value` should be expected to return
    #: (i.e. the length of the return list).
    return_size = 1

    @abstractmethod
    def get_value(self, point, data):
        """
        This method should return a single value (if :attr:`.Kernel.return_size` is 1) or a list of n values (if
        :attr:`.Kernel.return_size` is n) based on some calculation on the data given a single point.

        The data is deliberately left unspecified in the interface as it may be any type of data, however it is expected
        that each implementation will only work with a specific type of data (gridded, ungridded etc.) Note that this
        method will be called for every sample point and so could become a bottleneck for calculations, it is advisable
        to make it as quick as is practical. If this method is unable to provide a value (for example if no data points
        were given) a ValueError should be thrown.

        :param point: A single HyperPoint
        :param data: A set of data points to reduce to a single value
        :return: For return_size=1 a single value (number) otherwise a list of return values, which represents some
            operation on the points provided
        :raises ValueError: When the method is unable to return a value
        """

    def get_variable_details(self, var_name, var_long_name, var_standard_name, var_units):
        """Returns the details of all variables to be created from the outputs of a kernel.

        :param str var_name: Base variable name
        :param str var_long_name: Base variable long name
        :param str var_standard_name: Base variable standard_name
        :param str var_units: Base variable units
        :return: Tuple of tuples, each containing (variable name, variable long name, variable units)
        """
        return ((var_name, var_long_name, var_standard_name, var_units), )


class AbstractDataOnlyKernel(Kernel):
    """
    A Kernel that can work on data only, e.g. mean only requires the data values to calculate the mean, not the sampling
    point.
    """

    __metaclass__ = ABCMeta

    def get_value(self, point, data):
        """
        This method is redundant in the AbstractDataOnlyKernel and only serves as an interface to
        :meth:`.AbstractDataOnlyKernel`, removing the unnecessary point and checking for one or more data points.

        :param point: A single HyperPoint
        :param data: A set of data points to reduce to a single value
        :return: For return_size=1 a single value (number) otherwise a list of returns values, which represents some
            operation on the points provided
        """
        values = data.vals
        if len(values) == 0:
            raise ValueError
        return self.get_value_for_data_only(values)


    @abstractmethod
    def get_value_for_data_only(self, values):
        """
        This method should return a single value (if :attr:`.Kernel.return_size` is 1) or a list of n values
        (if :attr:`.Kernel.return_size` is n) based on some calculation on the the values (a numpy array).

        Note that this method will be called for every sample point in which data can be placed and so could become a
        bottleneck for calculations, it is advisable to make it as quick as is practical. If this method is unable to
        provide a value (for example if no data points were given) a ValueError should be thrown. This method will not
        be called if there are no values to be used for calculations.

        :param values: A numpy array of values (can not be none or empty)
        :return: A single data item if return_size is 1 or a list of items containing :attr:`.Kernel.return_size` items
        :raises ValueError: If there are any problems creating a value
        """


# noinspection PyPep8Naming
class mean(AbstractDataOnlyKernel):
    """
    Calculate mean of data points
    """

    def get_value_for_data_only(self, values):
        """
        return the mean
        """
        return np_mean(values)


# noinspection PyPep8Naming
class stddev(AbstractDataOnlyKernel):
    """
    Calculate the standard deviation
    """

    def get_value_for_data_only(self, values):
        """
        Return the standard deviation points
        """
        return np_std(values, ddof=1)


# noinspection PyPep8Naming,PyShadowingBuiltins
class min(AbstractDataOnlyKernel):
    """
    Calculate the minimum value
    """

    def get_value_for_data_only(self, values):
        """
        Return the minimum value
        """
        return np_min(values)


# noinspection PyPep8Naming,PyShadowingBuiltins
class max(AbstractDataOnlyKernel):
    """
    Calculate the maximum value
    """

    def get_value_for_data_only(self, values):
        """
        Return the maximum value
        """
        return np_max(values)


class sum(AbstractDataOnlyKernel):
    """
    Calculate the sum of the values
    """

    def get_value_for_data_only(self, values):
        """
        Return the sum of the values
        """
        return np_sum(values)


# noinspection PyPep8Naming
class moments(AbstractDataOnlyKernel):
    return_size = 3

    def __init__(self, mean_name='', stddev_name='', nopoints_name=''):
        self.mean_name = mean_name
        self.stddev_name = stddev_name
        self.nopoints_name = nopoints_name

    def get_variable_details(self, var_name, var_long_name, var_standard_name, var_units):
        """Sets name and units for mean, standard deviation and number of points variables, based
        on those of the base variable or overridden by those specified as kernel parameters.
        :param var_name: base variable name
        :param var_long_name: base variable long name
        :param var_standard_name: base variable standard name
        :param var_units: base variable units
        :return: tuple of tuples each containing (variable name, variable long name, variable units)
        """
        self.mean_name = var_name
        self.stddev_name = var_name + '_std_dev'
        stdev_long_name = 'Corrected sample standard deviation of %s' % var_long_name
        stddev_units = var_units
        self.nopoints_name = var_name + '_num_points'
        npoints_long_name = 'Number of points used to calculate the mean of %s' % var_long_name
        npoints_units = ''
        return ((self.mean_name, var_long_name, var_standard_name, var_units),
                (self.stddev_name, stdev_long_name, None, stddev_units),
                (self.nopoints_name, npoints_long_name, None, npoints_units))

    def get_value_for_data_only(self, values):
        """
        Returns the mean, standard deviation and number of values
        """

        return np_mean(values), np_std(values, ddof=1), np.size(values)


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
            if (haversine(np.asarray([point.longitude.item(), point.latitude.item()]),
                          nearest_point[['latitude', 'longitude']].values) >
                    haversine(np.asarray([point.longitude.item(), point.latitude.item()]),
                              data_point[['latitude', 'longitude']].values)):
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
        nearest_points = data[nearest_index]
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
        # TODO I might be able to use the index lookup - if the time index is available
        # return data.set_index(obs='time').sel(obs=point.time, method='nearest')
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