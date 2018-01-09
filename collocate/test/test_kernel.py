"""
Tests the various kernels
"""
import unittest
from numpy.testing import assert_almost_equal
import numpy as np
from collocate.test import mock


class TestFullAverage(unittest.TestCase):
    def test_basic_col_in_4d(self):
        from collocate.kernels import moments
        from collocate import collocate
        import datetime as dt

        data = mock.make_regular_4d_ungridded_data()
        # Note - This isn't actually used for averaging
        sample_points = mock.make_dummy_sample_points(latitude=[1.0], longitude=[1.0], altitude=[12.0],
                                                      time=[dt.datetime(1984, 8, 29, 8, 34)])

        new_data = collocate(sample_points, data, moments())
        means = new_data['var']
        std_dev = new_data['var_std_dev']
        no_points = new_data['var_num_points']

        assert_almost_equal(means.data[0], 25.5)
        assert_almost_equal(std_dev.data[0], np.sqrt(212.5))
        assert_almost_equal(no_points.data[0], 50)

    def test_basic_col_in_4d_with_pressure_not_altitude(self):
        from collocate.kernels import moments
        from collocate import collocate
        import datetime as dt

        data = mock.make_regular_4d_ungridded_data()
        # Note - This isn't actually used for averaging
        sample_points = mock.make_dummy_sample_points(latitude=[1.0], longitude=[1.0], altitude=[12.0],
                                                      time=[dt.datetime(1984, 8, 29, 8, 34)])

        new_data = collocate(sample_points, data, moments())
        means = new_data['var']
        std_dev = new_data['var_std_dev']
        no_points = new_data['var_num_points']

        assert_almost_equal(means.data[0], 25.5)
        assert_almost_equal(std_dev.data[0], np.sqrt(212.5))
        assert_almost_equal(no_points.data[0], 50)


class TestNNHorizontal(unittest.TestCase):
    def test_basic_col_in_2d(self):
        from collocate.kernels import nn_horizontal
        from collocate import collocate

        data = mock.make_regular_2d_ungridded_data()
        sample_points = mock.make_dummy_sample_points(latitude=[1.0, 4.0, -4.0], longitude=[1.0, 4.0, -4.0])
        new_data = collocate(sample_points, data, nn_horizontal())['var']
        assert_almost_equal(new_data.data[0], 8.0)
        assert_almost_equal(new_data.data[1], 12.0)
        assert_almost_equal(new_data.data[2], 4.0)

    def test_already_collocated_in_col_in_2d(self):
        from collocate.kernels import nn_horizontal
        from collocate import collocate

        data = mock.make_regular_2d_ungridded_data()
        # This point already exists on the cube with value 5 - which shouldn't be a problem
        sample_points = mock.make_dummy_sample_points(latitude=[0.0], longitude=[0.0])
        new_data = collocate(sample_points, data, nn_horizontal())['var']
        assert_almost_equal(new_data.data[0], 8.0)

    def test_coordinates_exactly_between_points_in_col_in_2d(self):
        """
            This works out the edge case where the points are exactly in the middle or two or more datapoints.
                The nn_horizontal algorithm will start with the first point as the nearest and iterates through the
                points finding any points which are closer than the current closest. If two distances were exactly
                the same  you would expect the first point to be chosen. This doesn't seem to always be the case but is
                probably down to floating points errors in the haversine calculation as these test points are pretty
                close together. This test is only really for documenting the behaviour for equidistant points.
        """
        from collocate.kernels import nn_horizontal
        from collocate import collocate

        data = mock.make_regular_2d_ungridded_data()
        sample_points = mock.make_dummy_sample_points(latitude=[2.5, -2.5, 2.5, -2.5], longitude=[2.5, 2.5, -2.5, -2.5])
        new_data = collocate(sample_points, data, nn_horizontal())['var']
        assert_almost_equal(new_data.data[0], 9.0)
        assert_almost_equal(new_data.data[1], 6.0)
        assert_almost_equal(new_data.data[2], 7.0)
        assert_almost_equal(new_data.data[3], 4.0)

    def test_coordinates_outside_grid_in_col_in_2d(self):
        from collocate.kernels import nn_horizontal
        from collocate import collocate

        data = mock.make_regular_2d_ungridded_data()
        sample_points = mock.make_dummy_sample_points(latitude=[5.5, -5.5, 5.5, -5.5], longitude=[5.5, 5.5, -5.5, -5.5])
        new_data = collocate(sample_points, data, nn_horizontal())['var']
        assert_almost_equal(new_data.data[0], 12.0)
        assert_almost_equal(new_data.data[1], 6.0)
        assert_almost_equal(new_data.data[2], 10.0)
        assert_almost_equal(new_data.data[3], 4.0)


class TestNNTime(unittest.TestCase):
    def test_basic_col_with_incompatible_points_throws_an_AttributeError(self):
        from collocate.kernels import nn_time
        from collocate import collocate

        data = mock.make_regular_2d_with_time_ungridded_data()
        # Make sample points with no time dimension specified
        sample_points = mock.make_dummy_sample_points(latitude=[1.0, 4.0, -4.0], longitude=[1.0, 4.0, -4.0])
        with self.assertRaises(AttributeError):
            new_data = collocate(sample_points, data, nn_time())['var']

    def test_basic_col_in_2d_with_time(self):
        from collocate.kernels import nn_time
        from collocate import collocate
        import datetime as dt

        data = mock.make_regular_2d_with_time_ungridded_data()
        sample_points = mock.make_dummy_sample_points(latitude=[1.0, 4.0, -4.0], longitude=[1.0, 4.0, -4.0],
                                                      time=np.asarray([dt.datetime(1984, 8, 29, 8, 34),
                                                            dt.datetime(1984, 9, 2, 1, 23),
                                                            dt.datetime(1984, 9, 4, 15, 54)],
                                                                      dtype='M8[ns]'))

        new_data = collocate(sample_points, data, nn_time())['var']
        assert_almost_equal(new_data.data[0], 3.0)
        assert_almost_equal(new_data.data[1], 7.0)
        assert_almost_equal(new_data.data[2], 10.0)

    def test_basic_col_with_time(self):
        from collocate.kernels import nn_time
        from collocate import collocate
        import numpy as np

        data = mock.make_dummy_sample_points(data=np.arange(4.0), latitude=[0.0, 0.0, 0.0, 0.0], longitude=[0.0, 0.0, 0.0, 0.0],
                                             time=[149754, 149762, 149770, 149778])

        sample_points = mock.make_dummy_sample_points(latitude=[0.0, 0.0, 0.0, 0.0], longitude=[0.0, 0.0, 0.0, 0.0],
                                                      time=[149751.369618055, 149759.378055556, 149766.373969907,
                                                            149776.375995371])

        ref = np.array([0.0, 1.0, 2.0, 3.0])

        new_data = collocate(sample_points, data, nn_time())['var']
        assert (np.equal(new_data.data, ref).all())

    def test_already_collocated_in_col_in_2d(self):
        from collocate.kernels import nn_time
        from collocate import collocate
        import datetime as dt
        import numpy as np

        data = mock.make_regular_2d_with_time_ungridded_data()

        sample_points = mock.make_dummy_sample_points(latitude=[0.0]*15, longitude=[0.0]*15,
                                                      time=[dt.datetime(1984, 8, 27) +
                                                            dt.timedelta(days=d) for d in range(15)])

        new_data = collocate(sample_points, data, nn_time())['var']
        assert (np.equal(new_data.data, np.arange(15) + 1.0).all())

    def test_coordinates_exactly_between_points_in_col_in_2d(self):
        """
            This works out the edge case where the points are exactly in the middle or two or more datapoints.
                The nn_time algorithm will start with the first point as the nearest and iterates through the
                points finding any points which are closer than the current closest. If two distances were exactly
                the same the first point to be chosen.
        """
        from collocate.kernels import nn_time
        from collocate import collocate
        import datetime as dt

        data = mock.make_regular_2d_with_time_ungridded_data()
        # Choose a time at midday
        sample_points = mock.make_dummy_sample_points(latitude=[0.0], longitude=[0.0], time=[dt.datetime(1984, 8, 29, 12)])
        new_data = collocate(sample_points, data, nn_time())['var']
        assert_almost_equal(new_data.data[0], 3.0)

    def test_coordinates_outside_grid_in_col_in_2d(self):
        from collocate.kernels import nn_time
        from collocate import collocate
        import datetime as dt

        data = mock.make_regular_2d_with_time_ungridded_data()
        sample_points = mock.make_dummy_sample_points(latitude=[0.0, 0.0, 0.0], longitude=[0.0, 0.0, 0.0],
                                                      time=[dt.datetime(1984, 8, 26),
                                                            dt.datetime(1984, 8, 26),
                                                            dt.datetime(1984, 9, 27)])
        new_data = collocate(sample_points, data, nn_time())['var']
        assert_almost_equal(new_data.data[0], 1.0)
        assert_almost_equal(new_data.data[1], 1.0)
        assert_almost_equal(new_data.data[2], 15.0)


class TestNNAltitude(unittest.TestCase):
    def test_basic_col_with_incompatible_points_throws_a_TypeError(self):
        from collocate.kernels import nn_altitude
        from collocate import collocate

        data = mock.make_regular_4d_ungridded_data()
        # Make sample points with no time dimension specified
        sample_points = mock.make_dummy_sample_points(latitude=[1.0, 4.0, -4.0], longitude=[1.0, 4.0, -4.0])
        with self.assertRaises(AttributeError):
            new_data = collocate(sample_points, data, nn_altitude())['var']

    def test_basic_col_in_4d(self):
        from collocate.kernels import nn_altitude
        from collocate import collocate
        import datetime as dt

        data = mock.make_regular_4d_ungridded_data()
        sample_points = mock.make_dummy_sample_points(latitude=[1.0, 4.0, -4.0], longitude=[1.0, 4.0, -4.0],
                                                      altitude=[12.0, 34.0, 89.0],
                                                      time=[dt.datetime(1984, 8, 29, 8, 34),
                                                            dt.datetime(1984, 9, 2, 1, 23),
                                                            dt.datetime(1984, 9, 4, 15, 54)])
        new_data = collocate(sample_points, data, nn_altitude())['var']
        assert_almost_equal(new_data.data[0], 6.0)
        assert_almost_equal(new_data.data[1], 16.0)
        assert_almost_equal(new_data.data[2], 46.0)

    def test_already_collocated_in_col_in_2d(self):
        from collocate.kernels import nn_altitude
        from collocate import collocate
        import datetime as dt

        data = mock.make_regular_4d_ungridded_data()
        sample_points = mock.make_dummy_sample_points(latitude=[0.0], longitude=[0.0], altitude=[80.0],
                                                      time=[dt.datetime(1984, 9, 4, 15, 54)])
        new_data = collocate(sample_points, data, nn_altitude())['var']
        assert_almost_equal(new_data.data[0], 41.0)

    def test_coordinates_exactly_between_points_in_col_in_2d(self):
        """
            This works out the edge case where the points are exactly in the middle or two or more datapoints.
                The nn_time algorithm will start with the first point as the nearest and iterates through the
                points finding any points which are closer than the current closest. If two distances were exactly
                the same the first point to be chosen.
        """
        from collocate.kernels import nn_altitude
        from collocate import collocate
        import datetime as dt

        data = mock.make_regular_4d_ungridded_data()
        # Choose a time at midday
        sample_points = mock.make_dummy_sample_points(latitude=[0.0], longitude=[0.0], altitude=[35.0],
                                                      time=[dt.datetime(1984, 8, 29, 12)])

        new_data = collocate(sample_points, data, nn_altitude())['var']
        assert_almost_equal(new_data.data[0], 16.0)

    def test_coordinates_outside_grid_in_col_in_2d(self):
        from collocate.kernels import nn_altitude
        from collocate import collocate
        import datetime as dt

        data = mock.make_regular_4d_ungridded_data()
        sample_points = mock.make_dummy_sample_points(latitude=[0.0, 0.0, 0.0], longitude=[0.0, 0.0, 0.0],
                                                      altitude=[-12.0, 91.0, 890.0],
                                                      time=[dt.datetime(1984, 8, 29, 8, 34),
                                                            dt.datetime(1984, 9, 2, 1, 23),
                                                            dt.datetime(1984, 9, 4, 15, 54)])

        new_data = collocate(sample_points, data, nn_altitude())['var']
        assert_almost_equal(new_data.data[0], 1.0)
        assert_almost_equal(new_data.data[1], 46.0)
        assert_almost_equal(new_data.data[2], 46.0)


class TestNNPressure(unittest.TestCase):
    def test_basic_col_with_incompatible_points_throws_an_AttributeError(self):
        from collocate.kernels import nn_pressure
        from collocate import collocate

        data = mock.make_regular_4d_ungridded_data()
        # Make sample points with no time dimension specified
        sample_points = mock.make_dummy_sample_points(latitude=[1.0, 4.0, -4.0], longitude=[1.0, 4.0, -4.0])

        with self.assertRaises(AttributeError):
            new_data = collocate(sample_points, data, nn_pressure())['var']

    def test_basic_col_in_4d(self):
        from collocate.kernels import nn_pressure
        from collocate import collocate
        import datetime as dt

        data = mock.make_regular_4d_ungridded_data()
        sample_points = mock.make_dummy_sample_points(latitude=[1.0, 4.0, -4.0], longitude=[1.0, 4.0, -4.0],
                                                      air_pressure=[12.0, 34.0, 89.0],
                                                      time=[dt.datetime(1984, 8, 29, 8, 34),
                                                            dt.datetime(1984, 9, 2, 1, 23),
                                                            dt.datetime(1984, 9, 4, 15, 54)])

        new_data = collocate(sample_points, data, nn_pressure())['var']
        assert_almost_equal(new_data.data[0], 6.0)
        assert_almost_equal(new_data.data[1], 16.0)
        assert_almost_equal(new_data.data[2], 46.0)

    def test_already_collocated_in_col_in_2d(self):
        from collocate.kernels import nn_pressure
        from collocate import collocate
        import datetime as dt

        data = mock.make_regular_4d_ungridded_data()
        sample_points = mock.make_dummy_sample_points(latitude=[0.0], longitude=[0.0], air_pressure=[80.0],
                                                      time=[dt.datetime(1984, 9, 4, 15, 54)])

        new_data = collocate(sample_points, data, nn_pressure())['var']
        assert_almost_equal(new_data.data[0], 41.0)

    def test_coordinates_exactly_between_points_in_col_in_2d(self):
        """
            This works out the edge case where the points are exactly in the middle or two or more datapoints.
                The nn_pressure algorithm will start with the first point as the nearest and iterates through the
                points finding any points which are closer than the current closest. If two distances were exactly
                the same the first point to be chosen.
        """
        from collocate.kernels import nn_pressure
        from collocate import collocate
        import datetime as dt

        data = mock.make_regular_4d_ungridded_data()
        # Choose a time at midday
        sample_points = mock.make_dummy_sample_points(latitude=[0.0], longitude=[0.0], air_pressure=[8.0],
                                                      time=[dt.datetime(1984, 8, 29, 12)])

        new_data = collocate(sample_points, data, nn_pressure())['var']
        assert_almost_equal(new_data.data[0], 1.0)

    def test_coordinates_outside_grid_in_col_in_2d(self):
        from collocate.kernels import nn_pressure
        from collocate import collocate
        import datetime as dt

        data = mock.make_regular_4d_ungridded_data()
        sample_points = mock.make_dummy_sample_points(latitude=[0.0, 0.0, 0.0], longitude=[0.0, 0.0, 0.0],
                                                      air_pressure=[0.1, 91.0, 890.0],
                                                      time=[dt.datetime(1984, 8, 29, 8, 34),
                                                            dt.datetime(1984, 9, 2, 1, 23),
                                                            dt.datetime(1984, 9, 4, 15, 54)])

        new_data = collocate(sample_points, data, nn_pressure())['var']
        assert_almost_equal(new_data.data[0], 1.0)
        assert_almost_equal(new_data.data[1], 46.0)
        assert_almost_equal(new_data.data[2], 46.0)


class TestMean(unittest.TestCase):
    def test_basic_col_in_4d(self):
        from collocate.kernels import mean
        from collocate import collocate
        import datetime as dt

        data = mock.make_regular_4d_ungridded_data()
        # Note - This isn't actually used for averaging
        sample_points = mock.make_dummy_sample_points(latitude=[1.0], longitude=[1.0], altitude=[12.0],
                                                      time=[dt.datetime(1984, 8, 29, 8, 34)])

        new_data = collocate(sample_points, data, mean())['var']
        assert_almost_equal(new_data.data[0], 25.5)


if __name__ == '__main__':
    unittest.main()
