"""
 Module to test the top-level collocation routine
"""
import unittest
import datetime as dt
import numpy as np
import xarray as xr
from collocate.kernels import moments
from collocate import collocate
from collocate.test import mock


class TestGeneralUngriddedCollocator(unittest.TestCase):

    def test_averaging_basic_col_in_4d(self):
        ug_data = mock.make_regular_4d_ungridded_data()
        # Note - This isn't actually used for averaging
        sample_points = mock.make_dummy_sample_points(latitude=[1.0], longitude=[1.0], altitude=[12.0], time=[dt.datetime(1984, 8, 29, 8, 34)])

        new_data = collocate(sample_points, ug_data, moments())
        means = new_data['var']
        std_dev = new_data['var_std_dev']
        no_points = new_data['var_num_points']

    def test_ungridded_ungridded_box_moments(self):
        data = mock.make_regular_2d_ungridded_data()
        sample = mock.make_dummy_sample_points(latitude=[1.0, 3.0, -1.0], longitude=[1.0, 3.0, -1.0], altitude=[12.0, 7.0, 5.0],
                                               time=[dt.datetime(1984, 8, 29, 8, 34),
                                                     dt.datetime(1984, 8, 29, 8, 34),
                                                     dt.datetime(1984, 8, 29, 8, 34)])

        kernel = moments()

        output = collocate(sample, data, kernel, h_sep=500)

        expected_result = np.array([28.0/3, 10.0, 20.0/3])
        expected_stddev = np.array([1.52752523, 1.82574186, 1.52752523])
        expected_n = np.array([3, 4, 3])
        assert isinstance(output, xr.Dataset)
        assert np.allclose(output['var'].data, expected_result)
        assert np.allclose(output['var_std_dev'].data, expected_stddev)
        assert np.allclose(output['var_num_points'].data, expected_n)

    def test_ungridded_ungridded_box_moments_missing_data_for_missing_sample(self):
        data = mock.make_regular_2d_ungridded_data()
        sample = mock.make_dummy_sample_points(latitude=[1.0, 3.0, -1.0], longitude=[1.0, 3.0, -1.0], altitude=[12.0, 7.0, 5.0],
                                               time=[dt.datetime(1984, 8, 29, 8, 34),
                                                     dt.datetime(1984, 8, 29, 8, 34),
                                                     dt.datetime(1984, 8, 29, 8, 34)])

        kernel = moments()

        # Set a missing sample data-value
        sample.data[1] = np.NaN

        output = collocate(sample, data, kernel, h_sep=500, missing_data_for_missing_sample=True)

        expected_result = np.array([28.0/3, np.NaN, 20.0/3])
        expected_stddev = np.array([1.52752523, np.NaN, 1.52752523])
        expected_n = np.array([3, np.NaN, 3])

        assert isinstance(output, xr.Dataset)
        assert np.allclose(output['var'].data, expected_result, equal_nan=True)
        assert np.allclose(output['var_std_dev'].data, expected_stddev, equal_nan=True)
        assert np.allclose(output['var_num_points'].data, expected_n, equal_nan=True)

    def test_ungridded_ungridded_box_moments_no_missing_data_for_missing_sample(self):
        data = mock.make_regular_2d_ungridded_data()
        sample = mock.make_dummy_sample_points(latitude=[1.0, 3.0, -1.0], longitude=[1.0, 3.0, -1.0], altitude=[12.0, 7.0, 5.0],
                                               time=[dt.datetime(1984, 8, 29, 8, 34),
                                                     dt.datetime(1984, 8, 29, 8, 34),
                                                     dt.datetime(1984, 8, 29, 8, 34)])

        kernel = moments()

        # Set a missing sample data-value
        sample.data[1] = np.NaN

        output = collocate(sample, data, kernel, h_sep=500, missing_data_for_missing_sample=False)

        expected_result = np.array([28.0/3, 10.0, 20.0/3])
        expected_stddev = np.array([1.52752523, 1.82574186, 1.52752523])
        expected_n = np.array([3, 4, 3])

        assert isinstance(output, xr.Dataset)
        assert np.allclose(output['var'].data, expected_result)
        assert np.allclose(output['var_std_dev'].data, expected_stddev)
        assert np.allclose(output['var_num_points'].data, expected_n)

    def test_list_ungridded_ungridded_box_mean(self):
        ug_data_1 = mock.make_regular_2d_ungridded_data()
        ug_data_2 = mock.make_regular_2d_ungridded_data(data_offset=3)
        ug_data_2.attrs['long_name'] = 'TOTAL SNOWFALL RATE: LS+CONV KG/M2/S'
        ug_data_2.attrs['standard_name'] = 'snowfall_flux'

        data_list = xr.Dataset({'precip': ug_data_1, 'snow': ug_data_2})
        sample_points = mock.make_regular_2d_ungridded_data()
        kernel = moments()
        output = collocate(sample_points, data_list, kernel, h_sep=500)

        expected_result = np.array(list(range(1, 16))).reshape((5, 3))
        expected_n = np.array(15 * [1]).reshape((5, 3))
        assert isinstance(output, xr.Dataset)
        assert output['snow'].name == 'snow'
        assert output['snow_std_dev'].name == 'snow_std_dev'
        assert output['snow_num_points'].name == 'snow_num_points'
        assert np.allclose(output['precip'].data, expected_result)
        assert np.isnan(output['precip_std_dev'].data).all()
        assert np.allclose(output['precip_num_points'].data, expected_n)
        assert np.allclose(output['snow'].data, expected_result + 3)
        assert np.isnan(output['snow_std_dev'].data).all()
        assert np.allclose(output['snow_num_points'].data, expected_n)
