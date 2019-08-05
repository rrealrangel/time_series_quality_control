# -*- coding: utf-8 -*-
"""
Created on Sun Aug 26 23:30:50 2018

@author: r.realrangel
"""

import numpy as np
import toml
import xarray as xr

import lib.data_manager as dmgr
import lib.quality_control_tests as qct

settings = toml.load(
    'C:/Users/rreal/Mega/projects/2019/01-drought_response/main/data/15014'
    '/sflo/config.toml'
    )

## TODO: Test all .mdb files in a given directory and subdirectories and
## export a report.
#input_list = dmgr.list_files(
#        parent_dir=settings['general']['input_dir'],
#        ext='.csv')
#
#for input_file in input_list:
X = xr.open_dataset(filename_or_obj=settings['general']['input_file'])

for var in X.var():
    # Perform tests to raw data.
    if settings['level_1_tests']['gross_range_test']:
        X[var + 'gross_range_test'] = qct.range_test(
            input_ts=X[var],
            threshold=4.89164,
            climatology=False)

    if settings['level_1_tests']['climatology_test']:
        X[var + '_climatology_test'] = qct.range_test(
            input_ts=X[var],
            threshold=4.89164,
            climatology=True)

    if settings['level_1_tests']['spikes_data_test']:
        X[var + '_spikes_data_test'] = qct.spikes_data_test(
            input_ts=X[var],
            threshold=4.89164,
            climatology=True)

    if settings['level_1_tests']['change_rate_test']:
        X[var + '_change_rate_test'] = qct.change_rate_test(
            input_ts=X[var],
            threshold=4.89164,
            climatology=True)

    if settings['level_1_tests']['flat_series_test']:
        X[var + '_flat_series_test'] = qct.flat_series_test(
            input_ts=X[var],
            value_tolerance=0.0,
            repetitions_tolerance=2,
            skipzero=True)

    if settings['level_1_tests']['tmp_outlier_test']:
        X[var + '_tmp_outlier_test'] = qct.tmp_outlier_test(
            input_ts=X[var],
            c=7.5,
            threshold=4.89164)

    # Remove suspicious values.
    tests = [i for i in X.var().keys() if i != var]
    is_outlier = X[tests[0]].copy()

    for test in tests:
        is_outlier = (is_outlier | X[test])

    X[var + '_filtered'] = X[var].copy()
    X[var + '_filtered'][is_outlier] = np.nan

    # Fill short missing periods (<=3 days)
    all_nans_filled = X.disc_filtered.interpolate_na(dim='time')
    missing_data_grouper = (
        X.disc_filtered.isnull() != X.disc_filtered.isnull().shift(time=1)
        ).astype(int).cumsum()

    for group in all_nans_filled.groupby(missing_data_grouper).groups.values():
        if (len(group) > 3) & (np.isnan(X.disc_filtered.values[group[0]])):
            all_nans_filled[{'time': group}] = np.nan

    X[var + '_filled'] = all_nans_filled

X.to_netcdf(settings['general']['output_file'])
