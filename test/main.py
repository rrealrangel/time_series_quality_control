# -*- coding: utf-8 -*-
"""
Created on Sun Aug 26 23:30:50 2018

@author: r.realrangel
"""

import lib.data_manager as dmgr
import lib.quality_control_tests as qct
import toml

with open('config.toml', 'rb') as fin:
    settings = toml.load(fin)

# TODO: Test all .mdb files in a given directory and subdirectories and
# export a report.
input_list = dmgr.list_files(
        parent_dir=settings['general']['input_dir'],
        ext='.csv')

for input_file in input_list:
    # X = dmgr.read_bandas_file(input_file=input_file)   # From BANDAS.
    X = dmgr.read_bdcn_file(input_file=input_file)   # From BDCN.

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

    X.to_netcdf(settings['general']['output_dir'] +
                '/' + input_file.stem + '.nc')
