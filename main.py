# -*- coding: utf-8 -*-
"""
Created on Sun Aug 26 23:30:50 2018

@author: r.realrangel
"""

import lib.data_manager as dmgr
import lib.quality_control_tests as qct
import numpy as np
import toml

with open('config.toml', 'rb') as fin:
    settings = toml.load(fin)

# TODO: Test all .mdb files in a given directory and subdirectories and
# export a report.
for input_file in dmgr.list_inputs(settings['general']['input_dir'])[128]:
    X = dmgr.read_mdb(input_file=input_file)   # Read data.

    # Perform tests to raw data.
    if settings['level_1_tests']['gross_range_test']:
        X['gross_range_test'] = qct.range_test(
                input_ts=X.main,
                threshold=4.89164,
                climatology=False)

    if settings['level_1_tests']['climatology_test']:
        X['climatology_test'] = qct.range_test(
                input_ts=X.main,
                threshold=4.89164,
                climatology=True)

    if settings['level_1_tests']['spikes_data_test']:
        X['spikes_data_test'] = qct.spikes_data_test(
                input_ts=X.main,
                threshold=4.89164,
                climatology=True)

    if settings['level_1_tests']['change_rate_test']:
        X['change_rate_test'] = qct.change_rate_test(
                input_ts=X.main,
                threshold=4.89164,
                climatology=True)

    if settings['level_1_tests']['flat_series_test']:
        X['flat_series_test'] = qct.flat_series_test(
                input_ts=X.main,
                value_tolerance=0.0,
                repetitions_tolerance=2,
                skipzero=True)

    if settings['level_1_tests']['tmp_outlier_test']:
        X['tmp_outlier_test'] = qct.tmp_outlier_test(
                input_ts=X.main,
                c=7.5,
                threshold=4.89164)

    tests = [i for i in X.var() if i != 'main']
    X['missing_suspect_value'] = X[tests[0]]

    for test in tests:
        # TODO: Tag as suspicious only if it is results as an outlier in
        # two of the following tests: climatological test, peak test, and
        # change rate test.
        X['missing_suspect_value'] = (X['missing_suspect_value'] | X[test])

    X['main_filtered'] = X.main.copy()
    X.main_filtered[X['missing_suspect_value']] = np.nan
    output_subdir = dmgr.load_dir(
            settings['general']['output_dir'] + '/rh' + input_file.stem[:2])
    X.to_netcdf(output_subdir / (input_file.stem + '.nc'))
