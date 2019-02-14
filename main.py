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

X = dmgr.read_mdb(settings['general']['input_file'])   # Read data.

# Perform tests to raw data.
if settings['level_1_tests']['gross_range_test']:
    X['gross_range_test'] = qct.gross_range_test(X.main)

if settings['level_1_tests']['climatology_test']:
    X['climatology_test'] = qct.climatology_test(X.main)

if settings['level_1_tests']['spikes_data_test']:
    X['spikes_data_test'] = qct.spikes_data_test(X.main)

if settings['level_1_tests']['change_rate_test']:
    X['change_rate_test'] = qct.change_rate_test(X.main)

if settings['level_1_tests']['flat_series_test']:
    X['flat_series_test'] = qct.flat_series_test(X.main)

tests = [i for i in X.var() if i != 'main']
X['filter'] = X[tests[0]]

for test in tests:
    X['filter'] = (X['filter'] | X[test])

X['main_filtered'] = X['main'].copy()
X['main_filtered'][X['filter']] = np.nan

# Perform tests to filtered data.
