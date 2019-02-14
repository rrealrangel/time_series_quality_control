# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 18:54:16 2019

@author: r.realrangel
"""
import numpy as np
import xarray as xr


def normal(X):
    return(xr.ufuncs.log(X))


def standard(X):
    return((X - X.mean()) / X.std())


def gross_range_test(X, threshold=3.8906):
    """ Test that data point exceeds min/max.

    Parameters
    ----------
        X: xarray.DataArray
            Time series of streamflow (or other variable).
        threshold: float (default is 3.8906)
            A threshold value to identifie outliers. It represents the
            deviation of the data point expressed as the number of
            times the standard deviation of the standardized time
            series. The default value (3.8906) guarantees a
            probability of 0.9999 of "normal" values (and a probability
            of 0.001 of outliers).

    Reference
    ---------
        Gudmundsson, L., & Seneviratne, S. I. (2016). Observation-based
            gridded runoff estimates for Europe (E-RUN version 1.1).
            Earth System Science Data, 8(2), 279–295.
            https://doi.org/10.5194/essd-8-279-2016

        IOOS. (2018). Manual for Real-Time Quality Control of Stream
            Flow Observations: A Guide to Quality Control and Quality
            Assurance for Stream Flow Observations in Rivers and
            Streams.
    """
    max_limit = standard(normal(X)) > threshold
    min_limit = standard(normal(X)) < -threshold
    return(min_limit | max_limit)


def climatology_test(X, threshold=3.8906):
    """ Test that data point falls within seasonal expectations.

    Parameters
    ----------
        X: xarray.DataArray
            Time series of streamflow (or other variable).
        threshold: float (default is 3.8906)
            A threshold value to identifie outliers. It represents the
            deviation of the data point expressed as the number of
            times the standard deviation of the standardized time
            series. The default value (3.8906) guarantees a
            probability of 0.9999 of "normal" values (and a probability
            of 0.001 of outliers).

    Reference
    ---------
        IOOS. (2018). Manual for Real-Time Quality Control of Stream
            Flow Observations: A Guide to Quality Control and Quality
            Assurance for Stream Flow Observations in Rivers and
            Streams.
    """
    tmp = []

    for month in range(1, 13):
        month_slices = X.isel(time=np.where(
                X.time.dt.month == month)[0])
        max_limit = standard(normal(month_slices)) > threshold
        min_limit = standard(normal(month_slices)) < -threshold
        tmp.append(min_limit | max_limit)

    test_flag = xr.concat(objs=tmp, dim='time')
    return(test_flag.reindex(time=sorted(test_flag.time.values)))


def spikes_data_test(X, threshold=3.8906):
    """Test that data point n-1 exceeds a selected threshold relative
    to the adjacent data point.

    Parameters
    ----------
        X: xarray.DataArray
            Time series of streamflow (or other variable).
        threshold: float (default is 3.8906)
            A threshold value to identifie outliers. It represents the
            deviation of the data point expressed as the number of
            times the standard deviation of the standardized time
            series. The default value (3.8906) guarantees a
            probability of 0.9999 of "normal" values (and a probability
            of 0.001 of outliers).

    Reference
    ---------
        IOOS. (2018). Manual for Real-Time Quality Control of Stream
            Flow Observations: A Guide to Quality Control and Quality
            Assurance for Stream Flow Observations in Rivers and
            Streams.
    """
    spike_reference_weight = xr.DataArray([0.50, 0.0, 0.50], dims=['window'])
    spike_reference = X.rolling(time=3, center=True).construct(
            'window').dot(spike_reference_weight)
    spikes = abs(X - spike_reference)
    spikes[spikes == 0] = np.nan
    return(standard(normal(spikes)) > threshold)


def change_rate_test(X, threshold=3.8906):
    """Excessive rise/fall test.

    Parameters
    ----------
        X: xarray.DataArray
            Time series of streamflow (or other variable).
        threshold: float (default is 3.8906)
            A threshold value to identifie outliers. It represents the
            deviation of the data point expressed as the number of
            times the standard deviation of the standardized time
            series. The default value (3.8906) guarantees a
            probability of 0.9999 of "normal" values (and a probability
            of 0.001 of outliers).

    Reference
    ---------
        IOOS. (2018). Manual for Real-Time Quality Control of Stream
            Flow Observations: A Guide to Quality Control and Quality
            Assurance for Stream Flow Observations in Rivers and
            Streams.

        Krajewski, W. F., & Krajewski, K. L. (1989). Real-time quality
            control of streamflow data: A simulation study. Journal of
            the American Water Resources Association, 25(2), 391–399.
            https://doi.org/10.1111/j.1752-1688.1989.tb03076.x
    """
    change_rate = abs(X.diff(dim='time'))
    change_rate = change_rate.reindex(time=sorted(X.time.values))
    change_rate[change_rate == 0] = np.nan
    return(standard(normal(change_rate)) > threshold)


def flat_series_test(X, value_tolerance=0.0, repetitions_tolerance=2,
                     skipzero=True):
    """ Invariant variable value.

    Parameters
    ----------
        X: xarray.DataArray
            Time series of streamflow (or other variable).
        tolerance: float (default is 0.0)
            Threshold of difference between two values to define if
            both are considered equivalent.
        repetitions_tolerance:
        skipzero:

    Reference
    ---------
        IOOS. (2018). Manual for Real-Time Quality Control of Stream
            Flow Observations: A Guide to Quality Control and Quality
            Assurance for Stream Flow Observations in Rivers and
            Streams.
    """
    invariant = (abs(X.diff(dim='time')) <= value_tolerance)

    for recurse in range(1, repetitions_tolerance+1):
        invariant = invariant * (abs(
                X.diff(dim='time', n=recurse)) <= value_tolerance)

    invariant = invariant.reindex(time=sorted(X.time.values)).astype('bool')

    if skipzero:
        invariant[X == 0] = False

    return(invariant)


def missd_ratio_test(X, threshold):
    """ Test if the missing data in the time series is below (True) the
    defined threshold or if it is above (False).

    Parameters
    ----------
        X: xarray.DataArray
            Time series of streamflow (or other variable).
        tolerance: float
            Threshold set up to missing data ratio (i. e., 
            missing_data / available_data). Values reported in the literature
            include 0.01, 0.05, 0.10, 0.15 and 0.20.

    Reference
    ---------
        Zhang, Y., & Post, D. (2018). How good are hydrological models
        for gap-filling streamflow data? Hydrology and Earth System
        Sciences, 22(8), 4593–4604.
        https://doi.org/10.5194/hess-22-4593-2018
    """
    gaps = float(X['main_filtered'].isnull().sum(dim='time'))
    return((gaps / X['main_filtered'].size) < threshold)
