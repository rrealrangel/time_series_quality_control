# -*- coding: utf-8 -*-
"""Quality control routines. Data management.
Author
------
    Roberto A. Real-Rangel (Institute of Engineering UNAM; Mexico)

License
-------
    GNU General Public License
"""
from pathlib2 import Path
from pyodbc import connect
import csv
import numpy as np
import sys
import xarray as xr


def load_dir(directory):
    """
    Parameters
    ----------
        directory: string
            Full path of the directory to be loaded.
    """
    if not Path(directory).exists():
        create_dir = raw_input(
                "The directory '{}' does not exist.\n"
                "Do you want to create it? [Y] Yes, [N] No. ".
                format(directory))

        if create_dir.lower() == 'y':
            Path(directory).mkdir(parents=True, exist_ok=True)

        else:
            sys.exit("Cannot continue without this directory. Aborting.")

    return(Path(directory))


def list_inputs(input_dir):
    return(list(Path(input_dir).glob(pattern='**/*.mdb')))


def slice_time_series(data, month):
    """Performs a slice from a dataset for a given month in the 'time'
        dimension.

    Parameters
    ----------
        data : xarray.Dataset
            A dataset to which it is desired to perform the slicing.
            Its temporal dimension needs to be called 'time'.
        month : integer
            The number of the month to slice (e. g., 1 for january, 2
            for fabruary, etc.)
    Returns
    -------
        xarray.Dataset
            A slice of the original dataset with only the data fform
            the month of interest.
    """
    return(data.isel(time=np.where(data.time.dt.month == month)[0]))


def read_csv(input_file):
    """ Reads a csv file and returns a xarray.DataArray with the time series.

    Parameters
    ----------
        input_file: string
            The path of the input csv file.

    Returns
    -------
        data_array: xarray.DataArray
            A xarray.DataArray containing the time series of the input_file.
    """
    with open(Path(input_file), 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        values = np.array([], dtype=np.float)
        time = np.array([], dtype=np.datetime64)

        for row in reader:
            year = row[0].zfill(4)
            month = row[1].zfill(2)

            for counter, value in enumerate(row[2:]):
                stdtime = year + '-' + month + '-' + str(counter + 1).zfill(2)

                try:
                    time = np.append(time, np.datetime64(stdtime))
                    values = np.append(values, value)

                except ValueError:
                    pass

    values[values == ''] = np.nan
    values = values.astype(np.float)
    X = xr.DataArray(data=values, dims=['time'], coords={
            'time': time})

    # Reindex to fill missing dates with nan.
    new_index = np.arange(
            time[0], time[-1] + np.timedelta64(1, 'D'), np.timedelta64(1, 'D'))
    return(X.reindex(time=new_index))


def read_mdb(input_file):
    """ Reads a Microsoft Data Base (mdb) file.
    """
    # TODO: Generalize thie function to read other data sources.
    connection = connect('DRIVER={DRIVER};DBQ={DBQ};PWD={PWD}'.format(
            DRIVER='{Microsoft Access Driver (*.mdb, *.accdb)}',
            DBQ=Path(input_file),
            PWD='pw'))
    cursor = connection.cursor()
    table_name = 'DD' + Path(input_file).resolve().stem
    query = 'SELECT * FROM ' + table_name
    rows = cursor.execute(query).fetchall()
    cursor.close()
    connection.close()
    values = np.array([], dtype=np.float)
    time = np.array([], dtype=np.datetime64)

    for row in rows:
        yyyy = str(row[0]).zfill(4)
        mm = str(row[1]).zfill(2)

        for counter, value in enumerate(row[2:]):
            stdtime = yyyy + '-' + mm + '-' + str(counter + 1).zfill(2)

            try:
                time = np.append(time, np.datetime64(stdtime))
                values = np.append(values, value)

            except ValueError:
                pass

    values[values == ''] = np.nan
    values = values.astype(np.float)
    X = xr.Dataset(
            data_vars={'main': (['time'], values)},
            coords={'time': time})

    # Reindex to fill missing dates with nan.
    new_index = np.arange(
            min(time), max(time) + np.timedelta64(1, 'D'),
            np.timedelta64(1, 'D'))
    return(X.reindex(time=new_index))
