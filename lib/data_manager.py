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


def list_inputs(input_dir, extension):
    return(sorted(list(Path(input_dir).glob(pattern='**/*' + extension))))


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


def read_bdcn_file(input_file):
    """ Extracts data from files from the National Climatologic Data
    Base (BDCN) of Mexico.

    Parameters
    ----------
        input_file: string
            The full path of the input file of the climatological records.
        remove_lines: integer (default value is 2)
            The number of lines that will be removed from the top of
            the file.
    """
    def date_builder(row):
        return(np.datetime64('-'.join([
                str(row['YEAR']).zfill(4),
                str(row['MONTH']).zfill(2),
                str(row['DAY']).zfill(2)]) + ' 08:00'))

    # Import CSV file.
    with open(str(input_file), 'r') as f:
        raw_text = f.read()

    raw_text = raw_text.replace('# ', '')
    metadata_rows = [i.split(',') for i in raw_text.splitlines()[:2]]
    variables = raw_text.splitlines()[2].split(',')
    data_type = np.empty(shape=np.shape(variables), dtype='|S6')

    for i, var in enumerate(variables):
        if (var == 'YEAR') or (var == 'MONTH') or (var == 'DAY'):
            data_type[i] = 'int'

        else:
            data_type[i] = 'f4'

    data = np.genfromtxt(
            fname=str(input_file),
            delimiter=',',
            skip_header=4,
            dtype=data_type,
            names=variables)
    time = np.asarray([date_builder(row=i) for i in data])

    # Remove the value of repeated dates.
    repeated = np.unique([i for i in time if (time == i).sum() > 1])

    if len(repeated) > 0:
        indices = []

        for i in repeated:
            indices = indices + list(np.where(time == i)[0])

        time = np.delete(time, indices)
        data = np.delete(data, indices)

    # Extract time series (ts).
    prec = xr.DataArray(
            data=data['PRECIP'],
            coords={'time': time},
            dims=['time'],
            attrs={'units': 'mm'})
    evap = xr.DataArray(
            data=data['EVAP'],
            coords={'time': time},
            dims=['time'],
            attrs={'units': 'mm'})
    tmin = xr.DataArray(
            data=data['TMIN'],
            coords={'time': time},
            dims=['time'],
            attrs={'units': 'C'})
    tmax = xr.DataArray(
            data=data['TMAX'],
            coords={'time': time},
            dims=['time'],
            attrs={'units': 'C'})

    dataset = xr.Dataset(
            data_vars={'precipitation': prec,
                       'evaporation': evap,
                       'temperature_min': tmin,
                       'temperature_max': tmax},
            attrs=dict(zip(metadata_rows[0], metadata_rows[1])))

    # Reindex to fill missing dates with nan.
    new_index = np.arange(
            min(time), max(time) + np.timedelta64(1, 'D'),
            np.timedelta64(1, 'D'))
    return(dataset.reindex(time=new_index))


def read_bandas_file(input_file):
    """ Reads the daily discharnge (DD) records from the National Database
    of Surface Water (BANDAS) of Mexico. Only works in Windows OS.
    """
    # TODO: Read all tables from input_file.
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

    # Remove the value of repeated dates.
    repeated = np.unique([i for i in time if (time == i).sum() > 1])

    if len(repeated) > 0:
        indices = []

        for i in repeated:
            indices = indices + list(np.where(time == i)[0])

        time = np.delete(time, indices)
        values = np.delete(values, indices)

    dataset = xr.Dataset(
            data_vars={'main': (['time'], values)},
            coords={'time': time})

    # Reindex to fill missing dates with nan.
    new_index = np.arange(
            min(time), max(time) + np.timedelta64(1, 'D'),
            np.timedelta64(1, 'D'))
    return(dataset.reindex(time=new_index))
