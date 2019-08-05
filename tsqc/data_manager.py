# -*- coding: utf-8 -*-
"""Quality control routines. Data management.
Author
------
    Roberto A. Real-Rangel (Institute of Engineering UNAM; Mexico)

License
-------
    GNU General Public License
"""
from collections import OrderedDict
from pyodbc import connect
import datetime as dt
import io
import numpy as np
import sys

from pathlib2 import Path
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


def list_files(parent_dir, ext):
    """List all files in a directory with a specified extension.

    Parameters
        parent_dir: string
            Full path of the directory of which the files are to be listed.
        ext: string
            Extension of the files to be listed.
    """
    return(sorted(list(Path(parent_dir).glob(pattern='**/*' + ext))))


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
#    def date_builder(row):
#        return(np.datetime64('-'.join([
#                str(row['YEAR']).zfill(4),
#                str(row['MONTH']).zfill(2),
#                str(row['DAY']).zfill(2)]) + ' 08:00'))

    # Import CSV file.
    with io.open(input_file, 'r', encoding='latin1') as f:
        raw_text = f.read().splitlines()

    # Separate header lines from data lines.
    separator = raw_text.index("---------- ------ ------ ------ ------")
    header = raw_text[:separator]
    data = raw_text[separator + 1:-1]

    # Process data rows.
    def flitem2array(input_list, item):
        return(np.array([
                np.nan if i == 'Nulo'
                else float(i)
                for i in [row[item] for row in input_list]]))

    def parse_date(date_string, add_time=' 00:00'):
        """Convert date strings from DD/MM/YYYY format to YYYY-MM-DD
        format.

        It also adds " 08:00" to the string. This is due to the BDC
        datasets values are measured at 8:00 h."""
        return('-'.join(list(reversed(date_string.split('/')))) + add_time)

    data_stripped = [[i for i in row.split(' ') if i != ''] for row in data]
    dates = [np.datetime64(parse_date(date_string=row[0], add_time=' 08:00'))
             for row in data_stripped]
    prec = flitem2array(data_stripped, 1)
    evap = flitem2array(data_stripped, 2)
    tmax = flitem2array(data_stripped, 3)
    tmin = flitem2array(data_stripped, 4)
    dataset = xr.Dataset(
            data_vars={'prec': (['time'], prec),
                       'evap': (['time'], evap),
                       'tmax': (['time'], tmax),
                       'tmin': (['time'], tmin)},
            coords={'time': dates})

    # Remove repeated dates.
    # !!! If a date appears more than one time in the record, only the
    # !!! first one is retained.
    _, index = np.unique(dataset['time'], return_index=True)
    dataset = dataset.isel(time=index)

    # Reindex to fill missing dates with nan.
    new_index = np.arange(
            min(dates), max(dates) + np.timedelta64(1, 'D'),
            np.timedelta64(1, 'D'))
    dataset = dataset.reindex(time=new_index)

    # Process header rows.
    metadata = OrderedDict()
    metadata['Title'] = header[1].title()
    metadata['Author'] = header[0]
    metadata['StationID'] = header[4].split(":")[-1].strip().zfill(5)
    metadata['StationName'] = header[5].split(":")[-1].strip().capitalize()
    metadata['State'] = header[6].split(":")[-1].strip().capitalize()
    metadata['Municipality'] = header[7].split(":")[-1].strip().capitalize()
    metadata['Operability'] = (
            'Working'
            if header[8].split(":")[-1].strip() == 'OPERANDO'
            else 'Not working')
    metadata['Owner'] = header[9].split(":")[-1].strip()
    metadata['WMOID'] = header[10].split(":")[-1].strip()
    metadata['Latitude'] = str(float(header[11].split(":")[-1].strip()[:-1]))
    metadata['Longitude'] = str(float(header[12].split(":")[-1].strip()[:-1]))
    metadata['Elevation'] = str(float(
            header[13].split(":")[-1].replace('msnm', '').replace(',', '').
            strip()))
    metadata['TemporalRange'] = str(min(dates)) + " -> " + str(max(dates))
    metadata['TemporalResolution'] = 'Daily (08:00 of the past day - 08:00 of the current day; local time)'
    metadata['ProductionDateTime'] = "Original file generated on " + str(
            np.datetime64(parse_date(header[15].split(":")[-1].strip())))
    metadata['Comment'] = (
            'Converted to NetCDF format on '
            + dt.datetime.now().isoformat()
            + ' by R. A. Real-Rangel (IIUNAM; rrealr@iingen.unam.mx)')
    metadata['Version'] = '1.0.0'

    # Assigning attributes to xr.Dataset and xr.DataArrays
    dataset.attrs = metadata
    prec_attrs = OrderedDict()
    prec_attrs['long_name'] = 'Total_precipitation'
    prec_attrs['units'] = 'mm'
    dataset.prec.attrs = prec_attrs
    evap_attrs = OrderedDict()
    evap_attrs['long_name'] = 'Total_evaporation'
    evap_attrs['units'] = 'mm'
    dataset.evap.attrs = evap_attrs
    tmax_attrs = OrderedDict()
    tmax_attrs['long_name'] = 'Maximum_temperature_in_the_last_24_hours'
    tmax_attrs['units'] = 'C'
    dataset.tmax.attrs = tmax_attrs
    tmin_attrs = OrderedDict()
    tmin_attrs['long_name'] = 'Minimum_temperature_in_the_last_24_hours'
    tmin_attrs['units'] = 'C'
    dataset.tmin.attrs = tmin_attrs
    return(dataset)


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
