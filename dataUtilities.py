###############################################################################################
# This file contains some utility functions that help in the pre-processing of raw data files #
###############################################################################################

import glob
import os
from datetime import datetime
import numpy as np
import pandas as pd
import re
from pytz import timezone as tz
from pandas.tseries.offsets import DateOffset

FILE_NAME_PATTERN_PYTHON = 'fileNamePatternPython'
FILE_NAME_PATTERN_WINDOWS = 'fileNamePatternWindows'
COMBINED_FILE_NAME = 'combinedFileName'
HEADER_ROWS = 'headerRows'
SOURCE_ENCODING = 'sourceEncoding'
SOURCE_SAMPLE_RATE = 'sourceSampleRate'
DELIMITER = 'delimiter'
COLUMN_OF_INTEREST = 'columnOfInterest'
GERMAN_NUMERALS_CONVERSION = 'germanNumeralsConversion'
SOURCE_TIMEZONE = 'sourceTimezone'
TARGET_TIMEZONE = 'targetTimezone'
DATETIME_COLUMNS = 'dateTimeColumns'
DATETIME_FORMATS = 'dateTimeFormats'
COLUMN_MAPPING = 'columnMapping'

"""
This is an example file configuration. it's a facility to define the specification of a raw file which will 
determine the pre-processing steps it has to go through.
FILE_NAME_PATTERN_PYTHON: (str) a regular expression that can uniquely identify this class of files.
FILE_NAME_PATTERN_WINDOWS: (str) dos-style wildcard expression to identify this class of files. 
COMBINED_FILE_NAME: (str) What should merged file be named
HEADER_ROWS: (int) How many rows to remove from the CSV file: 
SOURCE_ENCODING: (str) what is the file character encoding (e.g. 'UTF-8')
SOURCE_SAMPLE_RATE (str) what is the current sample rate (e.g. 1H, 15T, etc.)
DELIMITER: (str) the delimiter character (e.g. ',' ';' '\t')
COLUMN_OF_INTEREST: [str] which column contains the data of interest   
GERMAN_NUMERALS_CONVERSION: (boolean) does this file require the conversion from German to English number format  
SOURCE_TIMEZONE: (str) Current timezone for datetime fields (e.g. 'CET', 'UTC')
TARGET_TIMEZONE: (str) Which timezone to convert the datetime values to (e.g. 'CET', 'UTC')
DATETIME_COLUMNS: [str, str,] list of strings defining the column(s) that carry the datetime information. 
    Please note that order matters.
COLUMN_MAPPING: A dictionary defining how column names need to be changed. 
    The original column name is used as the key and the target column name as value. 
    This feature is specially useful when the same class of files uses inconsistent column naming. 
    The renaming step is performed first and hence all subsequent steps can use the new column names.
"""
gridFeedInFile = \
    {
        FILE_NAME_PATTERN_PYTHON: 'Netzeinspeisung_\d\d\d\d\.csv',
        FILE_NAME_PATTERN_WINDOWS: r'Netzeinspeisung_*.csv',
        COMBINED_FILE_NAME: 'Grid Feed in Combined.csv',
        HEADER_ROWS: 4,
        SOURCE_ENCODING: 'UTF-8',
        SOURCE_SAMPLE_RATE: '15T',
        DELIMITER: ';',
        COLUMN_OF_INTEREST: ['GRID_FEED_IN'],
        GERMAN_NUMERALS_CONVERSION: True,
        SOURCE_TIMEZONE: 'CET',
        TARGET_TIMEZONE: 'UTC',
        DATETIME_COLUMNS: ['DATUM', 'VON'],
        COLUMN_MAPPING: {'MW': 'GRID_FEED_IN', 'DATUM': 'DATUM', 'VON': 'VON', 'BIS': 'BIS'}
    }

solarPowerForecastedFile = \
    {
        FILE_NAME_PATTERN_PYTHON: 'Solarenergie_Prognose_\d\d\d\d\.csv',
        FILE_NAME_PATTERN_WINDOWS: r'Solarenergie_Prognose_*.csv',
        COMBINED_FILE_NAME: 'Solar Power Forecasted Combined.csv',
        HEADER_ROWS: 4,
        SOURCE_ENCODING: 'UTF-16-LE',
        SOURCE_SAMPLE_RATE: '15T',
        DELIMITER: ';',
        COLUMN_OF_INTEREST: ['SOLAR_POWER_FORECASTED'],
        GERMAN_NUMERALS_CONVERSION: True,
        SOURCE_TIMEZONE: 'CET',
        TARGET_TIMEZONE: 'UTC',
        DATETIME_COLUMNS: ['DATUM', 'VON'],
        COLUMN_MAPPING: {'MW': 'SOLAR_POWER_FORECASTED', 'DATUM': 'DATUM', 'VON': 'VON', 'BIS': 'BIS'}
    }

solarPowerActualFile = \
    {
        FILE_NAME_PATTERN_PYTHON: 'Solarenergie_Hochrechnung_\d\d\d\d\.csv',
        FILE_NAME_PATTERN_WINDOWS: r'Solarenergie_Hochrechnung_*.csv',
        COMBINED_FILE_NAME: 'Solar Power Actual Combined.csv',
        HEADER_ROWS: 4,
        SOURCE_ENCODING: 'UTF-16-LE',
        SOURCE_SAMPLE_RATE: '15T',
        DELIMITER: ';',
        COLUMN_OF_INTEREST: ['SOLAR_POWER_ACTUAL'],
        GERMAN_NUMERALS_CONVERSION: True,
        SOURCE_TIMEZONE: 'CET',
        TARGET_TIMEZONE: 'UTC',
        DATETIME_COLUMNS: ['DATUM', 'VON'],
        COLUMN_MAPPING: {'MW': 'SOLAR_POWER_ACTUAL', 'DATUM': 'DATUM', 'VON': 'VON', 'BIS': 'BIS'}
    }

windPowerForecastedFile = \
    {
        FILE_NAME_PATTERN_PYTHON: 'Windenergie_Prognose_\d\d\d\d\.csv',
        FILE_NAME_PATTERN_WINDOWS: r'Windenergie_Prognose_*.csv',
        COMBINED_FILE_NAME: 'Wind Power Forecasted Combined.csv',
        HEADER_ROWS: 4,
        SOURCE_ENCODING: 'UTF-16-LE',
        SOURCE_SAMPLE_RATE: '15T',
        DELIMITER: ';',
        COLUMN_OF_INTEREST: ['WIND_POWER_FORECASTED'],
        GERMAN_NUMERALS_CONVERSION: True,
        SOURCE_TIMEZONE: 'CET',
        TARGET_TIMEZONE: 'UTC',
        DATETIME_COLUMNS: ['DATUM', 'VON'],
        COLUMN_MAPPING: {'MW': 'WIND_POWER_FORECASTED', 'DATUM': 'DATUM', 'VON': 'VON', 'BIS': 'BIS',
                         'ONSHORE MW': 'ONSHORE MW',
                         'OFFSHORE MW': 'OFFSHORE MW'}
    }

windPowerActualFile = \
    {
        FILE_NAME_PATTERN_PYTHON: 'Windenergie_Hochrechnung_\d\d\d\d\.csv',
        FILE_NAME_PATTERN_WINDOWS: r'Windenergie_Hochrechnung_*.csv',
        COMBINED_FILE_NAME: 'Wind Power Actual Combined.csv',
        HEADER_ROWS: 4,
        SOURCE_ENCODING: 'UTF-16-LE',
        SOURCE_SAMPLE_RATE: '15T',
        DELIMITER: ';',
        COLUMN_OF_INTEREST: ['WIND_POWER_ACTUAL'],
        GERMAN_NUMERALS_CONVERSION: True,
        SOURCE_TIMEZONE: 'CET',
        TARGET_TIMEZONE: 'UTC',
        DATETIME_COLUMNS: ['DATUM', 'VON'],
        COLUMN_MAPPING: {'MW': 'WIND_POWER_ACTUAL', 'DATUM': 'DATUM', 'VON': 'VON', 'BIS': 'BIS',
                         'ONSHORE MW': 'ONSHORE MW',
                         'OFFSHORE MW': 'OFFSHORE MW'}
    }

verticalGridLoadFile = \
    {
        FILE_NAME_PATTERN_PYTHON: 'Vertikale_Netzlast_\d\d\d\d\.csv',
        FILE_NAME_PATTERN_WINDOWS: r'Vertikale_Netzlast_*.csv',
        COMBINED_FILE_NAME: 'Vertical Grid Load Combined.csv',
        HEADER_ROWS: 4,
        SOURCE_ENCODING: 'UTF-8',
        SOURCE_SAMPLE_RATE: '15T',
        DELIMITER: ';',
        COLUMN_OF_INTEREST: ['VERTICAL_GRID_LOAD'],
        GERMAN_NUMERALS_CONVERSION: True,
        SOURCE_TIMEZONE: 'CET',
        TARGET_TIMEZONE: 'UTC',
        DATETIME_COLUMNS: ['DATUM', 'VON'],
        COLUMN_MAPPING: {'VERTIKALE NETZLAST [MW]': 'VERTICAL_GRID_LOAD', 'DATUM': 'DATUM', 'VON': 'VON',
                         'BIS': 'BIS'}
    }

gridLossFile = \
    {
        FILE_NAME_PATTERN_PYTHON: 'Netzverluste_\d\d\d\d\.csv',
        FILE_NAME_PATTERN_WINDOWS: r'Netzverluste_*.csv',
        COMBINED_FILE_NAME: 'Grid Loss Combined.csv',
        HEADER_ROWS: 0,
        SOURCE_ENCODING: 'UTF-8',
        SOURCE_SAMPLE_RATE: '1H',
        DELIMITER: ';',
        COLUMN_OF_INTEREST: ['GRID_LOSS'],
        GERMAN_NUMERALS_CONVERSION: True,
        SOURCE_TIMEZONE: 'CET',
        TARGET_TIMEZONE: 'UTC',
        DATETIME_COLUMNS: ['DATUM'],
        COLUMN_MAPPING: {'VERLUSTE [MW]': 'GRID_LOSS', 'NETZVERLUSTE [MW]': 'GRID_LOSS', 'ZEIT_BERLIN': 'DATUM',
                         'VON': 'DATUM',
                         'BIS': 'BIS', 'DATUM': 'DATUM'}
    }

'''
FILE_CONFIG_LIST: A list of file configuration dictionaries that will be considered in the processing.
'''
FILE_CONFIG_LIST = [gridFeedInFile, solarPowerForecastedFile, solarPowerActualFile, windPowerForecastedFile,
                    windPowerActualFile, verticalGridLoadFile, gridLossFile]

'''
COLUMN_DATA_REGEX: a dictionary of columnNames (after column renaming) and their corresponding validation regular expression.
if a cell contains a value that matches the regular expression with extra leading/trailing characters, 
those leading/trailing characters will be automatically removed and only the matching portion will be kept. 
(for example: '02:00a' will be changed to '02:00' thus keeping only the time information)
'''
COLUMN_DATA_REGEX = {'DATUM': r'\d\d\.\d\d\.\d\d\d\d( \d\d){0,1}(\:\d\d){0,1}',
                     'VON': r'(\d\d\.\d\d\.\d\d\d\d ){0,1}\d\d\:\d\d',
                     'BIS': r'(\d\d\.\d\d\.\d\d\d\d ){0,1}\d\d\:\d\d',
                     'GRID_FEED_IN': r'\d+\.{0,1}\d*\,{0,1}\d*',
                     'VERTICAL_GRID_LOAD': r'-*\d+\.{0,1}\d*\,{0,1}\d*',
                     'SOLAR_POWER_ACTUAL': r'\d+\.{0,1}\d*\,{0,1}\d*',
                     'SOLAR_POWER_FORECASTED': r'\d+\.{0,1}\d*\,{0,1}\d*',
                     'WIND_POWER_ACTUAL': r'\d+\.{0,1}\d*\,{0,1}\d*',
                     'WIND_POWER_FORECASTED': r'\d+\.{0,1}\d*\,{0,1}\d*',
                     'GRID_LOSS': r'\d+\.{0,1}\d*\,{0,1}\d*',
                     'ZEIT_BERLIN': r'\d\d\.\d\d\.\d\d\d\d \d\d',
                     'ONSHORE MW': r'\d+\.{0,1}\d*\,{0,1}\d*',
                     'OFFSHORE MW': r'\d+\.{0,1}\d*\,{0,1}\d*'
                     }

'''
DATE_FORMAT_REGEX: This dictionary maps a datetime regex to a date format expression. 
It's used for inferring date format automatically.
'''
DATE_FORMAT_REGEX = \
    {
        r'\d*\d\.\d*\d\.\d\d\d\d \d*\d\:\d*\d': '%d.%m.%Y %H:%M',
        r'\d*\d\.\d*\d\.\d\d\d\d \d*\d\:\d*\d:\d*\d': '%d.%m.%Y %H:%M:%S',
        r'\d*\d\.\d*\d\.\d\d\d\d \d*\d': '%d.%m.%Y %H',
    }

'''
NUMERICAL_COLUMNS: This list defines which column names (after column renaming) contain numerical data. 
This designation will allow those columns to go through steps such as number format conversion. 
'''
NUMERICAL_COLUMNS = ['GRID_FEED_IN', 'VERTICAL_GRID_LOAD', 'SOLAR_POWER_ACTUAL', 'SOLAR_POWER_FORECASTED',
                     'WIND_POWER_ACTUAL', 'WIND_POWER_FORECASTED', 'GRID_LOSS', 'ONSHORE MW', 'OFFSHORE MW']


def filterMissingFiles(path, files):
    """
    This method filters out non-existing files.
    inputs:
    - path: (str) The path to the folder where the files reside
    - files: string list of file names to be checked for existence in the specified path:

    output:
    - string list of the subset of files that do exist, or empty list if no files exist.
    """
    os.chdir(path)
    filesInPath = glob.glob('*.*')
    return np.intersect1d(filesInPath, files)


def mergeCSVFiles(CSVFilesPath, fileNames=[], combinedFileName='combined_csv.csv', fileFilter='*.csv', delimiter=';',
                  returnDataFrame=False, axis=0):
    """
    This function merges a group of CSV files into one file.
    - All the individual files must have exactly the same number of columns and exactly the same column headers' names

    Inputs:
    - CSVFilesPath: Relative or absolute path where the individual CSV files are located. A string value.
    - fileNames: String List of file names to be merged. When passed the files names take precedence on
        the fileFilter argument.
    - combinedFileName: The name of the merged file. If relative path is used or only the filename is provided,
        the merged file will be stored in the same location as the individual files. Optional string with a default value
        of 'combined_csv.csv'.
    - fileFilter: DOS-style filename wildcard expression. Optional String with a default value of '*.csv'
    - delimiter: Which delimiting character to use with the CSV opening and saving.
        Optional String with a default value of ';'
    - returnDataFrame: Boolean value, if True, the merged data will not be saved but rather returned
        as a dataframe to the calling function. This is useful for in-memory pipeline processing.
    - axis: (int) optional. 0 means rows are concatenated, 1 means columns are concatenated.
    """

    os.chdir(CSVFilesPath)
    allFileNames = filterMissingFiles(CSVFilesPath, fileNames)
    if len(fileNames) == 0:
        allFileNames = filterMissingFiles(CSVFilesPath, [i for i in glob.glob(fileFilter)])
    if len(allFileNames) > 0:
        combined_csv = pd.concat((
            [pd.read_csv(f, delimiter=delimiter, encoding='unicode_escape', parse_dates=['TIME_STAMP'],
                         index_col='TIME_STAMP') for f in
             allFileNames]), axis=axis)

        combined_csv.sort_index(inplace=True)
        combined_csv = combined_csv[~combined_csv.index.duplicated(keep='first')]
        if not returnDataFrame:
            combined_csv.to_csv(combinedFileName, index=True, encoding='utf-8', sep=delimiter)
        else:
            return combined_csv


def containsGermanNumerals(series):
    """
    This method traverses a CSV column (pandas series) and checks if it meets any of the following conditions:
        1- contains at least one incident of a value with both a comma (German decimal character)
        as well as one or more dots (German thousands separator).
        2- Contains two or more dots, which means it's in German number format and has at least 7 digits.
    When any of the above is satisfied, this would be an indicative that the column is using german numbers convention
    and is thus eligible for conversion. This function provides a precondition for converting numbers from german to
    to English number format thus avoiding the situation when the conversion is applied more than once resulting
    in data corruption in the form of missing decimal point altogether.

    input(s):
    - series: a pandas series containing the numeric values to be checked.

    Output:
    Boolean value representing whether or not German number convention was discovered in the provided series.
        True means that German Convention was found.
    """

    for i in range(len(series)):
        tempVal = series[i]
        if (tempVal.find('.') > -1 and tempVal.find(',') > -1) or tempVal.count('.') > 1:
            return True
    return False


def convertGermanNumeral(series, forceConversion=False):
    """
    This function converts a CSV column (pandas float series) from the German number format
    (i.e. comma as a decimal point and dot as a thousands separator) into the English number format
    (i.e. dot as decimal point and no thousands separator)

    input(s):
    - series: A pandas float series that needs to be converted from German number format to English number format.
    - forceConversion: boolean flag when true, number conversion will take place even if no German number format was detected.
        This is useful when the column values are all integers and thus a German dot (for thousands)
        could be mistaken as an English decimal point, hence the need overriding the check.
    """

    if not (containsGermanNumerals(series) or forceConversion):
        return
    for i in range(len(series)):
        tempVal = series[i]
        series.at[i] = tempVal.replace('.', 't').replace(',', '.').replace('t', '')


def processGermanNumerals(sourceFileName='', sourceDataFrame=[], delimiter=';', forceConversion=False):
    """
    This function converts the user-defined columns of a CSV file from German number convention
    (i.e. comma as a decimal point and dot as a thousands separator) into the american convention
    (i.e. dot as decimal point and no thousands separator)

    input(s):
    - sourceFileName: The relative or absolute path to the CSV file that needs to be converted
    - sourceDataFrame: Optional pandas dataFrame that can be passed when no disk I/O is wished and only
        in-memory pipeline processing is required. passing this argument will also prevent
        the saving of the CSV file to disk and instead the data will be returned to the calling function.
    - delimiter: which delimiter should be used with the CSV file. Optional string with a default value of ';'
    - forceConversion: boolean flag when true, number conversion will take place even if no German number format was detected.
        This is useful when the column values are all integers and thus a German dot (for thousands)
        could be mistaken as an English decimal point, hence the need overriding the check.
    Output(s):
        Pandas DataFrame after processing
    """
    if len(sourceDataFrame) == 0:
        dataFrame = pd.read_csv(sourceFileName, sep=';', dtype=str)
    else:
        dataFrame = sourceDataFrame

    columnNames = np.intersect1d(dataFrame.columns, NUMERICAL_COLUMNS)

    for columnName in columnNames:
        convertGermanNumeral(dataFrame[columnName], forceConversion)
    if len(sourceDataFrame) == 0:
        dataFrame.to_csv(sourceFileName, sep=delimiter, index=False)
    return dataFrame


def checkDateGaps(timestampSequence, interval='15T'):
    """
    This function checks a user defined timestamps range for gaps.
    If gaps are found, they are printed to the standard output and the function returns False.

    Input(s):
    - timestampSequence: pandas series of a sequence of timestamps to be checked for gaps.
    - interval: The time interval between each two consecutive values. example values '1H' for 1 hour and '15T'
        for 15 minutes. A string value.

    Output(s):
    - Boolean value indicating whether the gaps check passed successfully or not. False means that gaps were found.
    """
    deltaShortfall = np.array([])
    deltaSurplus = np.array([])
    tempSource = np.array(timestampSequence)
    start = pd.Timestamp(tempSource[0])
    end = pd.Timestamp(tempSource[-1])

    correctTimestampSequence = np.array(pd.date_range(start, end, freq=interval))
    deltaShortfall = np.setdiff1d(correctTimestampSequence, tempSource)

    deltaSurplus = np.setdiff1d(tempSource, correctTimestampSequence)

    print('Date Gaps found at the following values:\n' + str(deltaShortfall))
    print('Date redundancies found at the following values:\n' + str(deltaSurplus))
    if len(deltaShortfall) == 0 and len(deltaSurplus) == 0:
        return True
    return False


def fixDateGaps(dataFrame, config):
    """
    This function checks a user defined timestamps range for gaps.
    If gaps are found, each missing entry is filled with the correct date timestamp and value that's
        copied from the previous row. This method excludes (i.e ignores and does not add) missing rows
        due to Day Light Saving Time (DST) change from winter to summer, those DST records are handled with the function
        'fixDSTKnownIssues' since they require extra caution.

    Input(s):
    - dataFrame: The dataFrame that needs to be checked and fixed for date gaps
    - config: The configuration dictionary corresponding to the data file as defined in the header section of
        this script.

    Output(s):
    - None. The function applies the correction directly to the dataFrame passed in the argument.
    """
    timestampSequence = dataFrame.index
    years = timestampSequence.year.drop_duplicates()
    dstSequenceSingle = []
    for year in years:
        dstChangeDate = getDstDates(year, config[SOURCE_TIMEZONE])[0]

        if max(timestampSequence) < dstChangeDate:  # Winter-Summer shift not reached in file, so do nothing
            return dataFrame

        dstSequenceSingle.append(
            np.array(
                pd.date_range(dstChangeDate, dstChangeDate + DateOffset(hours=1), freq=config[SOURCE_SAMPLE_RATE]))[
            0: -1])

    deltaShortfall = np.array([])
    deltaSurplus = np.array([])
    tempSource = np.array(timestampSequence)
    tempSource = np.append(tempSource, np.array(dstSequenceSingle).squeeze())
    tempSource.sort()
    start = pd.Timestamp(tempSource[0])
    end = pd.Timestamp(tempSource[-1])
    intervalMin = pd.Timestamp(tempSource[1]) - pd.Timestamp(tempSource[0])
    correctTimestampSequence = np.array(pd.date_range(start, end, freq=config[SOURCE_SAMPLE_RATE]))
    deltaShortfall = np.sort(np.setdiff1d(correctTimestampSequence, tempSource))
    for record in deltaShortfall:
        newRecord = dataFrame.loc[np.array(pd.DatetimeIndex([record - intervalMin]))].copy()
        newRecord.index = [record]
        dataFrame = dataFrame.append(newRecord)
        dataFrame.sort_index(inplace=True)
    return dataFrame


def resampleCSV(sourceFileName, targetFileName, delimiter=';', columnsToResample=[], newSampleRate='1H',
                oldSampleRate='15T', sourceDataFrame=[]):
    """
    This function converts values from one sample rate to another for example data sampled at 15-min time intervals can be
    sampled at 1-hour time intervals.

    Input(s):
    - sourceFileName: Relative or absolute path for the source CSV file
    - targetFileName: Relative or absolute path for the target (resampled) CSV file
    - delimiter: which delimiter is used. Optional string value with a default value of ';'
    - columnsToResample: A list of column header names foe the columns that need to be resampled.
        The columns must contain numeric or date values only. Unfortunately ue to a limitation in pandas,
        only one column can be passed as argument.
    - newSampleRate: What is the new sample rate. Optional string with a default value of '1H' i.e. 1 hour.
    - oldSampleRate: What is the current sample rate. Optional string with a default value of '15T' i.e. 15 minutes.
    - sourceDataFrame: a dataframe of the data that needs to be resampled. If passed, no CSV file will be
        opened or saved, instead tha data will be fetched from this dataFrame and will also be returned after processing.
        This is useful for in-memory pipeline processing.
    """

    if len(sourceDataFrame) > 0:
        dataset = sourceDataFrame
    else:
        dataset = pd.read_csv(sourceFileName, delimiter=delimiter, encoding='unicode_escape',
                              parse_dates=['TIME_STAMP'], index_col='TIME_STAMP')

    if len(columnsToResample) == 0:
        columnsToResample = dataset.columns

    dataset = dataset[columnsToResample].astype(float)
    dataset.fillna(0)
    dataset.drop_duplicates()
    dataset.sort_index()
    if oldSampleRate != newSampleRate:
        dataset = dataset.resample(newSampleRate).sum()
    if len(sourceDataFrame) > 0:
        return dataset
    else:
        dataset.to_csv(targetFileName, sep=delimiter)


def removeFileHeader(sourceFileName, targetPath, rowsToRemove='4', sourceEncoding='utf-16-le', targetEncoding='utf-8'):
    """
    This function remove any irrelevant lines at the beginning of a CSV file.
    Input(s):
    - fileName: relative or absolute path of the file to be cleaned.
    - rowsToRemove: How many rows to be removed from the file beginning. Integer optional value with a default value of 4
        that suits CSV files from 50Hertz.com.
    - sourceEncoding: Which character encoding is used by the source file. Optional string with a default value of
        'utf-16-le' as is used in files from 50hertz.com.
    - targetEncoding: Which encoding to be used after the file is cleaned. An optional String with
        a default value of 'utf-8'.
    """
    print("Attempting to remove " + str(rowsToRemove) + " rows.")
    f = open(sourceFileName, 'r', encoding=sourceEncoding)
    lines = f.readlines()
    updatedLines = lines[rowsToRemove:]
    f.close()
    os.chdir(targetPath)
    f = open(sourceFileName, 'w+', encoding=targetEncoding)
    f.writelines(updatedLines)
    f.close()
    print("Successfully removed " + str(rowsToRemove) + " rows.")


def csvSubset(sourceFileName, targetFileName, dateFrom, dateTo, dateFormat='%Y%m%d%H', dateColumn='MESS_DATUM',
              delimiter=','):
    custom_date_parser = lambda x: datetime.strptime(x, dateFormat)
    dataset = pd.read_csv(sourceFileName, delimiter=delimiter, encoding='unicode_escape',
                          parse_dates={'TIME_STAMP': [dateColumn]}, date_parser=custom_date_parser)
    dataset = dataset[dateFrom <= dataset.timestamp]
    dataset = dataset[dataset.timestamp <= dateTo]
    if checkDateGaps(dataset.timestamp, '1H'):
        dataset.to_csv(targetFileName, sep=delimiter, index=False)


def getDstDates(year, timezone):
    """
    This function returns the DayLight Saving Time (DST) date and hour for the change from Summer to Winter
    (usually in October in the northern hemisphere).
    Input(s):
    - year: (int) The year for which DST date is required
    - timezone: (str) the timezone to be used (e.g. 'CET', 'UTC',.etc.)

    output:
    If a match was found: an array with two records (winter-summer and summer-winter changes dates/times)
    otherwise: -1
    """
    tzone = tz(timezone)
    allDstChanges = tzone._utc_transition_times
    for i in range(len(allDstChanges)):
        if allDstChanges[i].year == year:
            return np.array(allDstChanges[i:i + 2]) + tzone.utcoffset(allDstChanges[i])
    return -1


def fixDSTKnownIssues(dataFrame, config):
    """
    This function fixed known issues related to DST. A file with correct DST handling would have the following features:
    - 1-hour-worth of missing records during DST from winter to summer
        (Clock is shifted forward thus leaving a 1-hour gap)
    - 1-hour-worth of duplicate records during DST from summer to winter
        (Clock is shifted backward thus having one hour twice)
    A file with the above characteristics would have a total number of records that's correct
    (from DST point of view, but could have other issues) and will therefore
    be eligible for timezone conversion without issues.


    Input(s):
    - dataFrame: The dataFrame that needs to be checked and fixed for date gaps dur to DST
    - config: The configuration dictionary corresponding to the data file as defined in the header section of
        this script.

    Output(s):
    - DataFrame after processing.
    """
    timestampSequence = dataFrame.index
    hourlyDataPoints = int(pd.Timedelta(hours=1) / (timestampSequence[1] - timestampSequence[0]))
    years = timestampSequence.year.drop_duplicates().values
    currentDstSequence = np.array(timestampSequence)
    for year in years:
        dstChangeDate = getDstDates(year, config[SOURCE_TIMEZONE])[-1]
        if max(timestampSequence) < dstChangeDate:  # Have not reached summer-winter shift yet in file, so do nothing
            return dataFrame
        dstIdentificationSequence = np.array(
            pd.date_range(dstChangeDate - DateOffset(hours=1),
                          dstChangeDate + DateOffset(hours=1) + DateOffset(hours=1),
                          freq=config[SOURCE_SAMPLE_RATE]))[
                                    0: -1]
        dstSequenceSingle = np.array(pd.date_range(dstChangeDate, dstChangeDate + DateOffset(hours=1),
                                                   freq=config[SOURCE_SAMPLE_RATE]))[
                            0: -1]
        dstSequenceDouble = np.concatenate((dstSequenceSingle, dstSequenceSingle))
        currentDstSequence = currentDstSequence[currentDstSequence >= dstIdentificationSequence[0]]
        currentDstSequence = currentDstSequence[currentDstSequence <= dstIdentificationSequence[-1]]
        currentDstSequenceSlice = timestampSequence.slice_locs(min(currentDstSequence), max(currentDstSequence))
        startIndex = currentDstSequenceSlice[0]
        endIndex = currentDstSequenceSlice[-1]
        if len(currentDstSequence) == hourlyDataPoints * 4:
            timestampSequence.values[startIndex + hourlyDataPoints: endIndex - hourlyDataPoints] = dstSequenceDouble
            dataFrame.set_index(timestampSequence, inplace=True)
            return dataFrame
        elif len(currentDstSequence) == hourlyDataPoints * 3:
            missingRecords = dataFrame[startIndex + hourlyDataPoints:endIndex - hourlyDataPoints].copy()
            dataFrame = dataFrame.append(missingRecords)
            dataFrame.sort_index(inplace=True)
            return fixDSTKnownIssues(dataFrame, config)
        else:
            ### to be implemented later
            pass


def getDateFormat(inputDate):
    """
    This function infers the date format of a given date

    Input(s):
    - inputDate: (str) date time values that needs format identification

    Output(s):
    - Date format of the passed date
    """
    for dateFormat in DATE_FORMAT_REGEX:
        if re.match(dateFormat, inputDate) is not None:
            return DATE_FORMAT_REGEX[dateFormat]


def convertTimezone(sourceFileName, config, sourceDataFrame=[]):
    """
    This function changes the timezone of a given file/ dataFrame.
    Before timezone changes are applied, date gaps and DST issues are fixed.
    After conversion all columns are dropped except the following columns:
    - 'timeStamp' column is kept as an index
    - Column(s) defined in the 'COLUMN_OF_INTEREST' configuration for each file type

    Input(s):
    - sourceFileName: The relative or absolute path to the CSV file that needs to be converted
    - config: The configuration dictionary relevant to the file type
    - sourceDataFrame: Optional pandas dataFrame that can be passed when no disk I/O is wished and only
    in-memory pipeline processing is required. passing this argument will also prevent
    the saving of the CSV file to disk and instead the data will be returned to the calling function.

    Output(s):
    - Pandas DataFrame after conversion of sourceDataFrame is not null otherwise,
        processed data is saved to the sourceFileName
    """
    if len(sourceDataFrame) > 0:
        dataset = sourceDataFrame
    else:
        dataset = pd.read_csv(sourceFileName, delimiter=config[DELIMITER], encoding='unicode_escape', dtype=str)

    timeStamp = dataset[config[DATETIME_COLUMNS][0]]
    for i in range(1, len(config[DATETIME_COLUMNS]), 1):
        timeStamp = timeStamp + ' ' + dataset[config[DATETIME_COLUMNS][i]]

    dtFormat = getDateFormat(timeStamp[0])

    timeStamp = pd.Index(
        np.array(pd.to_datetime(timeStamp, format=dtFormat, infer_datetime_format=False)))

    dataset.set_index(timeStamp, inplace=True)
    dataset.index.name = 'TIME_STAMP'

    checkDateGaps(timeStamp, config[SOURCE_SAMPLE_RATE])
    dataset = fixDateGaps(dataset, config)
    dataset = fixDSTKnownIssues(dataset, config)

    timeStamp = dataset.index.tz_localize(config[SOURCE_TIMEZONE], ambiguous='infer',
                                          nonexistent='shift_forward')

    timeStamp = timeStamp.tz_convert(config[TARGET_TIMEZONE])

    timeStamp = timeStamp.tz_localize(None)
    dataset.set_index(timeStamp, inplace=True)
    dataset.index.name = 'TIME_STAMP'

    checkDateGaps(timeStamp)
    columnsToDrop = np.setdiff1d(dataset.columns, config[COLUMN_OF_INTEREST])
    dataset = dataset.drop(columnsToDrop, axis=1)
    dataset.to_csv(sourceFileName, sep=config[DELIMITER], encoding='UTF-8', index=True)
    print("Finished timezone conversion")


def correctAnomalies(series, correctPattern):
    """
    This function checks the value inside a given column (pandas series) for anomalies/invalid values and tries to
    correct found issues by removing leading/trailing invalid characters. The validation is based on the regex defined
    for each column in the dictionary COLUMN_DATA_REGEX.

    If the data issue is beyond leading/trailing invalid characters,
    the entire value is deemed invalid and is replaced by the preceding value in the series

    Input(s):
    - series: Pandas string series to be checked for characters.
    - correctPattern: RexEx for the suitable for the passed series

    Output(s):
    - None. The corrections are made directly to the passed pandas series
    """
    for i in range(len(series)):
        try:
            tempVal = str(series[i])
            if len(re.sub(correctPattern, '', tempVal)) > 0:
                series.at[i] = re.match(correctPattern, tempVal).group()
        except Exception:
            series.at[i] = series.at[i - 1]


def csvInitialCleanup(config, sourceFileName='', sourceDataFrame=[]):
    """
    This methods performs the following preliminary cleanup actions:
    - Drops columns with all null values
    - Drops rows with all null values
    - Renames raw column names to standardized file names as defined in the configuration
    - Removes invalid characters and handle missing data by copying the value from the preceding row

    Input(s)
    - sourceFileName: The relative or absolute path to the CSV file that needs to be converted
    - config: The configuration dictionary relevant to the file type
    - sourceDataFrame: Optional pandas dataFrame that can be passed when no disk I/O is wished and only
         in-memory pipeline processing is required. passing this argument will also prevent
         the saving of the CSV file to disk and instead the data will be returned to the calling function.
    Output:
    - Updated dataFrame if sourceDataFrame is not empty, otherwise, the updated data is saved to the CSV file
    """
    delimiter = config[DELIMITER]
    columnMapping = config[COLUMN_MAPPING]

    if len(sourceDataFrame) == 0:
        dataFrame = pd.read_csv(sourceFileName, sep=';', dtype=str)
    else:
        dataFrame = sourceDataFrame

    # remove empty rows and columns
    print("Attempting to remove empty rows")
    dataFrame.dropna(axis=0, how='all', inplace=True)
    print("Successfully removed empty rows")

    print("Attempting to remove empty columns")
    dataFrame.dropna(axis=1, how='all', inplace=True, thresh=10)
    print("Successfully removed empty columns")

    columns = dataFrame.columns.values

    print("Attempting to change column names")
    for i in range(len(columns)):
        columns[i] = columnMapping[str.upper(columns[i])]
    print("Successfully changed column names")

    # remove invalid content in each cell
    print("Attempting to correct data anomalies")
    for columnName in dataFrame.columns:
        correctAnomalies(dataFrame[columnName], COLUMN_DATA_REGEX[columnName])
    print("Successfully corrected data anomalies")

    if len(sourceDataFrame) == 0:
        print("Saving to file: " + str(sourceFileName))
        dataFrame.to_csv(sourceFileName, sep=delimiter, index=False)
    return dataFrame


def processRawFiles(pathToFiles):
    """
    This method performs a one-stop turnkey pre-processing of a group of files.
    The user simply needs to define the location of the files.
    All the following situations are covered:
    - Files with various data types
    - Duplicate files with the same content
    - Files with missing data
    - Files with incorrect data
    - Files with incorrect DST entries
    - Files with German number format that needs conversion to English format
    - Files with inconsistent column names
    - Files with inconsistent date format
    - Files that need re-sampling
    - Files that need timezone conversion
    The method will automatically apply all needed actions in accordance to the configuration defined for each file type
    and then combine all the rows of similar files into one big file and finally combine the columns of all files
    into an even bigger file.

    Please note that any file type with no configuration will be ignored.

    input(s)
    - pathToFiles: the location of raw data files to be processed.
    """
    processedFilesPath = 'Processed Files'
    startTime = datetime.now()
    os.chdir(pathToFiles)
    if not os.path.exists(processedFilesPath):
        os.makedirs(processedFilesPath)
    for config in FILE_CONFIG_LIST:
        os.chdir(pathToFiles)
        rawFiles = [i for i in glob.glob(config[FILE_NAME_PATTERN_WINDOWS])]
        for rawFile in rawFiles:
            os.chdir(pathToFiles)
            print("processing file:" + str(rawFile))
            removeFileHeader(rawFile, processedFilesPath, config[HEADER_ROWS], config[SOURCE_ENCODING])
            csvInitialCleanup(config, rawFile)
            processGermanNumerals(rawFile, [], config[DELIMITER],
                                  config[GERMAN_NUMERALS_CONVERSION])
            convertTimezone(rawFile, config)
            resampleCSV(rawFile, rawFile, config[DELIMITER], config[COLUMN_OF_INTEREST], '1H',
                        config[SOURCE_SAMPLE_RATE])
        mergeCSVFiles(os.getcwd(), [], config[COMBINED_FILE_NAME], config[FILE_NAME_PATTERN_WINDOWS], config[DELIMITER]
                      , False)
    allFiles = []
    for config in FILE_CONFIG_LIST:
        allFiles.append(config[COMBINED_FILE_NAME])
    mergeCSVFiles(pathToFiles, allFiles, 'Final Combined Data.csv', '', config[DELIMITER], False, 1)

    print("All files processed successfully in " + str(datetime.now() - startTime))


#########################   Test   #########################
rawFilesPath = r'C:\Users\Owner\Desktop\Thesis\Temp\Misc'
processRawFiles(rawFilesPath)
