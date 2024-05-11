import datetime
import sys
import warnings
import pandas as pd
import numpy as np
from PySide6.QtWidgets import QApplication, QFileDialog

from BinaryParseV2 import binary_parse
import ErrorSignatures
class Patient:
    """
    A class to represent a Patient. Holds relevant patient info to grab from excel sheet.
    Methods
    -------
    bufferZeros(self, dataArray):
    Pads the input data array with zeros so the data starts and ends at midnight. Returns the buffered array.

    ptNormalizeSensor(self, dataArray):
        Normalizes the input array to lbs and then to a percentage of the patient's body weight. Returns the normalized array.
    """
    def __init__(self, ID, MRN, Procedure, DateOfSurgery, Wk2ApptTime, PreInjuryAmbulation, PreInjuryWork,
                 Wk2Pam, OneYrPam, OTAClass, Height, Weight, BMI, Gender, Race, DoB, Age):
        self.ID = ID
        self.MRN = MRN if MRN is not None else -1
        self.Procedure = Procedure if Procedure is not None else -1
        self.DateOfSurgery = DateOfSurgery if DateOfSurgery is not None else -1
        self.Wk2ApptTime = Wk2ApptTime if Wk2ApptTime is not None else -1
        self.PreInjuryAmbulation = PreInjuryAmbulation if PreInjuryAmbulation is not None else -1
        self.PreInjuryWork = PreInjuryWork if PreInjuryWork is not None else -1
        self.Wk2Pam = Wk2Pam if Wk2Pam is not None else -1
        self.OneYrPam = OneYrPam if OneYrPam is not None else -1
        self.OTAClass = OTAClass if OTAClass is not None else -1
        self.Height = Height if Height is not None else -1
        self.Weight = Weight if Weight is not None else -1
        self.BMI = BMI if BMI is not None else -1
        self.Gender = Gender if Gender is not None else -1
        self.Race = Race if Race is not None else -1
        self.DoB = DoB if DoB is not None else -1
        self.Age = Age if Age is not None else -1  # If age is not given, set to -1

    def __str__(self):
        return f'Patient {self.ID}: {self.Weight} lbs, {self.Age} years old'

    def bufferZeros(self, dataArray):
        # Pull a patients hour and minutes from the datetime object
        # try catch for case of patient not having recorded time of appointment
        try:
            if self.Wk2ApptTime.hour == 0 and self.Wk2ApptTime.minute == 0 \
                    and self.Wk2ApptTime.second == 0:
                raise ValueError
            hours = self.Wk2ApptTime.hour
            minutes = self.Wk2ApptTime.minute
            seconds = self.Wk2ApptTime.second
        except ValueError:
            defaultTime = 10
            print(f'Error: No appointment time found, using default time for appointment: {defaultTime}:00 AM')
            hours = defaultTime
            minutes = 0
            seconds = 0
        # Calculate number of zeros to use as buffer from midnight
        secondsFromMidnight = (hours * 60 * 60) + (minutes * 60) + seconds
        bufferZeros = secondsFromMidnight * 16  # 16 samples per second
        print(f'Buffering with {bufferZeros} zeros to start at midnight on {self.Wk2ApptTime.date() - datetime.timedelta(days=1)}')
        # Pad data array with zeros to ensure data starts at midnight
        # zeroArray = np.zeros(bufferZeros, dtype=int)
        # bufferedArray = np.append(zeroArray, dataArray)
        # Now determine the number of zeros to add to the end of the array
        # Determine number of days, hours, and minutes of recording
        # 1382400 indices per day, 57600 indices per hour, 960 indices per minute, 16 indices per second
        numOfDays = len(dataArray) // 1382400
        remainingIndices = len(dataArray) % 1382400
        numOfHours = remainingIndices // 57600
        remainingIndices %= 57600
        numOfMinutes = remainingIndices // 960
        remainingIndices %= 960
        numOfSeconds = remainingIndices // 16
        remainingIndices %= 16
        # Determine amount of indices before midnight of last day of recording
        # Total length of array after adding zeros at the beginning
        totalLength = len(dataArray)
        # Calculate the remainder when total length is divided by 1382400
        remainder = totalLength % 1382400
        # If remainder is not zero, calculate the difference to 1382400
        if remainder != 0:
            zerosToEnd = 1382400 - remainder
            # Pad data array with zeros to ensure data ends at midnight
            zeroEndArray = np.zeros(zerosToEnd, dtype=int)
            bufferedArray = np.append(dataArray, zeroEndArray)
        # Check if the resulting buffer length is a multiple of 1382400, error catch
        if len(bufferedArray) % 1382400 != 0:
            print(
                'Warning: buffered array length is not a multiple of 1382400. The buffered array isnt divisible into full days of data.')
        print('Patient recorded data for {} days, {} hours, and {} minutes'.format(numOfDays, numOfHours, numOfMinutes))
        return bufferedArray

    def ptNormalizeSensor(self, dataArray):
        # Normalize sensor counts to lbs
        normLbs = normalizeToLbs(dataArray)
        # Normalize sensor data to percentage of patients bodyweight
        print(f'Normalizing to {self.Weight} lbs')
        normPercent = ((normLbs / self.Weight) * 100)
        return normPercent


def setToAverage(i, dataArray):
    """
    Set the value at index i of dataArray to the average of its neighboring values.

    Parameters:
        i : int
            The index at which to set the value.
        dataArray : Array
            The data array in which to set the value.
    """
    if i == 0:
        dataArray[i] = (dataArray[i + 1])
    elif i == len(dataArray) - 1:
        dataArray[i] = ((dataArray[i] + dataArray[i - 1]) / 2)
    else:
        dataArray[i] = ((dataArray[i - 1] + dataArray[i + 1]) / 2)


def normalizeToLbs(counts):
    """
    Normalizes the input array to lbs, using a conversion factor derived from a maximum weight of 250 lbs.

    Parameters:
        counts : Array
            The data array to be normalized.

    Returns:
        Array : The normalized data array.
    """
    if counts is None:
        print('Error: counts is None')
        return None
    try:
        # Normalize sensor counts to 250 lbs(250/2^14)
        normed = counts * (250 / 2 ** 14)
        # Recheck data array for obvious errors
        for i in range(len(normed)):
            if normed[i] > 249:
                setToAverage(i, normed)
            elif normed[i] < 0:
                setToAverage(i, normed)
        return normed
    except TypeError:
        print('Error: counts not correct type')
        return counts


def trimTotal(dataArray, dataArrayTwo, dataArrayThree):
    """
    Accepts three data arrays, trims them to the size of the smallest array, adds up the corresponding values in
    the three arrays, and returns a total array.

    Parameters:
        dataArray : Array
            The first data array.
        dataArrayTwo : Array
            The second data array.
        dataArrayThree : Array
            The third data array.

    Returns:
        Array : The total array, which is the sum of the three input arrays.
    """
    length = min(len(dataArray), len(dataArrayTwo), len(dataArrayThree))
    totalArray = dataArray[:length] + dataArrayTwo[:length] + dataArrayThree[:length]
    return totalArray


def generatePatientDictionary():
    """
    Generates a dictionary of Patient objects using data from an Excel file. The keys are patient ID numbers.

    Returns:
        dict : The dictionary of Patient objects.
    """
    # Disable print area warning
    warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')
    # Create PySide6 Qt application
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    # Use QFileDialog to prompt the user to select a .bin file
    fileName, _ = QFileDialog.getOpenFileName(None, 'Select patient PRO and info excel', "","Excel files (*.xlsx *.xls)")
    # Store file path and prompt user for sheet name
    sheetName = input('Enter sheet name: ')
    excelFile = fileName
    # Read in excel into panda df, column names are row 2 thus header=1
    ptData = pd.read_excel(excelFile, sheet_name=sheetName, header=1, parse_dates=['Wk2Assessment Date and Time'])
    # Only keep rows containing patient data, C or A for CT and AB indexing
    ptData = ptData[ptData['ID#'].astype(str).str.startswith('C')]
    # Initialize patient dictionary
    ptDict = {}
    # Fill dictionary from df
    for index, row in ptData.iterrows():
        ID = row['ID#']
        MRN = row['MRN']
        Procedure = row['Procedure']
        DateOfSurgery = row['Date of Surgery']
        # Use try catch to handle invalid date time cells
        try:
            Wk2ApptTime = pd.to_datetime(row['Wk2Assessment Date and Time'])
        except ValueError:
            Wk2ApptTime = None
        PreInjuryAmbulation = row['Pre-Injury Ambulation']
        PreInjuryWork = row['Pre-Injury Work']
        Wk2Pam = row['Wk2PAM13 Level']
        OneYrPam = row['Yr1PAM 13 Level']
        OTAClass = row['OTA Classification']
        Height = row['Height (cm)']
        Weight = row['Weight (lb)']
        BMI = row['BMI']
        Gender = row['Gender']
        Race = row['Race/Ethnicity']
        DoB = row['DOB']
        Age = row['Age at Injury']
        # Create patient object and add to dictionary
        ptDict[ID] = Patient(ID, MRN, Procedure, DateOfSurgery, Wk2ApptTime, PreInjuryAmbulation, PreInjuryWork,
                             Wk2Pam, OneYrPam, OTAClass, Height, Weight, BMI, Gender, Race, DoB, Age)
    # Return the patient dictionary
    return ptDict


def generatePtDataDFs():
    """
    Generates sensor data DataFrames for a specified patient ID.

    Returns:
        Four DataFrames for total, heel, medial and lateral data respectively
        and the patient ID as the last element for .txt file labeling.
    """
    # Create patient dictionary
    ptDict = generatePatientDictionary()
    # Prompt user to enter a patient ID to generate sensor dataframes for
    ptID = input(f'Enter Pt ID to create sensor dataframes for(Example: CT02, CT14, and so on):')
    if ptID in ptDict.keys():
        # If the patient ID exists in the patient dictionary, process data for that patient
        DfTotal, DfHeel, DfMedial, DfLateral = addPtData(ptID, ptDict)
    else:
        # If the patient ID is not found in the patient dictionary, print an error and return None
        print('Error: patient ID not found! Or incorrect formatting')
        return None
    # Return the four dataframes and the patient ID
    return DfTotal, DfHeel, DfMedial, DfLateral, ptID


def addPtData(ptID, ptDict):
    """
    Processes and adds patient data to separate dataframes for total, heel, medial, and lateral sensor data.

    Args:
        ptID (str): The ID of the patient whose data is to be processed.
        ptDict (dict): The dictionary of Patient objects keyed by patient ID.

    Returns:
        Four pandas DataFrames for total, heel, medial, and lateral sensor data to be passed and returned by
        generatePtDataDfs function
    """
    # Extract counts from binary parse
    heelCounts, medialCounts, lateralCounts, _, _, _, _, _ = binary_parse()
    print("Binary data successfully parsed!")

    # Detect error signatures
    heelCountsChecked = ErrorSignatures.errorSignatures(heelCounts)
    medialCountsChecked = ErrorSignatures.errorSignatures(medialCounts)
    lateralCountsChecked = ErrorSignatures.errorSignatures(lateralCounts)
    print("Checked arrays for errors!")

    # Normalize data to percentage of patients bodyweight
    percentHeel = ptDict[ptID].ptNormalizeSensor(heelCountsChecked)
    percentMedial = ptDict[ptID].ptNormalizeSensor(medialCountsChecked)
    percentLateral = ptDict[ptID].ptNormalizeSensor(lateralCountsChecked)
    print("Sensor data has been normalized to percentage of pt weights!")

    # Buffer data to start at midnight and end at midnight
    bufferedHeel = ptDict[ptID].bufferZeros(percentHeel)
    bufferedMedial = ptDict[ptID].bufferZeros(percentMedial)
    bufferedLateral = ptDict[ptID].bufferZeros(percentLateral)

    # Ensure data is trimmed to same length, store total array
    bufferedTotal = ErrorSignatures.trimTotal(bufferedHeel, bufferedMedial, bufferedLateral)

    # Create empty dataframes for each sensor
    DfTotal = pd.DataFrame()
    DfHeel = pd.DataFrame()
    DfMedial = pd.DataFrame()
    DfLateral = pd.DataFrame()
    print("Adding and splicing patient data arrays!")

    # Split data into daily arrays and store into column of dataframe
    dayCount = 1
    for i in range(0, len(bufferedTotal), 1382400):  # iterate over 80 days
        # Cut data into daily arrays
        dailyArrayTotal = bufferedTotal[i: i + 1382400].tolist()
        dailyArrayHeel = bufferedHeel[i: i + 1382400].tolist()
        dailyArrayMedial = bufferedMedial[i: i + 1382400].tolist()
        dailyArrayLateral = bufferedLateral[i: i + 1382400].tolist()

        # Convert daily arrays to single-row dataframes
        dfDayTotal = pd.DataFrame([dailyArrayTotal])
        dfDayHeel = pd.DataFrame([dailyArrayHeel])
        dfDayMedial = pd.DataFrame([dailyArrayMedial])
        dfDayLateral = pd.DataFrame([dailyArrayLateral])

        # Add the daily dataframes to the total dataframes, transposing each daily df to be added as a new column
        DfTotal = pd.concat([DfTotal, dfDayTotal.T], axis=1)  # Transpose dfDayTotal
        DfHeel = pd.concat([DfHeel, dfDayHeel.T], axis=1)  # Transpose dfDayHeel
        DfMedial = pd.concat([DfMedial, dfDayMedial.T], axis=1)  # Transpose dfDayMedial
        DfLateral = pd.concat([DfLateral, dfDayLateral.T], axis=1)  # Transpose dfDayLateral

        # Increment day count for column labeling
        dayCount += 1
    # Rename the columns with day numbers
    DfTotal.columns = range(1, dayCount)
    DfHeel.columns = range(1, dayCount)
    DfMedial.columns = range(1, dayCount)
    DfLateral.columns = range(1, dayCount)

    print("Data added, checking DataFrames...")
    print("DfTotal:")
    print(DfTotal)
    print("DfHeel:")
    print(DfHeel)
    print("DfMedial:")
    print(DfMedial)
    print("DfLateral:")
    print(DfLateral)
    # Return dataframes
    return DfTotal, DfHeel, DfMedial, DfLateral


# Call pt data frame generator and enter patient ID to create DFs for, use:
DfTotal, DfHeel, DfMedial, DfLateral, ptID = generatePtDataDFs()
# In order to do computations on daily arrays of data from Data Frames:
# day5Data = np.array(DfTotal[5])
# Capture entire data array from df
# allData = DfTotal.transpose().values.flatten()
# allDataLateral = DfLateral.transpose().values.flatten()
# allDataHeel = DfHeel.transpose().values.flatten()
# plt.plot(allData)
# plt.plot(allDataHeel)
# plt.plot(allDataLateral)
# plt.show()
# Convert Df to tab delimited text file
# DfTotal.to_csv(f'Patient{ptID}SensorData.txt', sep='\t') # add parameter index=False to not include indices
