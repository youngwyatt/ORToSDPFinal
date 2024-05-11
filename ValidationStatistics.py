import os
import pandas as pd
from scipy.io import loadmat
from PatientDataMerger import generatePtDataDFs, addPtData
from scipy.stats import pearsonr
import numpy as np
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

def bufferZeros(dataArray):
    # Pull a patients hour and minutes from the datetime object
    # try catch for case of patient not having recorded time of appointment        # Calculate number of zeros to use as buffer from midnight
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
        print('Warning: buffered array length is not a multiple of 1382400. The buffered array isnt divisible into full days of data.')
    print('Patient recorded data for {} days, {} hours, and {} minutes'.format(numOfDays, numOfHours, numOfMinutes))
    return bufferedArray

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
                ErrorSignatures.setToAverage(i, normed)
            elif normed[i] < 0:
                ErrorSignatures.setToAverage(i, normed)
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



def generatePtDataDFs():
    """
    Generates sensor data DataFrames for a specified patient ID.

    Returns:
        Four DataFrames for total, heel, medial and lateral data respectively
        and the patient ID as the last element for .txt file labeling.
    """
    # Create patient dictionary
    # Prompt user to enter a patient ID to generate sensor dataframes for
    DfTotal, DfHeel, DfMedial, DfLateral = addPtData()
    # Return the four dataframes and the patient ID
    return DfTotal, DfHeel, DfMedial, DfLateral


def addPtData():
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

    # Normalize data to percentage of patients bodyweight
    Heel = normalizeToLbs(heelCounts)
    Medial = normalizeToLbs(medialCounts)
    Lateral = normalizeToLbs(lateralCounts)
    print("Sensor data has been normalized to percentage of pt weights!")

    # Buffer data to start at midnight and end at midnight
    bufferedHeel = bufferZeros(Heel)
    bufferedMedial = bufferZeros(Medial)
    bufferedLateral = bufferZeros(Lateral)

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
    DfTotal.columns = range(0, dayCount-1)
    DfHeel.columns = range(1, dayCount)
    DfMedial.columns = range(1, dayCount)
    DfLateral.columns = range(1, dayCount)

    print("Data added, checking DataFrames...")
    # print("DfTotal:")
    # print(DfTotal)
    # print("DfHeel:")
    # print(DfHeel)
    # print("DfMedial:")
    # print(DfMedial)
    # print("DfLateral:")
    # print(DfLateral)
    # Return dataframes
    return DfTotal, DfHeel, DfMedial, DfLateral
def importMat(fileName):
    #  Import Matlab workspaces to pandas dataframes
    mat_data = loadmat(fileName)
    data_matrix = mat_data['untotal']
    df = pd.DataFrame(data_matrix)
    print(df)
    return df
# Generate total df w python pipeline
def pythonDf():
    DfTotal, _, _, _ = generatePtDataDFs()
    return DfTotal
def checkConstantColumns(df):
    for col in df.columns:
        if len(df[col].unique()) == 1:
            # Add noise at every 10th row
            index = range(0, len(df), 10)
            df.loc[index, col] += 0.001
    return df

matFile = "CT11T.mat"
matlab = importMat(matFile)
matlab = checkConstantColumns(matlab)
python = pythonDf()
# Trim the Python Dataframe to match the number of columns in the matlab DataFrame
numMatlaCols = matlab.shape[1]
python = python.iloc[:, :numMatlaCols]
print(python)
python = checkConstantColumns(python)
# Pearson correlation coefficients
correlations = []
for col in python.columns:
    corr, _ = pearsonr(python[col], matlab[col])
    correlations.append(corr)
print(correlations)
avgCorr = sum(correlations) / len(correlations)
print("Average Pearson Correlation:", avgCorr)
correlationsDf = pd.DataFrame(correlations, columns=["Pearson Coefficients"], index=python.columns)
# Save to Excel
sheet_name = os.path.splitext(os.path.basename(matFile))[0]+'PC.xlsx'
correlationsDf.to_excel(sheet_name, sheet_name)
