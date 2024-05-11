import pandas as pd
from ErrorSignatures import peakFinder
import os
import sys
from PySide6.QtWidgets import QApplication, QFileDialog
import numpy as np
from BinaryParseV2 import fread


# Python file to generate error data analysis table in excel with supporting functions
# Author Wyatt Young
# Version June 25, 2023
def extractBinFiles():
    # Create .bin file array
    binArray = []
    number = int(input("How many patients do you have?: "))
    for i in range(number):
        # Create PySide6 Qt application
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)
        # Use QFileDialog to prompt the user to select a .bin file
        fileName, _ = QFileDialog.getOpenFileName(None, 'Open Binary File', "", "Binary files (*.bin)")
        binArray.append(fileName)
    return binArray


def parseAndExtractErrors(binArray):
    # Initialize list to nest data dictionary in
    errorData = []
    # Loop through each patients .bin to parse and find errors
    for binFile in binArray:
        try:
            print(f"Processing file: {binFile}")
            # Run binary parser for sensor arrays
            Heel, Lateral, Medial, _, _, _, _, _ = binary_parse_edited(binFile)
            print("Binary parsing completed")

            # Errors for heel
            saturations_H, sensorSpikes_H, sensorSpikesArray_H, triangleArtifacts_H, triangleArtifactsArray_H, dipsBetweenPeaks_H, dipsBetweenPeaksArray_H, _ \
                = errorSignaturesEdited(Heel)
            print("Heel error detection completed")

            # Errors for lateral
            saturations_L, sensorSpikes_L, sensorSpikesArray_L, triangleArtifacts_L, triangleArtifactsArray_L, dipsBetweenPeaks_L, dipsBetweenPeaksArray_L, _ \
                = errorSignaturesEdited(Lateral)
            print("Lateral error detection completed")

            # Errors for medial
            saturations_M, sensorSpikes_M, sensorSpikesArray_M, triangleArtifacts_M, triangleArtifactsArray_M, dipsBetweenPeaks_M, dipsBetweenPeaksArray_M, _ \
                = errorSignaturesEdited(Medial)
            print("Medial error detection completed")

            # Store data in dictionary
            binData = {
                "fileName": binFile,
                "Heel": {
                    "heelData": Heel,
                    "sensorSpikes_H": sensorSpikes_H,
                    "sensorSpikesArray_H": sensorSpikesArray_H,
                    "triangleArtifacts_H": triangleArtifacts_H,
                    "triangleArtifactsArray_H": triangleArtifactsArray_H,
                    "dipsBetweenPeaks_H": dipsBetweenPeaks_H,
                    "dipsBetweenPeaksArray_H": dipsBetweenPeaksArray_H,
                    "OverSaturations_H": saturations_H,
                },
                "Lateral": {
                    "lateralData": Lateral,
                    "sensorSpikes_L": sensorSpikes_L,
                    "sensorSpikesArray_L": sensorSpikesArray_L,
                    "triangleArtifacts_L": triangleArtifacts_L,
                    "triangleArtifactsArray_L": triangleArtifactsArray_L,
                    "dipsBetweenPeaks_L": dipsBetweenPeaks_L,
                    "dipsBetweenPeaksArray_L": dipsBetweenPeaksArray_L,
                    "OverSaturations_L": saturations_L,
                },
                "Medial": {
                    "medialData": Medial,
                    "sensorSpikes_M": sensorSpikes_M,
                    "sensorSpikesArray_M": sensorSpikesArray_M,
                    "triangleArtifacts_M": triangleArtifacts_M,
                    "triangleArtifactsArray_M": triangleArtifactsArray_M,
                    "dipsBetweenPeaks_M": dipsBetweenPeaks_M,
                    "dipsBetweenPeaksArray_M": dipsBetweenPeaksArray_M,
                    "OverSaturations_M": saturations_M
                }
            }
            print("Data dictionary created")

            # Append dictionary to list
            errorData.append(binData)
            print("Data appended to list")

        except Exception as e:
            print(f"Error in parsing data for {binFile}: {e}")
            break
    return errorData


def prepareErrorData(errorData):
    processedErrorData = []

    for binData in errorData:
        # Extract number of days recorded
        Heel = binData['Heel']['heelData']
        recordedDays = indicesToDaysEdited(Heel)
        print(f"Recorded days: {recordedDays}")

        # Initialize error day arrays
        spikesErrors_H = [0] * (recordedDays + 2)
        triangleErrors_H = [0] * (recordedDays + 2)
        dipsErrors_H = [0] * (recordedDays + 2)
        spikesErrors_L = [0] * (recordedDays + 2)
        triangleErrors_L = [0] * (recordedDays + 2)
        dipsErrors_L = [0] * (recordedDays + 2)
        spikesErrors_M = [0] * (recordedDays + 2)
        triangleErrors_M = [0] * (recordedDays + 2)
        dipsErrors_M = [0] * (recordedDays + 2)

        # Import errors into arrays at correct day
        for sensor, sensorKey in zip(['H', 'L', 'M'], ['Heel', 'Lateral', 'Medial']):
            if binData[sensorKey]['sensorSpikesArray_' + sensor]:  # Check if not empty
                for error in binData[sensorKey]['sensorSpikesArray_' + sensor]:
                    day = indexToDay(error)
                    eval(f'spikesErrors_{sensor}')[day] += 1
            if binData[sensorKey]['triangleArtifactsArray_' + sensor]:  # Check if not empty
                for error in binData[sensorKey]['triangleArtifactsArray_' + sensor]:
                    day = indexToDay(error)
                    eval(f'triangleErrors_{sensor}')[day] += 1
            if binData[sensorKey]['dipsBetweenPeaksArray_' + sensor]:  # Check if not empty
                for error in binData[sensorKey]['dipsBetweenPeaksArray_' + sensor]:
                    day = indexToDay(error)
                    eval(f'dipsErrors_{sensor}')[day] += 1

        # Store processed error data for each sensor
        for sensor in ['H', 'L', 'M']:
            allErrors = np.add(np.array(eval(f'spikesErrors_{sensor}')), np.array(eval(f'triangleErrors_{sensor}')),
                               np.array(eval(f'dipsErrors_{sensor}')))
            totalErrors = np.sum(allErrors)
            # Check for non-zero errors before assigning firstError and lastError
            if np.any(allErrors):
                firstError = np.nonzero(allErrors)[0][0] + 1
                lastError = np.nonzero(allErrors)[0][-1] + 1
                meanErrorDay = (np.sum(allErrors * np.arange(len(allErrors))) / np.sum(allErrors)) + 1
            else:
                firstError = None
                lastError = None
                meanErrorDay = None
            # Add to dictionary
            binData['totalErrors_' + sensor] = totalErrors
            binData['allErrors_' + sensor] = allErrors
            binData['firstError_' + sensor] = firstError
            binData['lastError_' + sensor] = lastError
            binData['meanErrorDay_' + sensor] = meanErrorDay

        processedErrorData.append(binData)

    return processedErrorData
def exportToExcel(processedErrorData, dataQueries):
    for data in processedErrorData:
        # Create dictionary with data queries
        queries = {
            'File': data['fileName'],
            'sensorSpikes_H': len(data['Heel']['sensorSpikesArray_H']),
            'sensorSpikes_L': len(data['Lateral']['sensorSpikesArray_L']),
            'sensorSpikes_M': len(data['Medial']['sensorSpikesArray_M']),
            'triangleArtifacts_H': len(data['Heel']['triangleArtifactsArray_H']),
            'triangleArtifacts_L': len(data['Lateral']['triangleArtifactsArray_L']),
            'triangleArtifacts_M': len(data['Medial']['triangleArtifactsArray_M']),
            'dipsBetweenPeaks_H': len(data['Heel']['dipsBetweenPeaksArray_H']),
            'dipsBetweenPeaks_L': len(data['Lateral']['dipsBetweenPeaksArray_L']),
            'dipsBetweenPeaks_M': len(data['Medial']['dipsBetweenPeaksArray_M']),
            'OverSaturations_H':data['Heel']['OverSaturations_H'],
            'OverSaturations_L': data['Lateral']['OverSaturations_L'],
            'OverSaturations_M': data['Medial']['OverSaturations_M'],
            'totalErrors_H': data['totalErrors_H'],
            'totalErrors_L': data['totalErrors_L'],
            'totalErrors_M': data['totalErrors_M'],
            'firstHeelErrorDay': data['firstError_H'],
            'firstLateralErrorDay': data['firstError_L'],
            'firstMedialErrorDay': data['firstError_M'],
            'lastHeelErrorDay': data['lastError_H'],
            'lastLateralErrorDay': data['lastError_L'],
            'lastMedialErrorDay': data['lastError_M'],
            'meanHeelErrorDay': data['meanErrorDay_H'],
            'meanLateralErrorDay': data['meanErrorDay_L'],
            'meanMedialErrorDay': data['meanErrorDay_M'],
        }
        # Extract file name instead of full path for indexing
        fileName = os.path.splitext(os.path.basename(data['fileName']))[0]
        # Convert data from dict to df and append to list
        dataQueries.append(pd.DataFrame(queries, index=[fileName]))
    return dataQueries


def errorTableGenerator():
    # Get .bin files
    binFiles = extractBinFiles()
    print("Bin files extracted!")

    # Extract error data from each .bin file
    errorData = parseAndExtractErrors(binFiles)
    print("Data parsed and errors extracted to errorData nested list")

    # Prepare the error data for exporting to Excel
    processedErrorData = prepareErrorData(errorData)
    print("Data prepped to be moved to excel")

    # Initialize an empty list for dataQueries
    dataQueries = []

    # Export data to Excel
    dataQueries = exportToExcel(processedErrorData, dataQueries)
    print("Data queries prepared to go to excel")

    # Concatenate all dataframes into a single one
    final_df = pd.concat(dataQueries)

    # Export to Excel
    final_df.to_excel("errorDataV2.xlsx")

def errorSignaturesEdited(dataArray, saturationThreshold=15000, neighborThreshold=11500, neighborRange=250,
                          spikeFactor=0.08):
    # Finding previously detected errors
    errorIndices = np.where((dataArray >= 16380) | (dataArray < 0))
    fullySaturated = len(errorIndices[0])
    print(f'Oversaturations: {fullySaturated}')
    # Extract peak indices and counts
    peakIndices, peakCounts = peakFinder(dataArray)
    # Initizalize error counters and arrays for plotting
    sensorSpikes = 0
    sensorSpikesArray = []
    triangleArtifacts = 0
    triangleArtifactsArray = []
    dipsBetweenPeaks = 0
    dipsBetweenPeaksArray = []
    print('Searching for error signatures!')
    for i in range(len(peakIndices) - 1):
        peakIndex = peakIndices[i]

        # Set triangle shape peaks to zero(error artifacts)
        if dataArray[peakIndex - 1] < 50 and dataArray[peakIndex + 1] < 50:
            triangleArtifacts += 1
            triangleArtifactsArray.append(peakIndex)
            continue

        # General sensor dips
        if dataArray[peakIndex + 1] <= 2 and dataArray[peakIndex + 2] >= 700:
            dipsBetweenPeaks += 1
            dipsBetweenPeaksArray.append(peakIndex + 1)
            continue
        if dataArray[peakIndex - 1] <= 2 and dataArray[peakIndex - 2] >= 700:
            dipsBetweenPeaks += 1
            dipsBetweenPeaksArray.append(peakIndex - 1)
            continue

        # Obvious sensor spikes to less than % of peak before or after peak
        if dataArray[peakIndex - 1] <= spikeFactor * dataArray[peakIndex]:
            if dataArray[peakIndex - 1] <= 2 and dataArray[peakIndex] >= 2000:
                sensorSpikes += 1
                sensorSpikesArray.append(peakIndex)
            else:
                sensorSpikes += 1
                sensorSpikesArray.append(peakIndex)
        elif dataArray[peakIndex + 1] <= spikeFactor * dataArray[peakIndex]:
            if dataArray[peakIndex + 1] <= 2 and dataArray[peakIndex] >= 2000:
                sensorSpikes += 1
                sensorSpikesArray.append(peakIndex)
            else:
                sensorSpikes += 1
                sensorSpikesArray.append(peakIndex)

        # Set dips between peaks to average
        if dataArray[peakIndex + 2] < 10 and dataArray[peakIndices[i + 1] - 2] < 10:
            dipsBetweenPeaks += 1
            dipsBetweenPeaksArray.append(peakIndex + 2)
        elif dataArray[peakIndex + 1] < 10 and dataArray[peakIndices[i + 1] - 1] < 10:
            dipsBetweenPeaks += 1
            dipsBetweenPeaksArray.append(peakIndex + 1)

        # Deal with positive and negative searches seperately(0 for 1,2,3 indices between peaks) start w longest range
        if dataArray[peakIndex + 1] < 2 and dataArray[peakIndex + 2] < 2 and dataArray[peakIndex + 3] < 2:
            dipsBetweenPeaks += 1
            dipsBetweenPeaksArray.append(peakIndex + 2)
        # flatten for 2 indices
        elif dataArray[peakIndex + 1] < 2 and dataArray[peakIndex + 2] < 2:
            dipsBetweenPeaks += 1
            dipsBetweenPeaksArray.append(peakIndex + 2)
        # Deal with negative search
        if dataArray[peakIndex - 1] < 2 and dataArray[peakIndex - 2] < 2 and dataArray[peakIndex - 3] < 2:
            dipsBetweenPeaks += 1
            dipsBetweenPeaksArray.append(peakIndex - 2)
        # flatten for 2 indices
        elif dataArray[peakIndex - 1] < 2 and dataArray[peakIndex - 2] < 2:
            dipsBetweenPeaks += 1
            dipsBetweenPeaksArray.append(peakIndex - 2)
        # Set specific sensor dips that occur later in array to average
        if dataArray[peakIndex] >= 2000 and dataArray[peakIndex - 2] < 2 and dataArray[peakIndex - 1] < 2:
            dipsBetweenPeaks += 1
            dipsBetweenPeaksArray.append(peakIndex - 2)
        elif dataArray[peakIndex] >= 2000 and dataArray[peakIndex + 2] < 2 and dataArray[peakIndex + 1] < 2:
            dipsBetweenPeaks += 1
            dipsBetweenPeaksArray.append(peakIndex + 2)
    totalErrors = sensorSpikes + triangleArtifacts + dipsBetweenPeaks
    print(f'Sensor Spikes: {sensorSpikes}, Triangle Artifacts: {triangleArtifacts}, Dips between peaks: {dipsBetweenPeaks}, Total Errors: {totalErrors}')
    return fullySaturated, sensorSpikes, sensorSpikesArray, triangleArtifacts, triangleArtifactsArray, dipsBetweenPeaks, dipsBetweenPeaksArray, totalErrors

def binary_parse_edited(fileName):
    """
    Parses a binary file from an Atlas module and returns parsed sampled data.

    The binary data structure is assumed as follows:
    * 4096-byte samples repeated throughout the file
    * Each sample contains:
      - 1 byte heel temp
      - 1 byte battery counts
      - 4 bytes rtc counts
      - 4 bytes x 409 samples index
      - 2 bytes x 409 samples heel counts
      - 2 bytes x 409 samples lateral counts
      - 2 bytes x 409 samples medial counts

    Returns:
    * Heel, Lateral, Medial: Arrays containing pressure data
    * heelTemperature: Array containing heel temperatures
    * batteryCounts: Array containing battery counts, converted to voltages
    * rtcCounts: Array containing RTC counts
    * index: Indexing array for the counts
    * Restart_positions: Array indicating the indices where the device was restarted
    """
    print("Parsing binary file: ", fileName)
    # Open binary file for parsing
    with open(fileName, 'rb') as fid:
        # Extract size of data
        filesize = os.path.getsize(fileName)
        # Allocate lists
        heelTemperature, batteryCounts, rtcCounts = [], [], []
        # Loop for heeltemps batterycounts and rtc, taking each 4096 byte sized sample in each iteration
        for n in range(0, filesize, 4096):
            # .Seek positions file, SEEK_SET seeks relative to beginning of the file
            fid.seek(n, os.SEEK_SET)
            # append values, B represents unsigned 8 bit integers, 1 byte read, skipping 0 bytes
            heelTemperature += fread(fid, 'B', 1, 0)
            # Repeat for battery counts and rtc, offset position by 1
            fid.seek(n + 1, os.SEEK_SET)
            batteryCounts += fread(fid, 'B', 1, 0)
            fid.seek(n + 2, os.SEEK_SET)
            # I represents unsigned 32 bit integers, < for little endian
            rtcCounts += fread(fid, '<I', 1, 0)
        # Trim zeros of heel temp and battery counts as well as conversion from list to numpy array
        heelTemperature = np.array(np.trim_zeros(heelTemperature, 'b'))
        batteryCounts = np.array(np.trim_zeros(batteryCounts, 'b'))
        # Convert to voltages, based off 19.6 mV/count
        batteryVoltages = batteryCounts * (19.6 / 1000)
        rtcCounts = np.array(rtcCounts)
        # Allocate index and pressure lists
        index, heelCounts, lateralCounts, medialCounts = [], [], [], []
        # Same looping structure, one sample each iteration for entire file size
        for n in range(0, filesize, 4096):
            fid.seek(n + 6, os.SEEK_SET)
            # Skip 6 bytes for index, grab 409 byte samples
            index += fread(fid, '<I', 409, 6)
            fid.seek(n + 10, os.SEEK_SET)
            # Skip 8 bytes for pressure readings, H represents 16 bit unsigned integers
            heelCounts += fread(fid, '<H', 409, 8)
            fid.seek(n + 12, os.SEEK_SET)
            lateralCounts += fread(fid, '<H', 409, 8)
            fid.seek(n + 14, os.SEEK_SET)
            medialCounts += fread(fid, '<H', 409, 8)
        # Convert to numpy arrays
        index = np.array(index)
        heelCounts = np.array(heelCounts)
        lateralCounts = np.array(lateralCounts)
        medialCounts = np.array(medialCounts)
    # Determine the maximum index where a non-zero value occurs and adjust array lengths
    maxLast = max(np.max(np.nonzero(heelCounts)[0]), np.max(np.nonzero(lateralCounts)[0]),
                      np.max(np.nonzero(medialCounts)[0])) + 1
    print('Trimming sensor arrays')
    # Sensor conditionals to trim each array to the proper length
    if len(heelCounts) < maxLast:
        padZerosH = maxLast - len(heelCounts)
        Heel = np.pad(heelCounts, (0, padZerosH), 'constant')
    else:
        Heel = heelCounts[:maxLast]

    if len(lateralCounts) < maxLast:
        padZerosL = maxLast - len(lateralCounts)
        Lateral = np.pad(lateralCounts, (0, padZerosL), 'constant')
    else:
        Lateral = lateralCounts[:maxLast]

    if len(medialCounts) < maxLast:
        padZerosM = maxLast - len(medialCounts)
        Medial = np.pad(medialCounts, (0, padZerosM), 'constant')
    else:
        Medial = medialCounts[:maxLast]
    # Determine restart positions by creating array of indices where index = 0
    restartPositions = np.array(np.where(index == 0)[0])
    # for i in range(len(restartPositions)):
    #     if len(restartPositions) < i + 1 and restartPositions[i + 1] - restartPositions[i] == 1:
    #         restartPositions = restartPositions[:i]

    # Testing prints :)
    # print(len(Medial), 'Medial', np.array2string(Medial, threshold=10, max_line_width=np.inf))
    # print(len(Lateral), 'Lateral', np.array2string(Lateral, threshold=10, max_line_width=np.inf))
    # print(len(Heel), 'Heel', np.array2string(Heel, threshold=100, max_line_width=np.inf))
    # print(len(heelTemperature), 'Heel Temperatures',np.array2string(heelTemperature, threshold=10, max_line_width=np.inf))
    # print(len(rtcCounts), 'rtc', np.array2string(rtcCounts, threshold=10, max_line_width=np.inf))
    # print(len(index), 'Index', np.array2string(index, threshold=10, max_line_width=np.inf))
    # print(len(batteryVoltages), 'Battery voltages',
    #       np.array2string(batteryVoltages, threshold=10, max_line_width=np.inf))
    # print(len(restartPositions), 'Restart Indices',
    #       np.array2string(restartPositions, threshold=10, max_line_width=np.inf))

    return Heel, Lateral, Medial, heelTemperature, batteryVoltages, rtcCounts, index, restartPositions

def indicesToDaysEdited(dataArray):
    indices = len(dataArray)
    numOfDays = 0
    numOfHours = 0
    # Indices equivalent to one day: 1382400
    while indices >= 1382400:
        indices -= 1382400
        numOfDays += 1
    # Indices equivalent to one hour: 57600
    while indices >= 57600:
        indices -= 57600
        numOfHours += 1
    return numOfDays


def indexToDay(index):
    numOfDays = 0
    numOfHours = 0
    # Indices equivalent to one day: 1382400
    while index >= 1382400:
        index -= 1382400
        numOfDays += 1
    # Indices equivalent to one hour: 57600
    while index >= 57600:
        index -= 57600
        numOfHours += 1
    return numOfDays


# Use:
errorTableGenerator()

