import os
import sys
from PySide6.QtWidgets import QApplication, QFileDialog
import numpy as np
import struct

from matplotlib import pyplot as plt


# Author Wyatt Young
# Version May 2, 2023
def fread(fid, uintCode, numValues, skipBytes):
    """
        Read and unpack binary data from a file.

        This function reads a specified number of values from a file, interpreting
        the data as a specified type and skipping a specified number of bytes
        between reads. The function also handles cases where the file ends before
        the requested number of values have been read.
        Parameters:
        fid (file): A patients binary file object.
        uintCode (str): The format code string for struct.unpack, specifying the
            type of data to be read and unpacked.
        numValues (int): The number of values to read from the file.
        skipBytes (int): The number of bytes to skip between reads.
        Returns:
        values (list): A list of the values that have been read and unpacked.
        """
    values = []
    fileSize = struct.calcsize(uintCode)  # Determine the size of the data type to be read
    for _ in range(numValues):
        data = fid.read(fileSize)  # Read the specified amount of data from the file
        # If we have reached the end of the file or the data read is less than the expected size, stop reading
        if not data or len(data) < fileSize:
            break
        # Unpack the read data and append the unpacked value to the list of values
        values.append(struct.unpack(uintCode, data)[0])
        # Move the file position forward by the number of bytes specified to skip
        fid.seek(skipBytes, 1)
    # Return the list of read and unpacked values
    return values
def binary_parse():
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
    # Create PySide6 Qt application
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    # Use QFileDialog to prompt the user to select a .bin file
    fileName, _ = QFileDialog.getOpenFileName(None, 'Open Binary File', "", "Binary files (*.bin)")
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
    print(len(Medial), 'Medial', np.array2string(Medial, threshold=10, max_line_width=np.inf))
    print(len(Lateral), 'Lateral', np.array2string(Lateral, threshold=10, max_line_width=np.inf))
    print(len(Heel), 'Heel', np.array2string(Heel, threshold=100, max_line_width=np.inf))
    print(len(heelTemperature), 'Heel Temperatures',np.array2string(heelTemperature, threshold=10, max_line_width=np.inf))
    print(len(rtcCounts), 'rtc', np.array2string(rtcCounts, threshold=10, max_line_width=np.inf))
    print(len(index), 'Index', np.array2string(index, threshold=10, max_line_width=np.inf))
    print(len(batteryVoltages), 'Battery voltages',
          np.array2string(batteryVoltages, threshold=10, max_line_width=np.inf))
    print(len(restartPositions), 'Restart Indices',
          np.array2string(restartPositions, threshold=10, max_line_width=np.inf))

    return Heel, Lateral, Medial, heelTemperature, batteryVoltages, rtcCounts, index, restartPositions


# Use:
# Heel, Lateral, Medial, heelTemperature, batteryVoltages, rtcCounts, index, restartPositions = binary_parse()

# plt.plot(Heel,label='Heel')
# plt.plot(Medial,label='Medial')
# plt.plot(Lateral,label='Lateral')
# plt.legend()
# plt.show()