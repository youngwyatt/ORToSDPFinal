import sys
import matplotlib.pyplot as plt
import numpy as np

from BinaryParseV2 import binary_parse
# Handling error signatures
def setToAverage(i, dataArray):
    if i == 0:
        dataArray[i] = (dataArray[i + 1])
    elif i == len(dataArray) - 1:
        dataArray[i] = ((dataArray[i] + dataArray[i - 1]) / 2)
    else:
        dataArray[i] = ((dataArray[i - 1] + dataArray[i + 1]) / 2)


# create reference to total counts, need to ensure each array from 3 sensors are same length in case one went to shit
# midway through period and was trimmed
def trimTotal(dataArray, dataArrayTwo, dataArrayThree):
    length = min(len(dataArray), len(dataArrayTwo), len(dataArrayThree))
    totalArray = dataArray[:length] + dataArrayTwo[:length] + dataArrayThree[:length]
    return totalArray


def indicesToDays(dataArray, index):
    if index is None:
        indices = len(dataArray)
    else:
        indices = index
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
    return '{} days and {} hours'.format(numOfDays, numOfHours)


# Function to detect peaks determined by a delta value
def peakdet(dataArray, delta, timeArray=None):
    """
    Eli Billauer, 3.4.05 (Explicitly not copyrighted).
    This function is released to the public domain; Any use is allowed.
    """
    # Arrays to store maxima and minima
    maxArray = []
    minArray = []
    # If timeArray is None, use indices of dataArray as x values
    if timeArray is None:
        timeArray = np.arange(len(dataArray))
    # Convert dataArray to numpy array for flexibility
    dataArray = np.asarray(dataArray)
    # Check if length of data array and time array match, else exit with error
    if len(dataArray) != len(timeArray):
        sys.exit('Input vectors data and time must have same length')
    # Delta must be a scalar and a positive number
    if not np.isscalar(delta):
        sys.exit('Input argument delta must be a scalar')
    if delta <= 0:
        sys.exit('Input argument delta must be positive')
    # Initialize minimum and maximum values
    mn = np.Inf
    mx = -np.Inf
    mnPos = np.NaN
    mxPos = np.NaN
    # Flag to indicate if the current position is looking for a maximum
    lookformax = True
    # Loop over dataArray
    for i in np.arange(len(dataArray)):
        this = dataArray[i]  # Current element
        # If this element is greater than current max, update max
        if this > mx:
            mx = this
            mxPos = timeArray[i]
        # If this element is smaller than current min, update min
        if this < mn:
            mn = this
            mnPos = timeArray[i]
        # If looking for max and this element is less than max - delta
        if lookformax:
            if this < mx - delta:
                # Append the max to maxArray
                maxArray.append((mxPos, mx))
                # Now look for min
                mn = this
                mnPos = timeArray[i]
                lookformax = False
        else:
            # If not looking for max and this element is greater than min + delta
            if this > mn + delta:
                # Append the min to minArray
                minArray.append((mnPos, mn))
                # Now look for max
                mx = this
                mxPos = timeArray[i]
                lookformax = True
    # Return maxima and minima arrays
    return np.array(maxArray), np.array(minArray)


# Function to capture peaks and return array of these counts
def peakFinder(dataArray, peakThreshold=1, delta=5, proximity=2):
    # Normalizing the data
    normArray = dataArray * (250 / 2 ** 14)
    # Squaring the data
    squaredArray = normArray ** 2
    # Capturing the indices of peaks
    maxArray, _ = peakdet(squaredArray, delta)
    #Check for if 0 peaks were detected, return empty arrays
    if maxArray.size == 0:
      print("No peaks detected")
      return np.array([]), np.array([])
    indices = maxArray[:, 0].astype(int)

    # Initializing empty arrays to store peak counts and indices
    peakCounts = np.zeros_like(indices, dtype=dataArray.dtype)
    peakIndices = np.zeros_like(indices)

    # Counter for actual number of peaks detected
    peakCount = 0
    # Calculating the difference between indices
    diffIndices = np.diff(indices).astype(float)

    # Appending infinity to handle first and last peaks
    diffIndices = np.insert(diffIndices, 0, float('inf'))
    diffIndices = np.append(diffIndices, float('inf'))

    # Loop to filter out larger peaks based on certain conditions
    for i in range(len(indices)):
        # The checks are only performed when there are elements at i-2 and i+2
        if i >= 2 and i < len(indices) - 2:
            # Conditions for a value to be considered as a peak
            if (normArray[indices[i]] >= peakThreshold and
                    # diffIndices[i] >= proximity and
                    # diffIndices[i - 1] >= proximity and
                    dataArray[indices[i]] > dataArray[indices[i] - 2] + 50 and
                    dataArray[indices[i]] > dataArray[indices[i] + 2] + 50):
                # If the condition passes, increment the peak count and store the peak
                peakCounts[peakCount] = dataArray[int(indices[i])]
                peakIndices[peakCount] = indices[i]
                peakCount += 1

    # Resizing the peakCounts and peakIndices arrays to actual size
    peakCounts = peakCounts[:peakCount]
    peakIndices = peakIndices[:peakCount]

    # Print the number of detected Peaks
    print("Number of detected Peaks: {}".format(peakCount))

    return peakIndices, peakCounts

def valueToDays(samples):
    seconds = samples / 16
    days = seconds / (60 * 60 * 24)
    return days
# 232605 -1st go, 181006 Has to be at least 20 indices lower +- 2, thresh=10, delta=10 +- 100 106324, +-2 +50, delta = 100 no threshold 121028

# Function to detect and correct errors in the data array
def errorSignatures(dataArray, saturationThreshold=11000, neighborThreshold=9500, neighborRange=250, spikeFactor=0.08):
    # Counters for different error types
    fullySaturated = 0
    sensorFailed = False

    # Finding the indices where errors might occur
    errorIndices = np.where((dataArray >= 16382) | (dataArray < 0))
    print(f"Number of obvious errors: {len(errorIndices[0])}")

    # Loop to correct error values
    for index in errorIndices[0]:
        setToAverage(index, dataArray)

    # Finding the indices of potential errors
    potentialErrorIndices = np.where(dataArray >= saturationThreshold)
    print(f"Number of potential errors: {len(potentialErrorIndices[0])}")

    # Loop to correct potential errors
    for i in potentialErrorIndices[0]:
        fullySaturated += 1
        neighborSum = 0
        satNeighborCount = 0

        # Check for saturation in the neighbor data points
        for j in range(i + 1, min(len(dataArray), i + neighborRange + 1)):
            if j != i and dataArray[j] >= neighborThreshold:
                neighborSum += dataArray[j]
                satNeighborCount += 1

        # If not enough neighbors are saturated downstream, set to average
        if satNeighborCount <= 3:
            setToAverage(i, dataArray)
            # neighborAvg = neighborSum / satNeighborCount
            # neighborDiff = dataArray[i] - neighborAvg
        # If sensor begins to fail ask to trim data from then on
        else:
            sensorFailed = True
            break
    # prompt user with troubled data plots and trim data if desired
    # if sensorFailed:
    #     ax1 = plt.subplot(211)
    #     ax1.plot(range(len(dataArray)), dataArray, label="Original Data")
    #     ax1.set_xlabel('Samples')
    #     ax1.set_ylabel('Counts')
    #     # Present data to be trimmed as 1000 indices before to ensure end of data isn't always peaked
    #     trim_start = max(0, i - 1000)
    #     ax1.plot(range(trim_start, len(dataArray)), dataArray[trim_start:], label="Data to be trimmed")
    #     ax1.axvline(x=trim_start, color='red', linestyle='--', linewidth=0.6)
    #     ax1.legend(loc='upper right')
    #     ax2 = plt.subplot(212, sharex=ax1)
    #     ax2.plot(range(trim_start), dataArray[:trim_start], label="Updated Data")
    #     ax1.set_xlabel('Samples')
    #     ax1.set_ylabel('Counts')
    #     ax2.legend(loc='upper right')
    #     plt.show(block=True)
    #     plt.pause(1.0)
    #     time = indicesToDays(None, i + 1)
    #     message = input("Sensor has failed at {} with {} over-saturated neighbor points within {} indices downstream. "
    #                     "Do you want to trim data from this point on? (y/n)".format(time, satNeighborCount,
    #                                                                                 neighborRange))
    #     if message.lower() == 'y':
    #         plt.close()
    #         # Trim data with error margin of 750 indices
    #         dataArray = dataArray[:i - 1000]
    #     else:
    #         plt.close()
    #         print("Data not trimmed! Now checking for sharp concavity changes.")
    # Start setting sharp peaks and +-2 indices dips to average between all 3 points
    # Extract peak indices and counts
    peakIndices, peakCounts = peakFinder(dataArray)
    # Testing: Sensor dips to low value
    # Initizalize data arrays for later comparison
    originalDataArray = dataArray.copy()
    adjustedDataArray = dataArray.copy()
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
            setToAverage(peakIndex, adjustedDataArray)
            triangleArtifacts += 1
            triangleArtifactsArray.append(peakIndex)
            continue

        # General sensor dips
        if dataArray[peakIndex + 1] <= 2 and dataArray[peakIndex + 2] >= 700:
            setToAverage(peakIndex + 1, adjustedDataArray)
            dipsBetweenPeaks += 1
            dipsBetweenPeaksArray.append(peakIndex + 1)
            continue
        if dataArray[peakIndex - 1] <= 2 and dataArray[peakIndex - 2] >= 700:
            setToAverage(peakIndex - 1, adjustedDataArray)
            dipsBetweenPeaks += 1
            dipsBetweenPeaksArray.append(peakIndex - 1)
            continue

        # Obvious sensor spikes to less than % of peak before or after peak
        if dataArray[peakIndex - 1] <= spikeFactor * dataArray[peakIndex]:
            if dataArray[peakIndex - 1] <= 6 and dataArray[peakIndex] >= 1000:
                setToAverage(peakIndex, adjustedDataArray)
                setToAverage(peakIndex - 1, adjustedDataArray)
                sensorSpikes += 1
                sensorSpikesArray.append(peakIndex)
            else:
                setToAverage(peakIndex, adjustedDataArray)
                sensorSpikes += 1
                sensorSpikesArray.append(peakIndex)
        elif dataArray[peakIndex + 1] <= spikeFactor * dataArray[peakIndex]:
            if dataArray[peakIndex + 1] <= 6 and dataArray[peakIndex] >= 1000:
                setToAverage(peakIndex, adjustedDataArray)
                setToAverage(peakIndex + 1, adjustedDataArray)
                sensorSpikes += 1
                sensorSpikesArray.append(peakIndex)
            else:
                setToAverage(peakIndex, adjustedDataArray)
                sensorSpikes += 1
                sensorSpikesArray.append(peakIndex)

        # Set dips between peaks to average
        if dataArray[peakIndex + 2] < 10 and dataArray[peakIndices[i + 1] - 2] < 10:
            setToAverage(peakIndex + 2, adjustedDataArray)
            dipsBetweenPeaks += 1
            dipsBetweenPeaksArray.append(peakIndex + 2)
        elif dataArray[peakIndex + 1] < 10 and dataArray[peakIndices[i + 1] - 1] < 10:
            setToAverage(peakIndex + 1, adjustedDataArray)
            dipsBetweenPeaks += 1
            dipsBetweenPeaksArray.append(peakIndex + 1)

        # Deal with positive and negative searches seperately(0 for 1,2,3 indices between peaks) start w longest range
        if dataArray[peakIndex + 1] < 2 and dataArray[peakIndex + 2] < 2 and dataArray[peakIndex + 3] < 2:
            setToAverage(peakIndex + 1, adjustedDataArray)
            setToAverage(peakIndex + 2, adjustedDataArray)
            setToAverage(peakIndex + 3, adjustedDataArray)
            dipsBetweenPeaks += 1
            dipsBetweenPeaksArray.append(peakIndex + 2)
        # flatten for 2 indices
        elif dataArray[peakIndex + 1] < 2 and dataArray[peakIndex + 2] < 2:
            setToAverage(peakIndex + 1, adjustedDataArray)
            setToAverage(peakIndex + 2, adjustedDataArray)
            dipsBetweenPeaks += 1
            dipsBetweenPeaksArray.append(peakIndex + 2)
        # Deal with negative search
        if dataArray[peakIndex - 1] < 2 and dataArray[peakIndex - 2] < 2 and dataArray[peakIndex - 3] < 2:
            setToAverage(peakIndex - 1, adjustedDataArray)
            setToAverage(peakIndex - 2, adjustedDataArray)
            setToAverage(peakIndex - 3, adjustedDataArray)
            dipsBetweenPeaks += 1
            dipsBetweenPeaksArray.append(peakIndex - 2)
        # flatten for 2 indices
        elif dataArray[peakIndex - 1] < 2 and dataArray[peakIndex - 2] < 2:
            setToAverage(peakIndex - 1, adjustedDataArray)
            setToAverage(peakIndex - 2, adjustedDataArray)
            dipsBetweenPeaks += 1
            dipsBetweenPeaksArray.append(peakIndex - 2)
        # Set specific sensor dips that occur later in array to average
        if dataArray[peakIndex] >= 2000 and dataArray[peakIndex - 2] < 2 and dataArray[peakIndex - 1] < 2:
            setToAverage(peakIndex - 2, adjustedDataArray)
            setToAverage(peakIndex - 1, adjustedDataArray)
            dipsBetweenPeaks += 1
            dipsBetweenPeaksArray.append(peakIndex - 2)
        elif dataArray[peakIndex] >= 2000 and dataArray[peakIndex + 2] < 2 and dataArray[peakIndex + 1] < 2:
            setToAverage(peakIndex + 2, adjustedDataArray)
            setToAverage(peakIndex + 1, adjustedDataArray)
            dipsBetweenPeaks += 1
            dipsBetweenPeaksArray.append(peakIndex + 2)
    if(triangleArtifacts != 0 and sensorSpikes != 0 and dipsBetweenPeaks != 0):
        # Isolating dropped steps
        # start_index = 78454140
        # end_index = 78454300
        # Adjusting plot statements to plot only the slice of data
        # plt.plot(range(end_index - start_index + 1), originalDataArray[start_index:end_index+1], color='#321E1E', alpha=0.6)
        # plt.plot(range(end_index - start_index + 1), adjustedDataArray[start_index:end_index + 1], color='#321E1E', alpha=0.6)
        # for spike in dipsBetweenPeaksArray:
        #     if start_index <= spike <= end_index:
        #         # Adjust the spike position to match the new x-axis that starts from 0
        #         adjusted_spike_position = spike - start_index
        #         plt.axvline(x=adjusted_spike_position, color='#CD1818', alpha=0.6, linestyle='--', linewidth=0.6, label="Dips between peaks" if spike == sensorSpikesArray[0] else "")
        # Plotting in days:
        totalSeconds = len(dataArray) / 16
        totalDays = totalSeconds / (60 * 60 * 24)
        # Create an array with the day values
        days = np.linspace(0, totalDays, num=len(dataArray))
        plt.plot(days, adjustedDataArray,color='#321E1E', alpha=0.7, label='Adjusted data')
        # Normal Plotting
        # plt.plot(range(len(originalDataArray)), originalDataArray, color='#321E1E', label='Original data')
        # plt.plot(range(len(adjustedDataArray)), adjustedDataArray,color='#321E1E', alpha=0.7, label='Adjusted data')
        # plt.scatter(peakIndices, peakCounts, color='red', s=4)
        # Plot errors
        if sensorSpikesArray:
            plt.axvline(x=valueToDays(sensorSpikesArray[0]), color=(17/255,120/255,130/255), alpha=1, linestyle='--', linewidth=0.6, label="Sensor dips")
            for i in range(1, len(sensorSpikesArray)):
                plt.axvline(x=valueToDays(sensorSpikesArray[i]), color='#116D6E', alpha=1.0, linestyle='--', linewidth=0.6)
        if triangleArtifactsArray:
            plt.axvline(x=valueToDays(triangleArtifactsArray[0]), color='#4E3636', linestyle='--', linewidth=0.6, label="Triangle error")
            for i in range(1, len(triangleArtifactsArray)):
                plt.axvline(x=valueToDays(triangleArtifactsArray[i]), color='#4E3636', linestyle='--', linewidth=0.6)
        if dipsBetweenPeaksArray:
            plt.axvline(x=valueToDays(dipsBetweenPeaksArray[0]), color='#CD1818', alpha=0.6, linestyle='--', linewidth=0.6, label="Dips between peaks")
            for i in range(1, len(dipsBetweenPeaksArray)):
                plt.axvline(x=valueToDays(dipsBetweenPeaksArray[i]), color='#CD1818', alpha=0.6, linestyle='--', linewidth=0.6)
        plt.legend(loc='upper left')
        plt.xlabel('Days', fontsize=11, fontweight='bold')
        plt.ylabel('Counts', fontsize=11, fontweight='bold')
    totalErrors = sensorSpikes + triangleArtifacts + dipsBetweenPeaks
    print(f'Sensor Spikes: {sensorSpikes}, Triangle Artifacts: {triangleArtifacts}, Dips between peaks: {dipsBetweenPeaks}, Total Errors: {totalErrors}')
    plt.tight_layout()
    plt.show()
    return dataArray
Heel, _, _, _, _, _,_,_ = binary_parse()
errorSignatures(Heel)

