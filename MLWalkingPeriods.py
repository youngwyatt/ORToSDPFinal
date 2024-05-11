import numpy as np
import pywt
import ImportMatData as data
import matplotlib.pyplot as plt
# windowsize in indices correlating to one day: 1,382,400
def preInverseDWT(dataArray, windowSize = 16, wavelet='db5'):
    # Obtain detail coefficients
    coeffs = pywt.dwt(dataArray, wavelet)
    _, detail = coeffs
    # Initialize feature arrays
    means = []
    variances = []
    maxCoeffs = []
    minCoeffs = []
    # Calculate features
    for i in range(0, len(coeffs) - windowSize, windowSize):
        window = coeffs[i:i + windowSize]
        means.append(np.mean(window))
        variances.append(np.var(window))
        maxCoeffs.append(np.max(window))
        minCoeffs.append(np.min(window))

    # Combine features into 2D array with rows as each window and columns as each feature
    featureArray = np.column_stack((means, variances, maxCoeffs, minCoeffs))
    return featureArray


def inverseDWT(dataArray, wavelet='db5'):
    # Perform DWT
    coeffs = pywt.dwt(dataArray, wavelet)

    # Perform inverse DWT
    reconstructed_data = pywt.idwt(*coeffs, wavelet)

    return reconstructed_data
def extract_steps(data, load_threshold):
    steps = []
    for i in range(1, len(data) - 1):  # Avoid the edges
        # Check for swing phase on both sides
        if data[i - 6] > load_threshold > data[i] and data[i + 6] > load_threshold > data[i]:
            # This index is the center of a swing phase
            # Check cadence, between 50 and 160 steps/minute
            # Cadence of 50 equates to 0.02 minutes per step
            time_diff = ((i) - (i - 6)) / 60  # Conv to minutes
            cadence = 60 / time_diff  # Steps per minute
            if 40 <= cadence <= 160:
                print("Cadence: " + str(cadence))
                # Cadence is within the desired range
                steps.append(i)
    return steps

total = data.heelCounts + data.lateralCounts + data.medialCounts
# Weight borne changes at 60000000
same = total[0:60000000]
time = data.indexCounts[0:60000000]
plt.plot(total)
plt.show()
# extract_steps(same, time, 100)
