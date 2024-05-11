import os
import numpy as np
import scipy.io
# Looping structure like function to quickly import matlab
# workspaces
def loadWorkspaces():
    data = {}
    files = os.listdir('.')
    for file in files:
        if file.endswith('.mat'):
            matData = scipy.io.loadmat(file)
            data[file] = np.array(matData[list(matData.keys())[3]]).flatten()
    return data
# workspaces = loadWorkspaces()
# heelCounts = workspaces['Heel.mat']
# indexCounts = workspaces['Index.mat']
# medialCounts = workspaces['medial.mat']
# lateralCounts = workspaces['lateral.mat']
# heelTemp = workspaces['heelTemp.mat']
# batteryVoltages = workspaces['battVolt.mat']
# RTC = workspaces['rtc.mat']
# restartPositions = workspaces['restartPositions.mat']

# Import matlab workspaces into numpy arrays
# heel = scipy.io.loadmat('Heel.mat')
# heelCounts = np.array(heel['Heel']).flatten()
# index = scipy.io.loadmat('Index.mat')
# indexCounts = np.array(index['index']).flatten()
# medial = scipy.io.loadmat('Medial.mat')
# medialCounts = np.array(medial['Medial']).flatten()
# lateral = scipy.io.loadmat('lateral.mat')
# lateralCounts = np.array(lateral['Lateral']).flatten()
# heelTemptemp = scipy.io.loadmat('heelTemp.mat')
# heelTemp = np.array(heelTemptemp['Heel_temperature']).flatten()
# batteryVoltagesTemp = scipy.io.loadmat('battVolt.mat')
# batteryVoltages = np.array(batteryVoltagesTemp['Batt_volt']).flatten()
# RTCtemp = scipy.io.loadmat('rtc.mat')
# RTC = np.array(RTCtemp['RTC']).flatten()
# restartPositionstemp = scipy.io.loadmat('restartPositions.mat')
# restartPositions = np.array(restartPositionstemp['Restart_positions']).flatten()
#
# Create total array for all 3 sensors
# totalCounts = heelCounts + lateralCounts + medialCounts