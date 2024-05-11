import sys

import numpy as np
import pandas as pd
from PySide6.QtWidgets import QApplication, QFileDialog
from matplotlib import pyplot as plt


def ortosReadTxt():
    # Create PySide6 Qt application
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    # Use QFileDialog to prompt the user to select a .bin file
    fileName, _ = QFileDialog.getOpenFileName(None, 'Open txt File', "", "txt files (*.txt)")
    df = pd.read_csv(fileName,delimiter='\t',header=None,skiprows=1)
    heel = df.iloc[:,0]
    medial = df.iloc[:,1]
    lateral = df.iloc[:,2]
    timestamp = df.iloc[:,3]
    total = heel + medial + lateral
    numOfDays = 0
    numOfHours = 0
    indices = len(heel)
    print(indices)
    # Indices equivalent to one day: 1382400
    while indices >= 1382400:
        indices -= 1382400
        numOfDays += 1
    # Indices equivalent to one hour: 57600
    while indices >= 57600:
        indices -= 57600
        numOfHours += 1
    print(f'Patient recorded data for {numOfDays} Days and {numOfHours} Hours.')
    return df, heel, medial, lateral, total, timestamp
df, h, m, l, t, time = ortosReadTxt()
plt.figure()
plt.plot(h,label='Heel Sensor')
plt.plot(m,label='Medial Sensor')
plt.plot(l,label='Lateral Sensor')
# plt.legend()
# plt.title('PT03')
# plt.show()
totalSeconds = len(t) / 16
totalDays = totalSeconds / (60 * 60 * 24)
# Create an array with the day values
days = np.linspace(0, totalDays, num=len(t))
for i in range(len(t)):
    if t[i] > 248:
        t[i] = 0
# plt.figure()
plt.plot(days, t, label='Total Load PT08')
plt.xlabel('Days')
plt.ylabel('lbs')
plt.legend()
plt.show()

