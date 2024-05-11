import os
import numpy as np
from scipy.stats import sem
import pandas as pd
from matplotlib import pyplot as plt
# # Pearson Figure
# directory = '/Users/wyattyoung/PycharmProjects/ORToSDataPL2023(UROP)'
# allFiles = os.listdir(directory)
# PCfiles = [file for file in allFiles if file.endswith('.csv')]
#
# pearsonMeans = []
# pearsonSEMs = []
# for fileName in PCfiles:
#     df = pd.read_csv(fileName,skiprows=1,header=None)
#     print(df)
#     mean = np.mean(df.iloc[:,1])
#     pearsonMeans.append(mean)
#     SE = sem(df.iloc[:,1])
#     pearsonSEMs.append(SE)
# file_labels = [name.replace('.csv', '') for name in PCfiles]
# plt.figure(figsize=(10, 5))
# # Plot scatter points with error bars
# plt.errorbar(file_labels, pearsonMeans, yerr=pearsonSEMs, fmt='o', color='green', elinewidth=5,capsize=1,ecolor='blue')
# # Annotate each dot with its mean value
# for i, txt in enumerate(pearsonMeans):
#     plt.annotate(f'{txt:.8f}',  # Format the text to show 2 decimal places
#                  (file_labels[i], pearsonMeans[i]),
#                  textcoords="offset points",  # how to position the text
#                  xytext=(0,10),  # distance from text to points (x,y)
#                  ha='center')  # horizontal alignment can be left, right or center
# plt.title('Pearson Coefficients Between Previous MATLAB Parsed Data and Pipeline Parsed Data')
# plt.ylim(0.3,1.1)
# plt.xlabel('Patient Datasets')
# plt.ylabel('Mean Pearson Coefficient')
# plt.show()
# Uncovered sensor errors figure
saturations = [3,1,1,0,3,17,62,1,1,1,0]
newErrors = [7,20,6,3,114,229,2804,373,1073,88,1966]
labels = ['CT01','CT02','CT07','CT09','CT10','CT12','CT13','CT14','CT15','CT17','AB28']
plt.figure()

width = 0.5  # width of the bars

# Calculate the positions for the bars
indices = np.arange(len(saturations))

# Determine the overlap by subtracting from the full width
# If width is 0.5, we subtract less than 0.5 for overlap
overlap = width / 1.5  # A small reduction from the full width for overlap

# Plot the bars with increased overlap
bars1 = plt.bar(indices - overlap, saturations, width=width, label='Previously Known Errors', color='#CD1818', alpha=0.9)
bars2 = plt.bar(indices, newErrors, width=width, label='Errors Found By Pipeline', color='#116D6E', alpha=0.9)

# Adding data labels
for bars in [bars1, bars2]:
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval, f'{int(yval)}',
                 va='bottom', ha='center', fontsize=8, fontweight='bold', color='black')

# Set the x-ticks to be in the middle of the two bars of a group, adjusted for overlap
plt.xticks(indices - overlap / 2, labels, fontsize=9, fontweight='bold')

plt.legend()
plt.ylabel('Errors', fontsize=11, fontweight='bold')
plt.xlabel('Patient Datasets', fontsize=11, fontweight='bold')
plt.title('Error Signatures Found via Pipeline Versus Prior Methods', fontsize=12, fontweight='bold')
plt.tight_layout()  # Adjusts subplot params so that the subplot(s) fits in to the figure area
plt.show()


