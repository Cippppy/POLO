import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np

# Define colors based on the unique colors extracted from the 'Color' column
color_map = {'blue': 'b', 'green': 'g', 'orange': 'orange', 'yellow': 'y', 'purple': 'purple', 'red': 'r', 'black': 'k', 'brown': 'brown', 'gray': 'gray'}

# Read the CSV file into a DataFrame
df = pd.read_csv('/home/cipoll17/POLO/wandb_export_2024-04-29T20_05_40.187-04_00.csv')

# Split the 'Name' column into 'Color' and 'Opacity' based on '_'
df[['Color', 'Opacity']] = df['Name'].str.split('_', expand=True)

# Group the DataFrame by 'Color'
grouped_by_color = df.groupby('Color')['map50_95']

# Calculate the mean and standard error for each color
mean_color = grouped_by_color.mean()
std_err_color = grouped_by_color.std()

# Plot the confidence intervals for each color and save to an image
for color in mean_color.index:
    color_value = color_map.get(color, 'black')
    mean = mean_color[color]
    error = 1.96 * (std_err_color[color] / np.sqrt(len(grouped_by_color.get_group(color))))
    if color == 'Baseline':
        color = 'Original'
        # Add a dotted horizontal line at the baseline
        # plt.axhline(y=mean, color='black', linestyle='--')
    plt.errorbar(color.capitalize(), mean, yerr=error, fmt='o', color=color_value, capsize=5)
plt.title('Mean mAP Score by Color with 95% Confidence Interval')
plt.xlabel('Color')
plt.ylabel('mAP50-95')
plt.grid(True)
plt.savefig('color_confidence_interval.png')
plt.close()

# Group the DataFrame by 'Opacity'
grouped_by_opacity = df.groupby('Opacity')['map50_95']

# Calculate the mean and standard error for each opacity
mean_opacity = grouped_by_opacity.mean()
std_err_opacity = grouped_by_opacity.std()

for opacity in mean_opacity.index:
    mean = mean_opacity[opacity]
    error = 1.96 * (std_err_opacity[opacity] / np.sqrt(len(grouped_by_opacity.get_group(opacity))))
    if opacity == 'Baseline':
        opacity = 100
    if int(opacity) == 80:
        plt.errorbar(20, mean, yerr=error, fmt='o', color='r', capsize=5)
    if int(opacity) == 20:
        plt.errorbar(80, mean, yerr=error, fmt='o', color='r', capsize=5)
    if int(opacity) == 60:
        plt.errorbar(40, mean, yerr=error, fmt='o', color='r', capsize=5)
    if int(opacity) == 40:
        plt.errorbar(60, mean, yerr=error, fmt='o', color='r', capsize=5)
    if int(opacity) == 0:
        plt.errorbar(100, mean, yerr=error, fmt='o', color='r', capsize=5)
    if int(opacity) == 100:
        # Add a dotted horizontal line at the baseline
        plt.axhline(y=mean, color='black', linestyle='--')
        plt.errorbar(0, mean, yerr=error, fmt='o', color='r', capsize=5)
    if int(opacity) == 10:
        plt.errorbar(90, mean, yerr=error, fmt='o', color='r', capsize=5)
    if int(opacity) == 90:
        plt.errorbar(10, mean, yerr=error, fmt='o', color='r', capsize=5)
    if int(opacity) == 30:
        plt.errorbar(70, mean, yerr=error, fmt='o', color='r', capsize=5)
    if int(opacity) == 70:
        plt.errorbar(30, mean, yerr=error, fmt='o', color='r', capsize=5)
    if int(opacity) == 50:
        plt.errorbar(50, mean, yerr=error, fmt='o', color='r', capsize=5)

# Plot the confidence intervals for each opacity and save to an image
# plt.errorbar(mean_opacity.index, mean_opacity, yerr=(ci_opacity[1]-ci_opacity[0])/2, fmt='o', color='r', capsize=5)
plt.title('Mean mAP Score by Opacity with 95% Confidence Interval')
plt.xlabel('Opacity')
plt.ylabel('mAP50-95')
plt.savefig('opacity_confidence_interval.png')
plt.close()
