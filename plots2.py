import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file into a DataFrame
df = pd.read_csv('/home/cipoll17/POLO/wandb_export_2024-04-29T20_05_40.187-04_00.csv')

# Split the 'Name' column into 'Color' and 'Opacity' based on '_'
df[['Color', 'Opacity']] = df['Name'].str.split('_', expand=True)

# Map the opacity values
opacity_map = {'Baseline': '0', '100': '0', '90': '10', '80': '20', '70': '30', '60': '40', '50': '50', '40': '60', '30': '70', '20': '80', '10': '90', '0': '100'}

# Replace opacity values with correct ones
df['Opacity'] = df['Opacity'].map(opacity_map)

# Group the DataFrame by 'Color' and 'Opacity'
grouped_by_color = df.groupby('Color')['map50_95']
grouped_by_opacity = df.groupby('Opacity')['map50_95']

# Calculate the mean for each color and opacity
mean_color = grouped_by_color.mean()
mean_opacity = grouped_by_opacity.mean()

# Define colors based on the unique colors extracted from the 'Color' column
color_map = {'blue': 'b', 'green': 'g', 'orange': 'orange', 'yellow': 'y', 'purple': 'purple', 'red': 'r', 'black': 'k', 'brown': 'brown', 'gray': 'gray'}

# Set font size for all text
plt.rc('font', size=15)

# Create bar chart for colors
plt.figure(figsize=(10, 10))
bars = plt.bar(mean_color.index, mean_color, color=[color_map.get(color, 'black') for color in mean_color.index])
plt.bar_label(bars, fmt='%.3f')
plt.title('Mean mAP Score by Color', fontsize=25)  # Adjust title font size
plt.xlabel('Color', fontsize=25)  # Adjust x-axis label font size
plt.ylabel('Mean mAP50-95', fontsize=25)  # Adjust y-axis label font size
# plt.xticks(fontsize=12)  # Adjust font size of x-axis tick labels
plt.yticks(fontsize=20)  # Adjust font size of y-axis tick labels
plt.savefig('color_mean_chart.png')
plt.close()

# Create bar chart for opacities
plt.figure(figsize=(10, 10))
bars = plt.bar(mean_opacity.index.astype(int), mean_opacity, color='black')
plt.bar_label(bars, fmt='%.3f')
plt.title('Mean mAP Score by Opacity', fontsize=25)  # Adjust title font size
plt.xlabel('Opacity', fontsize=25)  # Adjust x-axis label font size
plt.ylabel('Mean mAP50-95', fontsize=25)  # Adjust y-axis label font size
plt.xticks(fontsize=20)  # Adjust font size of x-axis tick labels
plt.yticks(fontsize=20)  # Adjust font size of y-axis tick labels
plt.savefig('opacity_mean_chart.png')
plt.close()
