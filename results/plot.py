import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV file into a DataFrame
df = pd.read_csv('plots/wandb_export_2024-04-29T20_05_40.187-04_00.csv')

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
plt.savefig('plots/color_mean_chart.png')
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
plt.savefig('plots/opacity_mean_chart.png')
plt.close()

# Convert mAP50-95 to numeric
df['map50_95'] = pd.to_numeric(df['map50_95'])

# Define the desired order of opacities (excluding 'Baseline')
opacity_order = ['10', '20', '30', '40', '50', '60', '70', '80', '90', '100']

# Filter out rows corresponding to 'Baseline' opacity
df_filtered = df[df['Opacity'] != '0']

# Reorder the rows in the pivot table based on the opacity order
pivot_df = df_filtered.pivot_table(index='Opacity', columns='Color', values='map50_95', aggfunc=np.mean)
pivot_df = pivot_df.reindex(opacity_order)

# Plot the heatmap with the 'coolwarm' colormap
plt.figure(figsize=(10, 10))
plt.imshow(pivot_df, cmap='coolwarm', interpolation='nearest')
plt.colorbar(label='mAP50-95')
plt.xlabel('Color', fontsize=25)
plt.ylabel('Opacity', fontsize=25)
plt.title('Heatmap: Opacity vs Color vs mAP50-95', fontsize=24)
# Adjusting tick positions manually
plt.xticks(np.arange(len(pivot_df.columns)) - 0.5, pivot_df.columns, rotation=45)
plt.yticks(np.arange(len(pivot_df.index)) - 0.5, pivot_df.index)

# Adding grid lines at the middle of each axis interval
plt.grid(axis='both', linestyle='-', linewidth=0.7, color='black')
plt.savefig('plots/heatmap.png')
plt.close()
