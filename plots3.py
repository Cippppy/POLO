import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Read the CSV file into a DataFrame
df = pd.read_csv('/home/cipoll17/POLO/wandb_export_2024-04-29T20_05_40.187-04_00.csv')

# Split the 'Name' column into 'Color' and 'Opacity' based on '_'
df[['Color', 'Opacity']] = df['Name'].str.split('_', expand=True)

# Map the opacity values
opacity_map = {'Baseline': '0', '100': '0', '90': '10', '80': '20', '70': '30', '60': '40', '50': '50', '40': '60', '30': '70', '20': '80', '10': '90', '0': '100'}

# Replace opacity values with correct ones
df['Opacity'] = df['Opacity'].map(opacity_map)

# Define colors based on the unique colors extracted from the 'Color' column
color_map = {'blue': 'b', 'green': 'g', 'orange': 'orange', 'yellow': 'y', 'purple': 'purple', 'red': 'r', 'black': 'k', 'brown': 'brown', 'gray': 'gray'}

# Create a directory to store plots if it doesn't exist
if not os.path.exists('plots'):
    os.makedirs('plots')

# Convert mAP50-95 to numeric
df['map50_95'] = pd.to_numeric(df['map50_95'])

# Define the desired order of opacities (excluding 'Baseline')
opacity_order = ['10', '20', '30', '40', '50', '60', '70', '80', '90', '100']

# Filter out rows corresponding to 'Baseline' opacity
df_filtered = df[df['Opacity'] != '0']

# Reorder the rows in the pivot table based on the opacity order
pivot_df = df_filtered.pivot_table(index='Opacity', columns='Color', values='map50_95', aggfunc=np.mean)
pivot_df = pivot_df.reindex(opacity_order)

# Set font size for all text
plt.rc('font', size=15)

# Plot the heatmap with the 'coolwarm' colormap
plt.figure(figsize=(10, 10))
plt.imshow(pivot_df, cmap='coolwarm', interpolation='nearest')
plt.colorbar(label='mAP50-95')
plt.xlabel('Color', fontsize=25)
plt.ylabel('Opacity', fontsize=25)
plt.title('Heatmap: Opacity vs Color vs mAP50-95', fontsize=24)
# plt.xticks(np.arange(len(pivot_df.columns)), pivot_df.columns, rotation=45)
# plt.yticks(np.arange(len(pivot_df.index)), pivot_df.index, fontsize=20)
# Adjusting tick positions manually
plt.xticks(np.arange(len(pivot_df.columns)) - 0.5, pivot_df.columns, rotation=45)
plt.yticks(np.arange(len(pivot_df.index)) - 0.5, pivot_df.index)

# Adding grid lines at the middle of each axis interval
plt.grid(axis='both', linestyle='-', linewidth=0.7, color='black')
# plt.grid(axis='both', linestyle='--', linewidth=0.5, color='black')  # Adding grid lines at axis intervals
plt.savefig('plots/heatmap.png')
plt.close()


# Box plot by Color and Opacity
plt.figure(figsize=(10, 6))
df.boxplot(column='map50_95', by=['Color', 'Opacity'], grid=False, figsize=(12, 8))
plt.title('Box Plot of mAP Scores by Color and Opacity')
plt.xlabel('Color & Opacity')
plt.ylabel('mAP50-95')
plt.xticks(rotation=45)
plt.savefig('plots/box_plot_color_opacity.png')
plt.show()

# Line plot by Color and Opacity
plt.figure(figsize=(10, 6))
for color in df['Color'].unique():
    color_df = df[df['Color'] == color]
    for opacity in df['Opacity'].unique():
        opacity_df = color_df[color_df['Opacity'] == opacity]
        plt.plot(opacity_df['Opacity'], opacity_df['map50_95'], label=f'{color} - Opacity {opacity}', marker='o')
plt.title('Line Plot of mAP Scores by Color and Opacity')
plt.xlabel('Opacity')
plt.ylabel('mAP50-95')
plt.legend()
plt.savefig('plots/line_plot_color_opacity.png')
plt.show()

# Scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(df['Opacity'], df['map50_95'], c=df['Color'].map(color_map))
plt.title('Scatter Plot of mAP Scores by Opacity')
plt.xlabel('Opacity')
plt.ylabel('mAP50-95')
plt.colorbar(label='Color')
plt.savefig('plots/scatter_plot_opacity.png')
plt.show()

# Histogram of mAP Scores
plt.figure(figsize=(10, 6))
plt.hist(df['map50_95'], bins=20, color='skyblue', edgecolor='black')
plt.title('Histogram of mAP Scores')
plt.xlabel('mAP50-95')
plt.ylabel('Frequency')
plt.savefig('plots/histogram_map_scores.png')
plt.show()