import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file into a DataFrame
df = pd.read_csv('/home/cipoll17/POLO/wandb_export_2024-04-29T20_05_40.187-04_00 copy.csv')

# Split the 'Name' column into 'Color' and 'Opacity' based on '_'
df[['Color', 'Opacity']] = df['Name'].str.split('_', expand=True)

# Map the opacity values
opacity_map = {'Baseline': '0', '100': '0', '90': '10', '80': '20', '70': '30', '60': '40', '50': '50', '40': '60', '30': '70', '20': '80', '10': '90', '0': '100'}

# Replace opacity values with correct ones
df['Opacity'] = df['Opacity'].map(opacity_map)

# Convert 'Opacity' column to numeric
df['Opacity'] = pd.to_numeric(df['Opacity'])

# Set font size for all text
plt.rc('font', size=15)

# Create scatter plot for Color vs. mAP Score
plt.figure(figsize=(10, 5))
plt.scatter(df['Color'], df['map50_95'], c=df['Opacity'], cmap='viridis', s=50)
plt.xlabel('Color', fontsize=15)
plt.ylabel('mAP Score', fontsize=15)
plt.title('mAP Score by Color and Opacity', fontsize=20)
plt.colorbar(label='Opacity')

# Save the plot to a file
plt.savefig('color_vs_map_score.png')
plt.close()

# Create scatter plot for Opacity vs. mAP Score
plt.figure(figsize=(10, 5))
plt.scatter(df['Opacity'], df['map50_95'], c=df['Color'], cmap='viridis', s=50)
plt.xlabel('Opacity', fontsize=15)
plt.ylabel('mAP Score', fontsize=15)
plt.title('mAP Score by Color and Opacity', fontsize=20)
plt.colorbar(label='Color')

# Save the plot to a file
plt.savefig('opacity_vs_map_score.png')
plt.close()
