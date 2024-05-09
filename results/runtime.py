import pandas as pd

# Read the CSV file into a DataFrame
df = pd.read_csv('results/wandb_export_2024-04-29T20_05_40.187-04_00.csv')

# Calculate the total number of images processed
# Each line in the CSV file represents 5,000 images
total_images = len(df) * 5000  

# Convert 'Runtime' column to timedelta and calculate the total runtime in seconds
total_runtime_seconds = df['Runtime'].sum()

# Calculate the average Frames Per Second (FPS)
fps = total_images / total_runtime_seconds

# Print the result
print("Average Frames Per Second (FPS):", fps)
