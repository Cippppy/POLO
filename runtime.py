import pandas as pd

# Read the CSV file into a DataFrame
df = pd.read_csv('/home/cipoll17/POLO/wandb_export_2024-04-29T20_05_40.187-04_00.csv')

# Calculate the total number of images processed
total_images = len(df) * 5000  # Each line represents 5,000 images

# Convert 'Runtime' column to timedelta and calculate the total runtime
total_runtime_seconds = df['Runtime'].sum()

# Calculate the average FPS
fps = total_images / total_runtime_seconds

print("Average Frames Per Second (FPS):", fps)