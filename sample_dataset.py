import pandas as pd
import random
import os

filename = "synthetic_ev_battery_5year_50cols_multimfg_3M.csv"
output_filename = "sampled_50cols_dataset.csv"

# Target sample size
target_rows = 20000
total_rows = 3000000

# Probability of keeping a row
p = target_rows / total_rows

print(f"Sampling ~{target_rows} rows from {total_rows} total rows...")
print("This uses memory-efficient stream processing and will take about 1 minute.")

# skiprows logic:
# i == 0 is the header, we always keep it (skiprows returns False)
# for i > 0, we randomly skip it based on probability 1 - p
df = pd.read_csv(
    filename, 
    header=0, 
    skiprows=lambda i: i > 0 and random.random() > p
)

print(f"Sampled {len(df)} rows. Saving to {output_filename}...")
df.to_csv(output_filename, index=False)

# Clean up the massive 1.7GB file to save space
print(f"Deleting the original 1.7GB file to save your disk space...")
os.remove(filename)

print("Done! The sampled dataset is ready.")
