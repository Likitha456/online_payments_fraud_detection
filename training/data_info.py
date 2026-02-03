import pandas as pd

# Load dataset
df = pd.read_csv("data/PS_20174392719_1491204439457_log.csv")

print("Shape of dataset:")
print(df.shape)

print("\nDataset Info:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())
