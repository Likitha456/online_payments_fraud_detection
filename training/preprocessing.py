import pandas as pd

# Load dataset
df = pd.read_csv("data/PS_20174392719_1491204439457_log.csv")

print("Initial shape:", df.shape)

# Drop ID columns (not useful for ML)
df = df.drop(['nameOrig', 'nameDest'], axis=1)

print("Shape after dropping ID columns:", df.shape)

# One-hot encode the 'type' column
df = pd.get_dummies(df, columns=['type'], drop_first=True)

print("Shape after encoding 'type':", df.shape)
# Separate features and target
# X = features
# y = target (isFraud)
# Separate features and target
X = df.drop('isFraud', axis=1)
y = df['isFraud']

print("X shape:", X.shape)
print("y shape:", y.shape)
