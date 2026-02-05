import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv(
    r"C:\Users\velak\OneDrive\Desktop\online-payments-fraud-detection\data\PS_20174392719_1491204439457_log.csv"
)

print(df.shape)
print(df.head())

sns.countplot(x='isFraud', data=df)
plt.title("Fraud vs Non-Fraud Transactions")
plt.show()

# Insight:
# Fraud cases are extremely fewer than non-fraud cases.
# This indicates severe class imbalance.

sns.countplot(x='type', hue='isFraud', data=df)
plt.title("Fraud by Transaction Type")
plt.show()

# Insight:
# Certain transaction types show higher fraud concentration.

sns.histplot(df['amount'], bins=50)
plt.title("Transaction Amount Distribution")
plt.show()

# Insight:
# Most transactions are small, few are very large.

sns.boxplot(x=df['amount'])
plt.title("Transaction Amount Outliers")
plt.show()

# Insight:
# The dataset is highly imbalanced.
# Non-fraud transactions (isFraud = 0) dominate the dataset.
# Fraud transactions (isFraud = 1) are extremely rare.
# This imbalance must be handled carefully during model training.
# Outliers are clearly present in transaction amounts.