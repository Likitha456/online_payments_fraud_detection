import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Load dataset
df = pd.read_csv("data/PS_20174392719_1491204439457_log.csv")

# Drop unnecessary ID columns
df = df.drop(['nameOrig', 'nameDest'], axis=1)

# Encode categorical column
df = pd.get_dummies(df, columns=['type'], drop_first=True)

# Separate features and target
X = df.drop('isFraud', axis=1)
y = df['isFraud']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 🔥 UPDATED BALANCED DECISION TREE
dt_model = DecisionTreeClassifier(
    random_state=42,
    class_weight='balanced',
    max_depth=8
)

dt_model.fit(X_train, y_train)

# Save the model
with open("decision_tree_model.pkl", "wb") as file:
    pickle.dump(dt_model, file)

print("Balanced Decision Tree model saved successfully as decision_tree_model.pkl")
