import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
df = pd.read_csv("data/PS_20174392719_1491204439457_log.csv")

# Drop ID columns
df = df.drop(['nameOrig', 'nameDest'], axis=1)

# Encode categorical column
df = pd.get_dummies(df, columns=['type'], drop_first=True)

# Separate features and target
X = df.drop('isFraud', axis=1)
y = df['isFraud']

print("Data prepared for training")
print("X shape:", X.shape)
print("y shape:", y.shape)
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Train-test split completed")
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
# Train Decision Tree model
dt_model = DecisionTreeClassifier(random_state=42)

dt_model.fit(X_train, y_train)

# Predictions
y_pred_dt = dt_model.predict(X_test)

# Evaluation
print("\nDecision Tree Results")
print("Accuracy:", accuracy_score(y_test, y_pred_dt))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_dt))
print("\nClassification Report:\n", classification_report(y_test, y_pred_dt))
from sklearn.ensemble import RandomForestClassifier

# Train Random Forest model (limited trees for performance)
rf_model = RandomForestClassifier(
    n_estimators=50,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

# Predictions
y_pred_rf = rf_model.predict(X_test)

# Evaluation
print("\nRandom Forest Results")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))
print("\nClassification Report:\n", classification_report(y_test, y_pred_rf))
