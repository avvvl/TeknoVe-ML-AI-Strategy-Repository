# Importing the necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from imblearn.ensemble import BalancedRandomForestClassifier

# Load data from CSV file
data = pd.read_csv('data.csv')  # Make sure the 'data.csv' file is in the same directory or provide the correct file path

# Preview the data
print(data.head())

# Define features (X) and target (y) columns
X = data.drop('target', axis=1)  # Replace 'target' with the name of the target column in your dataset
y = data['target']  # Replace 'target' with the name of the target column

# Split data into training (70%) and testing (30%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(f"Training data: {X_train.shape}, Test data: {X_test.shape}")

# Build the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Predict using the test set
y_pred = model.predict(X_test)

# Calculate model accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Print the classification report
report = classification_report(y_test, y_pred)
print(report)

# Get feature importance from the model
feature_importances = model.feature_importances_

# Sort the features by importance
feature_names = X.columns
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

print(importance_df)

# Plot the feature importance
sns.set(style="whitegrid")
plt.figure(figsize=(8, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='teal')
plt.xlabel('Importance')
plt.title('Feature Importance')
plt.show()

# If the data is imbalanced, use the Balanced Random Forest model
balanced_model = BalancedRandomForestClassifier(n_estimators=100, random_state=42)

# Train the balanced model
balanced_model.fit(X_train, y_train)

# Predict using the test set
y_pred_balanced = balanced_model.predict(X_test)

# Calculate the accuracy of the balanced model
balanced_accuracy = accuracy_score(y_test, y_pred_balanced)
print(f"Balanced Model Accuracy: {balanced_accuracy * 100:.2f}%")

# Save the trained model to a file
joblib.dump(model, 'random_forest_model.pkl')

# Load the saved model for future use
loaded_model = joblib.load('random_forest_model.pkl')
