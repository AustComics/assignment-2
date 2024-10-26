import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

# Load the dataset
data = pd.read_csv('road_accidents.csv')

# Convert categorical features to dummy variables
data = pd.get_dummies(data, columns=['weather_condition', 'time_of_day', 'road_type'], drop_first=True)

# Define the dependent variable and independent variables
X = data.drop(columns='accident_severity')
y = data['accident_severity']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the linear regression model
model = LinearRegression()

# Fit the model
model.fit(X_train, y_train)

# Save the model for future use
joblib.dump(model, 'accident_severity_model.pkl')

# Example of using the model to predict accident severity
# Make sure the hypothetical data aligns with the shape of X after encoding
hypothetical_data = np.array([[1, 0, 1, 0, 5]])  # Adjust values as per encoding columns in X
predicted_severity = model.predict(hypothetical_data)

print(f'Predicted Accident Severity: {predicted_severity[0]}')
