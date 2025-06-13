import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

# Load data
df = pd.read_csv('BostonHousing.csv')

# Print actual column names to verify
print(df.columns)

# Fix the column name if necessary
X = df.drop('MEDV', axis=1)
y = df['MEDV']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = LinearRegression()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'boston_model.pkl')
