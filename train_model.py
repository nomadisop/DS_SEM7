
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load feature-engineered data
DATA_PATH = 'cleaned_featured_books_dataset.csv'
df = pd.read_csv(DATA_PATH)

# Features and target
target = 'gross_sales'
drop_cols = ['gross_sales', 'book_name', 'author', 'genre', 'publisher', 'language_code']
features = [col for col in df.columns if col not in drop_cols]
X = df[features].copy()
y = df[target]


# Handle 'unknown' and non-numeric columns
for col in X.columns:
	if X[col].dtype == 'object':
		X[col] = X[col].replace('unknown', np.nan)
		X[col] = X[col].fillna('missing')
		X[col] = X[col].astype('category').cat.codes


# Impute missing values (NaN) with column mean for numeric, zero for others
for col in X.columns:
	if X[col].isnull().any():
		if np.issubdtype(X[col].dtype, np.number):
			X[col] = X[col].fillna(X[col].mean())
		else:
			X[col] = X[col].fillna(0)

# Impute any remaining NaNs with zero
X = X.fillna(0)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Train model
model = LinearRegression()
model.fit(X_train, y_train)


# Evaluate
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f"Test RMSE: {rmse:.2f}")
print(f"Test R2: {r2:.2f}")

# Save model
joblib.dump(model, 'model.pkl')
print("Model trained and saved as model.pkl")
