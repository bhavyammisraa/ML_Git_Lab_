import pandas as pd
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

# Load dataset
data = pd.read_csv("dataset.csv")

X = data[['feature']]
y = data['target']

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = DecisionTreeRegressor()
model.fit(X_scaled, y)

predictions = model.predict(X_scaled)

print("R2 Score:", r2_score(y, predictions))
