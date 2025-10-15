import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import joblib

# Step 1: Create data
hours = [1,2,3,4,5,6,7,8,9,10]
marks = [30,35,50,55,60,65,70,75,80,85]

# Step 2: Convert to DataFrame
df = pd.DataFrame({'Hours': hours, 'Marks': marks})

# Step 3: Train model
X = df[['Hours']]
y = df['Marks']
model = LinearRegression()
model.fit(X, y)

# Step 4: Save model
joblib.dump(model, 'marks_model.pkl')
print("✅ Model trained and saved successfully!")