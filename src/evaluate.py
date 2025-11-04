import os
import joblib
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error

model_path = os.path.join('models','model.pkl')
data_path = os.path.join('data','processed','housing.csv')

if not os.path.exists(model_path):
    print('Model not found. Run `python src/train.py` first.')
    raise SystemExit(1)

if not os.path.exists(data_path):
    print('Data not found. Run `python src/train.py` first to generate synthetic data.')
    raise SystemExit(1)

model = joblib.load(model_path)
df = pd.read_csv(data_path)
X = df[['rooms','area','age']].values
y = df['price'].values

# simple split: last 20% as test-like
split = int(len(X)*0.8)
X_test = X[split:]
y_test = y[split:]

preds = model.predict(X_test)
r2 = r2_score(y_test, preds)
mae = mean_absolute_error(y_test, preds)
print(f'R2: {r2:.4f}')
print(f'MAE: {mae:.2f}')
