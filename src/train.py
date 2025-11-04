import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

try:
    import mlflow
except Exception:
    mlflow = None

def generate_synthetic(path):
    from sklearn.datasets import make_regression
    import numpy as np
    X, y = make_regression(n_samples=500, n_features=3, noise=10.0, random_state=42)
    df = pd.DataFrame(X, columns=['rooms','area','age'])
    # make values positive and scale
    df['rooms'] = (abs(df['rooms']) * 3).round().astype(int) + 1
    df['area'] = (abs(df['area']) * 50).round().astype(int) + 20
    df['age'] = (abs(df['age']) * 10).round().astype(int)
    df['price'] = (y * 1000 + 50000).round(2)
    df.to_csv(path, index=False)
    print(f'✔ Synthetic dataset generated at {path}')

def main(preprocess_only=False):
    processed_path = os.path.join('data','processed','housing.csv')
    if not os.path.exists(processed_path):
        print('No processed dataset found — generating synthetic dataset for demo...')
        os.makedirs(os.path.dirname(processed_path), exist_ok=True)
        generate_synthetic(processed_path)

    if preprocess_only:
        print('Preprocess only: exiting after data preparation')
        return

    df = pd.read_csv(processed_path)
    X = df[['rooms','area','age']].values
    y = df['price'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/model.pkl')
    print('✅ Model trained and saved to models/model.pkl')

    # Try logging to MLflow if available
    if mlflow is not None:
        try:
            mlflow.set_experiment('mlops-example')
            with mlflow.start_run():
                score = model.score(X_test, y_test)
                mlflow.log_metric('r2_score', float(score))
                mlflow.log_artifact('models/model.pkl')
                print(f'✅ Logged run to MLflow (r2_score={score:.4f})')
        except Exception as e:
            print('⚠ MLflow logging failed:', e)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--preprocess-only', action='store_true', help='Only run preprocessing step')
    args = parser.parse_args()
    main(preprocess_only=args.preprocess_only)
