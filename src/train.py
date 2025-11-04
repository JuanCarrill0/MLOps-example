# Importamos las librerías necesarias
import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

# Intentamos importar MLflow para el seguimiento de experimentos
# Si no está disponible, mlflow será None
try:
    import mlflow
except Exception:
    mlflow = None

def generate_synthetic(path):
    """
    Genera un dataset sintético de viviendas con características y precios.
    
    Args:
        path: Ruta donde guardar el archivo CSV generado
    """
    from sklearn.datasets import make_regression
    import numpy as np
    
    # Generamos datos de regresión sintéticos: 500 muestras, 3 características
    X, y = make_regression(n_samples=500, n_features=3, noise=10.0, random_state=42)
    
    # Creamos un DataFrame con nombres de columnas descriptivos
    df = pd.DataFrame(X, columns=['rooms','area','age'])
    
    # Transformamos los valores para que sean realistas para viviendas
    df['rooms'] = (abs(df['rooms']) * 3).round().astype(int) + 1  # Habitaciones: 1-10
    df['area'] = (abs(df['area']) * 50).round().astype(int) + 20  # Área en m²: 20-300
    df['age'] = (abs(df['age']) * 10).round().astype(int)         # Edad en años: 0-50
    df['price'] = (y * 1000 + 50000).round(2)                    # Precio base + variación
    
    # Guardamos el dataset en formato CSV
    df.to_csv(path, index=False)
    print(f'✔ Synthetic dataset generated at {path}')

def main(preprocess_only=False):
    """
    Función principal que maneja el entrenamiento del modelo.
    
    Args:
        preprocess_only: Si es True, solo prepara los datos sin entrenar
    """
    # Definimos la ruta donde deben estar los datos procesados
    processed_path = os.path.join('data','processed','housing.csv')
    
    # Si no existen datos procesados, generamos datos sintéticos
    if not os.path.exists(processed_path):
        print('No processed dataset found — generating synthetic dataset for demo...')
        os.makedirs(os.path.dirname(processed_path), exist_ok=True)
        generate_synthetic(processed_path)

    # Si solo queremos preprocesar, salimos aquí
    if preprocess_only:
        print('Preprocess only: exiting after data preparation')
        return

    # Cargamos los datos procesados
    df = pd.read_csv(processed_path)
    
    # Separamos las características (X) de la variable objetivo (y)
    X = df[['rooms','area','age']].values  # Características: habitaciones, área, edad
    y = df['price'].values                 # Variable objetivo: precio

    # Dividimos los datos en entrenamiento (80%) y prueba (20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Creamos y entrenamos el modelo de regresión lineal
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Creamos el directorio de modelos si no existe y guardamos el modelo
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/model.pkl')
    print('✅ Model trained and saved to models/model.pkl')

    # Intentamos registrar el experimento en MLflow si está disponible
    if mlflow is not None:
        try:
            # Configuramos el experimento en MLflow
            mlflow.set_experiment('mlops-example')
            with mlflow.start_run():
                # Calculamos y registramos métricas
                score = model.score(X_test, y_test)  # R² score en datos de prueba
                mlflow.log_metric('r2_score', float(score))
                mlflow.log_artifact('models/model.pkl')  # Guardamos el modelo como artefacto
                print(f'✅ Logged run to MLflow (r2_score={score:.4f})')
        except Exception as e:
            print('⚠ MLflow logging failed:', e)

if __name__ == '__main__':
    # Configuramos los argumentos de línea de comandos
    parser = argparse.ArgumentParser()
    parser.add_argument('--preprocess-only', action='store_true', 
                       help='Only run preprocessing step')
    args = parser.parse_args()
    
    # Ejecutamos la función principal con los argumentos proporcionados
    main(preprocess_only=args.preprocess_only)
