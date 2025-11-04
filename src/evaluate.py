# Importamos las librerías necesarias para evaluación
import os
import joblib
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error

# Definimos las rutas de los archivos necesarios
model_path = os.path.join('models','model.pkl')        # Ruta del modelo entrenado
data_path = os.path.join('data','processed','housing.csv')  # Ruta de los datos

# Verificamos que el modelo existe
if not os.path.exists(model_path):
    print('Model not found. Run `python src/train.py` first.')
    raise SystemExit(1)

# Verificamos que los datos existen
if not os.path.exists(data_path):
    print('Data not found. Run `python src/train.py` first to generate synthetic data.')
    raise SystemExit(1)

# Cargamos el modelo entrenado y los datos
model = joblib.load(model_path)
df = pd.read_csv(data_path)

# Separamos las características (X) de la variable objetivo (y)
X = df[['rooms','area','age']].values  # Características: habitaciones, área, edad
y = df['price'].values                 # Variable objetivo: precio

# Creamos un conjunto de prueba usando el último 20% de los datos
# (simulando el mismo split que en entrenamiento)
split = int(len(X)*0.8)
X_test = X[split:]  # Características de prueba
y_test = y[split:]  # Precios reales de prueba

# Realizamos predicciones en el conjunto de prueba
preds = model.predict(X_test)

# Calculamos métricas de evaluación
r2 = r2_score(y_test, preds)                    # R² Score: qué tan bien explica la varianza
mae = mean_absolute_error(y_test, preds)        # Error Absoluto Medio: promedio de errores

# Mostramos los resultados
print(f'R2: {r2:.4f}')    # Coeficiente de determinación (0-1, más alto es mejor)
print(f'MAE: {mae:.2f}')  # Error absoluto medio en unidades de precio
