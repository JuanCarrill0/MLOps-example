# Importamos las librerías necesarias para la API y el modelo
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import os

class RequestBody(BaseModel):
    """
    Modelo de datos para validar las peticiones de predicción.
    Define la estructura esperada de los datos de entrada.
    """
    rooms: int      # Número de habitaciones
    area: float     # Área en metros cuadrados
    age: int        # Edad de la vivienda en años

# Creamos la aplicación FastAPI
app = FastAPI(title='mlops-example predict')

# Intentamos cargar el modelo al iniciar la aplicación
model_path = os.path.join('models','model.pkl')
if os.path.exists(model_path):
    model = joblib.load(model_path)  # Cargamos el modelo entrenado
else:
    model = None  # Si no existe el modelo, lo marcamos como None

@app.get('/health')
def health():
    """
    Endpoint para verificar el estado de la API.
    Retorna el estado del servicio y si el modelo está cargado.
    """
    return {'status':'ok', 'model_loaded': model is not None}

@app.post('/predict')
def predict(body: RequestBody):
    """
    Endpoint para realizar predicciones de precios de viviendas.
    
    Args:
        body: Datos de la vivienda (habitaciones, área, edad)
        
    Returns:
        dict: Precio predicho de la vivienda
        
    Raises:
        HTTPException: Si el modelo no está disponible
    """
    # Verificamos que el modelo esté cargado
    if model is None:
        raise HTTPException(status_code=503, 
                          detail='Model not available. Run training first.')
    
    # Preparamos los datos en el formato esperado por el modelo
    x = np.array([[body.rooms, body.area, body.age]])
    
    # Realizamos la predicción
    pred = model.predict(x)[0]
    
    # Retornamos el precio predicho
    return {'predicted_price': float(pred)}
