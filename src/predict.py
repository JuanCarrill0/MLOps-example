from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import os

class RequestBody(BaseModel):
    rooms: int
    area: float
    age: int

app = FastAPI(title='mlops-example predict')

model_path = os.path.join('models','model.pkl')
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    model = None

@app.get('/health')
def health():
    return {'status':'ok', 'model_loaded': model is not None}

@app.post('/predict')
def predict(body: RequestBody):
    if model is None:
        raise HTTPException(status_code=503, detail='Model not available. Run training first.')
    x = np.array([[body.rooms, body.area, body.age]])
    pred = model.predict(x)[0]
    return {'predicted_price': float(pred)}
