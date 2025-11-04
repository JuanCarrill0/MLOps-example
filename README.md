# mlops-example

Proyecto ejemplo de MLOps (entrenamiento, versionado mínimo, despliegue con FastAPI, CI/CD básico).

## Estructura
```
mlops-example/
│
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── models/
├── requirements.txt
├── dvc.yaml
├── .github/workflows/ci-cd.yml
├── Dockerfile
└── README.md
```

## Requisitos
- Python 3.9+
- git, dvc (opcional si quieres versionar datos)
- Docker (opcional para despliegue)

## Cómo ejecutar localmente (rápido)
1. Clona o descomprime este proyecto.
2. Crea un entorno virtual e instala dependencias:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # o .venv\Scripts\activate en Windows
   pip install -r requirements.txt
   ```
3. Entrenar modelo (genera datos sintéticos si no hay dataset):
   ```bash
   python src/train.py
   ```
4. Evaluar:
   ```bash
   python src/evaluate.py
   ```
5. Levantar la API de predicción:
   ```bash
   uvicorn src.predict:app --reload --host 0.0.0.0 --port 8000
   ```
   Endpoint POST `/predict` con JSON: `{"rooms":3,"area":75,"age":10}`

## CI/CD (GitHub Actions)
Se incluye un workflow ejemplo en `.github/workflows/ci-cd.yml` que instala dependencias, tira datos de DVC (comentado) y ejecuta `train` y `evaluate`.

---
Si quieres que cree un repositorio en GitHub o despliegue a Docker/Heroku/GCP, dime cuál prefieres y lo preparo.
