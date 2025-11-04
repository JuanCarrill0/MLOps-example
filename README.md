# 🧠 MLOps Example – Implementación práctica en Ingeniería de Software

Este proyecto es una **implementación funcional de MLOps (Machine Learning Operations)**, aplicada al contexto de la **ingeniería de software**.  
Su objetivo es demostrar cómo automatizar el ciclo de vida completo de un modelo de Machine Learning: desde el entrenamiento hasta el despliegue y la monitorización, usando herramientas abiertas y pipelines reproducibles.

---

## 🚀 ¿Qué hace este programa?

Este sistema entrena un modelo de **regresión lineal** para predecir el **precio estimado de una vivienda** a partir de variables simples:
- Número de habitaciones (`rooms`)
- Área en metros cuadrados (`area`)
- Antigüedad (`age`)

Incluye:
- Generación de datos sintéticos (si no hay dataset)
- Entrenamiento y evaluación del modelo
- API REST (con FastAPI) para realizar predicciones
- Control de versiones de datos y modelos con DVC
- Pipeline de CI/CD con GitHub Actions
- Integración opcional con MLflow para seguimiento de experimentos

---

## 🧩 Estructura del proyecto

```
mlops-example/
│
├── data/
│   ├── raw/                # Datos originales
│   └── processed/          # Datos procesados (versionados con DVC)
│
├── src/
│   ├── train.py            # Entrena el modelo y guarda modelo.pkl
│   ├── evaluate.py         # Evalúa el modelo (R2, MAE)
│   └── predict.py          # API FastAPI para predicciones
│
├── models/                 # Carpeta de modelos entrenados
│
├── .github/workflows/
│   └── ci-cd.yml           # Pipeline CI/CD de GitHub Actions
│
├── requirements.txt        # Dependencias del entorno
├── dvc.yaml                # Definición del pipeline de datos
├── Dockerfile              # Imagen Docker para desplegar la API
└── README.md
```

---

## ⚙️ ¿Cómo funciona el pipeline MLOps?

| Etapa | Descripción | Herramienta |
|-------|--------------|--------------|
| 1️⃣ Datos | Generación o carga de datos (`data/processed/housing.csv`) | DVC |
| 2️⃣ Entrenamiento | Entrena modelo de regresión y lo guarda (`models/model.pkl`) | scikit-learn |
| 3️⃣ Evaluación | Calcula métricas de desempeño | sklearn.metrics |
| 4️⃣ Versionamiento | Versiona datos y modelos | Git + DVC |
| 5️⃣ Despliegue | Expone modelo vía API REST (FastAPI) | Docker + Uvicorn |
| 6️⃣ CI/CD | Automatiza entrenamiento, evaluación y artefactos | GitHub Actions |

---

## 🛠️ Instalación y ejecución local

### 1️⃣ Clonar el repositorio
```bash
git clone https://github.com/<tu-usuario>/mlops-example.git
cd mlops-example
```

### 2️⃣ Crear entorno virtual e instalar dependencias
```bash
python -m venv .venv
source .venv/bin/activate   # En Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3️⃣ Entrenar el modelo
```bash
python src/train.py
```
Esto genera:
- `data/processed/housing.csv`
- `models/model.pkl`

### 4️⃣ Evaluar el modelo
```bash
python src/evaluate.py
```

### 5️⃣ Servir el modelo con API
```bash
uvicorn src.predict:app --reload
```
Luego abre en tu navegador:  
👉 [http://localhost:8000/docs](http://localhost:8000/docs)

Puedes probar el endpoint `POST /predict` con:
```json
{
  "rooms": 3,
  "area": 75,
  "age": 10
}
```

---

## 🧱 CI/CD con GitHub Actions

El archivo `.github/workflows/ci-cd.yml` ejecuta automáticamente:
1. Instalación de dependencias  
2. Entrenamiento del modelo  
3. Evaluación de métricas  
4. Subida del modelo entrenado como artefacto  

Cada vez que haces un `git push`, GitHub Actions lanza el pipeline y te mostrará los resultados en la pestaña **Actions** del repositorio.

---

## 💾 Control de versiones de datos y modelos con DVC

### 1️⃣ Inicializa DVC
```bash
dvc init
```

### 2️⃣ Versiona datos y modelos
```bash
dvc add data/processed/housing.csv
dvc add models/model.pkl
git add data/processed/housing.csv.dvc models/model.pkl.dvc .gitignore
git commit -m "Track data and model with DVC"
```

### 3️⃣ Configura el remoto (por ejemplo Google Drive)
Crea una carpeta en tu Drive llamada `mlops-storage` y copia su ID.

```bash
dvc remote add -d gdrive_remote gdrive://<ID>
dvc push
```

Así tus datos y modelos estarán almacenados fuera de GitHub, pero sincronizados.

---

## 🐳 Despliegue con Docker (opcional)

```bash
docker build -t mlops-example .
docker run -p 8000:8000 mlops-example
```

API disponible en:
```
http://localhost:8000/docs
```

---

## 📊 Monitoreo (opcional con MLflow)

Puedes registrar métricas automáticamente si tienes MLflow instalado:
```bash
mlflow ui
```
Y visualizar resultados de entrenamiento en:
👉 [http://localhost:5000](http://localhost:5000)

---

## 📚 Concepto: ¿Qué es MLOps?

> **MLOps** (Machine Learning Operations) es la práctica que combina **Machine Learning**, **DevOps** y **Data Engineering** para automatizar y mantener el ciclo de vida de los modelos de aprendizaje automático en producción.

En este ejemplo, MLOps permite:
- Automatizar el entrenamiento y evaluación del modelo  
- Versionar datasets y modelos con DVC  
- Desplegar un modelo como microservicio (FastAPI + Docker)  
- Asegurar reproducibilidad y trazabilidad con GitHub Actions  

---

## 👨‍💻 Autor
**Proyecto educativo de ejemplo – Ingeniería de Software y MLOps**  
Desarrollado por [Tu Nombre]  
Licencia: MIT

---

## 🧭 Próximos pasos
- Integrar monitoreo de “model drift” (Evidently AI)
- Conectar con MLflow remoto (tracking server)
- Implementar CI/CD completo con despliegue automático a Docker Hub o GCP

---

> “MLOps no es solo entrenar modelos; es llevarlos a producción de forma confiable, reproducible y escalable.”
