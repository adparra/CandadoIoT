import os
import logging
import pandas as pd
import numpy as np
from scipy.stats import kurtosis, skew
from scipy.signal import butter, lfilter
from sklearn.ensemble import RandomForestClassifier
import joblib
from database import get_db_connection
from typing import List

ASSETS_DIR = "ml_assets"
MODEL_PATH = os.path.join(ASSETS_DIR, "binary_anomaly_model.pkl") 
os.makedirs(ASSETS_DIR, exist_ok=True)

def load_model():
    """Carga el modelo de ML desde el archivo."""
    try:
        model = joblib.load(MODEL_PATH)
        logging.info(f"Modelo Binario '{MODEL_PATH}' cargado.")
        return model
    except FileNotFoundError:
        logging.warning(f"ADVERTENCIA: Archivo del modelo no encontrado en '{MODEL_PATH}'. Debe ser entrenado.")
        return None
    except Exception as e:
        logging.error(f"Error crítico al cargar el modelo: {e}")
        return None

model = load_model()

def butter_lowpass_filter(data, cutoff=3.0, fs=10.0, order=5):
    nyq = 0.5 * fs; normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return lfilter(b, a, data)

def extract_features_from_dataframe(data, window_size=20, overlap=0.5):
    features, labels = [], []
    step = int(window_size * (1 - overlap))
    cols_to_process = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']
    
    for i in range(0, len(data) - window_size, step):
        window = data.iloc[i:i+window_size]
        label = window['class_code'].mode()[0]
        window_features = []
        for col in cols_to_process:
            signal = window[col]
            window_features.extend([
                signal.mean(), signal.std(), np.sqrt(np.mean(signal**2)),
                signal.min(), signal.max(), skew(signal), kurtosis(signal)
            ])
        features.append(window_features)
        labels.append(label)
    return np.array(features), np.array(labels)


def train_model():
    """
    Servicio para entrenar el modelo binario desde la tabla 'official_training_data'.
    """
    global model
    logging.info("Iniciando servicio de re-entrenamiento con el dataset oficial de la BD...")
    
    db = get_db_connection()
    if not db:
        return {"status": "error", "message": "No se pudo conectar a la base de datos."}
    
    try:
        query = "SELECT ax, ay, az, gx, gy, gz, class_code FROM official_training_data"
        df_raw = pd.read_sql(query, db)
        
        if len(df_raw) < 1000:
            return {"status": "error", "message": f"Datos insuficientes en la BD. Se necesitan >1000, se encontraron {len(df_raw)}."}
        
        for col in ['ax', 'ay', 'az', 'gx', 'gy', 'gz']:
            df_raw[col] = butter_lowpass_filter(df_raw[col])

        X, y = extract_features_from_dataframe(df_raw, window_size=20, overlap=0.5) 
        
        if len(X) == 0:
            return {"status": "error", "message": "No se pudieron generar ventanas de características desde los datos de la BD."}

   
        clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        clf.fit(X, y)
        
      
        joblib.dump(clf, MODEL_PATH)
        model = clf
        
        return {"status": "success", "message": f"Modelo binario re-entrenado con {len(X)} muestras.", "model_path": MODEL_PATH}

    except Exception as e:
        logging.error(f"Error crítico durante el re-entrenamiento desde la BD: {e}")
        return {"status": "error", "message": str(e)}
    finally:
        if db.is_connected():
            db.close()


def predict_from_window(window_data: List[dict]):
    """Servicio para predecir una anomalía desde una ventana de datos."""
    if model is None:
        return {"status": "error", "message": "Modelo no disponible. Por favor, entrene el modelo primero."}

    if len(window_data) < 20:
        return {"status": "error", "message": f"Ventana de datos corta. Se necesitan 20 muestras."}
    
    df_window = pd.DataFrame(window_data)
    features_row = []
    for col in ['ax', 'ay', 'az', 'gx', 'gy', 'gz']:
        signal = df_window[col]
        features_row.extend([
            signal.mean(), signal.std(), np.sqrt(np.mean(signal**2)),
            signal.min(), signal.max(), skew(signal), kurtosis(signal)
        ])
    feature_vector = np.array(features_row).reshape(1, -1)
    
    prediction_code = model.predict(feature_vector)[0]
    prediction_proba = model.predict_proba(feature_vector)[0]
    
    class_map = {0: "Normal", 1: "Anomalia"}
    label = class_map.get(int(prediction_code), "Desconocido")
    confidence = prediction_proba.max() * 100
    
    return {
        "status": "success",
        "prediction_code": int(prediction_code),
        "label": label,
        "confidence_percent": round(confidence, 2)
    }