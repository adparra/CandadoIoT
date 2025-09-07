# main.py (Versión con guardado de predicciones)
import os
import logging
from fastapi import FastAPI, Request, HTTPException, status, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import RedirectResponse
from fastapi.responses import StreamingResponse  # Importar StreamingResponse
from typing import List
from collections import deque
from models import MPUData, MPUDataWithTimestamp, GPSData, GPSDataWithTimestamp, PredictionWindow
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.docs import get_redoc_html
import pandas as pd
import io
# --- Imports desde nuestros módulos locales ---
from database import get_db_connection
# from models import MPUData, MPUDataWithTimestamp, PredictionWindow # Asumiendo que crearás un modelo para PredictionLog
import ml_service
import numpy as np
import math
from datetime import datetime
from pydantic import BaseModel # Necesario para la declaración interna de modelos
# ==============================================================================
# SECCIÓN 1: CONFIGURACIÓN Y SEGURIDAD DE LA API
# ==============================================================================
# (Sin cambios aquí...)
app = FastAPI(title="ESP32 Smart Lock Backend", version="2.1.0", docs_url=None, redoc_url=None)
# --- Buffer de Datos en Memoria RAM ---
data_buffer = deque(maxlen=40) # Guarda las últimas 50 lecturas

# --- Control de Activación de Predicciones ---
PREDICTIONS_ENABLED = True # Por defecto, las predicciones están activadas
LOCK_STATE = False # False = Cerrado, True = Abierto
logging.basicConfig(filename="server.log", level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
@app.middleware("http")
async def log_requests(request: Request, call_next):
    client_host = "unknown"
    if request.client:
        client_host = request.client.host
    
    logging.info(f"REQUEST START: {request.method} {request.url} from {client_host}")
    response = await call_next(request)
    logging.info(f"REQUEST END: {request.method} {request.url} - STATUS {response.status_code}")
    return response

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

API_TOKEN = os.getenv('API_TOKEN', 'tu_super_token_secreto')
security = HTTPBearer()
async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != API_TOKEN:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Token inválido o ausente")

# ==============================================================================
# SECCIÓN 2: ENDPOINTS
# ==============================================================================
# --- Endpoints de Documentación y Raíz (sin cambios) ---
@app.get("/", include_in_schema=False)
async def root():
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not Found")
# 1. Endpoint protegido para servir el esquema OpenAPI (el JSON)
@app.get("/openapi.json", dependencies=[Depends(verify_token)], include_in_schema=False)
async def get_open_api_endpoint(req: Request):
    return req.app.openapi()

#@app.get("/docs", dependencies=[Depends(verify_token)], include_in_schema=False)
#async def get_documentation(req: Request):
#    return req.app.openapi()

@app.get("/docs", dependencies=[Depends(verify_token)], include_in_schema=False)
async def get_documentation():
    return get_swagger_ui_html(openapi_url="/openapi.json", title="Swagger UI")
# 3. Endpoint protegido para la UI de ReDoc (lo que pediste)
@app.get("/redoc", dependencies=[Depends(verify_token)], include_in_schema=False)
async def get_redoc_documentation():
    return get_redoc_html(openapi_url="/openapi.json", title="ReDoc")
    
# *** NUEVO ENDPOINT PARA EXPORTAR A CSV ***
@app.get("/data/export/{table_name}", dependencies=[Depends(verify_token)])
async def export_table_to_csv(table_name: str):
    """
    Exporta el contenido completo de una tabla de la BD a un archivo CSV.
    Tablas válidas: 'official_training_data', 'mpu_data', 'predictions', 'gps_data'
    """
    # Lista de tablas permitidas para evitar accesos no deseados
    allowed_tables = ['official_training_data', 'mpu_data', 'predictions', 'gps_data']
    if table_name not in allowed_tables:
        raise HTTPException(status_code=400, detail="Nombre de tabla no válido o no permitido.")

    db = get_db_connection()
    if not db:
        raise HTTPException(status_code=503, detail="No se pudo conectar a la base de datos.")
    
    try:
        # Usamos pandas para leer la tabla completa de forma sencilla
        query = f"SELECT * FROM {table_name}"
        df = pd.read_sql(query, db)
        
        # Convertimos el DataFrame a un string en formato CSV en memoria
        stream = io.StringIO()
        df.to_csv(stream, index=False)
        
        # Preparamos la respuesta para que el navegador la descargue
        response = StreamingResponse(
            iter([stream.getvalue()]),
            media_type="text/csv"
        )
        response.headers["Content-Disposition"] = f"attachment; filename={table_name}.csv"
        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al exportar la tabla: {e}")
    finally:
        if db.is_connected():
            db.close()

# --- Endpoint de Captura de Datos (Ahora con Buffer) ---
@app.post("/data/mpu", status_code=200, dependencies=[Depends(verify_token)])
async def save_mpu_data(data: MPUData):
    data_buffer.append(data.dict()) # Añade al buffer en RAM
    db = get_db_connection()
    if db:
        cursor = db.cursor()
        query = "INSERT INTO mpu_data (timestamp, ax, ay, az, gx, gy, gz) VALUES (NOW(), %s, %s, %s, %s, %s, %s)"
        try:
            cursor.execute(query, (data.ax, data.ay, data.az, data.gx, data.gy, data.gz))
            db.commit()
        except Exception as e:
            db.rollback(); logging.error(f"Error al guardar en BD: {e}")
        finally:
            cursor.close(); db.close()
    return {"status": "success", "message": "Dato recibido."}
@app.get("/data/mpu", response_model=List[MPUDataWithTimestamp], dependencies=[Depends(verify_token)])
async def get_mpu_data():
    db = get_db_connection()
    if not db:
        raise HTTPException(status_code=503, detail="No se pudo conectar a la base de datos.")
    cursor = db.cursor(dictionary=True)
    cursor.execute("SELECT * FROM mpu_data ORDER BY timestamp DESC LIMIT 50")
    results = cursor.fetchall()
    cursor.close(); db.close()
    return results

# Endpoints GPS

@app.post("/data/gps", status_code=201, dependencies=[Depends(verify_token)])
async def save_gps_data(data: GPSData):
    """
    Recibe una lectura de datos del GPS y la guarda en la base de datos.
    """
    db = get_db_connection()
    if not db:
        raise HTTPException(status_code=503, detail="No se pudo conectar a la base de datos.")
    
    cursor = db.cursor()
    query = """
        INSERT INTO gps_data (timestamp, latitude, longitude, altitude, speed)
        VALUES (NOW(), %s, %s, %s, %s)
    """
    try:
        cursor.execute(query, (data.latitude, data.longitude, data.altitude, data.speed))
        db.commit()
    except Exception as e:
        db.rollback()
        logging.error(f"Error al guardar datos GPS: {e}")
        raise HTTPException(status_code=500, detail=f"Error al guardar datos GPS: {e}")
    finally:
        cursor.close()
        db.close()
        
    return {"status": "success", "message": "Datos GPS guardados."}


@app.get("/data/gps", response_model=List[GPSDataWithTimestamp], dependencies=[Depends(verify_token)])
async def get_gps_data():
    """
    Obtiene los últimos 100 registros de datos del GPS desde la base de datos.
    """
    db = get_db_connection()
    if not db:
        raise HTTPException(status_code=503, detail="No se pudo conectar a la base de datos.")
        
    cursor = db.cursor(dictionary=True)
    try:
        query = "SELECT timestamp, latitude, longitude, altitude, speed FROM gps_data ORDER BY timestamp DESC LIMIT 50"
        cursor.execute(query)
        results = cursor.fetchall()
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al obtener datos GPS: {e}")
    finally:
        cursor.close()
        db.close()
# --- Endpoints de Control del Modelo ---
@app.post("/model/predict/enable", status_code=200, dependencies=[Depends(verify_token)])
async def enable_predictions():
    global PREDICTIONS_ENABLED
    PREDICTIONS_ENABLED = True
    return {"status": "success", "message": "Predicciones activadas."}

@app.post("/model/predict/disable", status_code=200, dependencies=[Depends(verify_token)])
async def disable_predictions():
    global PREDICTIONS_ENABLED
    PREDICTIONS_ENABLED = False
    return {"status": "success", "message": "Predicciones desactivadas."}

@app.get("/model/predict/now", status_code=200, dependencies=[Depends(verify_token)])
async def predict_now():
    """Realiza una predicción usando los últimos datos del buffer en RAM."""
    if not PREDICTIONS_ENABLED:
        return {"status": "disabled", "message": "Las predicciones están desactivadas."}

    if len(data_buffer) < 20:
        raise HTTPException(status_code=400, detail=f"Buffer insuficiente. Se necesitan 20 muestras, hay {len(data_buffer)}.")

    window_to_predict = list(data_buffer)[-20:]
    result = ml_service.predict_from_window(window_to_predict)
    
    if result["status"] == "error":
        raise HTTPException(status_code=500, detail=result["message"])
        
    # Guardar la predicción en la base de datos
    db = get_db_connection()
    if db:
        try:
            cursor = db.cursor()
            query = "INSERT INTO predictions (label, prediction_code, confidence_percent) VALUES (%s, %s, %s)"
            cursor.execute(query, (result['label'], result['prediction_code'], result['confidence_percent']))
            db.commit()
        except Exception as e:
            logging.error(f"No se pudo guardar la predicción: {e}")
        finally:
            cursor.close(); db.close()
            
    return result

@app.post("/model/retrain", status_code=200, dependencies=[Depends(verify_token)])
async def endpoint_retrain_model():
    result = ml_service.train_model()
    if result["status"] == "error": raise HTTPException(status_code=500, detail=result["message"])
    return result


# *** NUEVO ENDPOINT PARA EL DASHBOARD ***
@app.get("/model/predictions", dependencies=[Depends(verify_token)])
async def get_predictions():
    """
    Obtiene los últimos 100 registros de predicciones para ser mostrados en un dashboard.
    """
    db = get_db_connection()
    if not db:
        raise HTTPException(status_code=503, detail="No se pudo conectar a la base de datos.")
        
    try:
        cursor = db.cursor(dictionary=True)
        query = "SELECT * FROM predictions ORDER BY prediction_timestamp DESC LIMIT 100"
        cursor.execute(query)
        results = cursor.fetchall()
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al obtener predicciones: {e}")
    finally:
        if db.is_connected():
            cursor.close()
            db.close()
# ==============================================================================
# SECCIÓN 3: CONTROL DEL CANDADO
# ==============================================================================

@app.get("/lock/status", status_code=200, dependencies=[Depends(verify_token)])
async def get_lock_status():
    """
    Consulta el estado actual del candado. El ESP32 usará este endpoint.
    Retorna: {"lock_open": True} si está abierto, {"lock_open": False} si está cerrado.
    """
    return {"lock_open": LOCK_STATE}

@app.post("/lock/open", status_code=200, dependencies=[Depends(verify_token)])
async def open_lock():
    """
    Establece el estado del candado como 'Abierto' (True).
    """
    global LOCK_STATE
    LOCK_STATE = True
    logging.info("Comando recibido: ABRIR candado.")
    return {"status": "success", "message": "Candado abierto."}

@app.post("/lock/close", status_code=200, dependencies=[Depends(verify_token)])
async def close_lock():
    """
    Establece el estado del candado como 'Cerrado' (False).
    """
    global LOCK_STATE
    LOCK_STATE = False
    logging.info("Comando recibido: CERRAR candado.")
    return {"status": "success", "message": "Candado cerrado."}
class MadgwickAHRS:
    """
    Implementación de los algoritmos IMU y AHRS de Madgwick.
    """
    def __init__(self, sample_period: float = 1.0/256.0, beta: float = 0.1):
        self.sample_period = sample_period
        self.beta = beta
        self.q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    def update(self, gyro: np.ndarray, accel: np.ndarray):
        q = self.q
        if np.linalg.norm(accel) == 0: return
        # Normalizar el acelerómetro
        accel /= np.linalg.norm(accel)
        # Función objetivo del gradiente descendiente
        f = np.array([
            2*(q[1]*q[3] - q[0]*q[2]) - accel[0],
            2*(q[0]*q[1] + q[2]*q[3]) - accel[1],
            2*(0.5 - q[1]**2 - q[2]**2) - accel[2]
        ])
        # Matriz Jacobiana
        j = np.array([
            [-2*q[2],  2*q[3], -2*q[0], 2*q[1]],
            [ 2*q[1],  2*q[0],  2*q[3], 2*q[2]],
            [      0, -4*q[1], -4*q[2],      0]
        ])
        step = j.T @ f
        if np.linalg.norm(step) > 0:
            step /= np.linalg.norm(step)
        # Calcular la derivada del cuaternión
        q_dot = 0.5 * self.quaternion_multiply(q, np.array([0, gyro[0], gyro[1], gyro[2]])) - self.beta * step
        # Integrar para obtener el cuaternión actualizado
        self.q += q_dot * self.sample_period
        self.q /= np.linalg.norm(self.q)

    @staticmethod
    def quaternion_multiply(q1, q2):
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([-x1*x2 - y1*y2 - z1*z2 + w1*w2,
                         x1*w2 + y1*z2 - z1*y2 + w1*x2,
                        -x1*z2 + y1*w2 + z1*x2 + w1*y2,
                         x1*y2 - y1*x2 + z1*w2 + w1*z2], dtype=np.float64)

def rotate_vector_by_quaternion(v: np.ndarray, q: np.ndarray) -> np.ndarray:
    """ Rota un vector 'v' usando un cuaternión 'q'. """
    q_conj = np.array([q[0], -q[1], -q[2], -q[3]])
    v_q = np.array([0, v[0], v[1], v[2]])
    rotated_v = MadgwickAHRS.quaternion_multiply(MadgwickAHRS.quaternion_multiply(q, v_q), q_conj)
    return rotated_v[1:]

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """ Calcula la distancia en metros entre dos puntos geográficos. """
    R_earth = 6371000  # Radio de la Tierra en metros
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi / 2.0)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R_earth * c

# --- Gestión de Estado del Sensor ---
# Esta clase mantiene el estado del sensor entre llamadas a la API.
class SensorState:
    def __init__(self):
        self.ahrs = MadgwickAHRS(beta=0.05)
        self.vx, self.vy, self.x_pos, self.y_pos = 0.0, 0.0, 0.0, 0.0
        self.ref_lat, self.ref_lon = None, None
        self.last_mpu_timestamp = None
        self.R_earth = 6371000

# --- Instancia Global para el Estado de Dead Reckoning ---
# Agrega esta línea junto a tus otras variables globales como 'data_buffer'
dead_reckoning_state = SensorState()


# ==============================================================================
# NUEVOS ENDPOINTS PARA AGREGAR A LA SECCIÓN 2
# ==============================================================================
#

@app.post("/update_mpu", tags=["Dead Reckoning"], dependencies=[Depends(verify_token)])
async def update_mpu(data: MPUDataWithTimestamp):
    """
    Recibe datos del MPU para actualizar continuamente el estado de dead reckoning.
    """
    state = dead_reckoning_state # Usamos la instancia global
    
    if state.ref_lat is None or state.ref_lon is None:
        raise HTTPException(status_code=400, detail="Señal GPS no recibida aún. No se pueden procesar datos del MPU.")

    if state.last_mpu_timestamp is None:
        state.last_mpu_timestamp = data.timestamp
        return {"message": "Datos iniciales del MPU recibidos. Esperando más datos para actualizar."}

    dt = (data.timestamp - state.last_mpu_timestamp).total_seconds()
    if dt <= 0:
        return {"message": "Datos MPU duplicados o antiguos. No se realizó ninguna actualización."}

    state.last_mpu_timestamp = data.timestamp
    state.ahrs.sample_period = dt

    # Mapeo de ejes según la lógica original
    accel = np.array([-data.az, -data.ay, data.ax], dtype=np.float64)
    gyro = np.array([-data.gz, -data.gy, data.gx], dtype=np.float64)

    state.ahrs.update(gyro, accel)
    accel_world = rotate_vector_by_quaternion(accel, state.ahrs.q)
    accel_linear = accel_world - np.array([0, 0, 9.81])

    state.vx += accel_linear[0] * dt
    state.vy += accel_linear[1] * dt
    state.x_pos += state.vx * dt
    state.y_pos += state.vy * dt

    return {"status": "dead_reckoning_state_updated"}


@app.post("/update_gps", tags=["Dead Reckoning"], dependencies=[Depends(verify_token)])
async def update_gps(data: GPSDataWithTimestamp):
    """
    Recibe una nueva lectura de GPS, la compara con la posición actual de
    dead reckoning para verificar la divergencia y luego resetea la referencia.
    """
    state = dead_reckoning_state # Usamos la instancia global
    new_lat = data.latitude
    new_lon = data.longitude

    if state.ref_lat is None or state.ref_lon is None:
        state.ref_lat = new_lat
        state.ref_lon = new_lon
        state.last_mpu_timestamp = None
        return {"message": "Posición GPS inicial recibida. Dead reckoning inicializado."}

    dr_lat = state.ref_lat + (state.y_pos / state.R_earth) * (180 / np.pi)
    dr_lon = state.ref_lon + (state.x_pos / (state.R_earth * np.cos(np.pi * state.ref_lat / 180))) * (180 / np.pi)

    divergence = haversine_distance(new_lat, new_lon, dr_lat, dr_lon)
    
    alert = False
    if divergence > 50: # Umbral de 50 metros para generar alerta
        alert = True
        logging.warning(f"ALERTA DE DIVERGENCIA: {divergence:.2f} metros detectados!")

    response = {
        "dead_reckoning_estimate": {"latitude": dr_lat, "longitude": dr_lon},
        "new_gps_position": {"latitude": new_lat, "longitude": new_lon},
        "divergence_meters": round(divergence, 2),
        "alert_triggered": alert,
    }

    # Reseteamos la referencia a la nueva posición GPS
    state.ref_lat = new_lat
    state.ref_lon = new_lon
    state.vx, state.vy, state.x_pos, state.y_pos = 0.0, 0.0, 0.0, 0.0
    state.last_mpu_timestamp = None

    return response
