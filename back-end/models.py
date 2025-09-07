from pydantic import BaseModel
from typing import List
from datetime import datetime

class MPUData(BaseModel):
    ax: float
    ay: float
    az: float
    gx: float
    gy: float
    gz: float

class MPUDataWithTimestamp(MPUData):
    timestamp: datetime

class GPSData(BaseModel):
    latitude: float
    longitude: float
    altitude: float
    speed: float

class GPSDataWithTimestamp(GPSData):
    timestamp: datetime

class PredictionWindow(BaseModel):
    window_data: List[MPUData]