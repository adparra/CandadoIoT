#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <TinyGPS++.h>
#include <HardwareSerial.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include <time.h>
#include <Stepper.h>

// --- Configuración de Pines ---
#define I2C_SDA_PIN 25
#define I2C_SCL_PIN 26
#define GPS_RX_PIN 16
#define GPS_TX_PIN 17

// --- Configuración del Motor Stepper ---
const int STEPS_PER_REV = 2048;
#define IN1 19
#define IN2 18
#define IN3 5
#define IN4 23
const int STEPS_TO_MOVE = 14336;

// --- Configuración WiFi y Servidor ---
const char* ssid = XXX
const char* password = XXX;
const char* apiToken = XXX;
const char* serverUrl = XXX;

// --- Configuración NTP ---
const char* ntpServer = "pool.ntp.org";
const long gmtOffset_sec = -5 * 3600;
const int daylightOffset_sec = 0;

// --- Instancias de Sensores y Motor ---
Adafruit_MPU6050 mpu;
TinyGPSPlus gps;
HardwareSerial gpsSerial(2);
Stepper myStepper(STEPS_PER_REV, IN1, IN2, IN3, IN4);

// --- Variables Globales ---
unsigned long lastUploadTime = 0;
const long uploadInterval = 100;
bool timeConfigured = false;
unsigned long lastStatusCheck = 0;
const long statusCheckInterval = 5000; // Preguntar al servidor cada 5 segundos
bool isLocked = true;
bool wifiConnected = false;

// --- Funciones de Conexión ---
void connectToWiFi() {
  if (WiFi.status() == WL_CONNECTED) {
    wifiConnected = true;
    return;
  }
  
  Serial.println("Conectando a WiFi...");
  WiFi.disconnect();
  delay(1000);
  WiFi.begin(ssid, password);
  
  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 20) {
    delay(500);
    Serial.print(".");
    attempts++;
  }
  
  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\nConectado a WiFi!");
    Serial.print("Dirección IP: ");
    Serial.println(WiFi.localIP());
    configTime(gmtOffset_sec, daylightOffset_sec, ntpServer);
    timeConfigured = true;
    wifiConnected = true;
  } else {
    Serial.println("\nError al conectar a WiFi");
    timeConfigured = false;
    wifiConnected = false;
  }
}

String getISOTimestamp() {
  struct tm timeinfo;
  if (!getLocalTime(&timeinfo, 50)) { return "unavailable"; }
  char timestamp[25];
  strftime(timestamp, sizeof(timestamp), "%Y-%m-%dT%H:%M:%SZ", &timeinfo);
  return String(timestamp);
}

void sendDataToServer(const char* endpoint, const JsonDocument& doc) {
  if (WiFi.status() != WL_CONNECTED) { 
    Serial.println("WiFi no conectado, no se pueden enviar datos");
    return; 
  }
  
  HTTPClient http;
  String url = String(serverUrl) + endpoint;
  http.begin(url);
  http.addHeader("Content-Type", "application/json");
  http.addHeader("Authorization", String("Bearer ") + apiToken);
  
  String jsonString;
  serializeJson(doc, jsonString);
  
  int httpCode = http.POST(jsonString);
  if (httpCode > 0) {
    String payload = http.getString();
    Serial.printf("Datos enviados a %s [%d]\n", endpoint, httpCode);
  } else {
    Serial.printf("Error en la solicitud de datos: %s\n", http.errorToString(httpCode).c_str());
  }
  http.end();
}

// --- Función para verificar el estado del candado ---
void checkLockStatus() {
  Serial.println("Solicitando estado del candado...");
  
  HTTPClient http;
  String url = String(serverUrl) + "/lock/status";
  
  http.begin(url);
  http.addHeader("Authorization", String("Bearer ") + apiToken);
  http.addHeader("Content-Type", "application/json");
  http.setTimeout(10000);
  http.setReuse(false);
  
  int httpCode = http.GET();
  Serial.printf("Código HTTP recibido: %d\n", httpCode);
  
  if (httpCode > 0) {
    String payload = http.getString();
    Serial.print("Respuesta del servidor: ");
    Serial.println(payload);
    
    if (httpCode == 200) {
      StaticJsonDocument<200> doc;
      DeserializationError error = deserializeJson(doc, payload);
      
      if (error) {
        Serial.print("Error al parsear JSON: ");
        Serial.println(error.c_str());
      } else {
        bool lockOpen = doc["lock_open"];
        Serial.printf("Estado del servidor (lock_open): %s\n", lockOpen ? "true" : "false");
        Serial.printf("Estado local (isLocked): %s\n", isLocked ? "true" : "false");
        
        bool serverLocked = !lockOpen;
        
        if (serverLocked != isLocked) {
          if (serverLocked) {
            Serial.println("Cerrando candado...");
            myStepper.step(-STEPS_TO_MOVE);
          } else {
            Serial.println("Abriendo candado...");
            myStepper.step(STEPS_TO_MOVE);
          }
          
          isLocked = serverLocked;
          Serial.print("Nuevo estado local: ");
          Serial.println(isLocked ? "CERRADO" : "ABIERTO");
        } else {
          Serial.println("Estado sin cambios, no se requiere acción");
        }
      }
    } else {
      Serial.printf("Error HTTP: %s\n", http.errorToString(httpCode).c_str());
    }
  } else {
    Serial.printf("Error en la solicitud: %s\n", http.errorToString(httpCode).c_str());
    wifiConnected = false;
  }
  
  http.end();
  Serial.println("Conexión HTTP cerrada");
}

void setup() {
  Serial.begin(115200);
  delay(2000);
  
  Serial.println("Iniciando sistema de sensores y control de candado...");
  
  myStepper.setSpeed(15);

  Wire.begin(I2C_SDA_PIN, I2C_SCL_PIN);
  if (!mpu.begin(0x68)) {
    Serial.println("Error al inicializar MPU6050");
    while (1);
  }
  mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
  mpu.setGyroRange(MPU6050_RANGE_500_DEG);
  mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);

  gpsSerial.begin(9600, SERIAL_8N1, GPS_RX_PIN, GPS_TX_PIN);
  
  connectToWiFi();
}

void loop() {
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("WiFi desconectado - Intentando reconectar...");
    wifiConnected = false;
    connectToWiFi();
  } else if (!wifiConnected) {
    wifiConnected = true;
    Serial.println("WiFi reconectado exitosamente");
  }
  
  // --- Leer y enviar datos de sensores ---
  sensors_event_t a, g, temp;
  mpu.getEvent(&a, &g, &temp);

  bool gpsUpdated = false;
  unsigned long gpsStartTime = millis();
  while (millis() - gpsStartTime < 1000) {
    while (gpsSerial.available()) {
      if (gps.encode(gpsSerial.read())) {
        gpsUpdated = true;
      }
    }
  }

  if (millis() - lastUploadTime >= uploadInterval) {
    lastUploadTime = millis();
    String timestamp = getISOTimestamp();
    
    StaticJsonDocument<256> mpuDoc;
    mpuDoc["timestamp"] = timestamp;
    mpuDoc["ax"] = a.acceleration.x;
    mpuDoc["ay"] = a.acceleration.y;
    mpuDoc["az"] = a.acceleration.z;
    mpuDoc["gx"] = g.gyro.x;
    mpuDoc["gy"] = g.gyro.y;
    mpuDoc["gz"] = g.gyro.z;
    sendDataToServer("/data/mpu", mpuDoc);

    if (gpsUpdated && gps.location.isValid()) {
      StaticJsonDocument<256> gpsDoc;
      gpsDoc["timestamp"] = timestamp;
      gpsDoc["latitude"] = gps.location.lat();
      gpsDoc["longitude"] = gps.location.lng();
      gpsDoc["altitude"] = gps.altitude.meters();
      gpsDoc["speed"] = gps.speed.kmph();
      sendDataToServer("/data/gps", gpsDoc);
    }
  }
  
  // --- Verificar estado del candado en intervalos regulares ---
  unsigned long currentTime = millis();
  if (wifiConnected && (currentTime - lastStatusCheck >= statusCheckInterval || lastStatusCheck == 0)) {
    lastStatusCheck = currentTime;
    checkLockStatus();
  }
}
