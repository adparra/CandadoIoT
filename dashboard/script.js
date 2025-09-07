// --- CONFIGURACIÓN Y SERVICIOS ---

// 1. CONFIGURACIÓN DEFINIDA DIRECTAMENTE (REEMPLAZA A loadEnv)
const CONFIG = {
    // Usando los valores de tu archivo .env
    API_BASE_URL: "https://backtesis.terranet.com.ec",
    API_TOKEN: "8da75daf84811a7b0fb46a8e9c96145ddd964925959188889f7d1e01354830d1"
};

// Función genérica para hacer peticiones GET a la API con autenticación
async function fetchApi(endpoint, token) {
    // Se añade un manejador de errores para respuestas que no son JSON
    const response = await fetch(endpoint, {
        headers: { 'Authorization': `Bearer ${token}` }
    });
    if (!response.ok) {
        throw new Error(`Error en la petición ${response.status}`);
    }
    // Previene error si la respuesta está vacía
    const text = await response.text();
    return text ? JSON.parse(text) : {};
}

// Función genérica para hacer peticiones POST a la API con autenticación
async function postApi(endpoint, token) {
    const response = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Authorization': `Bearer ${token}` }
    });
    if (!response.ok) {
        throw new Error(`Error en la petición ${response.status}`);
    }
    const text = await response.text();
    return text ? JSON.parse(text) : {};
}

// --- LÓGICA DE LA INTERFAZ (UI) ---

// Muestra la página solicitada y oculta las demás
function showPage(pageId) {
    document.querySelectorAll('.page').forEach(page => page.classList.remove('active'));
    
    const newPage = document.getElementById(pageId);
    if (newPage) { // Verificación para evitar errores si el ID es incorrecto
        newPage.classList.add('active');
    }

    document.querySelectorAll('.nav-link').forEach(link => {
        link.classList.toggle('active', link.dataset.page === pageId);
    });
}

// Renderiza datos en una tabla
function renderTable(tbody, data, rowGenerator) {
    if (!tbody) return; // Evita error si la tabla no existe
    tbody.innerHTML = '';
    if (!data || data.length === 0) {
        const colSpan = tbody.closest('table').querySelector('thead tr').cells.length;
        tbody.innerHTML = `<tr><td colspan="${colSpan}">No hay datos disponibles.</td></tr>`;
        return;
    }
    data.forEach(item => tbody.appendChild(rowGenerator(item)));
}
// --- LÓGICA DEL CANDADO ---
async function updateLockStatus(apiUrl, apiToken) {
    const btn = document.getElementById('lock-control-btn');
    const statusText = document.getElementById('lock-status-text');

    try {
        const status = await fetchApi(`${apiUrl}/lock/status`, apiToken);
        if (status.lock_open) {
            // Estado: Abierto
            statusText.textContent = 'Estado: Abierto';
            btn.textContent = 'Cerrar Candado';
            btn.className = 'btn-cerrar';
        } else {
            // Estado: Cerrado
            statusText.textContent = 'Estado: Cerrado';
            btn.textContent = 'Abrir Candado';
            btn.className = 'btn-abrir';
        }
    } catch (error) {
        console.error("Error al obtener estado del candado:", error);
        statusText.textContent = 'Estado: Error';
        btn.textContent = 'Reintentar';
        btn.className = '';
    }
}

// --- CARGA DE DATOS PARA LAS PÁGINAS ---

// Carga los datos para el Dashboard principal
async function loadDashboardData(apiUrl, apiToken, map) {
    try {
        // Carga últimos 5 datos GPS
        const gpsData = await fetchApi(`${apiUrl}/data/gps?limit=5`, apiToken);
        const gpsTbody = document.querySelector('#gps-table tbody');
        renderTable(gpsTbody, gpsData, d => {
            const tr = document.createElement('tr');
            tr.innerHTML = `<td>${new Date(d.timestamp).toLocaleTimeString()}</td><td>${d.latitude.toFixed(6)}</td><td>${d.longitude.toFixed(6)}</td>`;
            return tr;
        });

        // Actualiza el mapa
        const latlngs = gpsData.map(d => [d.latitude, d.longitude]);
        if (latlngs.length > 0) {
            map.setView(latlngs[0], 16);
            L.marker(latlngs[0]).addTo(map).bindPopup('Última posición').openPopup();
            L.polyline(latlngs, { weight: 3, color: 'blue' }).addTo(map);
        }

        // Carga últimos 5 datos MPU
        const mpuData = await fetchApi(`${apiUrl}/data/mpu?limit=5`, apiToken);
        const mpuTbody = document.querySelector('#mpu-table tbody');
        renderTable(mpuTbody, mpuData, d => {
            const tr = document.createElement('tr');
            tr.innerHTML = `<td>${new Date(d.timestamp).toLocaleTimeString()}</td>
                            <td>${d.ax}</td><td>${d.ay}</td><td>${d.az}</td>
                            <td>${d.gx}</td><td>${d.gy}</td><td>${d.gz}</td>`;
            return tr;
        });
    } catch (error) {
        console.error("Error al cargar datos del dashboard:", error);
    }
}

// Carga los datos para las páginas de historial
async function loadHistoryPage(dataType, apiUrl, apiToken) {
    let endpoint = '';
    let tbody = null;
    let rowGenerator = null;

    if (dataType === 'mpu') {
        endpoint = `${apiUrl}/data/mpu`;
        tbody = document.querySelector('#mpu-history-table tbody');
        rowGenerator = d => {
            const tr = document.createElement('tr');
            tr.innerHTML = `<td>${new Date(d.timestamp).toLocaleString()}</td>
                            <td>${d.ax}</td><td>${d.ay}</td><td>${d.az}</td>
                            <td>${d.gx}</td><td>${d.gy}</td><td>${d.gz}</td>`;
            return tr;
        };
    } else if (dataType === 'gps') {
        endpoint = `${apiUrl}/data/gps`;
        tbody = document.querySelector('#gps-history-table tbody');
        rowGenerator = d => {
            const tr = document.createElement('tr');
            tr.innerHTML = `<td>${new Date(d.timestamp).toLocaleString()}</td>
                            <td>${d.latitude.toFixed(6)}</td><td>${d.longitude.toFixed(6)}</td>
                            <td>${d.altitude}</td><td>${d.speed}</td>`;
            return tr;
        };
    } else if (dataType === 'predictions') {
    endpoint = `${apiUrl}/model/predictions`;
    tbody = document.querySelector('#predictions-history-table tbody');
    rowGenerator = d => {
        const tr = document.createElement('tr');

        const label = d.label || 'Desconocido';
        const confidence = (d.confidence_percent || 0).toFixed(2);

        // FORMATEO DE FECHA MEJORADO
        let fechaFormateada = 'Fecha inválida';
        if (d.prediction_timestamp) {
            const fecha = new Date(d.prediction_timestamp);
            // Opciones para un formato como: 27/08/2025, 02:34:07 PM
            const opciones = {
                year: 'numeric', month: '2-digit', day: '2-digit',
                hour: '2-digit', minute: '2-digit', second: '2-digit',
                hour12: true
            };
            fechaFormateada = fecha.toLocaleString('es-EC', opciones);
        }

        // Se elimina la celda del ID (id) y se usa la nueva fecha
        tr.innerHTML = `<td>${fechaFormateada}</td>
                        <td>${label}</td>
                        <td>${confidence}%</td>`;
        return tr;
    };
}

    try {
        const data = await fetchApi(endpoint, apiToken);
        renderTable(tbody, data, rowGenerator);
    } catch (error) {
        console.error(`Error al cargar datos de ${dataType}:`, error);
        if (tbody) {
            tbody.innerHTML = `<tr><td colspan="100%">Error al cargar los datos. Revise la consola.</td></tr>`;
        }
    }
}


// --- LÓGICA PRINCIPAL Y EVENTOS ---

async function init() {
    try {
        const API_BASE_URL = CONFIG.API_BASE_URL;
        const API_TOKEN = CONFIG.API_TOKEN;
        // 👇 PEGA ESTE BLOQUE COMPLETO DENTRO DE init()
        // --- INICIO LÓGICA DEL CANDADO ---
        // Carga el estado inicial del candado al cargar la página
        await updateLockStatus(API_BASE_URL, API_TOKEN);

        // Añade el evento de clic al botón de control
        document.getElementById('lock-control-btn').addEventListener('click', async () => {
            const status = await fetchApi(`${API_BASE_URL}/lock/status`, API_TOKEN);

            if (status.lock_open) {
                // Si está abierto, enviamos la orden de cerrar
                await postApi(`${API_BASE_URL}/lock/close`, API_TOKEN);
            } else {
                // Si está cerrado, enviamos la orden de abrir
                await postApi(`${API_BASE_URL}/lock/open`, API_TOKEN);
            }

            // Actualiza la UI para reflejar el nuevo estado
            await updateLockStatus(API_BASE_URL, API_TOKEN);
        });
        // --- FIN LÓGICA DEL CANDADO ---
        // Inicializa mapa Leaflet
        const map = L.map('map').setView([0, 0], 2);
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png').addTo(map);

        // Carga inicial del dashboard
        await loadDashboardData(API_BASE_URL, API_TOKEN, map);

        // 2. CORRECCIÓN EN EL EVENT LISTENER (añadido 'async')
        document.querySelectorAll('.nav-link').forEach(link => {
            link.addEventListener('click', async (e) => {
                e.preventDefault();
                const pageId = e.target.dataset.page;
                showPage(pageId);

                if (pageId !== 'page-dashboard' && !e.target.dataset.loaded) {
                    const dataType = pageId.split('-')[2];
                    await loadHistoryPage(dataType, API_BASE_URL, API_TOKEN);
                    e.target.dataset.loaded = 'true';
                }
            });
        });

        // Configura los botones de exportación
        document.getElementById('export-mpu').addEventListener('click', () => window.open(`${API_BASE_URL}/data/export/mpu_data?token=${API_TOKEN}`));
        document.getElementById('export-gps').addEventListener('click', () => window.open(`${API_BASE_URL}/data/export/gps_data?token=${API_TOKEN}`));
        document.getElementById('export-predictions').addEventListener('click', () => window.open(`${API_BASE_URL}/data/export/predictions?token=${API_TOKEN}`));

        // Configura el interruptor de predicciones
        const toggle = document.getElementById('prediction-toggle');
        toggle.addEventListener('change', async (e) => {
            try {
                if (e.target.checked) {
                    await postApi(`${API_BASE_URL}/model/predict/enable`, API_TOKEN);
                    console.log('Predicciones activadas');
                } else {
                    await postApi(`${API_BASE_URL}/model/predict/disable`, API_TOKEN);
                    console.log('Predicciones desactivadas');
                }
            } catch (err) {
                alert('No se pudo cambiar el estado de las predicciones.');
            }
        });

        // Inicia el monitor en vivo
        const statusDiv = document.getElementById('live-status');
        setInterval(async () => {
            try {
                const result = await fetchApi(`${API_BASE_URL}/model/predict/now`, API_TOKEN);
                // 3. CORRECCIÓN: usa 'label' para consistencia con la base de datos
                if (result.label === 'Anomalia') {
                    statusDiv.textContent = '¡ANOMALÍA DETECTADA!';
                    statusDiv.className = 'status-box status-anomaly';
                } else {
                    statusDiv.textContent = 'Estado: Normal';
                    statusDiv.className = 'status-box status-normal';
                }
            } catch (err) {
                statusDiv.textContent = 'Error de conexión';
                statusDiv.className = 'status-box';
            }
        }, 5000);

    } catch (err) {
        console.error('Error en la inicialización:', err);
        document.body.innerHTML = `<div style="padding: 2rem; text-align: center;"><h2>Error Crítico</h2><p>No se pudo cargar la configuración o conectar con la API. Revise la consola.</p><p><i>${err.message}</i></p></div>`;
    }
}

window.addEventListener('DOMContentLoaded', init);