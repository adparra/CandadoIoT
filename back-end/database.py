import os
import mysql.connector
from dotenv import load_dotenv

load_dotenv()

def get_db_connection():
    """
    Establece y retorna una conexión a la base de datos MySQL.
    """
    try:
        return mysql.connector.connect(
            host=os.getenv('DATABASE_HOST'),
            user=os.getenv('DATABASE_USER'),
            password=os.getenv('DATABASE_PASSWORD'),
            database=os.getenv('DATABASE_NAME'),
            port=int(os.getenv('DATABASE_PORT', 3306))  # Puerto añadido aquí
        )
    except mysql.connector.Error as err:
        print(f"Error de base de datos: {err}")
        return None
