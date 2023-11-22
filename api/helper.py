import sqlite3
from pydantic import create_model
from typing import ClassVar

def create_connection():
    ## connection = sqlite3.connect("../db/ff_momentum.db")
    connection = sqlite3.connect("/Users/lnam/Georgia-Tech/CSE6242-DVA/FF-Momentum-Calculator/db/ff_momentum.db")
    return connection

def get_columns_types(table_name):
    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = cursor.fetchall()

    conn.close()
    type_mapping = {"TEXT": str, "INTEGER": int, "REAL": float}    
    return {column[1]: type_mapping[column[2].upper()] for column in columns}

def get_column_names_from_table(table): 
    
    conn = create_connection()
    cursor = conn.cursor()

    cursor.execute(f"PRAGMA table_info({table})")
    columns = [column[1] for column in cursor.fetchall()]

    conn.close()

    return columns 

def format_results(columns, results):
    return [
            dict(zip(columns, row))
            for row in results
        ]
