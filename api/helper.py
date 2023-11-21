import sqlite3

def create_connection():
    ## connection = sqlite3.connect("../db/ff_momentum.db")
    connection = sqlite3.connect("/Users/lnam/Georgia-Tech/CSE6242-DVA/FF-Momentum-Calculator/db/ff_momentum.db")
    return connection

def get_madden_columns():
    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info(madden_weekly)")
    columns = cursor.fetchall()

    column_names = [column[1] for column in columns]
    column_types = [column[2] for column in columns]    

    return column_names, column_types