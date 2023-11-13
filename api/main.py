from fastapi import FastAPI, HTTPException
from model import WeeklyProjection
import sqlite3

app = FastAPI()

def create_connection():
    connection = sqlite3.connect("../db/ff_momentum.db")
    return connection

def read_weekly_projections(week):
    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute("select * from weekly_projections where week=?", (week,))
    results = cursor.fetchall()
    conn.close()
    return [
            WeeklyProjection(
                data_source = row[0],
                player_id = row[1] if row[1] else 0,
                player_name = row[2],
                player_position = row[3],
                week = int(row[4]),
                standard_projected_points = float(row[5]) if row[5] else 0.00,
                half_ppr_projected_points = float(row[6]) if row[5] else 0.00,
                ppr_projected_points = float(row[7]) if row[7] else 0.00
            )
            for row in results 
        ]

@app.get("/")
def read_root():
    return "Welcome to FF Momentum Calculator" 

@app.get("/weeklyprojections/{week}")
def read_weekly(week: str):
    results = read_weekly_projections(week)

    if not results:
        raise HTTPException(status_code=404, detail=f"Week {week} not found")
    
    return results