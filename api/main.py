from fastapi import FastAPI, HTTPException
from model import WeeklyProjection
import sqlite3
import helper

app = FastAPI()
def _get_column_names_from_table(table): 
    
    conn = helper.create_connection()
    cursor = conn.cursor()

    cursor.execute(f"PRAGMA table_info({table})")
    columns = [column[1] for column in cursor.fetchall()]

    conn.close()

    return columns    

def read_weekly_projections_by_week(week):
    conn = helper.create_connection()
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

def read_weekly_madden(year, week):
    conn = helper.create_connection()
    cursor = conn.cursor()
    column_names, column_types = helper.get_madden_columns()
    cursor.execute("select * from madden_weekly where year=? and week=?", (year, week,))
    results = cursor.fetchall()
    weekly_madden_results = [WeeklyMadden(**dict(zip(column_names, row))) for row in results]

    conn.close()
    return weekly_madden_results[0] 

read_weekly_madden("2023", "1")

@app.get("/")
def read_root():
    return "Welcome to FF Momentum Calculator" 

@app.get("/weeklyprojections/{week}")
def read_weekly(week: str):
    results = read_weekly_projections_by_week(week)

    if not results:
        raise HTTPException(status_code=404, detail=f"Week {week} not found")
    
    return results

@app.get('/weeklymadden/')
def read_year_week(week: str, year: str):
    results = read_weekly_madden(year, week)

    if not results:
        raise HTTPException(status_code=404, detail=f"Year {year} or Week {week} not found")

    return results

    
