from fastapi import FastAPI, HTTPException
from model import WeeklyProjection, WeeklyMadden
import helper
import utils

app = FastAPI()

@app.get("/")
def read_root():
    return "Welcome to FF Momentum Calculator" 

@app.get("/weeklyprojections/{week}")
def read_weekly(week: str):
    results = utils.read_weekly_projections_by_week(week)

    if not results:
        raise HTTPException(status_code=404, detail=f"Week {week} not found")
    
    return results

@app.get('/weeklymadden/')
def read_year_week(week: str, year: str):
    results = utils.read_weekly_madden(year, week)

    if not results:
        raise HTTPException(status_code=404, detail=f"Year {year} or Week {week} not found")

    return results

@app.get("/weeklymadden/{player_name}")
def read_player(player_name: str):
    results = utils.read_weekly_madden_by_player(player_name)

    if not results:
        raise HTTPException(status_code=404, detail=f"Player {player_name} not found")

    return results

@app.get("/weeklystats/{week}")
def read_weekly_stats(week: str):
    results = utils.read_weekly_stats_by_week(week)

    if not results:
        raise HTTPException(status_code=404, detail=f"Week {week} not found")

    return results

@app.get("/seasonprojections/")
def read_season_projections():
    results = utils.read_season_projections()

    if not results:
        raise HTTPException(status_code=404, detail=f"Season projections not found")

    return results

@app.get("/historicalweeklystats/")
def read_historical_weekly_stats(week: str, year: str):
    results = utils.read_historical_weekly_stats(year, week)

    if not results:
        raise HTTPException(status_code=404, detail=f"Week {week} or Year {year} not found")

    return results

@app.get("/historicalweeklydefense/")
def read_historical_weekly_defense_stats(week: str, year: str):
    results = utils.read_historical_weekly_defense_stats(year, week)

    if not results:
        raise HTTPException(status_code=404, detail=f"Week {week} or Year {year} not found")

    return results
