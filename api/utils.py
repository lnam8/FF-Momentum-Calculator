import helper
from model import WeeklyProjection, WeeklyMadden

def read_weekly_projections_by_week(week):
    conn = helper.create_connection()
    cursor = conn.cursor()
    cursor.execute("select * from weekly_projections where week=?", (week,))
    column_names = helper.get_column_names_from_table("weekly_projections")
    results = cursor.fetchall()
    conn.close()
    return helper.format_results(column_names, results)

def read_weekly_madden(year, week):
    conn = helper.create_connection()
    cursor = conn.cursor()
    column_names = helper.get_column_names_from_table("madden_weekly")
    cursor.execute("select * from madden_weekly where year=? and week=?", (year, week,))
    results = cursor.fetchall()
    weekly_madden_results = [(zip(column_names, row)) for row in results]
    conn.close()
    return weekly_madden_results 

def read_weekly_madden_by_player(player_name): 
    conn = helper.create_connection()
    cursor = conn.cursor()
    column_names = helper.get_column_names_from_table("madden_weekly")
    cursor.execute("SELECT * FROM madden_weekly WHERE fullNameForSearch LIKE ?", (f"%{player_name}%",))
    results = cursor.fetchall()
    conn.close()
    return helper.format_results(column_names, results) 

def read_weekly_stats_by_week(week):
    conn = helper.create_connection()
    cursor = conn.cursor()
    column_names = helper.get_column_names_from_table("weekly_stats")
    cursor.execute("select * from weekly_stats where week=?", (week,))
    results = cursor.fetchall()
    conn.close()
    return helper.format_results(column_names, results)

def read_season_projections():
    conn = helper.create_connection()
    cursor = conn.cursor()
    column_names = helper.get_column_names_from_table("season_projections")
    cursor.execute("select * from season_projections")
    results = cursor.fetchall()
    conn.close()
    return helper.format_results(column_names, results)

def read_historical_weekly_stats(year, week):
    conn = helper.create_connection()
    cursor = conn.cursor()
    column_names = helper.get_column_names_from_table("historical_weekly_stats")
    cursor.execute("select * from historical_weekly_stats where year=? and week=?", (year, week,))
    results = cursor.fetchall()
    conn.close()
    return helper.format_results(column_names, results)

def read_historical_weekly_defense_stats(year, week):
    conn = helper.create_connection()
    cursor = conn.cursor()
    column_names = helper.get_column_names_from_table("historical_weekly_defense")
    cursor.execute("select * from historical_weekly_defense where year=? and week=?", (year, week,))
    results = cursor.fetchall()
    conn.close()
    return helper.format_results(column_names, results)