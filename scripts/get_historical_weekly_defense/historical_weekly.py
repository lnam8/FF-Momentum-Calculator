import pandas as pd
import argparse
from time import sleep

# Years that the opponent game logs that will be pulled
#years = [2022, 2021, 2020, 2019, 2018, 2017, 2016, 2015, 2014, 2013, 2012, 2011, 2010, 2009, 2008, 2007, 2006, 2005, 2004, 2003, 2002]
# Team mappings for Pro-Football Reference
team_map = {
    'crd': 'Arizona Cardinals' , 'atl': 'Atlanta Falcons', 'rav': 'Baltimore Ravens', 'buf': 'Buffalo Bills', 
    'car': 'Carolina Panthers', 'chi': 'Chicago Bears', 'cin': 'Cincinnati Bengals', 'cle': 'Cleveland Browns', 
    'dal': 'Dallas Cowboys', 'den': 'Denver Broncos', 'det': 'Detroit Lions', 'gnb': 'Green Bay Packers', 'htx': 'Houston Texans', 
    'clt': 'Indianapolis Colts', 'jax': 'Jacksonville Jaguars', 'kan': 'Kansas City Chiefs', 'rai': 'Las Vegas Raiders', 
    'sdg': 'Los Angeles Chargers', 'ram': 'Los Angeles Rams', 'mia': 'Miami Dolphins', 'min': 'Minnesota Vikings', 
    'nwe': 'New England Patriots', 'nor': 'New Orleans Saints', 'nyg': 'New York Giants', 'nyj': 'New York Jets', 
    'phi': 'Philadelphia Eagles', 'pit': 'Pittsburgh Steelers', 'sfo': 'San Francisco 49ers', 'sea': 'Seattle Seahawks', 
    'tam': 'Tampa Bay Buccaneers', 'oti': 'Tennessee Titans', 'was': 'Washington Commanders'
}
teams = list(team_map.keys())


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A script that gathers weekly Defense game logs from Pro-Football-Reference")
    parser.add_argument("--year", type=int, help="Year as an integer ex. 2022")
    args = parser.parse_args()
    year = args.year
    
    # Pro-Football Reference game log URL
    url = 'https://www.pro-football-reference.com/teams/{}/{}/gamelog/'
    
    # Column names
    dataframe_cols = ['Week', 'Day', 'Date', 'boxscore', 'Win/Loss', 'OT', 'at', 'Opponent', 'Team_Score', 'Opponent_Score', 
                      'Passing_Complete', 'Passing_Attempts', 'Passing_Yards', 'Passing_TD', 'Passing_Int', 'Sacks_Allowed', 
                      'Sacks_Yards_Allowed', 'Passing_Y/A', 'Passing_Net_Y/A', 'Passing_Completion%', 'QB_Rating', 
                      'Rushing_Attempts', 'Rushing_Yards', 'Rushing_Y/A', 'Rushing_TD', 'Field_Goals_Made', 'Field_Goals_Attempted', 
                      'Extra_Points_Made', 'Extra_Points_Attempted', 'Number_of_Punts', 'Punting_Yards', 'Third_Down_Conversions', 
                      'Third_Down_Attempts', 'Fourth_Down_Conversions', 'Fourth_Down_Attempts', 'Time_of_Possession'
                     ]
    
    # Loop through each year and pull in the opponent game logs for each team during the regular season
    #for year in years:
    data = pd.DataFrame()

    # Gathers the game logs and appends them to a DataFrame
    for team in teams:
        sleep(5)
        # Loads in the regular season opponent game log into a DataFrame
        final_url = url.format(team, year)
        pdf = pd.read_html(final_url)#, storage_options={'User-agent': 'your bot 0.1'})
        data_pdf = pdf[1] if len(pdf) == 2 else pdf[2]

        # Simple data cleaning and mapping
        data_pdf.columns = dataframe_cols
        data_pdf = data_pdf.drop(columns=['boxscore', 'at'])
        data_pdf.insert(loc=0, column='Year', value=year)
        data_pdf.insert(loc=6, column='Team', value=team_map.get(team))
        data_pdf['OT'] = data_pdf.OT.apply(lambda x: 0 if str(x) == 'nan' else 1)

        data = pd.concat([data, data_pdf])
        print(team, sep='\n', end=' ', flush=True)
    
    # Creates new CSV file and writes all data to that file
    print('\nAll data pulled, writing to a CSV.')
    filename = '{}_historical_weekly_data.csv'.format(year)
    print('Adding {} records to `{}_historical_weekly_data.csv`'.format(len(data), year))
    data.to_csv(filename, index=False, header=True)

