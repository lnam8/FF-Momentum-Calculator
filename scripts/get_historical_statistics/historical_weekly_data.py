import pandas as pd

# FantasyPros column names
wr_columns = ['Rank', 'player_name', 'receptions', 'targets', 'receiving_yards', 'receiving_yards_per_reception', 'LG', '20+', 'receiving_touchdowns', 'rushing_attempts', 'rushing_yards', 'rushing_touchdowns', 'FL', 'G', 'half_ppr_points', 'FPTS/G', 'ROST']
rb_columns = ['Rank', 'player_name', 'rushing_attempts', 'rushing_yards', 'rushing_yards_per_attempt', 'LG', '20+', 'rushing_touchdowns', 'receptions', 'targets', 'receiving_yards', 'receiving_yards_per_reception', 'receiving_touchdowns', 'FL', 'G', 'half_ppr_points', 'FPTS/G', 'ROST']
te_columns = ['Rank', 'player_name', 'receptions', 'targets', 'receiving_yards', 'receiving_yards_per_reception', 'LG', '20+', 'receiving_touchdowns', 'rushing_attempts', 'rushing_yards', 'rushing_touchdowns', 'FL', 'G', 'half_ppr_points', 'FPTS/G', 'ROST']

# FantasyPros URLs
wr = ['https://www.fantasypros.com/nfl/stats/wr.php?week={}&scoring=HALF&range=week&year={}', 'WR', wr_columns]
rb = ['https://www.fantasypros.com/nfl/stats/rb.php?week={}&scoring=HALF&range=week&year={}', 'RB', rb_columns]
te = ['https://www.fantasypros.com/nfl/stats/te.php?week={}&scoring=HALF&range=week&year={}', 'TE', te_columns]
position_groups = [wr, rb, te]

# Hisorical year and week range for each listed year
year_week = [(2022, 18), (2021, 18), (2020, 17), (2019, 17), (2018, 17), (2017, 17), (2016, 17), (2015, 17), (2014, 17), (2013, 17), (2012, 17), (2011, 17), (2010, 17), (2009, 17), (2008, 17), (2007, 17), (2006, 17), (2005, 17), (2004, 17), (2003, 17), (2002, 17)]

if __name__ == "__main__":

    # Loops through each year and returns the weekly statistics and exports them to individual CSV files
    for pair in year_week:
        year = pair[0]
        weeks = pair[1]
        data = pd.DataFrame()
        print('Year {}'.format(year))
        
        # Gathering the required weekly statistics data
        for week in range(1, weeks + 1):
            # Loop through position groups an pull top performers names
            for group in position_groups:
                fantasy_pros_url = group[0].format(week, year)
                position = group[1]
                df_columns = group[2]

                # Read in the top performing players per position
                df = pd.read_html(fantasy_pros_url)[0]

                # Format data so that it is similar to JSON above
                df.columns = df_columns
                df['player_team'] = df.player_name.apply(lambda x: ' '.join(x.split()[-1:])[1:-1])
                df['player_name'] = df.player_name.apply(lambda x: ' '.join(x.split()[:-1]))
                df['player_position'] = position
                df['week'] = week
                df['standard_points'] = float(0)
                df['ppr_points'] = float(0)
                if position in ['WR', 'TE']:
                    df['rushing_yards_per_attempt'] = df['rushing_yards'].div(df['rushing_attempts'], axis=0).round(1).fillna(0)
                df = df[['player_name', 'player_position', 'player_team', 'week', 'receptions', 'targets', 'receiving_yards', 'receiving_yards_per_reception', 'receiving_touchdowns', 'rushing_attempts', 'rushing_yards', 'rushing_yards_per_attempt', 'rushing_touchdowns', 'standard_points', 'half_ppr_points', 'ppr_points']]
                
                # Retrieving only records that have fantasy points scored in that week
                df = df[df.half_ppr_points > 0]

                data = pd.concat([data, df])
            print(week, sep='\n', end=' ', flush=True)
        
        # Creates new CSV file and writes all data to that file
        print()
        filename = '{}_weekly_data.csv'.format(year)
        print('Adding {} records to `{}_weekly_data.csv`'.format(len(data), year))
        data.to_csv(filename, index=False, header=True)
