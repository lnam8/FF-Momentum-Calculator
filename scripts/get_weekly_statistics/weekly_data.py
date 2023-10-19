import requests
import pandas as pd
import argparse

# API URL to make request to to pull in weekly data
url = 'https://api.sleeper.app/v1/stats/nfl/regular/2023/'

# Player information relative to Sleeper API
players = [
    {'id': '9493', 'name': 'Puka Nacua', 'position': 'WR', 'team': 'LAR'},
    {'id': '7553', 'name': 'Kyle Pitts', 'position': 'TE', 'team': 'ATL'},
    {'id': '5850', 'name': 'Josh Jacobs', 'position': 'RB', 'team': 'LV'},
    {'id': '4981', 'name': 'Calvin Ridley', 'position': 'WR', 'team': 'JAC'},
    {'id': '2749', 'name': 'Raheem Mostert', 'position': 'RB', 'team': 'MIA'},
    {'id': '1339', 'name': 'Zach Ertz', 'position': 'TE', 'team': 'ARI'}
]

# FantasyPros column names
wr_columns = ['Rank', 'player_name', 'receptions', 'targets', 'receiving_yards', 'receiving_yards_per_reception', 'LG', '20+', 'receiving_touchdowns', 'rushing_attempts', 'rushing_yards', 'rushing_touchdowns', 'FL', 'G', 'half_ppr_points', 'FPTS/G', 'ROST']
rb_columns = ['Rank', 'player_name', 'rushing_attempts', 'rushing_yards', 'rushing_yards_per_attempt', 'LG', '20+', 'rushing_touchdowns', 'receptions', 'targets', 'receiving_yards', 'receiving_yards_per_reception', 'receiving_touchdowns', 'FL', 'G', 'half_ppr_points', 'FPTS/G', 'ROST']
te_columns = ['Rank', 'player_name', 'receptions', 'targets', 'receiving_yards', 'receiving_yards_per_reception', 'LG', '20+', 'receiving_touchdowns', 'rushing_attempts', 'rushing_yards', 'rushing_touchdowns', 'FL', 'G', 'half_ppr_points', 'FPTS/G', 'ROST']


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A script that gathers weekly Fantasy Football Statistics")
    parser.add_argument("--week", type=int, help="Week as an integer ex. 1")
    args = parser.parse_args()
    week = args.week

    # Gathering the required weekly statistics data
    data = []
    top_performers_df = pd.DataFrame()

    r = requests.get(url + '{}'.format(week))
    d = r.json()
    
    # Appends weekly data for the requested 6 players above
    for player in players:
        try:
            d_info = d[player['id']]
            data.append({
                'player_name': player['name'],
                'player_position': player['position'],
                'player_team': player['team'],
                'week': int(week),
                'receptions': int(d_info.get('rec', 0)),
                'targets': int(d_info.get('rec_tgt', 0)),
                'receiving_yards': int(d_info.get('rec_yd', 0)),
                'receiving_yards_per_reception': float(d_info.get('rec_ypr', 0)),
                'receiving_touchdowns': int(d_info.get('rec_td', 0)),
                'rushing_attempts': int(d_info.get('rush_att', 0)),
                'rushing_yards': int(d_info.get('rush_yd', 0)),
                'rushing_yards_per_attempt': float(d_info.get('rush_ypa', 0)),
                'rushing_touchdowns': int(d_info.get('rush_td', 0)),
                'standard_points': float(d_info.get('pts_std', 0)),
                'half_ppr_points': float(d_info.get('pts_half_ppr', 0)),
                'ppr_points': float(d_info.get('pts_ppr', 0))
            })
        except KeyError as ke:
            print('Data for {} in Week {} not available yet'.format(player['name'], week))
    print('Added weekly data for requested players for Week {}.'.format(week))
    
    # Appends weekly data for the top WR, RB, and TE performers
    wr = ['https://www.fantasypros.com/nfl/stats/wr.php?week={}&scoring=HALF&range=week'.format(week), 'WR', wr_columns]
    rb = ['https://www.fantasypros.com/nfl/stats/rb.php?week={}&scoring=HALF&range=week'.format(week), 'RB', rb_columns]
    te = ['https://www.fantasypros.com/nfl/stats/te.php?week={}&scoring=HALF&range=week'.format(week), 'TE', te_columns]
    position_groups = [wr, rb, te]

    # Loop through position groups an pull top performers names
    for group in position_groups:
        fantasy_pros_url = group[0]
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

        top_performers_df = pd.concat([top_performers_df, df])

    pdf = pd.DataFrame(data)
    pdf = pd.concat([pdf, top_performers_df])

    # Appending any new records to the CSV
    if len(pdf) != 0:
        print('Adding {} records to `weekly_data.csv`'.format(len(pdf)))
        print(pdf)
        pdf.to_csv('weekly_data.csv', mode='a', index=False, header=False)
    else:
        print('No new records to add')
