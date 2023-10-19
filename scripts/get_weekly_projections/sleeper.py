import requests
import pandas as pd
import argparse

players = [
    {'id': '9493', 'name': 'Puka Nacua', 'position': 'WR'},
    {'id': '7553', 'name': 'Kyle Pitts', 'position': 'TE'},
    {'id': '5850', 'name': 'Josh Jacobs', 'position': 'RB'},
    {'id': '4981', 'name': 'Calvin Ridley', 'position': 'WR'},
    {'id': '2749', 'name': 'Raheem Mostert', 'position': 'RB'},
    {'id': '1339', 'name': 'Zach Ertz', 'position': 'TE'}
]


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A script that gathers weekly Sleeper Fantasy Football Projections")
    parser.add_argument("--week", type=int, help="Week as an integer ex. 1")
    args = parser.parse_args()
    week = args.week

    # Gathers weekly projection data for each player requested
    data = []
    for player in players:
        url = 'https://api.sleeper.com/projections/nfl/player/{}?season_type=regular&season=2023&grouping=week'.format(player['id'])
        r = requests.get(url)
        d = r.json()
        data.append(
            {
                'data_source': 'Sleeper',
                'player_id': int(player['id']),
                'player_name': player['name'],
                'player_position': player['position'],
                'week': week,
                'standard_projected_points': d[str(week)]['stats']['pts_std'],
                'half_ppr_projected_points': d[str(week)]['stats']['pts_half_ppr'],
                'ppr_projected_points': d[str(week)]['stats']['pts_ppr']
            }
        )

    pdf = pd.DataFrame(data)

    # Appending any new Sleeper records to the CSV
    if len(pdf) != 0:
        print('Adding {} records to `projection_data.csv`'.format(len(pdf)))
        print(pdf)
        pdf.to_csv('projection_data.csv', mode='a', index=False, header=False)
    else:
        print('No new records to add')
