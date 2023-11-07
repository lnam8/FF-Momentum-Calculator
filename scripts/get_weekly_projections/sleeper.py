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
        if d[str(week)] is not None:
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
        else:
            data.append(
                {
                    'data_source': 'Sleeper',
                    'player_id': int(player['id']),
                    'player_name': player['name'],
                    'player_position': player['position'],
                    'week': week,
                    'standard_projected_points': 0,
                    'half_ppr_projected_points': 0,
                    'ppr_projected_points': 0
                }
            )

    pdf = pd.DataFrame(data)

    # Get the projected points for all FLEX players over 3.0 points
    flex_data = []
    u = 'https://api.sleeper.app/projections/nfl/2023/1?season_type=regular&position[]=RB&position[]=TE&position[]=WR&order_by=ppr'
    re = requests.get(u)
    dd = re.json()
    d0 = [i for i in dd if len(i.get('stats')) != 1 and i.get('stats').get('pts_half_ppr') is not None and i.get('stats').get('pts_half_ppr') >= 3.0]
    for a in d0:
        flex_data.append(
            {
                'data_source': 'Sleeper',
                'player_id': int(a['player_id']),
                'player_name': ' '.join([a['player']['first_name'], a['player']['last_name']]),
                'player_position': a['player']['position'],
                'week': week,
                'standard_projected_points': a['stats']['pts_std'],
                'half_ppr_projected_points': a['stats']['pts_half_ppr'],
                'ppr_projected_points': a['stats']['pts_ppr']
            }
        )
    flex_pdf = pd.DataFrame(flex_data)

    # Concat and drop duplicates
    concat_pdf = pd.concat([pdf, flex_pdf])
    de_duped_pdf = concat_pdf.drop_duplicates()

    # Appending any new Sleeper records to the CSV
    if len(de_duped_pdf) != 0:
        print('Adding {} records to `projection_data.csv`'.format(len(de_duped_pdf)))
        print(de_duped_pdf.head(6))
        de_duped_pdf.to_csv('projection_data.csv', mode='a', index=False, header=False)
    else:
        print('No new records to add')
