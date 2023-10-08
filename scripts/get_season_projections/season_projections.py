import requests
import pandas as pd

if __name__ == "__main__":

    players = [
        {'id': '9493', 'name': 'Puka Nacua', 'position': 'WR'},
        {'id': '7553', 'name': 'Kyle Pitts', 'position': 'TE'},
        {'id': '5850', 'name': 'Josh Jacobs', 'position': 'RB'},
        {'id': '4981', 'name': 'Calvin Ridley', 'position': 'WR'},
        {'id': '2749', 'name': 'Raheem Mostert', 'position': 'RB'},
        {'id': '1339', 'name': 'Zach Ertz', 'position': 'TE'}
    ]

    # Retrieves season projections for players named
    data = []
    for player in players:
        url = 'https://api.sleeper.com/projections/nfl/player/{}?season_type=regular&season=2023'.format(player['id'])
        r = requests.get(url)
        d = r.json()

        data.append(
                {
                    'player_name': player['name'],
                    'player_position': player['position'],
                    'years_experience': d['player']['years_exp'],
                    'receptions': int(d['stats'].get('rec', 0)),
                    'receiving_yards': int(d['stats'].get('rec_yd', 0)),
                    'receiving_touchdowns': int(d['stats'].get('rec_td', 0)),
                    'rushing_attempts': int(d['stats'].get('rush_att', 0)),
                    'rushing_yards': int(d['stats'].get('rush_yd', 0)),
                    'rushing_touchdowns': int(d['stats'].get('rush_td', 0)),
                    'standard_points': d['stats'].get('pts_std', 0),
                    'half_ppr_points': d['stats'].get('pts_half_ppr', 0),
                    'ppr_points': d['stats'].get('pts_ppr', 0)
                }
            )

    pdf = pd.DataFrame(data)
    print(pdf.T)

    # Append any records to CSV
    #pdf.to_csv('season_projections.csv', mode='a', index=False, header=False)