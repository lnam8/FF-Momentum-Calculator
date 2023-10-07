from espn_api.football import League
import pandas as pd
import argparse

swid_pub = '{AA3E68CA-5D8E-46B6-BD21-3CA1A769BC3A}'
espn_s2_pub = 'AEBn%2BTZo5i4qxp2DmrNjiLeSE44bUdVOXiz52Xhwrj%2BPPGQ%2F0syAzKkBG%2BM2G9YqnL382KnFXWeVXhgxATkXQlIZD1vimsAaVWqFe2OG2MMDYxCKCjmsTbdbUu%2BUSRYvguDpTMa%2BghFGgxhbvfPtPB68N7g9WmWlm6BQ%2BjHqblWpGSvCebdkD%2BscgkHkTLucs7rTrj3wbq91yOaCLg4%2F0EFRvWLtR4CtFlW0Y5y1ExiAqnLLtumbljuS7AIOHz7m7qp7kX1VkCKUUqiUJa8x6Plu'


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A script that gathers weekly ESPN Fantasy Football Projections")
    parser.add_argument("--week", type=int, help="Week as an integer ex. 1")
    args = parser.parse_args()
    week = args.week

    # Connects to the ESPN League
    league = League(league_id=1014228097, year=2023, espn_s2=espn_s2_pub, swid=swid_pub)

    # Retrieves Player instances for players named in players_list
    player_list = ['Puka Nacua', 'Calvin Ridley', 'Kyle Pitts', 'Zach Ertz', 'Josh Jacobs', 'Raheem Mostert']

    # For each week, go through the Free Agent list and pull the projection amounts
    data = []
    positions = ['WR', 'RB', 'TE']
    for week in range(1, week + 1):
        for position in positions:
            fa = league.free_agents(position=position, week=week)
            filtered_players = [i for i in fa if i.name in player_list]
            for player in filtered_players:
                data.append(
                    {
                        'data_source': 'ESPN',
                        'player_id': player.playerId,
                        'player_name': player.name,
                        'player_position': player.position,
                        'week': week,
                        'standard_projected_points': None,
                        'half_ppr_projected_points': player.projected_points,
                        'ppr_projected_points': None
                    }
                )

    pdf = pd.DataFrame(data)

    # Appending any new ESPN records to the CSV
    historical_data = pd.read_csv('projection_data.csv')
    espn_only = historical_data[historical_data.data_source == 'ESPN']
    espn_only = espn_only.astype({'player_id': 'int'})
    complement = pd.concat([espn_only, pdf], ignore_index=True)
    complement.drop_duplicates(inplace=True, keep=False)
    if len(complement) != 0:
        print('Adding {} records to `projection_data.csv`'.format(len(complement)))
        print(complement.T)
        complement.to_csv('projection_data.csv', mode='a', index=False, header=False)
    else:
        print('No new records to add')
