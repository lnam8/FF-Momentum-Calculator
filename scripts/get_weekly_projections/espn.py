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

    # For the week provided, go through the Free Agent list and pull the projection amounts
    data = []
    positions = ['WR', 'RB', 'TE']
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

    # Get the overall projections for all FLEX players for that week over a 3 point threshold
    fa_data = []
    fas = [i for pos in positions for i in league.free_agents(position=pos, week=week) if i.projected_points >= 3.0]
    for pl in fas:
        fa_data.append(
            {
                'data_source': 'ESPN',
                'player_id': pl.playerId,
                'player_name': pl.name,
                'player_position': pl.position,
                'week': week,
                'standard_projected_points': None,
                'half_ppr_projected_points': pl.projected_points,
                'ppr_projected_points': None
            }
        )
    fa_pdf = pd.DataFrame(fa_data)

    # Concat and see if the new df has 6 less records after dropping duplicates
    concat_pdf = pd.concat([pdf, fa_pdf])
    de_duped_pdf = concat_pdf.drop_duplicates()

    # Appending new weekly ESPN records to the CSV
    if len(de_duped_pdf) != 0:
        print('Adding {} records to `projection_data.csv`'.format(len(de_duped_pdf)))
        print(de_duped_pdf.head(6))
        de_duped_pdf.to_csv('projection_data.csv', mode='a', index=False, header=False)
    else:
        print('No new records to add')
