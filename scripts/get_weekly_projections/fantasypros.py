import pandas as pd
import argparse

# Adds a new data entry for a unique Player Name and Week
def add_new_data(row, score_type, week):
    if score_type == 'HALF':
        data.append(
            {
                'data_source': 'FantasyPros', 'player_id': None, 'player_name': row.player_name,
                'player_position': players[row.player_name], 'week': week,
                'standard_projected_points': None, 'half_ppr_projected_points': row.projected_points,
                'ppr_projected_points': None
            }
        )
    elif score_type == 'PPR':
        data.append(
            {
                'data_source': 'FantasyPros', 'player_id': None, 'player_name': row.player_name,
                'player_position': players[row.player_name], 'week': week,
                'standard_projected_points': None, 'half_ppr_projected_points': None,
                'ppr_projected_points': row.projected_points
            }
        )
    else:
        data.append(
            {
                'data_source': 'FantasyPros', 'player_id': None, 'player_name': row.player_name,
                'player_position': players[row.player_name], 'week': week,
                'standard_projected_points': row.projected_points, 'half_ppr_projected_points': None,
                'ppr_projected_points': None
            }
        )


# Either adds a new weeks' data entry for an existing player, or updates a players existing data entry for a new score type
def update(row, score_type, week):
    players_in_data = [i['player_name'] for i in data]

    # Adding a new data entry for a new player
    if row.player_name not in players_in_data:
        add_new_data(row, score_type, week)
    
    # A record with that player's name exist. Check to see if week's data will be updated or if a new week will be added
    else:
        name_week_data = [item for item in data if item['player_name'] == row.player_name and item['week'] == week]
        
        # Existing player but new week. Add new entry
        if name_week_data == []:
            add_new_data(row, score_type, week)
        
        # Existing player and same week. Update dictionary with new projection data
        else:
            if score_type == 'HALF':
                name_week_data[0]['half_ppr_projected_points'] = row.projected_points
            elif score_type == 'PPR':
                name_week_data[0]['ppr_projected_points'] = row.projected_points
            else:
                name_week_data[0]['standard_projected_points'] = row.projected_points


scoring = ['STD', 'HALF', 'PPR']
players = {
    'Puka Nacua': 'WR',
    'Kyle Pitts': 'TE',
    'Josh Jacobs': 'RB',
    'Calvin Ridley': 'WR',
    'Raheem Mostert': 'RB',
    'Zach Ertz': 'TE'
}
data = []


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A script that gathers weekly FantasyPros Fantasy Football Projections")
    parser.add_argument("--week", type=int, help="Week as an integer ex. 1")
    args = parser.parse_args()
    weeks = args.week

    # Loops through and populates data set with projection data for each unique week and score type
    for week in range(1, weeks + 1):
        for score in scoring:
            url = 'https://www.fantasypros.com/nfl/projections/flex.php?scoring={}&week={}'.format(score, week)

            pdf = pd.read_html(url)[0]
            cols = ['player_name', 'position', 'RUSHING ATT', 'RUSHING YDS', 'RUSHING TDS', 'RECEIVING REC', 'RECEIVING YDS', 'RECEIVING TDS', 'MISC FL', 'projected_points']
            
            df = pdf.copy()
            df.columns = cols
            df = df[['player_name', 'position', 'projected_points']]
            df['team'] = df.player_name.apply(lambda x: x.split()[-1])
            df['position'] = df.position.apply(lambda x: x[:2])
            df['player_name'] = df.player_name.apply(lambda x: ' '.join(x.split()[:-1]))

            players_to_use = df[df['player_name'].isin(list(players.keys()))]
            players_to_use.apply(lambda x: update(x, score, week), axis=1)
            

    pdf = pd.DataFrame(data)

    # Appending any new FantasyPros records to the CSV
    historical_data = pd.read_csv('projection_data.csv')
    fantasypros_only = historical_data[historical_data.data_source == 'FantasyPros']
    complement = pd.concat([fantasypros_only, pdf], ignore_index=True)
    complement.drop_duplicates(inplace=True, keep=False)
    if len(complement) != 0:
        print('Adding {} records to `projection_data.csv`'.format(len(complement)))
        print(complement)
        complement.to_csv('projection_data.csv', mode='a', index=False, header=False)
    else:
        print('No new records to add')
