import requests
import pandas as pd
import argparse
import os 

def get_data_by_week(year: str, week_number: int) -> list:

    data = []
    
    offset = 0
    limit = 1000 # Max of 1000 results
    
    while True: 
        url = f"https://ratings-api.ea.com/v2/entities/m{year}-ratings?filter=iteration%3Aweek-{week_number}&sort=overall_rating&limit=1000&offset={offset}"

        response = requests.get(url)
        if response.status_code != 200:
            print("Request failed with status code:", response.status_code)

        res_json = response.json()
        players = res_json['docs']
        data += players 

        if len(players) != limit:
            break

        offset += limit
    
    print("Finished getting data from Madden")
    return data
    
def dir_path(path):
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"Dir:{path} is not a valid path")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A script that pulls historical Madden Player ratings by week from years 2022-2023")
    parser.add_argument("--dir", type=dir_path, help="Directory to save csv")

    args = parser.parse_args()
    dir = os.path.abspath(args.dir)
    years = ["22", "23"]

    for year in years:
        for j in range(18):
            list_players_data = get_data_by_week(year, j + 1)
        
            try:
                file = f"{dir}/year_{year}_week_{j + 1}_madden_ratings.csv"
                df = pd.DataFrame(list_players_data)
                df.to_csv(file, index=False)
                print(f"File successfully created: {file}")

            except Exception as e:
                print("Error occured while writing csv")

        