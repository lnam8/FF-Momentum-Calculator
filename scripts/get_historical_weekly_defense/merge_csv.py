import pandas as pd
import glob
import os

if __name__ == "__main__":

    final_csv_name = 'historical_weekly_data.csv'

    # Load in all CSVs into one
    files = sorted(glob.glob('{}/*_historical_weekly_data.csv'.format(os.getcwd())), reverse=True)
    dfs = [pd.read_csv(f) for f in files]
    pdf = pd.concat(dfs)
    print('Total number of records: {}'.format(len(pdf)))

    # Export to a single CSV
    pdf.to_csv(final_csv_name, index=False, header=True)

    