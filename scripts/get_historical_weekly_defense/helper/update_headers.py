import os
import pandas as pd

# Define the directory
directory = './'

for filename in os.listdir(directory):
    if filename.endswith(".csv"):  # we're assuming the files are CSVs
        # read the file
        df = pd.read_csv(os.path.join(directory, filename))
        
        # replace "Y/A" with "Y_A" in column headers
        df.columns = df.columns.str.replace('Passing_Completion%', 'Passing_Completion')

        # overwrite the file
        df.to_csv(os.path.join(directory, filename), index=False)
