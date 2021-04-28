import pandas as pd
import os

title_file = "filename_to_titles.csv"

# Preprocess data
vote_counter = -1
data = pd.DataFrame()

name_column = 'Bezeichnung'

# Load filename to 
filename_to_title_map = {}

with open(os.path.join("./de/csv/", title_file), 'r') as file:
    for line in file.readlines():
        filename, title = line.split(';', 2)
        filename_to_title_map[filename] = title



vote_column_to_title = {}

voting_features = ['ja', 'nein', 'Enthaltung', 'ungültig']
for dirname, _, filenames in os.walk('./de/csv'):
    for filename in filenames:
        if filename != title_file:
            vote_counter += 1
            print(os.path.join(dirname, filename))
            df = pd.read_csv(os.path.join(dirname, filename))
            
            # Give each voting behaviour type an identifier from 0 to len(voting_features) - 1
            for i, feature in enumerate(voting_features):
                df[feature] *= i
            vote_column_name = f'vote_{vote_counter}'
            # Map column name of vote to filename -> allows retrieving what the vote was about
            vote_column_to_title[vote_column_name] = filename_to_title_map[filename]
            
            # add feature for the vote
            df[vote_column_name] = df[voting_features].sum(axis=1)
            
            if data.empty:
                # if first file that is loaded set data equal to data from first file
                data = df[[name_column, vote_column_name]]
            else:
                # merge data with already loaded data 
                data = data.merge(df[[name_column, vote_column_name]], on=name_column)

print(vote_column_to_title)
print(data)
