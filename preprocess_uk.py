import pandas as pd
import os

# Preprocess data
vote_counter = -1
data = pd.DataFrame()

name_column = 'Member'
vote_column = 'Vote'

vote_column_to_title = {}

voting_features = {'Aye':0, 'No':1, 'No Vote Recorded':2}
for dirname, _, filenames in os.walk('./uk/csv'):
    for filename in filenames:
        vote_counter += 1
        print(os.path.join(dirname, filename))
        
        # Read title rows
        title_df = pd.read_csv(os.path.join(dirname, filename),nrows=(3),skip_blank_lines=True,header=None)
    
        # Read data rows
        df = pd.read_csv(os.path.join(dirname, filename),skiprows=(10))
        
        # Give each voting behaviour type an identifier from 0 to len(voting_features) - 1
        df[vote_column].replace(voting_features, inplace=True)
        
        #Replace the vote column name
        vote_column_name = f'vote_{vote_counter}'
        df=df.rename(columns={vote_column:vote_column_name})
         
        # Map column name of vote to title -> allows retrieving what the vote was about
        vote_column_to_title[vote_column_name] = title_df.iat[2,0]
                
        if data.empty:
            # if first file that is loaded set data equal to data from first file
            data = df[[name_column, vote_column_name]]
        else:
            # merge data with already loaded data 
            data = data.merge(df[[name_column, vote_column_name]], on=name_column)

print(vote_column_to_title)
print(data)
