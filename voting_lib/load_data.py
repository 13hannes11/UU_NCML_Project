#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np

def load_german_data():
    """
    Load German Parliament data
    return : Data with columns [Member, Party, vote_0, vote_1 etc]
    """  
    title_file = "filename_to_titles.csv"
    vote_counter = -1
    #data = pd.DataFrame()
    data = {}
    
    period_column_g = 'Wahlperiode'
    name_column_g = 'Bezeichnung'
    party_column_g = 'Fraktion/Gruppe'
    name_column = 'Member'
    party_column = 'Party'
    
    vote_column_to_title = {}
    
    voting_features = ['ja', 'nein', 'Enthaltung', 'ungültig']
    for dirname, _, filenames in os.walk('./de/csv'):
        for filename in filenames:
            if filename != title_file:
                
                print(filename)
                
                vote_counter += 1
                df = pd.read_csv(os.path.join(dirname, filename))
                
                # Give each voting behaviour type an identifier from 0 to len(voting_features) - 1
                for i, feature in enumerate(voting_features):
                    df[feature] *= i
                vote_column_name = f'vote_{vote_counter}'
                
                # Map column name of vote to filename -> allows retrieving what the vote was about
                vote_column_to_title[vote_column_name] = filename
                
                # add feature for the vote
                df[vote_column_name] = df[voting_features].sum(axis=1)
                
                df=df.rename(columns={name_column_g:name_column,party_column_g:party_column})
                
                period = df.iloc[0][period_column_g]
                                    
                if period in data:
                    # merge data with already loaded data 
                    data[period] = data[period].merge(df[[name_column, vote_column_name]], on=name_column)                    
                else:
                    # if first file that is loaded set data equal to data from first file
                    data[period] = df[[name_column, party_column, vote_column_name]]
                    
    print(data)
    return data


def load_uk_data(path):
    """
    Load German Parliament data
    return : Data with columns [Member, Party, vote_0, vote_1 etc]
    """  
    #print directory path
    print(path) 
    # Preprocess data
    vote_counter = -1
    data = pd.DataFrame()
    
    name_column = 'Member'
    party_column = 'Party'
    vote_column = 'Vote'
    
    column_to_filename = {}
    
    voting_features = {'Aye':0, 'Teller - Ayes':0, 'No':1, 'Teller - Noes':1, 'No Vote Recorded':2}
    for dirname, _, filenames in os.walk(path):
        for filename in filenames:
            vote_counter += 1
            
            # Read title rows
            # sep is set to new line so it never splits up the title cells
            title_df = pd.read_csv(os.path.join(dirname, filename), sep='\n',nrows=(3),skip_blank_lines=True,header=None)
        
            # Read data rows
            df = pd.read_csv(os.path.join(dirname, filename),skiprows=(10))
            
            # Give each voting behaviour type an identifier from 0 to len(voting_features) - 1
            df[vote_column].replace(voting_features, inplace=True)
            
            #Replace the vote column name
            vote_column_name = f'vote_{vote_counter}'
            df=df.rename(columns={vote_column:vote_column_name})
             
            # Map column name of vote to title -> allows retrieving what the vote was about
            column_to_filename[vote_column_name] = title_df.iat[2,0]
                    
            if data.empty:
                # if first file that is loaded set data equal to data from first file
                data = df[[name_column, party_column, vote_column_name]]
            else:
                # merge data with already loaded data 
                data = data.merge(df[[name_column, vote_column_name]], on=name_column)
    
    print(data)
    return data