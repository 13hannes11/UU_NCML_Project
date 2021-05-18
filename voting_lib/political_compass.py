import pandas as pd

def get_compass_parties(year=2017, country='de'):
    if country == 'de':
        if year == 2017:
            data  = [[5.5, 8],  [2, 4.5], [6, 6.25], [-2.5, -1.5], [7, 3], [3, 2.5]]
            index =  [  'AfD', 'BÜ90/GR', 'CDU/CSU', 'DIE LINKE.',  'FDP',   'SPD']
        elif year == 2013:
            data  = [[-3.5, -4],  [7, 6.5],   [-7, -6.5], [1, 2]]
            index =  ['BÜ90/GR', 'CDU/CSU', 'DIE LINKE.', 'SPD']
        elif year == 2005:
            # TODO: add data for 2011
            data  = [[-1.5, -1.5],  [9.5, 8],     [-6, -2], [3, 3.5]]
            index =  [  'BÜ90/GR', 'CDU/CSU', 'DIE LINKE.',   'SPD']
        else:
            raise Exception("Year " + str(year) + " does not exist for " + country)
    elif country == 'uk':
        if year == 2017:
            # TODO: add data
            data  = []
            index =  []
        elif year == 2015:
            # TODO: add data
            data  = []
            index =  []
        elif year == 2011:
            # TODO: add data
            data  = []
            index =  []
            pass
        else:
            raise Exception("Year " + str(year) + " does not exist for " + country)
    else:
        raise Exception("No data for " + country)
    
    return pd.DataFrame(data=data, index=index)