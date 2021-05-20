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
            data  = [[-1.5, -1.5],  [9.5, 8],     [-6, -2], [3, 3.5]]
            index =  [  'BÜ90/GR', 'CDU/CSU', 'DIE LINKE.',   'SPD']
        else:
            raise Exception("Year " + str(year) + " does not exist for " + country)
    elif country == 'uk':
        if year == 2019:
            data  = [[9.5, 7], [7, 8], [-3, -5], [-4.5, -1.5], [4, 2.5], [-0.5, -0.5], [-2, 1.5], [-0.5, -0.5]]
            index =  ['Conservative', 'Democratic Unionist Party', 'Green Party', 'Labour', 'Liberal Democrat', 'Plaid Cymru', 'Scottish National Party', 'Social Democratic & Labour Party']
        elif year == 2017:
            data  = [[8.5, 7.5], [5.5, 8], [-2.5, -4.5], [-4, -2], [3.5, 1], [-0.5, -1.5], [-1.5, 1.5]]
            index =  ['Conservative', 'Democratic Unionist Party', 'Green Party', 'Labour', 'Liberal Democrat', 'Plaid Cymru', 'Scottish National Party']
        elif year == 2015:
            data  = [[9, 6.5], [5, 8.5], [-4, -5], [4, 5.5], [5, 2.5], [-2.5, -1], [-0.5, 1.5], [-2, 4], [-2.5, -0.5], [8, 8]]
            index =  ['Conservative', 'Democratic Unionist Party', 'Green Party', 'Labour', 'Liberal Democrat', 'Plaid Cymru', 'Scottish National Party', 'Sinn Féin', 'Social Democratic & Labour Party', 'UK Independence Party']
        else:
            raise Exception("Year " + str(year) + " does not exist for " + country)
    else:
        raise Exception("No data for " + country)
    
    return pd.DataFrame(data=data, index=index)