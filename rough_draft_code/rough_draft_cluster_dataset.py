## Looing int the data sets. 
#In[]:
import pandas as pd 
import os 
import glob
import sqlite3          
WEEK_OF_THE_SEASON = 8
CURRENT_FOOTBALL_YEAR = 2023
THESE_PLAYERS = ['Puka Nakua','Josh Jacobs','Kyle Pitts','Zach Ertz','Raheem Mostert','Calvin Ridley'] 



#In[]:
conn = sqlite3.connect("C:\\Users\\15869\\Desktop\\FF_Teamwork\\FF-Momentum-Calculator\\db\\ff_momentum.db")
sql_query = """SELECT name FROM sqlite_master 
    WHERE type='table';"""
cursor = conn.cursor()
 
cursor.execute(sql_query)
tables = cursor.fetchall()

#In[]:

"""
###########################################################################################
Loading the data needed for the pull
data_weekly = all past weekly number data
cdata_weekly = Current Year ( defined above) data pulled for

"""
weekly_stats_query = """select player_name ,
            player_position,
            player_team ,
            year, 
            week ,
            receptions ,
            targets ,
            
           
            
            half_ppr_points 
              FROM """ +"historical_weekly_stats where year>2017 " 
cursor.execute(weekly_stats_query)
data_weekly = cursor.fetchall()

cols = ['player_name' ,
            'player_position',
            'player_team' ,
            'year', 
            'week' ,
            'receptions' ,
            'targets' ,
           #'receiving_yards' ,
            # 'receiving_yards_per_reception' ,
            # 'receiving_touchdowns' ,
            # 'rushing_attempts' ,
            # 'rushing_yards' ,
            # 'rushing_yards_per_attempt' ,
            # 'rushing_touchdowns' ,
            
            'half_ppr_points' 
            
            ]
df_data_weekly = pd.DataFrame(data_weekly,columns = cols)



#In[]:
cweekly_stats_query = """select DISTINCT 
            player_name ,
            player_position,
            player_team ,
            week, 
            receptions ,
            targets ,
           
            
            half_ppr_points
             FROM """ +"weekly_stats limit 5000 " 
cursor.execute(cweekly_stats_query)
cdata_weekly = cursor.fetchall()


cols2 = ['player_name' ,
            'player_position',
            'player_team' ,
            'week', 
            'receptions' ,
            'targets' ,
            #'receiving_yards' ,
            #'receiving_yards_per_reception' ,
            # 'receiving_touchdowns' ,
            # 'rushing_attempts' ,
            # 'rushing_yards' ,
            # 'rushing_yards_per_attempt' ,
            # 'rushing_touchdowns' ,
            
            'half_ppr_points' 
            
            ]
cdf_data_weekly = pd.DataFrame(cdata_weekly,columns = cols2)
cdf_data_weekly['year'] = CURRENT_FOOTBALL_YEAR

df_data_weekly = pd.concat([df_data_weekly,cdf_data_weekly]).reset_index(drop=True)

df_data_weekly['join_column'] = df_data_weekly.apply(lambda x:  str(x['player_name'] ) +" " + str(x['year'] ) + " " + str(x['week'] ),axis = 1) 
#In[]:


madden_weekly = """select * FROM """ +"madden_weekly  limit 5000 " 
data = cursor.execute(madden_weekly)
columns_in_madden = []
for column in data.description: 
    columns_in_madden.append(str(column[0]))
madden_weekly_data = cursor.fetchall()

madden_weekly_data = pd.DataFrame(madden_weekly_data,columns = columns_in_madden)
madden_weekly_data['join_column'] = madden_weekly_data.apply( lambda x: str(x['firstName']) + \
                                    " " + str(x['lastName']) + " " + \
                                    str(x['year']) +" " +  str( str(x['week']) ),axis  = 1)

#In[]:
if conn:
        # using close() method, we will close 
        # the connection
        conn.close()




#In[]:
import tqdm
years =  df_data_weekly['year'].unique()
all_players =  df_data_weekly['player_name'].unique()
each_player_by_year = {}
for x in tqdm.tqdm(years):
    print(x)

#In[]:
'''for y in tqdm.tqdm(years):
      for player in all_players:
            #print(y,player)
            temp_df = df_data_weekly[df_data_weekly['year'] == y]

            try: 
                one_player = temp_df[temp_df['player_name'] == player]
                temp_list = []
                for week_of_season in range(1,18):
                    try:
                        did_they_play = one_player[one_player['week'] == week_of_season]
                        if len(did_they_play) < 1:
                            temp_list.append(0)
                        #print(did_they_play['half_ppr_points'])
                        else:
                             temp_list.append(did_they_play['half_ppr_points'].iloc[0])

                    except:
                        temp_list.append(0)
                string_concat = str(player) + " " + str(y)
                each_player_by_year[string_concat] = (temp_list) 
            except:
                  pass # print('didnt play this year')
output_df_int_cluster = pd.DataFrame.from_dict(each_player_by_year).T

'''
piv_data = df_data_weekly.pivot(index= ['year','player_name','player_position'],columns = 'week',values = 'half_ppr_points').fillna(0).reset_index()
piv_data['index'] = piv_data.apply(lambda x: x['player_name'] + " " + str(x['year']), axis =1)
piv_data.drop(columns=['year','player_name','player_position'],inplace= True)
piv_data.set_index('index',inplace=True, drop=True)

output_df_int_cluster = pd.DataFrame.from_dict(each_player_by_year).T

#In[]:
good_indexes =  piv_data.sum(axis = 1) > 10

piv_data = piv_data[good_indexes]
#In[]:

COLUMNS_NEEDED = range(1,WEEK_OF_THE_SEASON )

modeling_output_df_int_cluster  = piv_data[COLUMNS_NEEDED]

from sklearn.neighbors import NearestNeighbors
for i in tqdm.tqdm([1]):
    neigh = NearestNeighbors(n_neighbors=5)
    neigh.fit(modeling_output_df_int_cluster)
    A = neigh.kneighbors_graph(modeling_output_df_int_cluster)
    B =A.toarray()
#In[]:
#from sklearn.neighbors import NearestNeighbors
import numpy as np
import matplotlib.pyplot as plt

new_df = modeling_output_df_int_cluster.reset_index('index')
future_df = piv_data.reset_index('index')


"""This section to be removed for real world application"""
TEMP_PROVE_POINT_COL = range(max(modeling_output_df_int_cluster.columns),max(piv_data.columns)+1)
future_output_df_int_cluster  = piv_data[TEMP_PROVE_POINT_COL]
temp_df = future_output_df_int_cluster.reset_index()




def findsuccess(id,WEEK_OF_THE_SEASON):
    hist_data = []
    labels = []
    plt.figure(figsize=(15,5))
    index_for_nn = np.where(B[:][id] ==1)[0]
    plt.plot(np.array(modeling_output_df_int_cluster.columns),np.array(new_df.iloc[id])[1:])
    #plt.plot(np.array(future_output_df_int_cluster.columns),np.array(temp_df.iloc[id])[1:],'r--')
    labels.append(np.array(new_df.iloc[id])[0])
    #labels.append(str(np.array(new_df.iloc[id])[0]) + ' Projected')
    plt.title('Players most like ' + str(np.array(new_df.iloc[id])[0]))
    for item in index_for_nn:
        if item != id: 
            print('Neighbors of id', new_df.iloc[item]['index'])
            plt.scatter(piv_data.columns,list(future_df.iloc[item])[1:])

            labels.append(np.array(future_df.iloc[item])[0])
            if sum(list(future_df.iloc[item])[WEEK_OF_THE_SEASON:]) > 5:
                for s in list(future_df.iloc[item])[1:]:
                    hist_data.append(s)
    plt.legend(labels)
    return hist_data







#In[]:
import re
list_of_people = []

for person in THESE_PLAYERS: 
    for item in modeling_output_df_int_cluster.index:
        person_year = person + " " +  str(CURRENT_FOOTBALL_YEAR)
        if re.findall(person_year,item) :
            
            temps = re.findall(person,item)
            for k in temps:
            

                list_of_people.append(new_df[new_df['index']== item ].index[0])


for number_id in list_of_people:

    hist_data = findsuccess(number_id,WEEK_OF_THE_SEASON)
    plt.figure(figsize=(5,5))
    plt.hist(list(hist_data))

#TODO Madden Data to be able to have the option to fiter the Random Forest by WR/ other
#Currently looks at only the points not the points vs hard opponents
