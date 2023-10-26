## Looing int the data sets. 
#In[]:
import pandas as pd 
import os 
import glob
import sqlite3          
WEEK_OF_THE_SEASON = 7 

#In[]:
conn = sqlite3.connect("C:\\Users\\15869\\Desktop\\FF_Teamwork\\FF-Momentum-Calculator\\db\\ff_momentum.db")
sql_query = """SELECT name FROM sqlite_master 
    WHERE type='table';"""
 
    # Creating cursor object using connection object
cursor = conn.cursor()
     
    # executing our sql query
cursor.execute(sql_query)
print("List of tables\n")
     
    # printing all tables list

tables = cursor.fetchall()

#In[]:


for item in tables:
    print(item[0])
    temp_query = """select * FROM """ +str(item[0]) +    "Limit 3"
    print(temp_query)
    cursor.execute(sql_query)
    print(cursor.fetchall())

#In[]:

weekly_stats_query = """select * FROM """ +"historical_weekly_stats" 
cursor.execute(weekly_stats_query)
data_weekly = cursor.fetchall()

cols = ['player_name' ,
            'player_position',
            'player_team' ,
            'week' ,
            'receptions' ,
            'targets' ,
            'receiving_yards' ,
            'receiving_yards_per_reception' ,
            'receiving_touchdowns' ,
            'rushing_attempts' ,
            'rushing_yards' ,
            'rushing_yards_per_attempt' ,
            'rushing_touchdowns' ,
            'standard_points' ,
            'half_ppr_points' ,
            'ppr_points' ]
df_data_weekly = pd.DataFrame(data_weekly,columns = cols)
df_data_weekly['year_shift']= df_data_weekly['week'].shift(1)
df_data_weekly['new_year'] = 1 * (df_data_weekly['year_shift'] != df_data_weekly['week']) * (df_data_weekly['week'] == 1)
df_data_weekly['what_year'] = df_data_weekly['new_year'].cumsum() + 2001

#In[]:
if conn:
        # using close() method, we will close 
        # the connection
        conn.close()
#In[]:
import tqdm
years =  df_data_weekly['what_year'].unique()
all_players =  df_data_weekly['player_name'].unique()
each_player_by_year = {}
for x in tqdm.tqdm(years):
    print(x)

#In[]:
for y in tqdm.tqdm(years):
      for player in all_players:
            #print(y,player)
            temp_df = df_data_weekly[df_data_weekly['what_year'] == y]

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



#In[]:

COLUMNS_NEEDED = range(0,WEEK_OF_THE_SEASON - 1)

modeling_output_df_int_cluster  = output_df_int_cluster[COLUMNS_NEEDED]

from sklearn.neighbors import NearestNeighbors
neigh = NearestNeighbors(n_neighbors=5)
neigh.fit(modeling_output_df_int_cluster)
A = neigh.kneighbors_graph(modeling_output_df_int_cluster)
B =A.toarray()
#In[]:
#from sklearn.neighbors import NearestNeighbors
import numpy as np
import matplotlib.pyplot as plt

new_df = modeling_output_df_int_cluster.reset_index()
future_df = output_df_int_cluster.reset_index()


"""This section to be removed for real world application"""
TEMP_PROVE_POINT_COL = range(max(modeling_output_df_int_cluster.columns),max(output_df_int_cluster.columns)+1)
future_output_df_int_cluster  = output_df_int_cluster[TEMP_PROVE_POINT_COL]
temp_df = future_output_df_int_cluster.reset_index()

def findsuccess(id):
    labels = []
    plt.figure(figsize=(15,5))
    index_for_nn = np.where(B[:][id] ==1)[0]
    plt.plot(np.array(modeling_output_df_int_cluster.columns),np.array(new_df.iloc[id])[1:])
    plt.plot(np.array(future_output_df_int_cluster.columns),np.array(temp_df.iloc[id])[1:],'r--')
    labels.append(np.array(new_df.iloc[id])[0])
    labels.append(str(np.array(new_df.iloc[id])[0]) + ' Projected')
    plt.title('Players most like ' + str(np.array(new_df.iloc[id])[0]))
    for item in index_for_nn:
        if item != id: 
            print('Neighbors of id', new_df.iloc[item]['index'])
            plt.scatter(output_df_int_cluster.columns,list(future_df.iloc[item])[1:])
            labels.append(np.array(future_df.iloc[item])[0])
    plt.legend(labels)
findsuccess(4)