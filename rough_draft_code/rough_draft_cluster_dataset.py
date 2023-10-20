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

weekly_stats_query = """select * FROM """ +"historical_weekly_stats LIMIT 3" 
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

years =  df_data_weekly['what_year'].unique()
all_players =  df_data_weekly['player_name'].unique()
each_player_by_year = {}

for y in years:
      for player in all_players:
            print(y,player)
            temp_df = df_data_weekly[df_data_weekly['what_year'] == y]

            try: 
                one_player = temp_df[temp_df['player_name'] == player]
                temp_list = []
                for week_of_season in range(1,WEEK_OF_THE_SEASON):
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
                  print('didnt play this year')





output_df_int_cluster = pd.DataFrame.from_dict(each_player_by_year).T