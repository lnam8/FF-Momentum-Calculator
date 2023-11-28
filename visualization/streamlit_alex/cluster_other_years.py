#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 20:19:07 2023

@author: alexsanner
"""

import pandas as pd 
import sqlite3
from sklearn import preprocessing
import numpy as np
from sklearn.cluster import KMeans
import streamlit as st
from sklearn.decomposition import PCA
import plotly.express as px
from plotly.offline import plot
import matplotlib.pyplot as plt
import re
class Query(object):
    def __init__(self):
        self.conn = sqlite3.connect("ff_momentum.db")
        sql_query = """SELECT name FROM sqlite_master 
            WHERE type='table';"""
            
        self.cursor = self.conn.cursor()
         
        self.cursor.execute(sql_query)
        self.tables = self.cursor.fetchall()

    def relevant_stats(self):
        ##JACOBS_DATA 
        weekly_stats_query = """select player_name ,
            player_position,
            player_team ,
            year, 
            week ,
            receptions ,
            targets ,
            
           
            
            half_ppr_points 
              FROM """ +"historical_weekly_stats where year>2017 " 
        
        
        
        ##ALREADY_THERE
        madden_query = '''
        select * From ''' + "madden_weekly"
        
        madden = pd.read_sql_query(madden_query, self.conn)
        madden['year'] = madden['year'].apply(lambda x: (int(x)-1)).astype(str)
        
        hist = '''select * from historical_weekly_stats'''            
        
        historical = pd.read_sql_query(hist,self.conn)
        historical = historical.reindex(sorted(historical.columns), axis=1)
        
        week = '''select * from weekly_stats'''
        
        weekly = pd.read_sql_query(week,self.conn)
        
        
        year = ['2023']*len(weekly)
        
        weekly['year'] = year
        weekly = weekly.reindex(sorted(weekly.columns), axis=1)
        
        
        new = pd.concat([weekly,historical]).reset_index(drop=True)
        new['week'] = new.week.astype(str)
        
        df = new.merge(madden, left_on=['year','week', 'player_name'], right_on = ['year','week','fullNameForSearch'])
        df = df.sort_values(by=['year','week','player_name']).reset_index(drop=True)
        
        df = df[(df.player_position == 'WR') & (df.year == '2023')]
        
        df1 = df.set_index('player_name')
        
        
        df1 = df1.apply(pd.to_numeric, errors='coerce').dropna(axis=1)
        
        clean = df1.drop(['primaryKey', 'year', 'jerseyNum', 'signingBonus', 'totalSalary', 'standard_points',
                          'breakSack_rating','plyrPortrait','kickPower_rating','throwUnderPressure_rating',
                          'passBlockFinesse_rating', 'throwPower_rating',
                          'kickReturn_rating', 'leadBlock_rating', 'bCVision_rating',
                          'playAction_rating','weight', 'yearsPro','throwAccuracyShort_rating',
                          'throwOnTheRun_rating', 'manCoverage_rating', 'stiffArm_rating',
                          'powerMoves_rating', 'passBlockPower_rating', 'impactBlocking_rating','throwAccuracyDeep_rating',
                          'teamId', 'age'], axis =1)
        
        clean = clean.replace(['inf', 'nan', '-inf'], 0)
        clean = clean[np.isfinite(clean).all(1)]
        
        return df, clean
    def historical_cluster_stats(self,CURRENT_FOOTBALL_YEAR):
        self.conn = sqlite3.connect("ff_momentum.db")
        cursor = self.conn.cursor()
        weekly_stats_query = """select player_name ,
            player_position,
            player_team ,
            year, 
            week ,
            receptions ,
            targets ,
            rushing_attempts,
           
            
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
            'rushing_attempts' ,
            # 'rushing_yards' ,
            # 'rushing_yards_per_attempt' ,
            # 'rushing_touchdowns' ,
            
            'half_ppr_points' 
            
            ]
        df_data_weekly = pd.DataFrame(data_weekly,columns = cols)
        cweekly_stats_query = """select DISTINCT 
            player_name ,
            player_position,
            player_team ,
            week, 
            receptions ,
            targets ,
           rushing_attempts,
            
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
                     'rushing_attempts' ,
                    # 'rushing_yards' ,
                    # 'rushing_yards_per_attempt' ,
                    # 'rushing_touchdowns' ,
                    
                    'half_ppr_points' 
                    
                    ]
        cdf_data_weekly = pd.DataFrame(cdata_weekly,columns = cols2)
        cdf_data_weekly['year'] = 2023

        df_data_weekly = pd.concat([df_data_weekly,cdf_data_weekly]).reset_index(drop=True)

        df_data_weekly['join_column'] = df_data_weekly.apply(lambda x:  str(x['player_name'] ) +" " + str(x['year'] ) + " " + str(x['week'] ),axis = 1) 
        return df_data_weekly
    def clean_historical_cluster(self, df_data_weekly):
        piv_data = df_data_weekly.pivot(index= ['year','player_name','player_position'],columns = 'week',values = 'half_ppr_points').fillna(0).reset_index()
        piv_data['index'] = piv_data.apply(lambda x: x['player_name'] + " " + str(x['year']), axis =1)
        piv_data.drop(columns=['year','player_name','player_position'],inplace= True)
        piv_data.set_index('index',inplace=True, drop=True)
        COLUMNS_NEEDED = range(1,WEEK_OF_THE_SEASON )
        piv_data_catches = df_data_weekly.pivot(index= ['year','player_name','player_position'],columns = 'week',values = 'targets').fillna(0).reset_index()
        piv_data_catches['index'] = piv_data_catches.apply(lambda x: x['player_name'] + " " + str(x['year']), axis =1)
        piv_data_catches.drop(columns=['year','player_name','player_position'],inplace= True)
        piv_data_catches.set_index('index',inplace=True, drop=True)
        piv_data_catches['total_catches'] = piv_data_catches[COLUMNS_NEEDED].sum(axis = 1)

        COLUMNS_NEEDED = range(1,WEEK_OF_THE_SEASON )
        piv_data_ra = df_data_weekly.pivot(index= ['year','player_name','player_position'],columns = 'week',values = 'rushing_attempts').fillna(0).reset_index()
        piv_data_ra['index'] = piv_data_ra.apply(lambda x: x['player_name'] + " " + str(x['year']), axis =1)
        piv_data_ra.drop(columns=['year','player_name','player_position'],inplace= True)
        piv_data_ra.set_index('index',inplace=True, drop=True)
        piv_data_ra['total_rushing_attempts'] = piv_data_ra[COLUMNS_NEEDED].sum(axis = 1)



        new_piv_data = piv_data.join(piv_data_catches['total_catches'] ,how = 'left')
        even_newer_piv_data = new_piv_data.join(piv_data_ra['total_rushing_attempts'] ,how = 'left')

       
        good_indexes =  even_newer_piv_data.sum(axis = 1) > 10

        even_newer_piv_data = even_newer_piv_data[good_indexes]
        return even_newer_piv_data
    def cluster_historical(self, piv_data,WEEK_OF_THE_SEASON):
        
        
        
        COLUMNS_NEEDED = range(1,WEEK_OF_THE_SEASON)
        piv_data['sum_total'] = piv_data[COLUMNS_NEEDED].sum(axis = 1)
        piv_data['stddev'] = piv_data[COLUMNS_NEEDED].std(axis = 1)

        list_of_columns = []
        for item in range(1,WEEK_OF_THE_SEASON):
            list_of_columns.append(item)
        
        list_of_columns.append('sum_total')
        list_of_columns.append('stddev')
        list_of_columns.append('total_catches')
        list_of_columns.append('total_rushing_attempts')
        
        modeling_output_df_int_cluster  = piv_data[list_of_columns]

        from sklearn.neighbors import NearestNeighbors
        for i in [1]:
            neigh = NearestNeighbors(n_neighbors=11)
            neigh.fit(modeling_output_df_int_cluster)
            A = neigh.kneighbors_graph(modeling_output_df_int_cluster)
            B =A.toarray()
        return B
    def prevyearcompare(self,B,id,WEEK_OF_THE_SEASON,piv_data):
        ##redundant but currently nessisasy 
        COLUMNS_NEEDED = range(1,WEEK_OF_THE_SEASON )
        PLOT_COLUMNS = range(1,16)
        modeling_output_df_int_cluster  = piv_data[COLUMNS_NEEDED]
        new_df = modeling_output_df_int_cluster.reset_index('index')
        future_df = piv_data[PLOT_COLUMNS].reset_index('index')
        ##Main Code 
        hist_data = []
        labels = []
        plt_figure = plt.figure(figsize=(15,10))
        index_for_nn = np.where(B[:][id] ==1)[0]
        plt.plot(np.array(modeling_output_df_int_cluster.columns),np.array(new_df.iloc[id])[1:])
        #plt.plot(np.array(future_output_df_int_cluster.columns),np.array(temp_df.iloc[id])[1:],'r--')
        labels.append(np.array(new_df.iloc[id])[0])
        #labels.append(str(np.array(new_df.iloc[id])[0]) + ' Projected')
        plt.title('Players most like ' + str(np.array(new_df.iloc[id])[0]))
        for item in index_for_nn:
            if item != id: 
                #print('Neighbors of id', new_df.iloc[item]['index'])
                plt.scatter(piv_data[PLOT_COLUMNS].columns,list(future_df.iloc[item])[1:])

                labels.append(np.array(future_df.iloc[item])[0])
                if sum(list(future_df.iloc[item])[WEEK_OF_THE_SEASON:]) > 5:
                    for s in list(future_df.iloc[item])[1:]:
                        hist_data.append(s)
        plt.legend(labels)
        return hist_data,plt_figure

    def cluster(self,df,clean):
        
    
        min_max_scale = preprocessing.MinMaxScaler()
        scaled = min_max_scale.fit_transform(clean)
        
        scaled = scaled[np.isfinite(scaled).all(1)]
        
        scaled= pd.DataFrame(scaled)
        
        pca = PCA(n_components = 3)
        df_new = pd.DataFrame(pca.fit_transform(scaled))
        #Check how many clusters should we pick that have the least cost function value
        
        
        
        # Vary the Number of Clusters
        min_clust = 1
        max_clust = 40
        
        # Compute Within Cluster Sum of Squares
        within_ss = []
        for num_clust in range(min_clust, max_clust+1):
            km = KMeans(n_clusters = num_clust)
            km.fit(df_new)
            within_ss.append(km.inertia_)
        
        
        # f, axes = plt.subplots(1, 1, figsize=(16,4))
        # plt.plot(range(min_clust, max_clust+1), within_ss)
        # plt.xlabel('Number of Clusters')
        # plt.ylabel('Within Cluster Sum of Squares')
        # plt.xticks(np.arange(min_clust, max_clust+1, 1.0))
        # plt.grid(which='major', axis='y')
        # plt.show()
        
        #Using KMeans Algorithm for machine learning modelling
        km = KMeans(n_clusters = 3) #specify number of clusters
        km = km.fit(df_new) #fit the dataset
        labels = km.predict(df_new) #predict which players belong to a certain cluster
        cluster = km.labels_.tolist() #obtain the cluster number for each players
        
        #Displaying the output of predicted cluster
        df_new.insert(loc = 0, column = 'Name', value = clean.index)
        df_new.insert(loc = 1, column = 'Cluster',value = cluster)
        df_new.columns = ['Name', 'Cluster', 'Feature 1', 'Feature 2', 'Feature 3']
        # df_new.columns = ['Name', 'Cluster', 'Feature 1', 'Feature 2']
        
        
        df_new = df_new.merge(df, 
                     how = 'left', 
                     left_on = ['Name'], 
                     right_on = ['player_name']).drop_duplicates(subset=['Name'])
        
        
        #fig = px.scatter_3d(df_new,x="Feature 1",y="Feature 2", z='Feature 3', color ='Cluster', text="Name", title="")
        fig = px.scatter(df_new,x="Feature 1",y="Feature 2", color ='Cluster', text="Name", title="")
        fig.update_traces(textposition='top center')
        
        return df_new, fig
    
    
    def closest_points(self, df, POI, neighbors):
        
        
        
        return
    def id_from_player_name(self,players,piv_data,WEEK_OF_THE_SEASON):
        COLUMNS_NEEDED = range(1,WEEK_OF_THE_SEASON )
        modeling_output_df_int_cluster  = piv_data[COLUMNS_NEEDED]
        new_df = modeling_output_df_int_cluster.reset_index('index')
        for item in modeling_output_df_int_cluster.index:
            person_year = players + " " + str(2023)
            if re.findall(person_year,item):
                print(item)
                id_name = new_df[new_df['index']== item ].index[0]   
                print(id_name)


                    
        return id_name
if __name__ == '__main__':
    
    q = Query()
    df, clean = q.relevant_stats()
    WEEK_OF_THE_SEASON = st.number_input("Current Week Number", value=3, placeholder=3,step=1)

    st.write('The current week Number is ', WEEK_OF_THE_SEASON)
    df_new, figure = q.cluster(df,clean)
    df_data_weekly = q.historical_cluster_stats(WEEK_OF_THE_SEASON)
    
    
    clean_hist_df = q.clean_historical_cluster(df_data_weekly)
    hist_array   = q.cluster_historical(clean_hist_df,WEEK_OF_THE_SEASON)
    max_week_of_ = df_data_weekly[df_data_weekly['year'] == 2023]
    def update(player):
        df_new, figure = q.cluster(df[df.player_name == player], clean)
        return df_new, figure
    
    players = st.selectbox('Available Players', set(df.player_name.unique().tolist()))
                          #index = None,
                          #on_change = update(index))

    print(players)
    id_name = q.id_from_player_name(players,clean_hist_df,WEEK_OF_THE_SEASON)
    
    st.header('Previous Years Understanding for ' + str(players) + ' in week ' + str(WEEK_OF_THE_SEASON))

    histo_hist,plot_figure = q.prevyearcompare(hist_array,id_name,WEEK_OF_THE_SEASON,clean_hist_df)
    st.pyplot(plot_figure)
    
    hist_figure = plt.figure(figsize=(10,10))
    plt.title('Most Likely Outcomes for ' + str(players) + 'after week ' +str(WEEK_OF_THE_SEASON))
    plt.xlabel('half ppr points scored')
    plt.ylabel('Expected chance of score')
    plt.hist(list(histo_hist))
    st.pyplot(hist_figure)
    
    # st.plotly_chart(figure)
