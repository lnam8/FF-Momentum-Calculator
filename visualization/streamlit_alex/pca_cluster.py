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
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)



class Query(object):
    def __init__(self):
        self.conn = sqlite3.connect("../../db/ff_momentum.db")
        sql_query = """SELECT name FROM sqlite_master 
            WHERE type='table';"""
            
        self.cursor = self.conn.cursor()
         
        self.cursor.execute(sql_query)
        self.tables = self.cursor.fetchall()

    def relevant_stats(self, position):
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
        
        df = df[(df.player_position == position) & (df.year == '2023')]
        
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
                     right_on = ['player_name']).drop_duplicates(subset=['Name']).reset_index(drop=True)
        
        
        return df_new
        
        
    def plot_cluster(self, df):
        #fig = px.scatter_3d(df_new,x="Feature 1",y="Feature 2", z='Feature 3', color ='Cluster', text="Name", title="")
        fig = px.scatter(df,x="Feature 1",y="Feature 2", color ='Cluster', text="Name", title="")
        fig.update_traces(textposition='top center')
        
        return fig
    
    
    def closest_points(self, df, POI, neighbors):
        #POI = 'Stefon Diggs'
        
        x = df[df['player_name'] == POI]['Feature 1'].reset_index(drop=True).tolist()
        y = df[df['player_name'] == POI]['Feature 2'].reset_index(drop=True).tolist()
        z = df[df['player_name'] == POI]['Feature 3'].reset_index(drop=True).tolist()
        
        df['distance'] = df.apply(lambda row: np.sqrt( (row['Feature 1'] - x[0])**2 + (row['Feature 2'] - y[0])**2 + (row['Feature 3'] - z[0])**2 ), axis=1)
        
        df = df.sort_values(by=['distance']).reset_index(drop=True)
        
        
        return df[0:neighbors+1].reset_index(drop=True)
        
        
if __name__ == '__main__':
    
    q = Query()

    
    if 'player_position' not in st.session_state:
        st.session_state['player_position'] = None
    
        
    position = st.selectbox('Position', ('WR', 'RB', 'TE'), index = st.session_state.player_position)
    
    if position == None:
        df, clean = q.relevant_stats('WR')
    else:
        df,clean = q.relevant_stats(position)
        
 
    
    if 'selection' not in st.session_state:
        st.session_state['selection'] = None
    
    player = st.selectbox('Available Players', df.player_name.unique().tolist(),
                           index = st.session_state.selection)
    
    
    
    
    
    
    
    
    if player == None:
        df_new = q.cluster(df,clean)
        figure = q.plot_cluster(df_new)
        st.plotly_chart(figure)
    else:
        df_new = q.cluster(df,clean)
        close = q.closest_points(df_new,player,10)
                
        figure = q.plot_cluster(close)
        st.plotly_chart(figure)

 
    