#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 20:19:07 2023

@author: alexsanner
"""

import pandas as pd 
import sqlite3
from sklearn import preprocessing
# from scikit-learn import preprocessing
import numpy as np
from sklearn.cluster import KMeans
# from scikit-learn.cluster import KMeans
import streamlit as st
from sklearn.decomposition import PCA
# from scikit-learn.decomposition import PCA
import plotly.express as px
from plotly.offline import plot
import plotly.graph_objects as go

import matplotlib.pyplot as plt
import warnings

import re
warnings.simplefilter(action='ignore', category=FutureWarning)


class Query(object):
    def __init__(self):
        #db = st.secrets["db"]
        db = 'ff_momentum.db'
        self.conn = sqlite3.connect(db)
        sql_query = """SELECT name FROM sqlite_master 
            WHERE type='table';"""
            
        self.cursor = self.conn.cursor()
         
        self.cursor.execute(sql_query)
        self.tables = self.cursor.fetchall()

    def relevant_stats(self,position):
        madden_query = '''
        select * From ''' + "madden_weekly"
        
        madden = pd.read_sql_query(madden_query, self.conn)
        madden['year'] = madden['year'].apply(lambda x: (int(x)-1)).astype(str)
        
        hist = '''select * from historical_weekly_stats'''            
        
        historical = pd.read_sql_query(hist,self.conn)
        historical = historical.reindex(sorted(historical.columns), axis=1)
        
        week = '''select * from weekly_stats'''
        
        weekly = pd.read_sql_query(week,self.conn)

        unmodified_weekly = weekly
        
        
        year = ['2023']*len(weekly)
        
        weekly['year'] = year
        weekly = weekly.reindex(sorted(weekly.columns), axis=1)
        
        
        new = pd.concat([weekly,historical]).reset_index(drop=True)
        new['week'] = new.week.astype(str)
        
        df = new.merge(madden, left_on=['year','week', 'player_name'], right_on = ['year','week','fullNameForSearch'])
        df = df.sort_values(by=['year','week','player_name']).reset_index(drop=True)
        
        if position == 'FLX':
            df = df[df['player_position'].isin(['WR','RB','TE']) & (df.year == '2023')]
        else:
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
        
        return df, clean, unmodified_weekly
    

    def cusum(self, weekly_stats):
        weekly_stats = weekly_stats.drop_duplicates(subset=['player_name', 'week'], keep='first').reset_index(drop=True)
        weekly_stats['pts_cumsum'] = weekly_stats.groupby('player_name')['half_ppr_points'].cumsum()
        weekly_stats['activity'] = weekly_stats['targets']+weekly_stats['rushing_attempts']
        weekly_stats['is_active'] = np.where((weekly_stats['activity']>0) | (weekly_stats['half_ppr_points']>2), 1, 0)
        weekly_stats['weeks_active'] = weekly_stats.groupby('player_name')['is_active'].cumsum()



        return weekly_stats

    
    def historical_cluster_stats(self,CURRENT_FOOTBALL_YEAR):
        #db = st.secrets["db"]
        db = 'ff_momentum.db'
        self.conn = sqlite3.connect(db)
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

        modeling_output_df_int_cluster.columns = [str(x) for x in modeling_output_df_int_cluster.columns]

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
                     right_on = ['player_name']).drop_duplicates(subset=['Name']).reset_index(drop=True)
        
        
        
        return df_new
        
    def plot_cluster(self, df):

        if len(df) > 50:
            title = 'Database of Players Clustered Using Principle Component Analysis'
            df['Cluster'] = df['Cluster'].astype(str)

            #fig = px.scatter_3d(df_new,x="Feature 1",y="Feature 2", z='Feature 3', color ='Cluster', text="Name", title="")
            
            fig1 = px.scatter(df,x="Feature 1",y="Feature 2", color ='Cluster', text="Name", title=title,
                            color_discrete_map={'1': 'red', '0': 'blue', '2': 'white'})

            fig1.update_traces(textposition='top center', textfont_size=12)
            
            return fig1

        else:
            title = 'Players Most Similar to {}'.format(df.player_name[0])


            df['Cluster'] = df['Cluster'].astype(str)

            df['Cluster'][0] = df.player_name[0]

            #fig = px.scatter_3d(df_new,x="Feature 1",y="Feature 2", z='Feature 3', color ='Cluster', text="Name", title="")
            
            fig1 = px.scatter(df,x="Feature 1",y="Feature 2", color ='Cluster', text="Name", title=title,
                            color_discrete_map={'1': 'red', '0': 'blue', '2': 'white', df.player_name[0]:'limegreen'})

            fig1.update_traces(textposition='top center', textfont_size=12)
            
            return fig1
    
    def closest_points(self, df, POI, neighbors):
    
        x = df[df['player_name'] == POI]['Feature 1'].reset_index(drop=True).tolist()
        y = df[df['player_name'] == POI]['Feature 2'].reset_index(drop=True).tolist()
        z = df[df['player_name'] == POI]['Feature 3'].reset_index(drop=True).tolist()
        
        df['distance'] = df.apply(lambda row: np.sqrt( (row['Feature 1'] - x[0])**2 + (row['Feature 2'] - y[0])**2 + (row['Feature 3'] - z[0])**2 ), axis=1)
        
        df = df.sort_values(by=['distance']).reset_index(drop=True)
        
        
        return df[0:neighbors+1].reset_index(drop=True)

    def id_from_player_name(self,players,piv_data,WEEK_OF_THE_SEASON):
        COLUMNS_NEEDED = range(1,WEEK_OF_THE_SEASON )
        modeling_output_df_int_cluster  = piv_data[COLUMNS_NEEDED]
        new_df = modeling_output_df_int_cluster.reset_index('index')
        for item in modeling_output_df_int_cluster.index:
            person_year = players + " " + str(2023)
            if re.findall(person_year,item):
                #print(item)
                id_name = new_df[new_df['index']== item ].index[0]   
                #print(id_name)


                    
        return id_name
if __name__ == '__main__':
    
    st.title('Fantasy Football Player Sit/Start Analyzer')
    q = Query()


    option = st.selectbox(
    'Select position',
    ('WR', 'RB', 'TE', 'FLX'))

    st.write('You selected:', option)
    position_select = option

        ##Alex
 #   if 'player_position' not in st.session_state:
 #       st.session_state['player_position'] = 1 
    
        
  #  position = st.selectbox('Position', ('WR', 'RB', 'TE'), index = st.session_state.player_position)
 #   print(position)
    
    df,clean, weekly_stats = q.relevant_stats(option)

    weekly_stats = q.cusum(weekly_stats)

    def position_tables(df=weekly_stats, pos=position_select):
        pos_up = pos.upper()
        if pos_up == 'FLX':
            pos_table = df[df['player_position'].isin(['WR','RB','TE'])].copy()
        else:
            pos_table = df[df['player_position']==pos_up].copy()
            # pos_table = df.loc[df['player_position']=='RB'].copy()
        pos_table['weekly_rank'] = pos_table.groupby('week')['pts_cumsum'].rank('dense', ascending=False)
        pos_table.sort_values(['weekly_rank'], ascending=False).groupby('week')
        return pos_table # RETURNS df of formatted player data for selected position

    # Set up bubble tables

    bubble_range = {
        'wr': {'wr_max':40, 'wr_min':20},
        'rb':{'rb_max':40, 'rb_min':20},
        'te':{'te_max':15, 'te_min':6},
        'flx':{'flx_max':60, 'flx_min':40}
        }

    def bubble_tables(df):
        '''RETURNS table of bubble avg points per week and stdev of bubble player points'''
        if len(df['player_position'].unique()) == 1:
            pos = df['player_position'].unique()[0].lower()
        else:
            pos = 'flx'
        bubbles = df.loc[(df['weekly_rank']<=bubble_range[pos][pos+'_max']) & (df['weekly_rank']>=bubble_range[pos][pos+'_min'])]
        bubble_std = bubbles['half_ppr_points'].std()
        bubble_avg_pos = bubbles.groupby('week')['pts_cumsum'].mean().reset_index()
        bubble_avg_pos.rename(columns={'pts_cumsum': 'bubble_avg'}, inplace=True)
        bubble_avg_pos['bubble_avg'] = bubble_avg_pos['bubble_avg']+bubble_std
        return bubble_avg_pos, bubble_std

    # Join bubble info to main table and calculate breakout trigger

    def breakouts(position_table, bubble_table, bubble_std):
        '''
        INPUTS: output of "position_tables", table from "bubble_tables", std from "bubble_tables" 
        OUTPUT: Table with breakout info added
        '''
        breakout_check = pd.merge(position_table, bubble_table, left_on='weeks_active', right_on='week', how='left')
        breakout_check['cumsum_diff'] = breakout_check['pts_cumsum']-breakout_check['bubble_avg']
        breakout_check['breakout'] = np.where(breakout_check['cumsum_diff']>0, 1, 0)
        return breakout_check

    # RUN THROUGH

    position_table = position_tables()
    bubble_avg, bubble_std = bubble_tables(position_table)
    breakout_table = breakouts(position_table=position_table, bubble_table=bubble_avg, bubble_std=bubble_std)
    #print(len(breakout_table))
    breakout_table.head(20)



    df = breakout_table


    # Highlight Player

    players = sorted(list(df['player_name'].unique()))

    player_selection = st.selectbox(
        'Select player to spotlight',
        players,
        index=None)

    st.write('You selected:', player_selection)

    st.markdown('''  
        ### Fantasy Football Cumulative Data by Weeks Played
        Players marked with :green[green] bubbles have surpassed the breakout threshold.  
        Players with :gray[grey] bubbles are under the breakout threshold.   
    ''')



    # VIZ


    # Preprocess data to create line dash styles
    df['line_dash'] = df['breakout'].apply(lambda x: 'solid' if x == 1 else 'dash')

    # Create a figure
    fig = go.Figure()

    sorted_player_names = sorted(df['player_name'].unique())

    # Group by player_name and add traces
    for player in sorted_player_names:
        player_data = df[df['player_name'] == player]
        marker_fill = ['green' if breakout == 1 else 'rgba(178, 178, 178, 0.8)' for breakout in player_data['breakout']]
        marker_size = [8 if breakout == 1 else 4 for breakout in player_data['breakout']]
        fig.add_trace(go.Scatter(x=player_data['weeks_active'], y=player_data['pts_cumsum'],
                                mode='lines+markers', name=player, line=dict(dash='solid', width=0.8),
                                marker=dict(size=marker_size, symbol='circle', line=dict(color='white', width=1),
                                            opacity=1, line_color='black', color=marker_fill),
                                text=[player_data['player_name']]))

    # Add average line
    fig.add_trace(go.Scatter(x=bubble_avg['week'], y=bubble_avg['bubble_avg'], line_dash='dot',
                            marker=dict(color='grey', size=8, symbol='circle'),
                            mode='lines', line=dict(color='yellow', width=5), name='Threshold'))

    # Update if player selected for highlighting
    if player_selection:
        player_select_data = df[df['player_name'] == player_selection]
        selected_marker_fill = ['green' if breakout == 1 else 'rgba(0, 0, 0, 0.2)' for breakout in player_select_data['breakout']]
        selected_marker_size = [12 if breakout == 1 else 8 for breakout in player_select_data['breakout']]
        fig.add_trace(go.Scatter(x=player_select_data['weeks_active'], y=player_select_data['pts_cumsum'],
                                    mode='lines+markers', name=player_selection, line=dict(dash='solid', width=4, color='white'),
                                    marker=dict(size=selected_marker_size, symbol='circle', line=dict(color='white', width=1),
                                                opacity=1, line_color='white', color=selected_marker_fill)))
        
        # Add green or red box annotation based on the breakout value of the last data point
        last_point_breakout = player_select_data['breakout'].iloc[-1]
        last_point_week = player_select_data['weeks_active'].iloc[-1]
        box_color = 'green' if last_point_breakout == 1 else 'red'
        box_text = 'START' if last_point_breakout == 1 else 'SIT'
        fig.add_shape(
            type='rect',
            label=dict(
                text=f'{player_selection}<br>{box_text}',
                font=dict(
                    size=38,
                    family='Arial'
                ),
                textposition='middle center'
            ),
            x0=-0.4,
            x1=max(df['weeks_active'])*3/5,
            y0=max(df['pts_cumsum']),
            y1=max(df['pts_cumsum'])*3/4,
            fillcolor=box_color,
            opacity=1,
            layer='above',
        )

    # Customize layout
    fig.update_layout(xaxis_title='Weeks Played', yaxis_title='Cumulative Points Scored',
                    legend_title='Player Name', height=1000, width=1000)

    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=20),
    )

    # Display the chart using Streamlit
    st.plotly_chart(fig)

    # def update(player):
    #     df_new, figure = q.cluster(df[df.player_name == player], clean)
    #     return df_new, figure
    
    #players = st.selectbox('Available Players', set(df.player_name.unique().tolist()))
                          #index = None,
                          #on_change = update(index))

    # print(players)
    # id_name = q.id_from_player_name(players,clean_hist_df,WEEK_OF_THE_SEASON)
    
    # st.header('Previous Years Understanding for ' + str(players) + ' in week ' + str(WEEK_OF_THE_SEASON))

    # histo_hist,plot_figure = q.prevyearcompare(hist_array,id_name,WEEK_OF_THE_SEASON,clean_hist_df)
    # st.pyplot(plot_figure)
    
    # hist_figure = plt.figure(figsize=(10,10))
    # plt.title('Most Likely Outcomes for ' + str(players) + 'after week ' +str(WEEK_OF_THE_SEASON))
    # plt.xlabel('half ppr points scored')
    # plt.ylabel('Expected chance of score')
    # plt.hist(list(histo_hist))
    # st.pyplot(hist_figure)
    
    if player_selection == None:
        df_new = q.cluster(df,clean)
        figure = q.plot_cluster(df_new)
        st.plotly_chart(figure)
    else:
        WEEK_OF_THE_SEASON = st.selectbox("Current Week Number", sorted(list(weekly_stats.week.unique())), index=max(weekly_stats.week)-1)
        st.write('The current week Number is ', WEEK_OF_THE_SEASON)

        
        

        
        # df_new, figure = q.cluster(df,clean)
        df_data_weekly = q.historical_cluster_stats(WEEK_OF_THE_SEASON)
        
        
        try:
            clean_hist_df = q.clean_historical_cluster(df_data_weekly)
            hist_array   = q.cluster_historical(clean_hist_df,WEEK_OF_THE_SEASON)
            max_week_of_ = df_data_weekly[df_data_weekly['year'] == 2023]
            id_name = q.id_from_player_name(player_selection,clean_hist_df,WEEK_OF_THE_SEASON)
        
            st.header('Previous Years Understanding for ' + str(player_selection) + ' in week ' + str(WEEK_OF_THE_SEASON))
    
            histo_hist,plot_figure = q.prevyearcompare(hist_array,id_name,WEEK_OF_THE_SEASON,clean_hist_df)
            st.pyplot(plot_figure)
            
            hist_figure = plt.figure(figsize=(10,10))
            plt.title('Most Likely Outcomes for ' + str(player_selection) + ' after week ' +str(WEEK_OF_THE_SEASON))
            plt.xlabel('half ppr points scored')
            plt.ylabel('Expected chance of score')
            plt.hist(list(histo_hist))
            st.pyplot(hist_figure)
        except:
            st.text(f'Not enough data for {player_selection}')
        try:
            if 'neighbor' not in st.session_state:
                st.session_state['neighbor'] = 10

            neighbor_dropdown = st.selectbox('Number of Similar Players', np.arange(50),
                        index = st.session_state.neighbor)
            df_new = q.cluster(df,clean)
            close = q.closest_points(df_new,player_selection,10)
                    
            figure = q.plot_cluster(close)
            st.plotly_chart(figure)
        except:
            pass
