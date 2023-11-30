import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
import streamlit as st
# from streamlit_echarts import st_echarts
import plotly.express as px
import plotly.graph_objects as go
# import os

# Set up database connections
# CHANGE FOR STREAMLIT MODEL
con = sqlite3.connect('/Users/ericitokazu/Documents/GT OMSA/CSE 6242/FF-Momentum-Calculator-main/db/ff_momentum.db')

cur = con.cursor()

table_fetch = cur.execute("SELECT name FROM sqlite_schema WHERE type='table'").fetchall()

# tables = [i[0] for i in table_fetch]

# ---- Construct base tables ----

hist_weekly = pd.read_sql("SELECT * FROM historical_weekly_stats", con=con)

weekly_stats = pd.read_sql("SELECT * FROM weekly_stats", con=con)

weekly_stats = weekly_stats.drop_duplicates(subset=['player_name', 'week'], keep='first').reset_index(drop=True)
weekly_stats['pts_cumsum'] = weekly_stats.groupby('player_name')['half_ppr_points'].cumsum()
weekly_stats['activity'] = weekly_stats['targets']+weekly_stats['rushing_attempts']
weekly_stats['is_active'] = np.where(weekly_stats['activity']>0, 1, 0)
weekly_stats['weeks_active'] = weekly_stats.groupby('player_name')['is_active'].cumsum()

# select position - make selectable on streamlit

option = st.selectbox(
    'Select position',
    ('WR', 'RB', 'TE', 'FLX'))

st.write('You selected:', option)

position_select = option
# position_select = 'wr'


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
print(len(breakout_table))
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
    Players with :gray[empty] bubbles are under the breakout threshold.   
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
    marker_fill = ['green' if breakout == 1 else 'rgba(0, 0, 0, 0.2)' for breakout in player_data['breakout']]
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

