import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
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
position_select = 'wr'


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
    return bubble_avg_pos, bubble_std

# Join bubble info to main table and calculate breakout trigger

def breakouts(position_table, bubble_table, bubble_std):
    '''
    INPUTS: output of "position_tables", table from "bubble_tables", std from "bubble_tables" 
    OUTPUT: Table with breakout info added
    '''
    # if pos.lower() == 'flx':
        # bubble_col = pos.lower()+'_bubble_avg'
        # bub_std = bubble_table[]
        # subset_table = flx_subsets.copy()
    # breakout_check = pd.merge(position_table, bubble_table, on='week', how='left')
    breakout_check = pd.merge(position_table, bubble_table, left_on='weeks_active', right_on='week', how='left')
    breakout_check['cumsum_diff'] = breakout_check['pts_cumsum']-breakout_check['bubble_avg']
    breakout_check['breakout'] = np.where(breakout_check['cumsum_diff']>bubble_std, 1, 0)
    # else:    
    #     bubble_col = pos.lower()+'_bubble_avg'
    #     bub_pos = bubble_weekly[['week', bubble_col]].copy()
    #     subset_table = flx_subsets.loc[flx_subsets['player_position']==pos.upper()].copy()
    #     breakout_check = pd.merge(subset_table, bub_pos, on='week', how='left')
    #     breakout_check['cumsum_diff'] = breakout_check['pts_cumsum']-breakout_check[bubble_col]
    #     breakout_check['breakout'] = np.where(breakout_check['cumsum_diff']>bub_stds.loc[bub_stds['position']==pos.upper()]['stds'].values[0], 1, 0)
    return breakout_check

# RUN THROUGH

position_table = position_tables()
bubble_avg, bubble_std = bubble_tables(position_table)
breakout_table = breakouts(position_table=position_table, bubble_table=bubble_avg, bubble_std=bubble_std)
print(len(breakout_table))
breakout_table.head(20)

# test = position_table.loc[position_table['player_name']=='Puka Nacua']

# test.head(20)



# VIZ

# pos = None
# while not pos:
#     pos_input = input('Select position (wr, rb, te, flx): ')
#     if pos_input.lower() not in ['wr', 'rb', 'te', 'flx']:
#         pos = None
#     else:
#         pos = pos_input.lower()
df = breakout_table
# bubble_col_chart = pos+'_bubble_avg'
# Group the data by the 'Group' column
grouped = df.groupby('player_name')

# Create a multi-line chart
plt.figure(figsize=(10, 6))

for name, group in grouped:
    # plt.plot(group['weeks_active'], group['pts_cumsum'], linewidth=1, alpha=0.5)
    plt.plot(group['week_x'], group['pts_cumsum'], linewidth=1, alpha=0.5)
plt.plot(bubble_avg['week'], bubble_avg['bubble_avg'], label='Bubble Avg', color='black', linewidth=4, marker='o', markerfacecolor='red', markersize=6)
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Multi-Line Chart by Groups')
plt.legend()
plt.grid(True)
plt.show()



