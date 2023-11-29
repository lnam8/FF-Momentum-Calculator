import sqlite3
import pandas as pd
import requests
import streamlit as st
import altair as alt
from bs4 import BeautifulSoup

def get_conn():
    return sqlite3.connect('/Users/rahim/School/CSE6242/FF-Momentum-Calculator/db/ff_momentum.db')

def get_projections(conn, week, position):
    query = "select * from weekly_projections where week = {} and player_position = '{}'".format(week, position)
    data = pd.read_sql_query(query, conn)
    data = data[['player_name', 'player_position', 'week', 'data_source', 'half_ppr_projected_points']].drop_duplicates().reset_index()
    data['player_avg'] = data.groupby('player_name')['half_ppr_projected_points'].transform('mean').round(2)
    return data

def get_stats(conn, week, position):
    query = "select * from weekly_stats where week = {} and player_position = '{}'".format(week, position)
    data = pd.read_sql_query(query, conn)
    data = data[['player_name', 'player_position', 'week', 'data_source', 'half_ppr_projected_points']].drop_duplicates()
    data['player_avg'] = data.groupby('player_name')['half_ppr_projected_points'].transform('mean').round(2)
    return data

def get_position_code(pos):
    mapp = {'Wide Receiver': ('WR', ('Puka Nacua', 'Calvin Ridley')),
            'Running Back': ('RB', ('Josh Jacobs', 'Raheem Mostert')),
            'Tight End': ('TE', ('Sam LaPorta', 'Zach Ertz'))
            }
    return mapp[pos]

def player_highlighting(data, player_name, chart):
    pdf = data[data['player_name'] == player_name]
    yahoo_url = get_yahoo_url(conn, player_option)
    image_url = get_profile_image(yahoo_url)
    st.image(image_url)
    return alt.Chart(pdf).mark_circle().encode(
        x='index',
        y='half_ppr_projected_points',
        color=alt.value("#FFAA00"),
        tooltip=['player_name', 'half_ppr_projected_points', 'data_source']
        )

def get_yahoo_url(conn, player_name):
    query = "select player_id from weekly_projections where player_name = '{}' and data_source = 'Sleeper' limit 1".format(player_name)
    data = pd.read_sql_query(query, conn)
    player_id = data.iloc[0]['player_id']
    sleeper_url = 'https://api.sleeper.app/v1/players/nfl'
    r = requests.get(sleeper_url)
    yahoo_id = r.json()[str(player_id)]['yahoo_id']
    r.close()
    yahoo_url = 'https://sports.yahoo.com/nfl/players/{}/'.format(yahoo_id)
    return yahoo_url

def get_profile_image(url):
    r = requests.get(url)
    html = r.text
    r.close()
    soup = BeautifulSoup(html, 'html.parser')
    img_element = soup.find('img', {'class': 'Pos(a) B(0) M(a) H(100%) T(10%) H(90%)! Start(-15%)'})
    img_url = img_element['src']
    return img_url

if __name__ == "__main__":

    conn = get_conn()

    st.write('# Highlighted Player Projections vs. Overall Averages')
    position_option = st.selectbox('Pick a position group:', ('Running Back', 'Wide Receiver', 'Tight End'))
    position, player_names = get_position_code(position_option)
    week_option = st.selectbox('Pick a week:', (1, 2, 3, 4, 5, 6))
    player_option = st.selectbox('Select a player to highlight:', (player_names))

    proj = get_projections(conn, week_option, position)
    a = proj[['player_name', 'player_avg']].drop_duplicates().reset_index()
    sc = alt.Chart(a).mark_circle().encode(
        x='index',
        y='player_avg',
        tooltip=['player_name', 'player_avg']
        )
    ph = player_highlighting(proj, player_option, sc)
    c = alt.layer(sc, ph).encode(alt.Y(title='Players vs. Player Average'), alt.X(title=''))
    st.altair_chart(c)
    