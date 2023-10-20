#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 20:20:33 2023

@author: alexsanner
"""

from dash import Dash, html, dcc, Input, Output, callback
import pandas as pd
import plotly.express as px

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__, external_stylesheets=external_stylesheets)


ff = pd.read_csv('weekly_data.csv')


ff = ff.assign(standard_points=lambda x: (x.receiving_yards*0.1)\
                                 + (x.rushing_yards*0.1) + (x.receiving_touchdowns*6)\
                                 + (x.rushing_touchdowns*6))

ff = ff.assign(ppr_points=lambda x: (x.receiving_yards*0.1) + (x.receptions*1)\
                                 + (x.rushing_yards*0.1) + (x.receiving_touchdowns*6)\
                                 + (x.rushing_touchdowns*6))
    
#%%
ff = ff.drop_duplicates(subset=['player_name', 'week']).reset_index(drop=True)

df1 = ff[['player_name', 'player_position', 'player_team', 'week', 'receptions',
       'targets', 'receiving_yards', 'receiving_yards_per_reception',
       'receiving_touchdowns', 'rushing_attempts', 'rushing_yards',
       'rushing_yards_per_attempt', 'rushing_touchdowns', 'standard_points']]

df1.columns = ['player_name', 'player_position', 'player_team', 'week', 'receptions',
       'targets', 'receiving_yards', 'receiving_yards_per_reception',
       'receiving_touchdowns', 'rushing_attempts', 'rushing_yards',
       'rushing_yards_per_attempt', 'rushing_touchdowns',
       'total_points']

df1['point_system'] = ['standard']*len(df1)



df2 = ff[['player_name', 'player_position', 'player_team', 'week', 'receptions',
       'targets', 'receiving_yards', 'receiving_yards_per_reception',
       'receiving_touchdowns', 'rushing_attempts', 'rushing_yards',
       'rushing_yards_per_attempt', 'rushing_touchdowns', 'ppr_points']]

df2.columns = ['player_name', 'player_position', 'player_team', 'week', 'receptions',
       'targets', 'receiving_yards', 'receiving_yards_per_reception',
       'receiving_touchdowns', 'rushing_attempts', 'rushing_yards',
       'rushing_yards_per_attempt', 'rushing_touchdowns',
       'total_points']

df2['point_system'] = ['ppr']*len(df2)


df3 = ff[['player_name', 'player_position', 'player_team', 'week', 'receptions',
       'targets', 'receiving_yards', 'receiving_yards_per_reception',
       'receiving_touchdowns', 'rushing_attempts', 'rushing_yards',
       'rushing_yards_per_attempt', 'rushing_touchdowns',
       'half_ppr_points']]

df3.columns = ['player_name', 'player_position', 'player_team', 'week', 'receptions',
       'targets', 'receiving_yards', 'receiving_yards_per_reception',
       'receiving_touchdowns', 'rushing_attempts', 'rushing_yards',
       'rushing_yards_per_attempt', 'rushing_touchdowns',
       'total_points']

df3['point_system'] = ['half']*len(df3)

df = pd.concat([df1,df2,df3], axis=0)

df = df[['player_name', 'point_system', 'week', 'total_points']]

df.columns = ['Country Name', 'Indicator Name', 'Week', 'Value']
#%%
app.layout = html.Div([
    html.Div([

        html.Div([
            dcc.Dropdown(
                df['Indicator Name'].unique(),
                'half',
                id='crossfilter-xaxis-column',
            ),
            dcc.RadioItems(
                ['Linear', 'Log'],
                'Linear',
                id='crossfilter-xaxis-type',
                labelStyle={'display': 'inline-block', 'marginTop': '5px'}
            )
        ],
        style={'width': '49%', 'display': 'inline-block'}),

        html.Div([
            dcc.Dropdown(
                df['Indicator Name'].unique(),
                'standard',
                id='crossfilter-yaxis-column'
            ),
            dcc.RadioItems(
                ['Linear', 'Log'],
                'Linear',
                id='crossfilter-yaxis-type',
                labelStyle={'display': 'inline-block', 'marginTop': '5px'}
            )
        ], style={'width': '49%', 'float': 'right', 'display': 'inline-block'})
    ], style={
        'padding': '10px 5px'
    }),

    html.Div([
        dcc.Graph(
            id='crossfilter-indicator-scatter',
            hoverData={'points': [{'customdata': 'Puka Nacua'}]}
        )
    ], style={'width': '49%', 'display': 'inline-block', 'padding': '0 20'}),
    html.Div([
        dcc.Graph(id='x-time-series'),
        dcc.Graph(id='y-time-series'),
    ], style={'display': 'inline-block', 'width': '49%'}),

    html.Div(dcc.Slider(
        df['Week'].min(),
        df['Week'].max(),
        step=None,
        id='crossfilter-year--slider',
        value=df['Week'].max(),
        marks={str(year): str(year) for year in df['Week'].unique()}
    ), style={'width': '49%', 'padding': '0px 20px 20px 20px'})
])


@callback(
    Output('crossfilter-indicator-scatter', 'figure'),
    Input('crossfilter-xaxis-column', 'value'),
    Input('crossfilter-yaxis-column', 'value'),
    Input('crossfilter-xaxis-type', 'value'),
    Input('crossfilter-yaxis-type', 'value'),
    Input('crossfilter-year--slider', 'value'))

def update_graph(xaxis_column_name, yaxis_column_name,
                 xaxis_type, yaxis_type,
                 year_value):
    dff = df[df['Week'] == year_value]

    fig = px.scatter(x=dff[dff['Indicator Name'] == xaxis_column_name]['Value'],
            y=dff[dff['Indicator Name'] == yaxis_column_name]['Value'],
            hover_name=dff[dff['Indicator Name'] == yaxis_column_name]['Country Name']
            )

    fig.update_traces(customdata=dff[dff['Indicator Name'] == yaxis_column_name]['Country Name'])

    fig.update_xaxes(title=xaxis_column_name, type='linear' if xaxis_type == 'Linear' else 'log')

    fig.update_yaxes(title=yaxis_column_name, type='linear' if yaxis_type == 'Linear' else 'log')

    fig.update_layout(margin={'l': 40, 'b': 40, 't': 10, 'r': 0}, hovermode='closest')

    return fig


def create_time_series(dff, axis_type, title):

    fig = px.scatter(dff, x='Week', y='Value')

    fig.update_traces(mode='lines+markers')

    fig.update_xaxes(showgrid=False)

    fig.update_yaxes(type='linear' if axis_type == 'Linear' else 'log')

    fig.add_annotation(x=0, y=0.85, xanchor='left', yanchor='bottom',
                       xref='paper', yref='paper', showarrow=False, align='left',
                       text=title)

    fig.update_layout(height=225, margin={'l': 20, 'b': 30, 'r': 10, 't': 10})

    return fig


@callback(
    Output('x-time-series', 'figure'),
    Input('crossfilter-indicator-scatter', 'hoverData'),
    Input('crossfilter-xaxis-column', 'value'),
    Input('crossfilter-xaxis-type', 'value'))
def update_x_timeseries(hoverData, xaxis_column_name, axis_type):
    country_name = hoverData['points'][0]['customdata']
    dff = df[df['Country Name'] == country_name]
    dff = dff[dff['Indicator Name'] == xaxis_column_name]
    title = '<b>{}</b><br>{}'.format(country_name, xaxis_column_name)
    return create_time_series(dff, axis_type, title)


@callback(
    Output('y-time-series', 'figure'),
    Input('crossfilter-indicator-scatter', 'hoverData'),
    Input('crossfilter-yaxis-column', 'value'),
    Input('crossfilter-yaxis-type', 'value'))
def update_y_timeseries(hoverData, yaxis_column_name, axis_type):
    dff = df[df['Country Name'] == hoverData['points'][0]['customdata']]
    dff = dff[dff['Indicator Name'] == yaxis_column_name]
    return create_time_series(dff, axis_type, yaxis_column_name)


if __name__ == '__main__':
    app.run(debug=True)
