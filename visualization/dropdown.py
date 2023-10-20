# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 21:26:15 2023

@author: asann
"""

import dash
import dash_core_components as dcc
import dash_html_components as html

import pandas as pd

df = pd.read_csv('weekly_data.csv')

def generate_table(dataframe, max_rows=10):
    return html.Table(
        # Header
        [html.Tr([html.Th(col) for col in dataframe.columns])] +

        # Body
        [html.Tr([
            html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
        ]) for i in range(min(len(dataframe), len(dataframe)))]
    )

app = dash.Dash()

app.layout = html.Div(children=[
    html.H4(children='Fantasy Football'),
    dcc.Dropdown(id='dropdown', options=[
        {'label': i, 'value': i} for i in df.player_name.unique()
    ], multi=True, placeholder='Filter by Player Name...'),
    html.Div(id='table-container')
])

@app.callback(
    dash.dependencies.Output('table-container', 'children'),
    [dash.dependencies.Input('dropdown', 'value')])
def display_table(dropdown_value):
    if dropdown_value is None:
        return generate_table(df)

    dff = df[df.player_name.str.contains('|'.join(dropdown_value))]
    return generate_table(dff)

app.css.append_css({"external_url": "https://codepen.io/chriddyp/pen/bWLwgP.css"})

if __name__ == '__main__':
    app.run_server(debug=True)