# -*- coding: utf-8 -*-
"""
Created on Sun May 17 09:27:32 2020

@author: jalbert
"""

import dash
import dash_html_components as html
import dash_core_components as dcc
import webbrowser
from threading import Timer

app = dash.Dash(__name__)

app.layout = html.Div(children=[
    html.H1(children='Hello Dash'),

    html.Div(children='''
        Dash: A web application framework for Python.
    '''),

    dcc.Graph(
        id='example-graph',
        figure={
            'data': [
                {'x': [1, 2, 3], 'y': [4, 1, 2], 'type': 'bar', 'name': 'SF'},
                {'x': [1, 2, 3], 'y': [2, 4, 5], 'type': 'bar', 'name': u'Montréal'},
            ],
            'layout': {
                'title': 'Dash Data Visualization'
            }
        }
    )
])

def open_browser():
    webbrowser.open_new('http://127.0.0.1:5000/')

if '__name__'=='__main__':
    Timer(1, open_browser).start()
    app.run_server(debug=True, port=5000)