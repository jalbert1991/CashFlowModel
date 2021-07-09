# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 22:34:16 2020

@author: jalbert
"""

import dash
import dash_bootstrap_components as dbc

#external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
external_stylesheets = [dbc.themes.LUX]
#external_stylesheets = [dbc.themes.BOOTSTRAP]

#app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

#server = app.server
#app.config.suppress_callback_exceptions = True

app = dash.Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=external_stylesheets)
server = app.server

app.config.suppress_callback_exceptions = True
