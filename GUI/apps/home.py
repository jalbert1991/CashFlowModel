# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 09:43:35 2020

@author: jalbert
"""

import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_daq as daq

layout = html.Div([
    dbc.Container([
        dbc.Row([
            dbc.Col(html.H1('Welcome to the FRA Cash Flow Engine', className='text-center'),
                     className='mb-5 mt-5')
            ]),
        #dbc.Row([html.H5('Please select an option below to continue', className='mb-4')
        #    ]),
        #dbc.Row([
        dbc.CardDeck([
            #dbc.Col(
                dbc.Card(children=[
                    html.H4(children='Create New Model', className='text-center'),
                    html.P('Create a brand new model from scratch.', className='card-text'),
                    dbc.Button('New Model',
                                color='dark',
                                href='/apps/create_new_model'
                                ),
                    ], 
                body=True, color='dark', outline=True)
                ,# width=4),
            #dbc.Col(
                dbc.Card(children=[
                    html.H4('Download Existing Model', className='text-center'),
                    html.P('Download a prior model from the database. Can be used to rerun a prior model or continue working on an incomplete model.', className='card-text'),
                    dbc.Button('Load Model',
                                color='dark',
                                href='/apps/build_model'
                        ),
                    ], 
                body=True, color='dark', outline=True)
                , #width=4),
            #dbc.Col(
                dbc.Card(children=[
                    html.H4('Refresh Model', className='text-center'),
                    html.P('Download the configuration for a prior model to rerun with a new cutoff date. This will create a new model with settings predefined.', className='card-text'),
                    dbc.Button('Refresh Model',
                                color='dark',
                                href='/apps/build_model'
                        ),
                    ], 
                body=True, color='dark', outline=True)
                , #width=4),
            ]),
        ])
    ])