# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 23:12:07 2020

@author: jalbert
"""

import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_daq as daq

    
config_tabs = html.Div(children=[
        dcc.Tabs(id='tabs', value='rate_curves', children=[
            #dcc.Tab(label='Create Model', value='create_model'),
            dcc.Tab(label='Curves', value='rate_curves'),
            dcc.Tab(label='Config Model', value='model_config'),
            dcc.Tab(label='Run Model', value='model_run'),
            dcc.Tab(label='Results', value='model_eval')
        ]),
        html.Div(id='tabs-content')
    ])


home = html.Div([
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



"""
landing_page = html.Div(
        id='cf_model-create-container',
        className='cf-model-create-body',
        children=[
            html.Div(
                id='cf-model-create',
                className='three columns',
                children=[
                    html.P('Select Options to create a new model or download a pre-existing model'),
                    dcc.RadioItems(
                            options=[
                            {'label':'Create New Model', 'value':'new'},
                            {'label':'Load Model', 'value':'load'}
                            ],
                        value='new',
                        labelStyle={'display':'inline-block'}),
                    ]
                ),
            html.Div(
                id='cf-model-datatape',
                className='three columns',
                children=[
                    html.P('Download Source'),
                    dcc.Tabs(
                        id='datatape',
                        className='datatape-tabs',
                        children=[
                            dcc.Tab(className='datatape-tab', label='CDW', value='cdw'),
                            dcc.Tab(className='datatape-tab', label='Custom SQL', value='sql'),
                            dcc.Tab(className='datatape-tab', label='Excel', value='excel')
                            ]
                        )
                    ])
        #html.H1('Create/Load Models'),
        #html.P('Select Options to create a new model or download a pre-existing model'),
            #dcc.RadioItems(
            #    options=[
            #            {'label':'Create New Model', 'value':'new'},
            #            {'label':'Load Model', 'value':'load'}
            #    ],
            #    value='new',
            #    labelStyle={'display':'inline-block'}),
        #dcc.Tabs(id='create-model-tabs', className='model_tabs_container', value='new', children=[
        #    dcc.Tab(label='New Model', value='new', className='model_tab'),
        #    dcc.Tab(label='Load Model', value='load', className='model_tab'),
        #    dcc.Tab(label='Refresh', value='refresh', className='model_tab')
        #    ]),
        #    html.Div(id='create-model-content') #style={'border-style': 'solid', 'border-color': 'gray'}
    ],
    #style={'display': 'inline-block', 'width': 'flex'}
   )
"""

model_name = dbc.FormGroup([
    #dbc.Label('Model Name', html_for='model_name_form'), #, className='text-center'
    dbc.Col(
        dbc.Input(placeholder='Enter Model Name', type='text', id='model-name'),
        ),
    ], #className='mr-3'
)

model_type = dbc.FormGroup([
    dbc.Label('Model Type'),
    dcc.Dropdown(
        options=[
            {'label':'Template', 'value':1},
            {'label':'Pricing', 'value':2},
            {'label':'RUW', 'value':3},
            ],
        value=3,
        id='model_type_form',
        )
    ])

asset_class = dbc.FormGroup([
    dbc.Label('Asset Class'),
    dcc.Dropdown(
        options=[
            {'label':'Solar', 'value':'solar'},
            {'label':'Home Improvement', 'value':'home improvement'},
            ],
        value='',
        id='asset_class_form'
        )
    ])

deal_ids = dbc.FormGroup([
    dbc.Label('Deal IDs'),
    dcc.Dropdown(
        options=[],
        multi=True,     
        id='deal_ids_form',
        ),
    ])

batch_keys = dbc.FormGroup([
    dbc.Label('Batch Keys'),
    dcc.Dropdown(
        options=[],
        multi=True,     
        id='batch_keys_form',
        ),
    ], )

model_template = dbc.FormGroup([
    dbc.Checkbox(id='download_template_form', className='form-check-input', checked=True),
    dbc.Label('Import Model Template', 
              html_for='download_template_form',
              className='form-check-label',
              ),
        ], 
    )

data_tape_selector = dbc.FormGroup([
    dbc.Label('Select Data Tape Source'),
    dbc.RadioItems(
            options=[
                {"label": "CDW", "value": 1},
                {"label": "Custom SQL", "value": 2},
                {"label": "Load From File", "value": 3},
            ],
            value=1,
            id="datatape-input",
            inline=True,
        ),
    ])

custom_sql_input = dbc.FormGroup([
    dbc.Label('Custom SQL Query'),
    dbc.Textarea(
        cols=6,
        rows=10,
        ),
    ])

upload_sql_input= dbc.FormGroup([
    dbc.Label('Upload Excel File'),
    dcc.Upload(
        id='data-tape-upload',
        children=[html.Div([
            'Drag and Drop or ',
            html.A('Select Files'),
            ]),
            ],
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        multiple=False,
        )
    ])

#create model form

load_model_form = dbc.Row([
    dbc.Row(dbc.Col(data_tape_selector))
    #model_id, scenario_list=[]
    ])

refresh_model_form = dbc.Row([
    #model_name, source_model_id, scenario_list=[]
    dbc.Row([model_name]),
    dbc.Row([deal_ids, batch_keys]),
    ],
    form=True
)



create_model_form = dbc.Form([
        model_name,
        dbc.Row([
            dbc.Col(model_type),
            dbc.Col(asset_class),
            ], form=True),
        deal_ids,
        batch_keys,
        dbc.Row([
            dbc.Col(model_template, className='text-center'),
            ]),
    ])

load_data_tape_form = dbc.Form([
    dbc.Row([
        dbc.Col(data_tape_selector),
        ], className='text-center'),
    dbc.Row([
        dbc.Col(custom_sql_input, width=6),
        dbc.Col(upload_sql_input, width=6),
        ])
    ])

create_model = dbc.Container([
    dbc.Row(
        dbc.Col(html.H4('Create New Model'), className='mb-4 text-center')
        ),
    dbc.Row([
        dbc.Col([
            #create_model_form
            dbc.Card([
                create_model_form
                ], body=True),
            ]),
        ]),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                load_data_tape_form
                ], body=True)
            ])
        ])
    ])
"""

#import data tape form
data_tape_form = dbc.Row([
    
    ])
"""





rate_curves = html.Div([
    html.H3('Tab Content - Rate Curves')
    ])
        
model_config = html.Div([
    html.H3('Tab Content - Model Config')
    ])
        
model_run = html.Div([
    html.H3('Tab Content - Model Run')
    ])
        
model_eval = html.Div([
    html.H3('Tab Content - Model Eval')
    ])

layout1 = html.Div([
    html.H3('App 1'),
    dcc.Dropdown(
        id='app-1-dropdown',
        options=[
            {'label': 'App 1 - {}'.format(i), 'value': i} for i in [
                'NYC', 'MTL', 'LA'
            ]
        ]
    ),
    html.Div(id='app-1-display-value'),
    dcc.Link('Go to App 2', href='/apps/app2')
])

layout2 = html.Div([
    html.H3('App 2'),
    dcc.Dropdown(
        id='app-2-dropdown',
        options=[
            {'label': 'App 2 - {}'.format(i), 'value': i} for i in [
                'NYC', 'MTL', 'LA'
            ]
        ]
    ),
    html.Div(id='app-2-display-value'),
    dcc.Link('Go to App 1', href='/apps/app1')
])