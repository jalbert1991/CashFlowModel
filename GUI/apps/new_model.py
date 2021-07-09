# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 17:50:28 2020

@author: jalbert
"""
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_daq as daq


model_name = dbc.FormGroup([
    #dbc.Label('Model Name', html_for='model_name_form'), #, className='text-center'
    dbc.Col(
        dbc.Input(placeholder='Enter Model Name', type='text', id='model-name', persistence=True),
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
        id='model-type-form',
        )
    ])

asset_class = dbc.FormGroup([
    dbc.Label('Asset Class'),
    dcc.Dropdown(
        options=[
            {'label':'Solar', 'value':'solar'},
            {'label':'Home Improvement', 'value':'home improvement'},
            {'label':'Marketplace', 'value':'marketplace'},
            {'label':'Student', 'value':'student'},
            {'label':'Mortgage', 'value':'mortgage'},
            ],
        value='',
        id='asset-class-form'
        )
    ])

#deal_ids = 

#batch_keys = 


data_tape_selector = dbc.FormGroup([
    dbc.Label('Select Data Tape Source'),
    dbc.RadioItems(
            options=[
                {"label": "CDW", "value": 'cdw'},
                {"label": "Custom SQL", "value": 'sql'},
                {"label": "Load From File", "value": 'excel'},
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


def return_tab_create_model(deal_id_dict, batch_key_dict):
    create_model_form = dbc.Form([
            model_name,
            dbc.Row([
                dbc.Col(model_type),
                dbc.Col(asset_class),
                ], form=True),
            dbc.FormGroup([
                dbc.Label('Select Deals'),
                dcc.Dropdown(
                    options=deal_id_dict,
                    multi=True,     
                    id='deal-ids-form',
                    ),
                ]),
            dbc.FormGroup([
                dbc.Label('Select Batches'),
                dcc.Dropdown(
                    options=batch_key_dict,
                    multi=True,
                    id='batch-keys-form',
                    ),
                ], 
                ),
            ]
        )
    
    return html.Div([
            create_model_form,
            dbc.Button('Create Model', color='dark', block=True, id='new-model-create-button'), #width={"size": 6, "offset": 3}
            #dbc.Collapse([
                dbc.Alert(['Creating Model ', dbc.Spinner(id='create-model-spinner', size='sm')],
                          color='dark', 
                          id='create-model-alert',
                          is_open=False,
                          dismissable=True,
                          className='mt-2'
                          ),
                #dbc.Row([
                #    dbc.Col(dbc.Button('Back', id='new-model-button-continue', color='dark')),
                #    dbc.Col(dbc.Button('Continue', id='new-model-button-continue', color='dark'))
                #    ]),
                #], id='create-model-collapse')
            ])

def return_tab_data_tape():
    data_tape_form = dbc.Form([
        dbc.Row([
            dbc.Col(data_tape_selector),
            ], className='text-center'),
        dbc.Row([
            dbc.Col(custom_sql_input, width=6),
            dbc.Col(upload_sql_input, width=6),
            ]),
        ]
    )
    
    return data_tape_form



build_model_tabs = dbc.Container([
        dbc.Row(
            dbc.Col(html.H4('Build New Model'), className='mb-4 text-center')
            ),
        #dbc.Col([
        #dbc.Row([
            #dbc.Col([
            #    dbc.ListGroup([
            #            dbc.ListGroupItem("1) Model Info", color="success"),
            #            dbc.ListGroupItem("2) Data Tape", color="secondary"),
            #            dbc.ListGroupItem("3) Curve Groups", color="secondary"),
            #        ]),
            #    ], width=2),
            dbc.Row([
                dbc.Tabs(
                    [
                    dbc.Tab(label="Create Model", tab_id="tab-create"),
                    dbc.Tab(label="Data Tape", tab_id="tab-data-tape", disabled=False),
                    dbc.Tab(label="Curve Groups", tab_id="tab-curve-groups", disabled=True),
                    dbc.Tab(label="Model Config", tab_id="tab-model-config", disabled=True),
                    dbc.Tab(label="Scenarios", tab_id="tab-scenarios", disabled=True)
                    ],
                    id="build-model-tabs",
                    active_tab="tab-create",
                    className='mb-3'
                    ),
                ]),
            dbc.Row([
                    dbc.Col([
                    #create_model_form
                    dbc.Card([
                        html.Div(id="build-model-tab-content"),
                        ], body=True),
                    ]),
                ]),
            #]),
        #dcc.Store(),
        #dcc.Store(),
        #html.Div(id="build-model-tab-content")
        #dbc.Row([
        #    dbc.Col([
        #        #create_model_form
        #        dbc.Card([
        #            html.Div(id="build-model-tab-content"),
        #            ], body=True),
        #        ]),
        #    ]),
            #])
        ])

def return_sub_layout_create_model(deal_id_dict, batch_key_dict):
    create_model_form = dbc.Form([
            model_name,
            dbc.Row([
                dbc.Col(model_type),
                dbc.Col(asset_class),
                ], form=True),
            dbc.FormGroup([
                dbc.Label('Select Deals'),
                dcc.Dropdown(
                    options=deal_id_dict,
                    multi=True,     
                    id='deal-ids-form',
                    ),
                ]),
            dbc.FormGroup([
                dbc.Label('Select Batches'),
                dcc.Dropdown(
                    options=batch_key_dict,
                    multi=True,
                    id='batch-keys-form',
                    ),
                ], 
                ),
            dbc.Row([
                dbc.Col(data_tape_selector),
                ], className='text-center'),
            dbc.Row([
                dbc.Col(custom_sql_input, width=6),
                dbc.Col(upload_sql_input, width=6),
                ]),
            html.Div([
                dbc.Button('Build Model', color='dark', block=True, id='new-model-create-button'), #width={"size": 6, "offset": 3})
                #dbc.Collapse([
                #    dbc.Card([
                #        dbc.Row(["Creating Model ", dbc.Spinner(html.Div(id='create-model-spinner'), size="sm")]),
                #        dbc.Row(["Downloading Data Tape ", dbc.Spinner(html.Div(id='download-data-tape-spinner'), size="sm")]),
                #        ], body=True)
                #    ], id='new-model-status')
                dbc.Modal([
                    dbc.ModalBody([
                        dbc.Alert(['Creating Model ', dbc.Spinner(id='create_model-spinner', size='sm')], color='dark', id='create-model-alert'),
                        dbc.Alert(['Downloading Data Tape ', dbc.Spinner(id='download-data-tape-spinner', size='sm')], color='dark', id='data-tape-alert'),
                        #dbc.Row(["Creating Model ", dbc.Spinner(html.Div(id='create-model-spinner'), size="sm")]),
                        #dbc.Row(["Downloading Data Tape ", dbc.Spinner(html.Div(id='download-data-tape-spinner'), size="sm")]),
                        ]),
                    dbc.ModalFooter([
                        dbc.Row([
                            dbc.Col(dbc.Button('Back', id='new-model-button-continue', color='dark')),
                            dbc.Col(dbc.Button('Continue', id='new-model-button-continue', color='dark'))
                            ]),
                        ]),
                    ],
                    id='new-model-status-modal',
                    centered=True,
                    ),
                dcc.Store(id='new-model-trigger'),
                ])
        ])
    
    return create_model_form


def return_layout_create_model(deal_dict, batch_dict):
    create_model = dbc.Container([
        dbc.Row(
            dbc.Col(html.H4('Create New Model'), className='mb-4 text-center')
            ),
        dbc.Row([
            dbc.Col([
                #create_model_form
                dbc.Card([
                    return_sub_layout_create_model(deal_dict, batch_dict)
                    ], body=True),
                ]),
            ]),
        #dbc.Row([
        #    dbc.Col([
        #        dbc.Card([
        #            
        #            ])
        #        ])
        #    ]),
        dbc.Row([
            html.Div(id='create-model-test')
            ])
        ])
    
    return html.Div([sidebar, create_model]) #sidebar style={"padding":"4rem"}


