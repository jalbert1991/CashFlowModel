# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 22:55:21 2020

@author: jalbert
"""

import time

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State, MATCH, ALL

from GUI.app import app
from GUI.app import server

#import callbacks
from GUI.apps import home, new_model

#import layouts 
#from GUI.layouts import landing_page, create_model, config_tabs, rate_curves, model_config, model_run, model_eval
#import GUI.layouts as layouts

#import and create blank CF Model Instance
from GUI.cf_model import cf_model


navbar = dbc.Navbar(
    dbc.Container(
        [
            html.A(
                # Use row and col to control vertical alignment of logo / brand
                dbc.Row(
                    [
                        dbc.Col(html.Img(src="/assets/credigy-large-trans+(1)_white.png", height="30px")),
                        dbc.Col(dbc.NavbarBrand("FRA Cash Flow Engine", className="ml-2")),
                    ],
                    align="center",
                    no_gutters=True,
                ),
                href="/apps/home",
            ),
            dbc.NavbarToggler(id="navbar-toggler1"),
            dbc.Collapse(
                dbc.Nav(
                    # right align dropdown menu with ml-auto className
                    [dbc.NavItem(dbc.NavLink("Home", href='/apps/home')),
                     dbc.NavItem(dbc.NavLink("Dashboard", href='/apps/dashboard')),
                        ], navbar=True, className="ml-auto",
                ),
                id="navbar-collapse1",
                navbar=True,
            ),
        ]
    ),
    color="dark",
    dark=True,
    className="mb-4",
    fixed="top",
    style={"padding":".5rem 1rem"}
)

def make_sidebar_item(item_name):
    return dbc.Card([
        dbc.CardHeader(
            html.H2(
                dbc.Button(item_name,
                           #color='',
                           id=f"sidebar-item-{item_name}")
                )
            
            ), 
        dbc.Collapse(
            dbc.CardBody(children=[],
                id={'type':'sidebar-collapse-item',
                    'id':f"sidebar-collapse-{item_name}"}
                )
            ),
        ], )

sidebar_sections = ['Model Info', 'Data Tape', 'Curve Groups', 'Model Config', 'Scenarios']
sidebar_accordion = html.Div([make_sidebar_item(x) for x in sidebar_sections], 
                                id={'type':'sidebar-sections',
                                    'index':ALL})

sidebar = html.Div(
    [
        html.H4("Model Details"), #, className="display-4"),
        html.Hr(),
        #dbc.Nav(
        #    [
        #        dbc.NavLink("Page 1", href="/page-1", id="page-1-link"),
        #        dbc.NavLink("Page 2", href="/page-2", id="page-2-link"),
        #        dbc.NavLink("Page 3", href="/page-3", id="page-3-link"),
        #    ],
        #    vertical=True,
        #    pills=True,
        #),
        #dbc.ListGroup([
        #    dbc.ListGroupItem([
        #        dbc.ListGroupItemHeading("1) Model Info"),
        #        dbc.ListGroupItemText(id='sidebar-info-text'),
        #        ], color="secondary", id='sidebar-info'),
        #    dbc.ListGroupItem([
        #        dbc.ListGroupItemHeading("2) Data Tape"),
        #        dbc.ListGroupItemText(id='sidebar-data-tape'),
        #        ], color="secondary"),
        #    dbc.ListGroupItem("3) Curve Groups", color="secondary", id='sidebar-curve-groups'),
        #    ]),
        #sidebar_accordion,
    ],
    style={
        "position": "fixed",
        "top": 0,
        "left": 0,
        "bottom": 0,
        #"width": "16rem",
        "padding": "6rem 1rem 2rem",
        #"padding": "2rem 1rem",
        "background-color": "#f8f9fa",
    },
    className='sidebar',
)

app.layout = html.Div([
        dcc.Location(id='url', refresh=False),
        navbar,
        sidebar,
        html.Div(id='page-content', style={"padding-top": "6rem",}),
        dcc.Interval(id='interval-start-up', max_intervals=1, n_intervals=0)
        ]
    )

##################################
#       Page Selection
@app.callback(Output('page-content', 'children'),
              Input('url', 'pathname'),
              )
def display_page(pathname):
    if pathname == '/apps/create_new_model':
        #return layouts.create_model
        #return new_model.return_layout_create_model(cf_model.unique_deal_ids(), cf_model.unique_batch_keys())
        return new_model.build_model_tabs
    #if pathname == '/apps/build_model':
    #     return layouts.create_model
    else:
        return home.layout

##########################################
#              Sidebar Accordion
@app.callback(#[Output(f"sidebar-collapse-{x}", "is_open") for x in sidebar_sections],
              #[Input(f"sidebar-item-{x}", "n_clicks") for x in sidebar_sections],
              #[State(f"sidebar-collapse-{x}", "is_open") for x in sidebar_sections],
              Output({'type':'sidebar-collapse-item', 'id':MATCH}, 'is_open'),
              Input({'type':'sidebar-sidebar-item', 'id':MATCH}, 'n_clicks'),
              State({'type':'sidebar-sidebar-item', 'id':MATCH}, 'is_open'),
              prevent_initial_call=True)
def sidebar_accordian_toggle(*args):
    ctx = dash.callback_context
    button_id = ctx.triggered[0]['prop_id'].split(".")[0]
    outputs
    
    #if not ctx.triggered:
    #    [False for x in ]
    
    #component_names = item.component_id for item in 


##########################################
#           New Model Tabs 
@app.callback(Output('build-model-tab-content', 'children'),
              Input('build-model-tabs', 'active_tab')
              )
def switch_tab(at):
    if at=='tab-create':
        return new_model.return_tab_create_model(cf_model.unique_deal_ids(), cf_model.unique_batch_keys())
    elif at=='tab-data-tape':
        return new_model.return_tab_data_tape()
    

#@app.callback(#Output('new-model-status-modal', 'is_open'),
#              #Output('new-model-trigger', 'data'),
#              Output('create-model-alert', 'is_open'),
#              Input('new-model-create-button', 'n_clicks'),
#              prevent_initial_call=True)
#def toggle_new_model_collapse(n_click):
#    print('trigger 1')
#    if n_click:
#        return True #, 'new model start'

@app.callback(#Output('new-model-create-button', 'children'),
              Output('create-model-spinner', 'children'), 
              #Output('create-model-alert', 'color'),
              Output('sidebar-info', 'color'),
              Output('sidebar-info-text', 'children'),
              Input('new-model-create-button', 'n_clicks'),
              #Input('create-model-alert', 'is_open'),
              #Input('new-model-trigger', 'data'),
              #Input('new-model-status-modal', 'is_open'),
              State('model-name', 'value'),
              State('asset-class-form', 'value'),
              State('model-type-form', 'value'),
              State('deal-ids-form', 'value'),
              State('batch-keys-form', 'value'),
              #State('datatape-input', 'value'), 
              prevent_initial_call=True)
def create_model_inputs(d, model_name, asset_class, model_type, deal_ids, batch_keys):
    print('trigger 1')
    if d:
        deal_ids = [] if not deal_ids else deal_ids
        batch_keys = [] if not batch_keys else batch_keys
        
        output_text = html.Div([
                dbc.Row(model_name),
                dbc.Row(asset_class),
                dbc.Row(model_type),
                dbc.Row(deal_ids),
                dbc.Row(batch_keys),
            ])
        
        #''.join([str(model_name), str(asset_class), str(model_type), str(deal_ids), str(batch_keys), str(datatape_source)])
        
        #global cash_flow_model
        #cash_flow_model = cf_model.create_model('new', model_name, deal_ids, batch_keys, model_type, asset_class)
        print('time before')
        time.sleep(3)
        print('time 1')
        #return html.Div(msg)
        #button_loading_status = [dbc.Spinner(size="sm"), " Loading..."]
        model_status = [' - Done']
        return model_status, 'success', output_text
    
@app.callback(Output('build-model-tabs', 'active_tab'),
              Input('create-model-alert', 'color'),
              prevent_initial_call=True
              )
def create_model_next_tab(color):
    if color=='success':
        time.sleep(1)
        return 'tab-data-tape'
    
@app.callback(Output('download-data-tape-spinner', 'children'),
              Input('create-model-spinner', 'children'),
              State('datatape-input', 'value'),
              prevent_initial_call=True
              )
def download_data_tape(input_change, data_tape_source):
    #if data_tape_source in ['cdw','sql']:
    #    cash_flow_model.import_data_tape_sql(source=data_tape_source, query='')
    #elif data_tape_source in ['excel']:
    #    cash_flow_model.import_data_tape_excel()
    
    time.sleep(5)
    print('time 2')
    return [' - Done']
        
#@app.callback([Output('deal-ids-form', 'options'),
#               Output('batch-keys-form', 'options')
#               ],
#              [Input('interval-start-up', 'n_intervals')]
#              )
#def add_deal_batch():
#    deal_ids = cf_model.unique_deal_ids()
#    batch_keys = cf_model.unique_batch_keys()
#    return deal_ids, batch_keys

#@app.callback(Output('create-model-content', 'children'),
#              [Input('create-model-tabs', 'value')]
#              )
#def display_create_model_tab(tab):
#    if tab == 'new':
#        return landing_page_new
#    else:
#        return ''



#@app.callback(Output('tabs-content', 'children'),
#              [Input('tabs', 'value')])
#def display_tab(tab):
#    if tab == 'rate_curves':
#        return layouts.rate_curves
#    elif tab == 'model_config':
#        return layouts.model_config
#    elif tab == 'model_run':
#        return layouts.model_run
#    elif tab == 'model_eval':
#        return layouts.model_eval
#    else:
#        return '404'

#import GUI.layouts as layouts


if __name__ == '__main__':
    app.run_server(debug=False)




"""
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

# see https://community.plot.ly/t/nolayoutexception-on-deployment-of-multi-page-dash-app-example-code/12463/2?u=dcomfort
from app import server
from app import app
from layouts import layout_birst_category, layout_ga_category, layout_paid_search, noPage, layout_display, layout_publishing, layout_metasearch
import callbacks

# see https://dash.plot.ly/external-resources to alter header, footer and favicon
app.index_string = ''' 
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>CC Performance Marketing Report</title>
        {%favicon%}
        {%css%}
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
        </footer>
        <div>CC Performance Marketing Report</div>
    </body>
</html>
'''

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

# Update page
# # # # # # # # #
@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/cc-travel-report' or pathname == '/cc-travel-report/overview-birst/':
        return layout_birst_category
    elif pathname == '/cc-travel-report/overview-ga/':
        return layout_ga_category
    elif pathname == '/cc-travel-report/paid-search/':
        return layout_paid_search
    elif pathname == '/cc-travel-report/display/':
        return layout_display
    elif pathname == '/cc-travel-report/publishing/':
        return layout_publishing
    elif pathname == '/cc-travel-report/metasearch-and-travel-ads/':
        return layout_metasearch
    else:
        return noPage

# # # # # # # # #
external_css = ["https://cdnjs.cloudflare.com/ajax/libs/normalize/7.0.0/normalize.min.css",
                "https://cdnjs.cloudflare.com/ajax/libs/skeleton/2.0.4/skeleton.min.css",
                "//fonts.googleapis.com/css?family=Raleway:400,300,600",
                "https://codepen.io/bcd/pen/KQrXdb.css",
                "https://maxcdn.bootstrapcdn.com/font-awesome/4.7.0/css/font-awesome.min.css",
                "https://codepen.io/dmcomfort/pen/JzdzEZ.css"]

for css in external_css:
    app.css.append_css({"external_url": css})

external_js = ["https://code.jquery.com/jquery-3.2.1.min.js",
               "https://codepen.io/bcd/pen/YaXojL.js"]

for js in external_js:
    app.scripts.append_script({"external_url": js})

if __name__ == '__main__':
    app.run_server(debug=True)
    
"""
