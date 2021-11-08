# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import CashFlowModelMain as cf_main
import PySimpleGUI as sg
import json
import gc
gc.collect()

pd.set_option('display.max_columns', 500)
#import DataPrep.DataMover as data_mover
#import openpyxl as xl

#import GUI.index as gui
gui.app.run_server(debug=False)

pd.options.display.float_format = '{:20,.2f}'.format


import DataPrep.DataMover as data_prep
prep = data_prep.DataPrep()

asset_class = 'solar'
sql="""
    SELECT TOP 1 model_key
        ,model_name
    	--,scenario_key
    	--,seq_order
    FROM fra.cf_model.vw_scenario
    WHERE 1=1
    	AND uw_type='template'
    	--AND asset_class='{}'
    ORDER BY seq_order
""".format(asset_class)

sql = """
SELECT top 1 uw_month
                    , scenario_key
                from fra.cf_model.vw_scenario
                	where 1=1
                		and model_name='{}'
                    """.format('Solar Template')


model_attr = prep.sql_engine_import.execute(sql, output=True)
model_attr.squeeze()
len(model_attr)

model_attr


#########################################
#            testing
#########################################
import CashFlowModelMainV2 as cf_main

############################################################
#Build Solar Asset Class Template
solar_template_model = cf_main.CashFlowModel.new_model(model_name='Solar Template', deal_ids=[], batch_keys=[], asset_class='solar',uw_month=None, uw_type='template')

#import sample data
solar_template_model.import_data_tape_sql(source='cdw')

#create curve group
solar_template_model.create_curve_group('Solar Asset Class - 201909')
file_path = r"\\SRV-NWUS-SHR\BA$\Credit Strategy\4. RUW\2. Deliverables\2019\201908 - Mosaic\Model\Mosaic RUW 201908 V11.xlsm"
solar_template_model.import_rate_curves_excel('Solar Asset Class - 201909', 'default', 'base', file_path, 'MDR', 'C3:NF363', key_cols=['Key'])
solar_template_model.import_rate_curves_excel('Solar Asset Class - 201909', 'prepay', 'base', file_path, 'MPR', ws_range='C3:NF363', key_cols=['Key'])
solar_template_model.import_rate_curves_excel('Solar Asset Class - 201909', 'curtail', 'base', file_path, 'MCR', ws_range='C3:NF363', key_cols=['Key'])
solar_template_model.create_recovery_curve('Solar Asset Class - 201909', (18, 0.25))

#create segments
solar_template_model.create_segment(use_gui=False, curve_group_name='Solar Asset Class - 201909', segment_type='default', OriginationTerm=[], OriginationCreditScore=[700,750,800]) 
solar_template_model.create_segment(use_gui=False, curve_group_name='Solar Asset Class - 201909', segment_type='prepay', OriginationCreditScore=[700,750,800]) 
solar_template_model.create_segment(use_gui=False, curve_group_name='Solar Asset Class - 201909', segment_type='curtail', OriginationTerm=[], OriginationCreditScore=[700,750,800], OriginationMonth=[]) 

solar_template_model.return_curve_map('Solar Asset Class - 201909')

#manual map for missing
manual_map = {
    'default': {'120|700-750': '120|700-750',
  '120|750-800': '120|750-800',
  '120|<700': '120|<700',
  '180|700-750': '180|700-750',
  '180|750-800': '180|750-800',
  '180|<700': '180|<700',
  '240|700-750': '240|700-750',
  '240|750-800': '240|750-800',
  '240|<700': '240|<700',
  '300|700-750': '300|700-750',
  '300|750-800': '300|750-800',
  '300|<700': '300|<700',
  '120|>800': '120|800-850',
  '144|>800': '144|800-850',
  '180|>800': '180|800-850',
  '240|>800': '240|800-850',
  '300|>800': '300|800-850'},
 'prepay': {'700-750': '700-750',
  '750-800': '750-800',
  '<700': '<700',
  '>800': '800-850'},
 'curtail': {'120|700-750|1': '120|700-750|1',
  '120|700-750|10': '120|700-750|10',
  '120|700-750|11': '120|700-750|11',
  '120|700-750|12': '120|700-750|12',
  '120|700-750|2': '120|700-750|2',
  '120|700-750|3': '120|700-750|3',
  '120|700-750|4': '120|700-750|4',
  '120|700-750|5': '120|700-750|5',
  '120|700-750|6': '120|700-750|6',
  '120|700-750|7': '120|700-750|7',
  '120|700-750|8': '120|700-750|8',
  '120|700-750|9': '120|700-750|9',
  '120|750-800|1': '120|750-800|1',
  '120|750-800|10': '120|750-800|10',
  '120|750-800|11': '120|750-800|11',
  '120|750-800|12': '120|750-800|12',
  '120|750-800|4': '120|750-800|4',
  '120|750-800|5': '120|750-800|5',
  '120|750-800|6': '120|750-800|6',
  '120|750-800|7': '120|750-800|7',
  '120|750-800|8': '120|750-800|8',
  '120|750-800|9': '120|750-800|9',
  '120|<700|1': '120|<700|1',
  '120|<700|10': '120|<700|10',
  '120|<700|11': '120|<700|11',
  '120|<700|12': '120|<700|12',
  '120|<700|2': '120|<700|2',
  '120|<700|3': '120|<700|3',
  '120|<700|4': '120|<700|4',
  '120|<700|5': '120|<700|5',
  '120|<700|6': '120|<700|6',
  '120|<700|7': '120|<700|7',
  '120|<700|8': '120|<700|8',
  '120|<700|9': '120|<700|9',
  '180|700-750|1': '180|700-750|1',
  '180|700-750|10': '180|700-750|10',
  '180|700-750|11': '180|700-750|11',
  '180|700-750|12': '180|700-750|12',
  '180|700-750|2': '180|700-750|2',
  '180|700-750|3': '180|700-750|3',
  '180|700-750|4': '180|700-750|4',
  '180|700-750|5': '180|700-750|5',
  '180|700-750|6': '180|700-750|6',
  '180|700-750|7': '180|700-750|7',
  '180|700-750|8': '180|700-750|8',
  '180|700-750|9': '180|700-750|9',
  '180|750-800|1': '180|750-800|1',
  '180|750-800|10': '180|750-800|10',
  '180|750-800|11': '180|750-800|11',
  '180|750-800|12': '180|750-800|12',
  '180|750-800|4': '180|750-800|4',
  '180|750-800|5': '180|750-800|5',
  '180|750-800|6': '180|750-800|6',
  '180|750-800|7': '180|750-800|7',
  '180|750-800|8': '180|750-800|8',
  '180|750-800|9': '180|750-800|9',
  '180|<700|1': '180|<700|1',
  '180|<700|10': '180|<700|10',
  '180|<700|11': '180|<700|11',
  '180|<700|12': '180|<700|12',
  '180|<700|2': '180|<700|2',
  '180|<700|3': '180|<700|3',
  '180|<700|4': '180|<700|4',
  '180|<700|5': '180|<700|5',
  '180|<700|6': '180|<700|6',
  '180|<700|7': '180|<700|7',
  '180|<700|8': '180|<700|8',
  '180|<700|9': '180|<700|9',
  '240|700-750|1': '240|700-750|1',
  '240|700-750|10': '240|700-750|10',
  '240|700-750|11': '240|700-750|11',
  '240|700-750|12': '240|700-750|12',
  '240|700-750|2': '240|700-750|2',
  '240|700-750|3': '240|700-750|3',
  '240|700-750|4': '240|700-750|4',
  '240|700-750|5': '240|700-750|5',
  '240|700-750|6': '240|700-750|6',
  '240|700-750|7': '240|700-750|7',
  '240|700-750|8': '240|700-750|8',
  '240|700-750|9': '240|700-750|9',
  '240|750-800|1': '240|750-800|1',
  '240|750-800|10': '240|750-800|10',
  '240|750-800|11': '240|750-800|11',
  '240|750-800|12': '240|750-800|12',
  '240|750-800|4': '240|750-800|4',
  '240|750-800|5': '240|750-800|5',
  '240|750-800|6': '240|750-800|6',
  '240|750-800|7': '240|750-800|7',
  '240|750-800|8': '240|750-800|8',
  '240|750-800|9': '240|750-800|9',
  '240|<700|1': '240|<700|1',
  '240|<700|10': '240|<700|10',
  '240|<700|11': '240|<700|11',
  '240|<700|12': '240|<700|12',
  '240|<700|2': '240|<700|2',
  '240|<700|3': '240|<700|3',
  '240|<700|4': '240|<700|4',
  '240|<700|5': '240|<700|5',
  '240|<700|6': '240|<700|6',
  '240|<700|7': '240|<700|7',
  '240|<700|8': '240|<700|8',
  '240|<700|9': '240|<700|9',
  '300|700-750|1': '300|700-750|1',
  '300|700-750|10': '300|700-750|10',
  '300|700-750|11': '300|700-750|11',
  '300|700-750|12': '300|700-750|12',
  '300|700-750|2': '300|700-750|2',
  '300|700-750|3': '300|700-750|3',
  '300|700-750|4': '300|700-750|4',
  '300|700-750|5': '300|700-750|5',
  '300|700-750|6': '300|700-750|6',
  '300|700-750|7': '300|700-750|7',
  '300|700-750|8': '300|700-750|8',
  '300|700-750|9': '300|700-750|9',
  '300|750-800|1': '300|750-800|1',
  '300|750-800|10': '300|750-800|10',
  '300|750-800|11': '300|750-800|11',
  '300|750-800|12': '300|750-800|12',
  '300|750-800|2': '300|750-800|2',
  '300|750-800|3': '300|750-800|3',
  '300|750-800|4': '300|750-800|4',
  '300|750-800|5': '300|750-800|5',
  '300|750-800|6': '300|750-800|6',
  '300|750-800|7': '300|750-800|7',
  '300|750-800|8': '300|750-800|8',
  '300|750-800|9': '300|750-800|9',
  '300|<700|1': '300|<700|1',
  '300|<700|10': '300|<700|10',
  '300|<700|11': '300|<700|11',
  '300|<700|12': '300|<700|12',
  '300|<700|2': '300|<700|2',
  '300|<700|3': '300|<700|3',
  '300|<700|4': '300|<700|4',
  '300|<700|5': '300|<700|5',
  '300|<700|6': '300|<700|6',
  '300|<700|7': '300|<700|7',
  '300|<700|8': '300|<700|8',
  '300|<700|9': '300|<700|9',
  '120|>800|1': '120|800-850|1',
  '120|>800|10': '120|800-850|10',
  '120|>800|11': '120|800-850|11',
  '120|>800|12': '120|800-850|12',
  '120|>800|2': '120|800-850|2',
  '120|>800|3': '120|800-850|3',
  '120|>800|4': '120|800-850|4',
  '120|>800|5': '120|800-850|5',
  '120|>800|6': '120|800-850|6',
  '120|>800|7': '120|800-850|6',
  '120|>800|8': '120|800-850|8',
  '120|>800|9': '120|800-850|9',
  '144|>800|1': '144|800-850|1',
  '144|>800|10': '144|800-850|10',
  '144|>800|11': '144|800-850|11',
  '144|>800|12': '144|800-850|12',
  '144|>800|2': '144|800-850|2',
  '144|>800|3': '144|800-850|3',
  '144|>800|4': '144|800-850|4',
  '144|>800|5': '144|800-850|5',
  '144|>800|6': '144|800-850|6',
  '144|>800|7': '144|800-850|6',
  '144|>800|8': '144|800-850|8',
  '144|>800|9': '144|800-850|9',
  '180|>800|1': '180|800-850|1',
  '180|>800|10': '180|800-850|10',
  '180|>800|11': '180|750-800|11',
  '180|>800|12': '180|800-850|12',
  '180|>800|2': '180|800-850|2',
  '180|>800|3': '180|800-850|3',
  '180|>800|4': '180|800-850|4',
  '180|>800|5': '180|800-850|5',
  '180|>800|6': '180|800-850|6',
  '180|>800|7': '180|800-850|7',
  '180|>800|8': '180|800-850|8',
  '180|>800|9': '180|800-850|9',
  '240|>800|1': '240|800-850|1',
  '240|>800|10': '240|800-850|10',
  '240|>800|11': '240|800-850|11',
  '240|>800|12': '240|800-850|12',
  '240|>800|2': '240|800-850|2',
  '240|>800|3': '240|800-850|3',
  '240|>800|4': '240|800-850|4',
  '240|>800|5': '240|800-850|5',
  '240|>800|6': '240|800-850|6',
  '240|>800|7': '240|800-850|7',
  '240|>800|8': '240|800-850|8',
  '240|>800|9': '240|800-850|9',
  '300|>800|1': '300|800-850|1',
  '300|>800|10': '300|800-850|10',
  '300|>800|11': '300|800-850|11',
  '300|>800|12': '300|800-850|12',
  '300|>800|2': '300|800-850|2',
  '300|>800|3': '300|800-850|3',
  '300|>800|4': '300|800-850|4',
  '300|>800|5': '300|800-850|5',
  '300|>800|6': '300|800-850|6',
  '300|>800|7': '300|800-850|7',
  '300|>800|8': '300|800-850|8',
  '300|>800|9': '300|800-850|9'},
 'recovery': {},
 'collection': {}}

solar_template_model.map_curves(curve_group_name='Solar Asset Class - 201909', manual_map=manual_map)

#create model configuration
config={}
config['amort_formula'] = {}
config['amort_timing'] = {}

config['amort_timing']['promo'] = "['MonthsToAcquisition'] == 0" #(['MonthsOnBook'] < 18) & (['ProjectionMonth']==1)
config['amort_formula']['promo'] = "np.pmt(['InterestRate'], ['OriginationTerm']-['PromoTerm'], ['PromoTargetBalance'])"

config['amort_timing']['promo end'] = "['MonthsOnBook'] == ['PromoTerm']"
config['amort_formula']['promo end'] = ''
config['default_amort_type'] = 'scale'

solar_template_model.create_model_config(model_config_name='Solar Asset Class', config_type=1, config_dict=config)

#create Base Case Sceanrio
solar_template_model.create_cf_scenario('Base Case', cutoff_date=None, curve_group='Solar Asset Class - 201909', index_projection_date=None, model_config='Solar Asset Class')


#######################################################################################
## create Mosaic Model
mosaic = cf_main.CashFlowModel.new_model(model_name='Mosaic', deal_ids=[171], batch_keys=[], asset_class='solar', uw_type='ruw', uw_month='201909')

mosaic.create_cf_scenario('Backtest', cutoff_date='2016-10-31', curve_group='Solar Asset Class - 201909', index_projection_date=None, model_config='Solar Asset Class')
mosaic.create_cf_scenario('Base Case', cutoff_date='2019-08-31', curve_group='Solar Asset Class - 201909', index_projection_date=None, model_config='Solar Asset Class')
mosaic.run_cash_flows()

mosaic.eval.create_plot('ScheduledPaymentAmount', end_period=150)
mosaic.eval.create_table('ContractualPrincipalPayment', end_period=120, rate=True)
mosaic.eval.create_plot('Historical ChargeOff Recovery', end_period=200)



stress = {
    'default': [(0, 0.50), (12, 0.50), (24, 0)],
    'recovery': [(0, -0.25)]
    }

##########################################################################################
## Create Galileo 1
cf_model = cf_main.CashFlowModel.new_model(model_name='Galileo', deal_ids=[193], batch_keys=[], asset_class='solar', uw_type='ruw', uw_month='201909')

#import curves
#cf_model.import_rate_curves(use_gui=True)

#import model template
#cf_model.download_model_template()


#create a new curve set
#cf_model.copy_curves('Galileo - Default', 'Solar Asset Class - 201909') #curve_type=['prepay', 'curtail']

#create new curve group and load with base solar curves
cf_model.create_curve_group('Galileo Defaults')
cf_model.import_rate_curves_sql(curve_group_name='Galileo Defaults', model_name='Solar Template', scenario_name='Base Case', curve_type='all')

#import Galileo Specific Curves
file_path = r"\\SRV-NWUS-SHR\BA$\Credit Strategy\4. RUW\2. Deliverables\2019\201909 - Galileo 1\Model\Galileo 1 Solar RUW 201909 V5 (CPR Adj).xlsm"
cf_model.import_rate_curves_excel('Galileo Defaults', 'default', 'base', file_path, 'MDR', ws_range='C3:NF363', key_cols=['Key'])
#curtail adjust
cf_model.import_rate_curves_excel('Galileo Defaults', 'curtail', 'adjust', file_path, 'MCR', ws_range='ABQ3:APT363', key_cols=['Key'])
#prepay adjust
cf_model.import_rate_curves_excel('Galileo Defaults', 'prepay', 'adjust', file_path, 'MPR', ws_range='ABQ3:APT363', key_cols=['Key'])

cf_model.create_cf_scenario('Base Case', cutoff_date='2019-08-31', curve_group='Galileo Defaults', model_config='Solar Asset Class')
cf_model.create_cf_scenario('Backtest', cutoff_date='min', curve_group='Galileo Defaults', model_config='Solar Asset Class')

cf_model.run_cash_flows()

#check actuals/projections
cf_model.eval.create_plot('TotalPaymentMade')


##########################################################################################
## Download Test

import CashFlowModelMain as cf_main
import numpy as np
import pandas as pd

new_template = cf_main.CashFlowModel.create_template('solar')
new_template

template = cf_main.CashFlowModel.load_model(model_name='Solar Template')

download_model = cf_main.CashFlowModel.load_model(model_name='Faraday', uw_month=None) 
download_model.set_final_scenario('Base2')


download_model.import_data_tape_sql() 
download_model.run_cash_flows() 

download_model.eval.create_avp()


refresh_model = cf_main.CashFlowModel.refresh_model(model_name='Faraday', uw_month=202108)
refresh_model.run_cash_flows()
refresh_model.eval.create_avp()

refresh_model.cf_scenarios



month_end_date = datetime.date(int(str(uw_month)[0:4]), int(str(uw_month)[4:6]), 1) + relativedelta(months=-1, day=31)
new_cutoff = month_end_date.strftime("%Y-%m-%d")


download_model.rate_curve_groups['Solar Asset Class - 201909'].map_segments_to_curves('recovery')
download_model.rate_curve_groups['Solar Asset Class - 201909'].segments['recovery'].segment_rules __dict__.keys()

download_model= cf_main.CashFlowModel.load_model(model_name='Casser')
download_model.run_cash_flows(['Casser 052021 Backtest v2'])
download_model.import_rate_curves_sql(curve_group_name='single_curve_test', source_curve_group_name='FaradayOrigP2', update_curve_map=False)
download_model.rate_curve_groups['FaradayOrigP1'].segment_curve_map
download_model.rate_curve_groups['FaradayOrigP1'].segment_map_manual

download_model = cf_main.CashFlowModel.load_model(model_name='Voyager2')
download_model.run_cash_flows()
download_model.cf_scenarios
download_model.run_single_account(scenario='Base', account_id=20618302)
download_model.eval.create_avp()
download_model.cf_scenarios
download_model.set_final_scenario('Stress2')







download_model.eval.create_plot('PostChargeOffCollections')
download_model.eval.create_table('PrincipalFullPrepayment')
download_model.eval.create_avp()

download_model.eval.return_metric_list()
download_model.rate_curve_groups['VoyagerFFOrigP'].curves[('recovery', 'base')].data_rate_curves['rate'].values
download_model.model_configs['VoyagerFFBase'].config_dict

download_model.data_tape.cutoff_tape

download_model.cf_scenarios['Base'][1]._cf_data.keys()


stress = {'default': [(0, 0.1), (12, 0.1), (24, 0.0)],
    'prepay': [(0, -0.25), (12, -0.25), (24, 0.0)]}

import json
stress_json = json.dumps(stress)
stress_convert = json.loads(stress_json)

{k:[tuple(i) for i in v] for (k, v) in stress_convert.items()}

download_model.rate_curve_groups['VoyagerFFOrigP'].curves[('default','base')].data_rate_curves
download_model.data_tape.cutoff_tape['BOM_PrincipalBalance'].sum()
download_model.data_tape.cutoff_tape['TotalPrincipalBalance'].sum()


download_model.import_rate_curves(use_gui=True)
download_model.rate_curve_groups
download_model.rate_curve_groups['Galileo Defaults'].segment_account_map
download_model.rate_curve_groups['Galileo Defaults'].curve_account_map
download_model.rate_curve_groups['Galileo Defaults'].transition_keys #curve_keys, transition_keys
download_model.rate_curve_groups['Galileo Defaults'].curve_keys
download_model.rate_curve_groups['Galileo Defaults'].transition_keys
download_model.rate_curve_groups['Galileo Defaults'].curve_type_info

download_model.rate_curve_groups['Galileo Defaults']

download_model.rate_curve_groups['CasserCurves'].


stress_test = '{"default": [[0, 0.1], [12, 0.1], [24, 0.0]], "prepay": [[0, -0.25], [12, -0.25], [24, 0.0]]}'

import json
json.loads(stress_test)

promo_types = {'No Interest': (0, 1, )}





download_model.eval.create_avp()


os.startfile(in_mem_file)


download_model.rate_curve_groups['Galileo Defaults'].return_curve_group(['default', 'prepay'])
download_model.rate_curve_groups['Galileo Defaults'].curve_account_map
curve_keys = download_model.rate_curve_groups['Galileo Defaults'].curve_keys
trans_keys = download_model.rate_curve_groups['Galileo Defaults'].transition_keys
download_model.rate_curve_groups['Galileo Defaults'].return_transition_matrix2()
download_model.rate_curve_groups['Galileo Defaults'].return_rate_matrix('recovery')
download_model.rate_curve_groups['Galileo Defaults'].curves[('default','base')].data_rate_curves

curve_keys[curve_keys['curve_type']=='rollrate']

rr_matrix = download_model.rate_curve_groups['Galileo Defaults'].return_transition_matrix()
curtail_matrix = download_model.rate_curve_groups['Galileo Defaults'].return_rate_matrix('curtail')
rate_data = download_model.rate_curve_groups['Galileo Defaults'].return_curve_group(['default','prepay','rollrate'])

rate_data.droplevel(['curve_type', 'curve_key'])

rr_matrix[0, :, 1, 8]


trans_keys[trans_keys['curve_type']=='curtail']



rr_segment = download_model.rate_curve_groups['Galileo Defaults'].return_transition_matrix()

rr_segment[1, :, 0, 8]

#isolate the default rolls
default_ix = (rr_segment['from_status']==1) & (rr_segment['to_status']==8)
rr_duplicate = pd.concat([rr_segment[default_ix.values]]*6) # , ignore_index=True)
rr_duplicate['from_status'] = rr_duplicate.groupby(['rollrate']).cumcount()+2


rr_duplicate


batch_array = download_model.eval.cf_output.index.get_level_values('BatchKey').values 
np.in1d(batch_array, [504, 522]))



download_model.eval.return_metric_list()

download_model.eval.create_plot('PrincipalPartialPrepayment')

download_model.eval.create_plot('PrincipalFullPrepayment')
download_model.eval.create_plot('PrincipalTotalPrepayment')

download_model.eval.create_plot('ChargeOffAmount')
download_model.eval.create_table('ChargeOffAmount')

download_model.eval.create_plot('TotalPaymentMade')
download_model.eval.create_plot('ScheduledPaymentMade')

download_model.eval.create_plot('BOM_PrincipalBalance')
download_model.eval.create_plot('ContractualPrincipalPayment')
download_model.eval.create_plot('InterestPayment')

download_model.eval.return_metric_list()

trans_matrix = download_model.rate_curve_groups['Galileo Defaults'].return_transition_matrix()
download_model.rate_curve_groups['Galileo Defaults'].curves
download_model.rate_curve_groups['Galileo Defaults'].curve_account_map
download_model.rate_curve_groups['Galileo Defaults'].segment_account_map
download_model.rate_curve_groups['Galileo Defaults'].return_rate_matrix('default')
download_model.rate_curve_groups['Galileo Defaults'].return_curve_group('prepay')
download_model.rate_curve_groups['Galileo Defaults'].curves[('default','base')].data_rate_curves


download_model.rate_curve_groups['Galileo Defaults'].__dict__.keys()
download_model.rate_curve_groups['Galileo Defaults'].curve_keys

download_model.import_rate_curves(use_gui=True)
download_model.map_curves(use_gui=True, curve_group_name='Galileo - Default')

download_model.rate_curve_groups['Galileo - Default'].segment_curve_map[['segment_type','segment_name','curve_name']]
download_model.rate_curve_groups['Galileo - Default'].curve_keys[['curve_name', 'curve_type']].drop_duplicates()

download_model.data_tape.raw_tape

proj_test = download_model.data_prep.import_index_projections('2020-01-31')
name_ix = proj_test.raw_projections['IndexName'].isin(proj_test.index_list.values())
#projection_select = proj_test.raw_projections.loc[name_ix].copy()
projection_select = proj_test.raw_projections.copy()

#index_name_swap = {v:k for k,v in proj_test.index_list.items()}
#projection_select['IndexID'] = projection_select['IndexName'].map(index_name_swap)


#################################################################################
## RUW test

galileo_refresh = cf_main.CashFlowModel.ruw_model(model_name='Galileo', uw_month=202011, scenario_list=['Base Case'])

type(galileo_refresh.model_id)
galileo_refresh.uw_month

####################################################################################
## Roll Rate Test

rr_test = cf_main.CashFlowModel.new_model(model_name='ja adhoc', deal_ids=[], batch_keys=[], asset_class=None, uw_month=None, uw_type='test')

rr_test.create_curve_group(curve_group_name='rr_test')
rr_test.import_rate_curves_sql(curve_group_name='rr_test', source_curve_group_name='Express 201706')

rr_test.rate_curve_groups['rr_test'].curves[('rollrate','base')].data_rate_curves
rr_test.rate_curve_groups['rr_test'].curves[('rollrate','base')].return_transition_matrix()


rr_test.rate_curve_groups['rr_test'].curve_keys
rr_test.rate_curve_groups['rr_test'].curve_account_map




curve_frame = rr_test.rate_curve_groups['rr_test'].curves[('rollrate','base')].data_rate_curves.set_index(['from_status', 'to_status'], append=True)

dict_test = curve_frame.xs('Career', level='curve_id')['rate'].groupby(['from_status','to_status']).apply(list).to_dict()
dict_len = len(next(iter(dict_test.values())))

import numpy as np
rr_array = np.ones(shape=[15,15,dict_len])

for key, value in dict_test.items():
    rr_array[key[0], key[1]] = value
    
rr_array[1, 2] 
    

np.fromiter
rr_test.rate_curve_groups
download_model.rate_curve_groups['Galileo Defaults'].data_tape.account_status_list

len(curve_frame)['Career'])


rate_dict = {l: rate_data_fill.xs(l, level=0).groupby('curve_key').apply(list).to_dict() for l in rate_data_fill.index.levels[0]}









import DataPrep.DataMover as data_prep

xl_engine = data_prep.ExcelEngine()
sql_engine = data_prep.SQLEngine()

file_path = r"M:\Credigy\Financial Risk Analytics\Re-Underwrite\Deliverables\2021\202103 - Express\Express - PM - RUW 61217 v6 (Revert to 3-16 RUW after 12 months).xlsm"
file_path = r"R:\PL\1. US\A. Banking\Acquired\3. Performing Loans\2015\11-2014 - Express\3. RUW 6-17\Express RUW\Express - PM - RUW 61217 v6 (Revert to 3-16 RUW after 12 months).xlsm"
rr_excel_range = xl_engine.from_excel(file_path=file_path, ws_name='RR', ws_range='E4:GF')
rr_excel_range.dropna(axis=0, inplace=True)
rr_excel_range.drop(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M'], axis=1, inplace=True)
rr_pivot = rr_excel_range.melt(id_vars= ['Month', 'Segment'], var_name='transition', value_name='rate')

rr_pivot[['from_status', 'to_status']] = rr_pivot['transition'].str.split(' to ', 1, expand=True)
rr_pivot.drop('transition', axis=1, inplace=True)
rr_pivot['curve_name']=rr_pivot['Segment']

column_map = {
    'Segment':'curve_id',
    'Month':'period'
    }

rr_pivot.rename(column_map, axis=1, inplace=True)
rr_pivot['curve_type']='rollrate'
rr_pivot['curve_sub_type']='base'
rr_pivot = rr_pivot.astype({'period':'int32'})

rr_pivot.set_index(['curve_id','period'], inplace=True)
rr_pivot.set_index(['from_status', 'to_status'], append=True, inplace=True)

rr_pivot.xs(('Career',59,'5','5'))

express_rr.

express_rr = data_prep.CurveGroup(curve_group_key=3, curve_group_name='Express 201706', data_tape=None, period_type='calendar') 

express_rr.add_curve(rr_pivot, 'rollrate', 'base', 'calendar', 'Excel Custom Curves')
express_rr.curves[('rollrate', 'base')]
express_rr.curves[('rollrate', 'base')].data_rate_curves.set_index(['from_status','to_status'], append=True).xs(('Career',59,'5'))

curve_group_key = express_rr.curve_group_key
rate_json = express_rr.curves[('rollrate', 'base')].curve_json()
#look for corresponding segment if exists
#segment_key = self.rate_curve_groups[curve_group].segments[curve_type].segment_key
segment_key=None

sql_cmd = "exec fra.cf_model.usp_upload_curve_rates ?, ?, ?, ?, ?, ?, ?"
params = [curve_group_key, 'rollrate', 'base', 'calendar', 'Excel Custom Curves', rate_json, segment_key]
sql_engine.execute(sql_cmd, params)


pd.read_json(rate_json).set_index(['period','from_status','to_status']).xs((1, 0, 0))


cf_main.CashFlowModel.new_model('JA AdHoc')


download_model.rate_curve_groups['Galileo Defaults'].curves[('default', 'base')].data_rate_curves



projection_select['curve_type'] = 'index'
projection_select['curve_sub_type'] = 'base'
projection_select['from_status'] = -1
projection_select['to_status'] = -1
column_map = {'IndexName':'curve_id',
              'ProjectionMonth':'period',
              }
projection_select.rename(column_map, axis=1, inplace=True)

projection_select


projection_select = projection_select[['curve_type','curve_sub_type','curve_id', 'period','rate', 'from_status', 'to_status']].set_index(['curve_type','curve_sub_type','curve_id', 'period']).sort_index()
projection_select.xs('1 Month LIBOR Rate')

download_model.rate_curve_groups['Galileo - Default'].curves[('default', 'base')].data_rate_curves #curve_ids



download_model.cf_scenarios['Base Case']
download_model.run_cash_flows(['Backtest'], False)
rollrates = download_model.cf_scenarios['Base Case'][1]._cf_modules['rate_curves'].cf_model_data['rate_arrays']['rollrate']
pd.DataFrame(rollrates[1, 1])

np.zeros([14,14])

download_model.data_tape.cutoff_tape


download_model.cf_scenarios['Base Case'][1]._model_config
download_model.cf_scenarios['Base Case'][1]._cf_data
download_model.cf_scenarios['Base Case'][1]._cf_modules['time'].cf_model_data
download_model.cf_scenarios['Base Case'][1]._cf_metrics
download_model.cf_scenarios['Base Case'][1].model_state_output['MonthsOnBook']


#self._cf_data[key][0] = np.nan #value

download_model.cf_scenarios['Base Case'][1]._cf_data['month_of_year'][0] = np.nan
download_model.cf_scenarios['Base Case'][1]._cf_data['month_of_year'][0].dtype
for key in download_model.cf_scenarios['Base Case'][1]._cf_data.keys():
    print(download_model.cf_scenarios['Base Case'][1]._cf_data[key].dtype)

download_model.cf_scenarios['Base Case'][1]._cf_data.keys()  'upb_trans_agg' 'PrincipalPartialPrepayment'
pd.DataFrame(download_model.cf_scenarios['Base Case'][1]._cf_data['upb_trans_agg'][1,1]) .shape
pd.DataFrame(np.sum(download_model.cf_scenarios['Base Case'][1]._cf_data['upb_trans_agg'][1,1], axis=0))

download_model.cf_scenarios['Base Case'][1]._cf_data['PrincipalPartialPrepayment'].shape
download_model.cf_scenarios['Base Case'][1]._cf_data['TotalPrincipalBalance'].shape
prepay = download_model.cf_scenarios['Base Case'][1]._cf_data['PrincipalPartialPrepayment']

download_model.cf_scenarios['Base Case'][1]._cf_data.keys()

download_model.eval.

pd.DataFrame(np.sum(download_model.cf_scenarios['Backtest'][1].cf_input_data['BOM_PrincipalBalance'], axis=0))
download_model.cf_scenarios['Base Case'][1].cf_input_data['TotalPrincipalBalance'].shape


prepay_sum = np.sum(prepay, axis=0)
prepay_sum.shape

download_model.cf_scenarios['Backtest'][1]._cf_data['AsOfDate'][0].shape

import ECL.RollRateConvert as rr_convert

ecl_rr = rr_convert.RollRateConvert(variance_step=6)
ecl_rr.calibrate_rolls(download_model.cf_scenarios['Backtest'][1])

test_frame = pd.DataFrame(index=download_model.cf_scenarios['Backtest'][1]._cf_data['AsOfDate'][0])
test_frame['co_target'] = ecl_rr.scenario_metrics['co_target']
test_frame['goal_seek_output'] = ecl_rr.co_output

test_frame.plot.line()
test_frame.sum()

po_frame = pd.DataFrame(index=download_model.cf_scenarios['Backtest'][1]._cf_data['AsOfDate'][0])
po_frame['po_target'] = ecl_rr.scenario_metrics['po_target']
po_frame['goal_seek_output'] = ecl_rr.po_output

po_frame.plot.line()
po_frame.sum()

pd.DataFrame(ecl_rr.full_roll_matrix[0])
pd.DataFrame(ecl_rr.forward_drift_matrix)
pd.DataFrame(ecl_rr.combined_mask)

ecl_rr.cf_metrics['eom_upb_sum'][:, 8]

ecl_rr.base_rolls

np.sum(ecl_rr.full_roll_matrix, axis=2)

np.sum(ecl_rr.full_roll_matrix[0], axis=0)
pd.DataFrame(ecl_rr.full_roll_matrix[0])
pd.DataFrame(ecl_rr.combined_mask)

1-np.sum(ecl_rr.full_roll_matrix, axis=2, where=ecl_rr.combined_mask[np.newaxis, :, :])


ecl_rr.rr_inputs_flatten()
ecl_rr.rr_inputs_reshape(ecl_rr.goal_seek_input)
len(inputs_reshape[3])
ecl_rr.po_drift_matrix.shape

ecl_rr.goal_seek_input
np.array_split(ecl_rr.goal_seek_input, [15,10, 20])

np.sum(ecl_rr.full_roll_matrix, axis=2)

ecl_rr.cf_metrics.keys()
ecl_rr.cf_metrics['C'].shape
ecl_rr.cf_metrics['ppmt'][1]

test_output = np.multiply(ecl_rr.cf_metrics['bom_upb'][0], ecl_rr.full_roll_matrix[0])
pd.DataFrame(ecl_rr.cf_metrics['bom_upb'][0])
pd.DataFrame(np.sum(test_output, axis=0))

pd.DataFrame(ecl_rr.full_roll_matrix[0])

ecl_rr.cf_metrics['eom_upb'][0].shape

ecl_rr.scenario

metric = download_model.cf_scenarios['Backtest'][1]._cf_data['upb_trans_agg']
np.nan_to_num(np.sum(metric[:, :, 13,1], axis=0))

curtail = download_model.cf_scenarios['Base Case'][1]._cf_data['PrincipalPartialPrepayment']

download_model.cf_scenarios['Base Case'][1]._cf_data['TotalPrincipalBalance'][0,0]
download_model.cf_scenarios['Base Case'][1].cf_input_data['BOM_PrincipalBalance'].shape

curtail.shape
metric.shape

def eom_status_sum(metric):
    #sum accounts
    metric_sum = np.sum(metric, axis=0)
    #sum begin status
    metric_sum = np.sum(metric_sum, axis=2)
    return metric_sum

upb_final = eom_status_sum(metric)
upb_final.shape


#target
scenario output 
    Status 8 = CO
    Status 12 + Curtail = PO
    
#month loop. 
#find total month end values
bom_principal*roll rates = upb_trans
upb_trans - contractualprincipalpayment = upb_final

loop output
    Status 8 CO
    Status 12 PO

#goal to minimize variance 
    abs(co_output-co_goal)
    abs(po_output-po_goal)
    


pd.DataFrame(metric_sum[4])

pd.DataFrame(metric_sum[1])
metric

eom_balance = np.sum(download_model.cf_scenarios['Base Case'][1]._cf_data['TotalPrincipalBalance'], axis=0)
eom_balance[4]

################################################################################
## new model test

galileo_test = cf_main.CashFlowModel.new_model('Galileo Test', [193], [], 4, 'solar')
galileo_test.download_model_template()


galileo_refresh = cf_main.CashFlowModel.refresh_model('202010 - Galileo 1 Refresh', 13, scenario_list=['Base Case'])
galileo_refresh.run_cash_flows()

galileo_refresh.eval.create_plot('ChargeOffAmount')

#Scenario with Mosaic Default curve
galileo_refresh.copy_curves('Mosaic Defaults', 'Galileo 1 - Default')
galileo_refresh.import_rate_curves_sql('Mosaic Defaults', model_id=11, scenario_name='Base Case', curve_type='default')
galileo_refresh.create_cf_scenario('Mosaic Defaults', cutoff_date='max', curve_group='Mosaic Defaults', model_config='Solar Asset Class')

#scenario with +50% default Stress
stress = {
    'default': [(0, 0.50), (12, 0.50), (24, 0.0)]
    }
galileo_refresh.create_cf_scenario('Default Stress +50%', cutoff_date='max', curve_group='Galileo 1 - Default', curve_stress=stress, model_config='Solar Asset Class')


galileo_refresh.run_cash_flows()

galileo_refresh.eval.create_plot('ChargeOffAmount', end_period=120)


g1_refresh = cf_main.CashFlowModel.new_model('202010 - Galileo 1 Refresh', [193], [], 3, 'solar')
g1_refresh.import_data_tape_sql()
g1_refresh.download_model_template()
g1_refresh.download_cf_scenario(13, scenario_list=['Base Case'])
g1_refresh.create_cf_scenario('Base Case', cutoff_date='max', curve_group='Galileo 1 - Default', model_config='Solar Asset Class')

g1_refresh.run_cash_flows()

g1_refresh.create_curve_group('mosaic defaults')
g1_refresh.copy_curves('mosaic defaults', 'Galileo 1 - Default')
g1_refresh.import_rate_curves_sql('mosaic defaults', model_id=8, scenario_name='Base Case', curve_type='default')

g1_refresh.rate_curve_groups['mosaic defaults'].curve_type_info
g1_refresh.create_cf_scenario('Mosaic Defaults', cutoff_date='max', curve_group='mosaic defaults', model_config='Solar Asset Class')

g1_refresh.run_cash_flows()
g1_refresh.cf_scenarios.keys()

g1_refresh.eval.create_plot('PrincipalFullPrepayment')






np.zeros



















#import curves manually
#create curve object
cf_model.create_curve_group('Solar Asset Class - 201909')

#file_path = r"\\SRV-NWUS-SHR\BA$\Credit Strategy\4. RUW\2. Deliverables\2019\201908 - Mosaic\Model\Mosaic RUW 201908 V11.xlsm"
file_path = r"\\SRV-NWUS-SHR\BA$\Credit Strategy\4. RUW\2. Deliverables\2019\201909 - Galileo 1\Model\Galileo 1 Solar RUW 201909 V5 (CPR Adj).xlsm"
cf_model.import_rate_curves_excel('Solar Asset Class', 'default', 'base', file_path, 'MDR', ws_range='C3:NF363', key_cols=['Key'])
cf_model.create_segment(use_gui=False, curve_group_name='Solar Asset Class', segment_type='default', OriginationTerm=[], OriginationCreditScore=[700,750,800]) 

cf_model.import_rate_curves_excel('Solar Asset Class', 'prepay', 'base', file_path, 'MPR', ws_range='C3:NF363', key_cols=['Key'])
cf_model.import_rate_curves_excel('Solar Asset Class', 'prepay', 'adjust', file_path, 'MPR', ws_range='ABQ3:APT363', key_cols=['Key'])
cf_model.create_segment(use_gui=False, curve_group_name='Solar Asset Class', segment_type='prepay', OriginationCreditScore=[700,750,800]) 

cf_model.import_rate_curves_excel('Solar Asset Class', 'curtail', 'base', file_path, 'MCR', ws_range='C3:NF363', key_cols=['Key'])
cf_model.import_rate_curves_excel('Solar Asset Class', 'curtail', 'adjust', file_path, 'MCR', ws_range='ABQ3:APT363', key_cols=['Key'])
cf_model.create_segment(use_gui=False, curve_group_name='Solar Asset Class', segment_type='curtail', OriginationTerm=[], OriginationCreditScore=[700,750,800], OriginationMonth=[]) 

#download curve test
cf_model.create_curve_group('Download Test')
cf_model.import_rate_curves_sql('Download Test', 1, 'Base Case')

#manually map any missing
cf_model.map_curves(curve_group_name='Solar Asset Class', manual_map=manual_map)
cf_model.map_curves(curve_group_name='Download Test', manual_map=manual_map)

#manually map any missing curves
cf_model.map_curves(use_gui=True)

#generate model
config={}
config['amort_formula'] = {}
config['amort_timing'] = {}

config['amort_timing']['promo'] = "['MonthsToAcquisition'] == 0" #(['MonthsOnBook'] < 18) & (['ProjectionMonth']==1)
#config['amort_formula']['promo'] = "np.pmt(['rate'],['OriginationTerm']-['PromoEndMOB'], ['OriginationBalance']-['PromoTargetBalance']))))" #"np.pmt(['rate'],['OriginationTerm'], (-['BOM_PrincipalBalance']*(1+['rate'])+np.pv(['rate'], ['PromoEndMOB'], 0, (['OriginationBalance']-['PromoTargetBalance']))))"
#config['amort_formula']['promo'] = "np.pmt(['rate'],['RemainingTerm']-['PromoTerm'], ['BOM_PrincipalBalance']-(['OriginationBalance']-['PromoTargetBalance']))"
#config['amort_formula']['promo'] = "np.pmt(['InterestRate'], ['OriginationTerm']-['PromoTerm'], ['BOM_PrincipalBalance']-(['OriginationBalance']-['PromoTargetBalance']))"
config['amort_formula']['promo'] = "np.pmt(['InterestRate'], ['OriginationTerm']-['PromoTerm'], ['PromoTargetBalance'])"


config['amort_timing']['promo end'] = "['MonthsOnBook'] == ['PromoTerm']"
config['amort_formula']['promo end'] = ''
config['default_amort_type'] = 'scale'

cf_model.create_model_config('Solar Asset Class', 1, config)

stress = {'default': [(0, 0.10), (12, 0.10), (24, 0)],
             'prepay': [(0, -0.25), (12, -0.25), (24, 0)],
             'curtail': [(0, -0.25), (12, -0.25), (24, 0)]
             }
stress_test = cf_model.create_curve_stress(stress)
stress_array = stress_test.return_stress_array(['curtail'], 301)
stress_array['curtail'][np.newaxis, :]

cf_model.cf_scenarios['Base Case'][1]._cf_modules['rate_curves'].cf_model_data['rate_arrays']['rollrate'].shape
cf_model.cf_scenarios['Base Case'][1]._cf_modules['rate_curves'].cf_model_data['cur_month_rates']['curtail']
cf_model.cf_scenarios['Base Case'][1]._cf_modules['payments'].cf_model_data.keys()
cf_model.cf_scenarios['Base Case'][1]._cf_modules['payments'].cf_model_data['ScheduledPaymentCalc'][500]
cf_model.cf_scenarios['Base Case'][1]._cf_modules['payments'].cf_model_data['ScheduledPaymentAmount'][500]
cf_model.cf_scenarios['Base Case'][1].cf_input_data['ScheduledPaymentAmount'][500]
cf_model.cf_scenarios['Base Case'][1]._cf_modules['payments'].cf_model_data['amort_rule_used'][500]
cf_model.cf_scenarios['Base Case'][1]._cf_modules['payments'].amort_timing
cf_model.cf_scenarios['Base Case'][1]._cf_modules['payments'].cf_input_data['ProjectionMonth']
cf_model.cf_scenarios['Base Case'][1]._cf_data['PrincipalPartialPrepayment'][0][1]

cf_model.cf_scenarios['Base Case'][1]._model_config.keys()

cf_model.cf_scenarios['Base Case'][1]._cf_modules['time'].cf_model_data['MonthsOnBook'][500]

cf_model.data_prep.model_configs['Solar Asset Class'].config_dict['amort_timing']


cf_model.create_cf_scenario(scenario_name='Base Case', cutoff_date='2019-09-30', curve_group='Solar Asset Class', model_config='Solar Asset Class')
cf_model.create_cf_scenario(scenario_name='Covid Stress', cutoff_date='2019-09-30', curve_group='Solar Asset Class', model_config='Solar Asset Class', curve_stress=stress)
cf_model.create_cf_scenario(scenario_name='Download Test', curve_group='Download Test', model_config=config) #cutoff_date='2019-09-30'



cf_model.run_cash_flows()

cf_model.cf_scenarios
cf_model.save_model('Base Case')

#eval output
cf_model.eval.create_plot('ChargeOffAmount')
cf_model.eval.create_table('ChargeOffAmount')




cf_model.data_prep.rate_curve_groups['Solar Asset Class'].curves[('default','base')].data_rate_curves
cf_model.data_prep.rate_curve_groups['Solar Asset Class'].__dict__.keys()
cf_model.data_prep.rate_curve_groups['Solar Asset Class'].curve_type_info    

segment_group.create_account_map()

segment_group.segment_account_map    
    

import DataPrep.DataMover as ETL
sql_engine = ETL.SQLEngine(db_name='FRA')
import sqlalchemy as sa








rows = [(2, 'curtail', 'base', 14, '10028 - Galileo Test - Base Case', 'MOB'),
 (2, 'curtail', 'adjust', 15, '10028 - Galileo Test - Base Case', 'MOB')]
cols = ['new_curve_group', 'curve_type', 'curve_sub_type', 'curve_set_key', 'curve_source', 'period_type']

pd.DataFrame(rows, columns=cols)


cf_model.cf_scenarios['Covid Stress']._cf_modules['payments']._cf_input_fields
cf_model.cf_scenarios['Base Case']._cf_modules['payments'].cf_model_data['ScheduledPaymentCalc'][400]
cf_model.cf_scenarios['Base Case']._cf_modules['payments'].cf_model_data['ScheduledPaymentAmount'][400]
cf_model.cf_scenarios['Base Case']._cf_modules['payments'].cf_model_data['TotalPaymentMade'][400]
cf_model.cf_scenarios['Base Case']._cf_modules['payments'].cf_model_data['sch_pmt_trans'][400]
pd.DataFrame(cf_model.cf_scenarios['Base Case']._cf_modules['balance'].cf_model_data['upb_trans_agg'][0])

cf_model.cf_scenarios['Base Case']._model_config.keys()


cf_model.cf_scenarios['Base Case']._cf_data.keys()
pd.DataFrame(cf_model.cf_scenarios['Base Case']._cf_data['ScheduledPaymentCalc'][400])
pd.DataFrame(cf_model.cf_scenarios['Base Case']._cf_data['ScheduledPaymentAmount'][400])
cf_model.cf_scenarios['Base Case']._cf_data['TotalPaymentMade'][400,2]
pd.DataFrame(cf_model.cf_scenarios['Base Case']._cf_data['sch_pmt_trans'][400,1])
pd.DataFrame(cf_model.cf_scenarios['Base Case']._cf_data['amort_rule_used'][400])
pd.DataFrame(cf_model.cf_scenarios['Base Case']._cf_data['rate'][400])
pd.DataFrame(cf_model.cf_scenarios['Base Case']._cf_data['upb_trans_agg'][400,2])
cf_model.cf_scenarios['Base Case']._cf_data['sch_pmt_trans']

pd.DataFrame(cf_model.cf_scenarios['Base Case']._cf_data['BOM_PrincipalBalance'][400,0])
cf_model.cf_scenarios['Base Case']._cf_data['MonthsToAcquisition'][400]

cf_model.cf_scenarios['Base Case'].cf_input_data['ScheduledPaymentAmount'][400]
cf_model.cf_scenarios['Base Case'].cf_input_data['OriginationBalance'][400]
cf_model.cf_scenarios['Base Case'].cf_input_data['PromoTargetBalance'][400]
cf_model.cf_scenarios['Base Case'].cf_input_data['OriginationTerm'][400]
cf_model.cf_scenarios['Base Case'].cf_input_data['RemainingTerm'][400]
cf_model.cf_scenarios['Base Case'].cf_input_data['InterestRate'][400]

cf_model.cf_scenarios['Base Case']._cf_data['MonthsOnBook'][:,0]

cf_model.data_prep.prior_uw_projections.head()
cf_model.eval.cf_output.index.names
cf_model.eval.comb_index_names


cf_model.save_model()
json.dumps(cf_model.model_config)

#aggregate results

cf_model.data_prep.data_tape.raw_tape.columns

cf_model.cf_scenarios['Base Case']._cf_modules['payments'].amort_formula
cf_model.cf_scenarios['Base Case']._cf_modules['payments'].amort_formula

np.where(cf_model.cf_scenarios['Base Case']._cf_data['BOM_PrincipalBalance'][:,200,13]>0)

cf_model.cf_scenarios['Base Case']._cf_data['MonthsToAcquisition'][:,0]
cf_model.cf_scenarios['Base Case']._cf_data['MonthsOnBook'][:,0]
cf_model.cf_scenarios['Base Case']._cf_data['BOM_PrincipalBalance'][:,4,13]
cf_model.cf_scenarios['Base Case']._cf_data['TotalPrincipalBalance'][:,1,13]
cf_model.cf_scenarios['Base Case']._cf_data['ScheduledPaymentAmount'][0,17,:]
pd.DataFrame(np.sum(cf_model.cf_scenarios['Base Case']._cf_data['sch_pmt_trans'][:,10,:,:], axis=1))


cf_model.cf_scenarios['Base Case'][1]._cf_modules['rate_curves'].cf_model_data['rate_arrays']['curtail']
pd.DataFrame(cf_model.cf_scenarios['Base Case']._cf_modules['rate_curves'].cf_model_data['cur_month_rates']['rollrate'][0])
pd.DataFrame(cf_model.cf_scenarios['Base Case']._cf_data['bom_final_trans'][0]) , 200]



cf_model.cf_scenarios['Base Case']._cf_data['TotalPrincipalBalance'][987][3]
cf_model.cf_scenarios['Base Case']._cf_data['MonthsOnBook'][972][1]
cf_model.cf_scenarios['Base Case']._cf_modules['time'].cf_model_data['ProjectionMonth']

cf_model.cf_scenarios['Base Case']._cf_data['AsOfDate']
cf_model.cf_scenarios['Base Case']._cf_data['eom_upb'][972][4]
pd.DataFrame(cf_model.cf_scenarios['Base Case']._cf_modules['roll_rates'].cf_model_data['rate_arrays']['transition'][987][1])
cf_model.cf_scenarios['Base Case']._cf_modules['time'].cf_model_data['MonthsOnBook'][987]

np.arange(len(cf_model.cf_scenarios['Base Case']._cf_data['BOM_PrincipalBalance']))

pd.DataFrame(cf_model.cf_scenarios['Base Case']._cf_data['sch_pmt'][972])
cf_model.cf_scenarios['Base Case']._cf_data['sch_pmt_trans'][972][3][1]
cf_model.cf_scenarios['Base Case']._cf_data['upb_trans'][972][3]
cf_model.cf_scenarios['Base Case']._cf_data['upb_trans_agg'][972][3]

cf_model.cf_scenarios['Base Case']._cf_data['int_accrue'][972][4]
pd.DataFrame(cf_model.cf_scenarios['Base Case']._cf_data['bom_units'][972])
cf_model.cf_scenarios['Base Case']._cf_data['pmt_made'][972][4]
cf_model.cf_scenarios['Base Case']._cf_data['curtail'][972][3]
cf_model.cf_scenarios['Base Case']._cf_data['rate'][972][2]
cf_model.cf_scenarios['Base Case']._cf_data.keys()

cf_model.cf_scenarios['Base Case']._cf_modules['payments'].amort_timing
cf_model.cf_scenarios['Base Case']._cf_modules['payments'].pmt_matrix
pd.DataFrame(cf_model.cf_scenarios['Base Case']._cf_data['amort_rule_used'][972])


cf_model.cf_scenarios['Base Case']._cf_data.keys()

cf_model.cf_scenarios['Base Case']._model_config['data_tape'].iloc[987]
cf_model.cf_scenarios['Base Case'].cf_input_data['MonthsOnBook'][987]
cf_model.cf_scenarios['Base Case']._cf_modules['time'].cf_model_data['MonthsOnBook'][987]
cf_model.cf_scenarios['Base Case']._cf_modules['time'].cf_model_data['month_to_acq'][987]

rule_df = cf_model.data_prep.segment_rules_combined
cf_model.data_prep.segment_input

import numpy as np
np.where(cf_model.cf_scenarios['Base Case'].cf_input_data['MonthsOnBook']<0)
min(cf_model.cf_scenarios['Base Case'].cf_input_data['MonthsOnBook'])<0
cf_model.cf_scenarios['Base Case']._model_config['rate_curves']
pd.DataFrame(cf_model.cf_scenarios['Base Case']._cf_modules['roll_rates'].cf_model_data['rate_arrays']['transition'][972,-1,:,:])
cf_model.cf_scenarios['Base Case']._cf_modules['roll_rates'].cf_model_data['rate_arrays']['transition'][:,-1,:,:]

test_range = list(range(0,2+2))



#cheat for now
manual_map = {
    'default': {'120|700-750': '120|700-750',
  '120|750-800': '120|750-800',
  '120|<700': '120|<700',
  '180|700-750': '180|700-750',
  '180|750-800': '180|750-800',
  '180|<700': '180|<700',
  '240|700-750': '240|700-750',
  '240|750-800': '240|750-800',
  '240|<700': '240|<700',
  '300|700-750': '300|700-750',
  '300|750-800': '300|750-800',
  '300|<700': '300|<700',
  '120|>800': '120|800-850',
  '144|>800': '144|800-850',
  '180|>800': '180|800-850',
  '240|>800': '240|800-850',
  '300|>800': '300|800-850'},
 'prepay': {'700-750': '700-750',
  '750-800': '750-800',
  '<700': '<700',
  '>800': '800-850'},
 'curtail': {'120|700-750|1': '120|700-750|1',
  '120|700-750|10': '120|700-750|10',
  '120|700-750|11': '120|700-750|11',
  '120|700-750|12': '120|700-750|12',
  '120|700-750|2': '120|700-750|2',
  '120|700-750|3': '120|700-750|3',
  '120|700-750|4': '120|700-750|4',
  '120|700-750|5': '120|700-750|5',
  '120|700-750|6': '120|700-750|6',
  '120|700-750|7': '120|700-750|7',
  '120|700-750|8': '120|700-750|8',
  '120|700-750|9': '120|700-750|9',
  '120|750-800|1': '120|750-800|1',
  '120|750-800|10': '120|750-800|10',
  '120|750-800|11': '120|750-800|11',
  '120|750-800|12': '120|750-800|12',
  '120|750-800|4': '120|750-800|4',
  '120|750-800|5': '120|750-800|5',
  '120|750-800|6': '120|750-800|6',
  '120|750-800|7': '120|750-800|7',
  '120|750-800|8': '120|750-800|8',
  '120|750-800|9': '120|750-800|9',
  '120|<700|1': '120|<700|1',
  '120|<700|10': '120|<700|10',
  '120|<700|11': '120|<700|11',
  '120|<700|12': '120|<700|12',
  '120|<700|2': '120|<700|2',
  '120|<700|3': '120|<700|3',
  '120|<700|4': '120|<700|4',
  '120|<700|5': '120|<700|5',
  '120|<700|6': '120|<700|6',
  '120|<700|7': '120|<700|7',
  '120|<700|8': '120|<700|8',
  '120|<700|9': '120|<700|9',
  '180|700-750|1': '180|700-750|1',
  '180|700-750|10': '180|700-750|10',
  '180|700-750|11': '180|700-750|11',
  '180|700-750|12': '180|700-750|12',
  '180|700-750|2': '180|700-750|2',
  '180|700-750|3': '180|700-750|3',
  '180|700-750|4': '180|700-750|4',
  '180|700-750|5': '180|700-750|5',
  '180|700-750|6': '180|700-750|6',
  '180|700-750|7': '180|700-750|7',
  '180|700-750|8': '180|700-750|8',
  '180|700-750|9': '180|700-750|9',
  '180|750-800|1': '180|750-800|1',
  '180|750-800|10': '180|750-800|10',
  '180|750-800|11': '180|750-800|11',
  '180|750-800|12': '180|750-800|12',
  '180|750-800|4': '180|750-800|4',
  '180|750-800|5': '180|750-800|5',
  '180|750-800|6': '180|750-800|6',
  '180|750-800|7': '180|750-800|7',
  '180|750-800|8': '180|750-800|8',
  '180|750-800|9': '180|750-800|9',
  '180|<700|1': '180|<700|1',
  '180|<700|10': '180|<700|10',
  '180|<700|11': '180|<700|11',
  '180|<700|12': '180|<700|12',
  '180|<700|2': '180|<700|2',
  '180|<700|3': '180|<700|3',
  '180|<700|4': '180|<700|4',
  '180|<700|5': '180|<700|5',
  '180|<700|6': '180|<700|6',
  '180|<700|7': '180|<700|7',
  '180|<700|8': '180|<700|8',
  '180|<700|9': '180|<700|9',
  '240|700-750|1': '240|700-750|1',
  '240|700-750|10': '240|700-750|10',
  '240|700-750|11': '240|700-750|11',
  '240|700-750|12': '240|700-750|12',
  '240|700-750|2': '240|700-750|2',
  '240|700-750|3': '240|700-750|3',
  '240|700-750|4': '240|700-750|4',
  '240|700-750|5': '240|700-750|5',
  '240|700-750|6': '240|700-750|6',
  '240|700-750|7': '240|700-750|7',
  '240|700-750|8': '240|700-750|8',
  '240|700-750|9': '240|700-750|9',
  '240|750-800|1': '240|750-800|1',
  '240|750-800|10': '240|750-800|10',
  '240|750-800|11': '240|750-800|11',
  '240|750-800|12': '240|750-800|12',
  '240|750-800|4': '240|750-800|4',
  '240|750-800|5': '240|750-800|5',
  '240|750-800|6': '240|750-800|6',
  '240|750-800|7': '240|750-800|7',
  '240|750-800|8': '240|750-800|8',
  '240|750-800|9': '240|750-800|9',
  '240|<700|1': '240|<700|1',
  '240|<700|10': '240|<700|10',
  '240|<700|11': '240|<700|11',
  '240|<700|12': '240|<700|12',
  '240|<700|2': '240|<700|2',
  '240|<700|3': '240|<700|3',
  '240|<700|4': '240|<700|4',
  '240|<700|5': '240|<700|5',
  '240|<700|6': '240|<700|6',
  '240|<700|7': '240|<700|7',
  '240|<700|8': '240|<700|8',
  '240|<700|9': '240|<700|9',
  '300|700-750|1': '300|700-750|1',
  '300|700-750|10': '300|700-750|10',
  '300|700-750|11': '300|700-750|11',
  '300|700-750|12': '300|700-750|12',
  '300|700-750|2': '300|700-750|2',
  '300|700-750|3': '300|700-750|3',
  '300|700-750|4': '300|700-750|4',
  '300|700-750|5': '300|700-750|5',
  '300|700-750|6': '300|700-750|6',
  '300|700-750|7': '300|700-750|7',
  '300|700-750|8': '300|700-750|8',
  '300|700-750|9': '300|700-750|9',
  '300|750-800|1': '300|750-800|1',
  '300|750-800|10': '300|750-800|10',
  '300|750-800|11': '300|750-800|11',
  '300|750-800|12': '300|750-800|12',
  '300|750-800|2': '300|750-800|2',
  '300|750-800|3': '300|750-800|3',
  '300|750-800|4': '300|750-800|4',
  '300|750-800|5': '300|750-800|5',
  '300|750-800|6': '300|750-800|6',
  '300|750-800|7': '300|750-800|7',
  '300|750-800|8': '300|750-800|8',
  '300|750-800|9': '300|750-800|9',
  '300|<700|1': '300|<700|1',
  '300|<700|10': '300|<700|10',
  '300|<700|11': '300|<700|11',
  '300|<700|12': '300|<700|12',
  '300|<700|2': '300|<700|2',
  '300|<700|3': '300|<700|3',
  '300|<700|4': '300|<700|4',
  '300|<700|5': '300|<700|5',
  '300|<700|6': '300|<700|6',
  '300|<700|7': '300|<700|7',
  '300|<700|8': '300|<700|8',
  '300|<700|9': '300|<700|9',
  '120|>800|1': '120|800-850|1',
  '120|>800|10': '120|800-850|10',
  '120|>800|11': '120|800-850|11',
  '120|>800|12': '120|800-850|12',
  '120|>800|2': '120|800-850|2',
  '120|>800|3': '120|800-850|3',
  '120|>800|4': '120|800-850|4',
  '120|>800|5': '120|800-850|5',
  '120|>800|6': '120|800-850|6',
  '120|>800|7': '120|800-850|6',
  '120|>800|8': '120|800-850|8',
  '120|>800|9': '120|800-850|9',
  '144|>800|1': '144|800-850|1',
  '144|>800|10': '144|800-850|10',
  '144|>800|11': '144|800-850|11',
  '144|>800|12': '144|800-850|12',
  '144|>800|2': '144|800-850|2',
  '144|>800|3': '144|800-850|3',
  '144|>800|4': '144|800-850|4',
  '144|>800|5': '144|800-850|5',
  '144|>800|6': '144|800-850|6',
  '144|>800|7': '144|800-850|6',
  '144|>800|8': '144|800-850|8',
  '144|>800|9': '144|800-850|9',
  '180|>800|1': '180|800-850|1',
  '180|>800|10': '180|800-850|10',
  '180|>800|11': '180|750-800|11',
  '180|>800|12': '180|800-850|12',
  '180|>800|2': '180|800-850|2',
  '180|>800|3': '180|800-850|3',
  '180|>800|4': '180|800-850|4',
  '180|>800|5': '180|800-850|5',
  '180|>800|6': '180|800-850|6',
  '180|>800|7': '180|800-850|7',
  '180|>800|8': '180|800-850|8',
  '180|>800|9': '180|800-850|9',
  '240|>800|1': '240|800-850|1',
  '240|>800|10': '240|800-850|10',
  '240|>800|11': '240|800-850|11',
  '240|>800|12': '240|800-850|12',
  '240|>800|2': '240|800-850|2',
  '240|>800|3': '240|800-850|3',
  '240|>800|4': '240|800-850|4',
  '240|>800|5': '240|800-850|5',
  '240|>800|6': '240|800-850|6',
  '240|>800|7': '240|800-850|7',
  '240|>800|8': '240|800-850|8',
  '240|>800|9': '240|800-850|9',
  '300|>800|1': '300|800-850|1',
  '300|>800|10': '300|800-850|10',
  '300|>800|11': '300|800-850|11',
  '300|>800|12': '300|800-850|12',
  '300|>800|4': '300|800-850|4',
  '300|>800|5': '300|800-850|5',
  '300|>800|6': '300|800-850|6',
  '300|>800|7': '300|800-850|7',
  '300|>800|8': '300|800-850|8',
  '300|>800|9': '300|800-850|9'},
 'recovery': {},
 'collection': {}}









