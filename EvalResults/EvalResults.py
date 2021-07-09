# -*- coding: utf-8 -*-
"""
Created on Wed May 13 20:56:21 2020

@author: jalbert
"""

import numpy as np
import pandas as pd
import os
import pathlib
import io
import openpyxl as xl
import sys

import importlib.resources as pkg_resources
import tempfile
import atexit
import shutil
import Templates as templates

class AVP(object):
    
    def __init__(self, scenarios):
        self.scenarios = scenarios
        #self.data_tape = data_tape
        self.single_account=False
        
        self.metric_list=['AccountStatusCode','BOM_PrincipalBalance','InterestBalance', 'TotalPrincipalBalance','ScheduledPaymentAmount'
                          ,'ScheduledPaymentMade', 'TotalPaymentMade', 'ContractualPrincipalPayment','InterestPayment','PrincipalPartialPrepayment'
                          ,'PrincipalFullPrepayment', 'ChargeOffAmount', 'eom_units', 'int_accrue'] #
        
        self.other_metrics=['PostChargeOffCollections', 'hist_defaults']
        
        self.column_map = {
                    #'as_of_date':'AsOfDate'
                    #'bom_upb':'BOM_PrincipalBalance'
                    #'eom_upb':'TotalPrincipalBalance'
                    #'eom_int':'InterestBalance'
                    #,'sch_pmt':'ScheduledPaymentAmount'
                    #,'tot_sch_pmt':'tot_sch_pmt'
                    #,'TotalPaymentMade':'TotalPaymentMade'
                    #,'ipmt':'sch_int_pmt'
                    #,'ppmt':'sch_prin_pmt'
                    'default':'ChargeOffAmount'
                    ,'curtail':'PrincipalPartialPrepayment'
                    ,'prepay':'PrincipalFullPrepayment'
                    ,'recovery':'PostChargeOffCollections'
                    ,'bk':'Bankruptcy'
                    }
        
        self.cf_output_status = pd.DataFrame()
        self.cf_output = pd.DataFrame()
        self.account_status_list=[]
        self.account_status_active=[]
        
        self.line_colors = {'Actuals': 'black', 'Original Pricing': 'gray', 'Current RUW': 'steelblue'}
        self.line_style = {'Actuals': '-', 'Original Pricing': '-', 'Current RUW': '-'}
        
        #empty temp file dict and register cleanup function
        sysTemp = tempfile.gettempdir()
        self.temp_folder = os.path.join(sysTemp,'cf_hvp')
        #tempfile.mkdtemp(dir=self.temp_folder)
        #check if temp folder exists
        if not os.path.exists(self.temp_folder):
            #create directory
            os.makedirs(self.temp_folder)
        
        #close and delete files
        for file in os.listdir(self.temp_folder):
            file_path = os.path.join(self.temp_folder, file)
            try:
                #file.close()
                os.remove(file_path)
            except:
                pass
            
        self.temp_files=[]
        atexit.register(self.delete_temp_files, self.temp_files)
        
    def output_actuals(self, data_tape_object, metric_list=None, period_type='calendar', account=False, batch=True, segment=False, account_id=None):
        #copy datatape to not alter original
        data_tape = data_tape_object.raw_tape.copy()
        scenario_name='Actuals'
        if account_id:
            filter_ix = data_tape['AccountID']==account_id
            data_tape = data_tape.loc[filter_ix]
            #scenario_name=str(account_id)
        
        self.account_status_list = data_tape_object.account_status_list
        self.account_status_active = data_tape_object.account_status_active
        
        #self.drop_scenario('Actuals')
        self.drop_scenario(scenario_name)
        
        if not metric_list:
            metric_list = metric_list = ['BOM_PrincipalBalance','InterestBalance', 'TotalPrincipalBalance','ScheduledPaymentAmount', 'ScheduledPaymentMade', 'TotalPaymentMade', 'ContractualPrincipalPayment','InterestPayment','PrincipalPartialPrepayment','PrincipalFullPrepayment', 'PostChargeOffCollections']
        
        #reset index if needed
        if not pd.RangeIndex(stop=(len(data_tape))).equals(data_tape.index):
            data_tape.reset_index(inplace=True)
        
        #set new index
        if account:
            self.index_set('AccountKey', data_tape)
        if batch:
            self.index_set('BatchKey', data_tape)
        if segment:
            self.index_set('segment_group_id', data_tape)
        if period_type=='calendar':
            self.index_set('AsOfDate', data_tape)
        else:
            self.index_set('MonthsOnBook', data_tape)
            
        #add the statuses
        self.index_set('AccountStatusCode', data_tape)
        #add scenario
        #data_tape['Scenario']='Actuals'
        data_tape['Scenario'] = scenario_name
        data_tape.set_index('Scenario', append=True, inplace=True)
        
        #group metris to index
        dt_grouped = data_tape[metric_list].groupby(data_tape.index.names, axis=0).sum()
        
        #reconfigure DF, uppivot metrics into rows, then pivot status into columns
        dt_pivot=pd.melt(dt_grouped.reset_index(), id_vars=dt_grouped.index.names, var_name='Metric')
        index_name=list(dt_grouped.index.names)
        index_name.append('Metric')
        dt_pivot.set_index(index_name, inplace=True)
        dt_unstack = dt_pivot.unstack(level='AccountStatusCode', fill_value=0)
        dt_unstack.columns= dt_unstack.columns.droplevel(0)
        
        #rename columns to plain text
        dt_unstack.rename(columns=self.account_status_list, inplace=True)
                
        if len(self.cf_output_status)==0:
            self.cf_output_status=dt_unstack
        else:
            
            self.cf_output_status.drop('actuals',level='Scenario',inplace=True, errors='ignore')    
            self.cf_output_status = self.cf_output_status.append(dt_unstack)
            
        # create combined output
        #self.output_combined('Actuals')
        self.output_combined(scenario_name)
        
    def output_single_account(self, scenario, account_id):
        """
        output data for a single account only
        """
        self.single_account=True
        
        
        
    def output_hist_proj(self, data_tape):
        """
        Reshape and store metrics for prior projections (Original Pricing, Prior RUW, etc.)
        """
        #unpivot metric columns into rows
        metric_cols = [c for c in data_tape.columns if c not in ['ProjectionName','LoadDate'] and not c.startswith('check_')]
        df_unpivot = pd.melt(data_tape[metric_cols], id_vars=['Scenario','BatchKey','AsOfDate'], var_name='Metric',value_name='Value')
        #df_unpivot.rename(columns={'Scenario':'scenario'}, inplace=True)
        df_unpivot.set_index(['BatchKey','AsOfDate','Scenario','Metric'], inplace=True)
        df_unpivot = df_unpivot.astype('float64')

        if len(self.cf_output)==0:
            self.cf_output = df_unpivot
        else:
            scenario_list = list(df_unpivot.index.get_level_values('Scenario').unique())
            #self.cf_output.drop(scenario_list, axis=0, level='Scenario', errors='ignore', inplace=True)
            self.drop_scenario(scenario_list)
            self.cf_output = self.cf_output.append(df_unpivot)
            self.cf_output.sort_index(inplace=True)
        #return df_unpivot
    
    def output_proj_all(self, scenario):
        """
        Aggregates all final cf metrics to monthly totals.
        
        Parameters
        ==================
        scenario: str
            scenario name to process
        """
        
        scenario_data = self.scenarios[scenario][1]
        
        self.account_status_list = scenario_data._model_config['account_status_list']
        self.account_status_active = scenario_data._model_config['account_status_active']
        #drop any current entries for this scenario
        if len(self.cf_output_status)>0:
            #self.cf_output_status.drop(scenario, level='Scenario', inplace=True, errors='ignore')
            self.drop_scenario(scenario)
        
        #get array shapes for this scenario
        self.array_shape = scenario_data._cf_data['MonthsOnBook'].shape
        
        #create generic index
        self.row_index = self.create_index(scenario)
        
        #loop through each metric
        metric_dfs = []
        for metric in self.metric_list:    
            metric_single = self.proj_by_status(scenario, metric)
            metric_dfs.append(metric_single) if metric_single is not None else None

        metrics_df_combined = pd.concat(metric_dfs)
        
        #append metric to df 
        if len(self.cf_output_status)==0:
            self.cf_output_status=metrics_df_combined
        else:
            self.cf_output_status.drop(scenario, level='Scenario', inplace=True, errors='ignore')
            self.cf_output_status = self.cf_output_status.append(metrics_df_combined)
            self.cf_output_status.sort_index(inplace=True)
            
        #consolidate status into one output
        self.output_combined(scenario)
        
        #recoveries
        if scenario_data._model_config.get('PostChargeOffCollections'):
            self.proj_recovery(scenario)
    
    def index_set(self, column, data_tape):
        #if index is the default range then replace
        if pd.RangeIndex(stop=(len(data_tape))).equals(data_tape.index):
            data_tape.set_index(column, append=False, inplace=True)
        #if index exists then append
        else:
            data_tape.set_index(column, append=True, inplace=True)
    
    def drop_scenario(self, scenario):
        
        if len(self.cf_output_status)>0:
            self.cf_output_status.drop(scenario, axis=0, level='Scenario', errors='ignore', inplace=True)
        if len(self.cf_output)>0:
            self.cf_output.drop(scenario, axis=0, level='Scenario', errors='ignore', inplace=True)
    
       
    
    def create_index(self, scenario, period_type='calendar', account=False, batch=True, segment=False):
        index_list = []
        index_name = []
        
        scenario_data = self.scenarios[scenario][1]
        
        if account:
            index_list.append(np.repeat(scenario_data._model_config['data_tape'].index.values, self.array_shape[1]))
            index_name.append('AccountKey')
        if batch:
            index_list.append(np.repeat(scenario_data._model_config['data_tape']['BatchKey'].values, self.array_shape[1]))
            index_name.append('BatchKey')
        if segment:
            index_list.append(np.repeat(scenario_data._model_config['data_tape']['segment_group_id'].values, self.array_shape[1]))
            index_name.append('segment_group_id')
        if period_type=='calendar':
            #as of date
            index_list.append(scenario_data._cf_data['AsOfDate'].flatten())
            index_name.append('AsOfDate')
        else:
            #months on book
            index_list.append(scenario_data._cf_data['MonthsOnBook'].flatten())
            index_name.append('MonthsOnBook')
        
        return pd.MultiIndex.from_arrays(index_list, names=index_name)
  
    def proj_by_status(self, scenario, metric): # row_names, column_names
        """
        Takes 3D/4D array input and converts into a dataframe with a multiindex. 
        Dataframe will have account/month/metric as rows and Account status as columns
        """
        
        scenario_data = self.scenarios[scenario][1]
        
        #if metric not found just exit
        try:
            array = scenario_data._cf_data[metric]
        except:
            return
            
        input_shape=array.shape
        #stack loan ids and date
        if len(input_shape)==3:
            array_reshape=np.reshape(array, (input_shape[0]*input_shape[1], input_shape[2]))
        elif len(input_shape)==4:
            array = np.sum(array, axis=2)
            array_reshape = np.reshape(array, (input_shape[0]*input_shape[1], input_shape[2]))
        
        #metrics by status
        output_by_status = pd.DataFrame(data=array_reshape, index=self.row_index).groupby(by=list(self.row_index.names)).sum()
        #output_by_status.columns = output_by_status.columns.astype(str)
        
        #rename columns to plain text
        output_by_status.rename(columns=self.account_status_list, inplace=True)
        
        #add scenario and metric to index
        output_by_status['Scenario']=scenario
        output_by_status['Metric']=self.column_map.get(metric,metric)
        output_by_status.set_index(['Scenario','Metric'], append=True, inplace=True)
        
        return output_by_status
    
    def proj_4d_array(self, scenario, input_array, output_metric_nm, from_status=None, to_status=None):
        """
        Converts 4D array into 2D output with indexes. Output returns one single column
        """
        scenario_data = self.scenarios[scenario][1]
        
        #if metric not found exit
        try: 
            array = scenario_data._cf_data[input_array]
        except:
            return
        
        if not from_status:
            from_status=list(self.account_status_list.keys())
        if not to_status:
            to_status=list(self.account_status_list.keys())
            
        #convert input to array
        from_status = np.array(from_status)
        to_status = np.array(to_status)
        
        #sum over axis #2 and return result
        sum_array = array[:,:,from_status[:, np.newaxis],to_status].sum(axis=2)
        
        input_shape=sum_array.shape
        array_reshape= np.reshape(sum_array, (input_shape[0]*input_shape[1])) #(input_shape[0]*input_shape[1], input_shape[2])
        metric_output = pd.DataFrame(data=array_reshape, columns=['Value'], index=self.row_index).groupby(by=list(self.row_index.names)).sum()
        metric_output['Metric']=output_metric_nm
        metric_output['Scenario']=scenario
        metric_output.set_index(['Scenario','Metric'], append=True, inplace=True)
        
        return metric_output
    
    def proj_recovery(self, scenario):
        """
        Aggregates expected recoveries from prior defaults & projected defaults
        
        Prior Defaults: recovery curve is applied to historical defaults and then truncated up to the CutOffDate, so we only have future projected amount
        
        Projected Defaults: sum up recovery amounts in diagonals. Model produces entire recovery curve against
        each default month. To sum up totals you have to sum in a diagonal offset.

        Parameters
        ============================
        scenario : str
            scenario to process

        """
        
        scenario_data = self.scenarios[scenario][1]
        
        projection_periods = scenario_data._model_config['projection_periods']
        
        #historical defaults
        hist_default = scenario_data._cf_data.get('hist_default')
        hist_recovery = scenario_data._cf_data.get('hist_recovery')
        hist_recovery = hist_recovery[:, :projection_periods+1]
        #hist_default index
        
        
        index_list = []
        index_name = []
        index_list.append(np.repeat(hist_default.index.get_level_values('BatchKey').values, projection_periods+1))
        index_name.append('BatchKey')
        index_list.append(np.tile(scenario_data._cf_data['AsOfDate'][0], len(hist_default)))
        index_name.append('AsOfDate')
        hist_recovery_index = pd.MultiIndex.from_arrays(index_list, names=index_name)
        
        hist_recovery_df = pd.DataFrame(data = hist_recovery.flatten(), index=hist_recovery_index, columns=['Value']).groupby(by=list(hist_recovery_index.names)).sum() #.rename('Value')
        #hist_recovery_df['Scenario'] = scenario
        #hist_recovery_df['Metric'] = 'Historical ChargeOff Recovery'
        #hist_recovery_df.set_index(['Scenario'], append=True, inplace=True)
        
        #projected defaults
        proj_recovery_raw =  scenario_data._cf_data.get('PostChargeOffCollections')
        proj_recovery_raw = np.nan_to_num(proj_recovery_raw)
        
        #trace only works left to right, so we need to flip array then start with offsets on right
        proj_recovery_flip = np.fliplr(proj_recovery_raw)
        
        #have to do trace in a loop. no way around it. trace only does one diagonal at a time
        recovery_sum = [np.trace(proj_recovery_flip, offset=i, axis1=1, axis2=2) for i in range(-(proj_recovery_flip.shape[1]-1), proj_recovery_flip.shape[2]-1)]
        recovery_array = np.array(recovery_sum)
        
        recovery_swap = np.swapaxes(recovery_array, 1, 0)
        #cutoff recoveries at number of projection months
        recovery_swap = recovery_swap[:, :projection_periods+1]
        #create index and sum 
        proj_recovery_df = pd.DataFrame(data = recovery_swap.flatten(), index=self.row_index, columns=['Value']).groupby(by=list(self.row_index.names)).sum() #.rename('Value')
        #proj_recovery_df['Metric'] = 'Projected ChargeOff Recovery'
        #proj_recovery_df.set_index(['Scenario'], append=True, inplace=True)
        
        #combine and sum into one final output
        combined_recovery_df = pd.concat([hist_recovery_df, proj_recovery_df]).groupby(by=list(self.row_index.names)).sum()
        combined_recovery_df.sort_index(inplace=True)
        
        #add metric name
        hist_recovery_df['Scenario'] = scenario
        hist_recovery_df['Metric'] = 'Historical ChargeOff Recovery'
        hist_recovery_df.set_index(['Scenario', 'Metric'], append=True, inplace=True)
        proj_recovery_df['Scenario'] = scenario
        proj_recovery_df['Metric'] = 'Projected ChargeOff Recovery'
        proj_recovery_df.set_index(['Scenario', 'Metric'], append=True, inplace=True)
        combined_recovery_df['Scenario'] = scenario
        combined_recovery_df['Metric'] = 'PostChargeOffCollections'
        combined_recovery_df.set_index(['Scenario', 'Metric'], append=True, inplace=True)
        
        
        #load to final DF
        self.cf_output = self.cf_output.append(hist_recovery_df)
        self.cf_output = self.cf_output.append(proj_recovery_df)
        self.cf_output = self.cf_output.append(combined_recovery_df)
        self.cf_output.sort_index(inplace=True)
        
        #return proj_recovery_df
        
    def output_combined(self, scenario):
        """
        Aggregates the metrics by status and sums to one total for each period.
        
        Parameters
        =============
        scenario: str
            scenario to process
        """
       
        active = self.account_status_active
        status = self.account_status_list
        status_swap = {v:k for k,v in status.items()}
        
        col_list = self.cf_output_status.columns.values
        active_status = [v for k, v in status.items() if active[k]==1 and v in col_list]
        
        #active status only matters for certail metrics. add logic to only apply on those rules
        metrics_balance = ['BOM_PrincipalBalance','TotalPrincipalBalance','InterestBalance','eom_units', 'ScheduledPaymentAmount']
        metrics_other = list(self.cf_output_status.index.unique('Metric'))
        metrics_other = [x for x in metrics_other if x not in metrics_balance]
        
        #make copy of selected scenario
        scenario_raw = self.cf_output_status.xs(scenario, level='Scenario', drop_level=False).sort_index().copy()
        
        #active status only for balances. for model projections 
        idx = pd.IndexSlice
        scenario_balance = scenario_raw.loc[idx[:,:,:,metrics_balance], idx[:]][active_status].sum(axis=1).rename('Value')#. rename(scenario)
        #sum all for other metrics
        scenario_other = scenario_raw.loc[idx[:,:,:,metrics_other], idx[:]].sum(axis=1).rename('Value') #.rename(scenario)
        
        #combine back into one df
        scenario_final = pd.concat([scenario_balance, scenario_other])
        
        self.comb_index_names = list(scenario_final.index.names)
        
        #add metrics for status balalaces defaults, prepays, etc.
        if scenario=='Actuals':
            status_select = ['default','bk']
            status_select = [v for v in status_select if v in col_list]
            balance_by_status = self.cf_output_status.xs([scenario, 'BOM_PrincipalBalance'], level=['Scenario','Metric'])[status_select].reset_index()
            #balance_by_status = self.cf_output_status.xs([scenario, 'TotalPrincipalBalance'], level=['Scenario','Metric'])[status_select].reset_index()
            balance_by_status['Scenario']='Actuals'
            balance_by_status.rename(columns=self.column_map, inplace=True)
            balance_by_status = pd.melt(balance_by_status, id_vars=['BatchKey','AsOfDate','Scenario'], var_name='Metric', value_name = 'Value').set_index(['BatchKey','AsOfDate','Scenario','Metric'])
            balance_by_status = balance_by_status.astype('float64')
        else:
            #Monthly balance by status is cumulative. once UPB rolls into prepay or default it stays there permanently
            #for monthly output we only want to show NEW balance rolling into these final statuses
            status_select = ['default','bk','prepay']
            status_select = [v for v in status_select if v in col_list]
                
            combined_metric_list = []
            for s in status_select:
                to_status=status_swap[s] #selected status
                from_status = [x for x in status.keys() if x!=to_status] #all statuses except for selected
                combined_metric_list.append(self.proj_4d_array(scenario=scenario, input_array='upb_trans_agg', output_metric_nm=self.column_map.get(s,s), from_status=from_status, to_status=to_status))
            
            balance_by_status = pd.concat(combined_metric_list)
        
        scenario_final = balance_by_status.append(scenario_final.to_frame()) #, verify_integrity=True)
        scenario_final.sort_index(inplace=True)
        #consolidate additional metrics from model output
        if scenario != 'Actuals':
           
            #scheduled payment made
            sch_pmt = scenario_final.loc[idx[:, :, scenario, ['ContractualPrincipalPayment', 'InterestPayment']], :].groupby(['BatchKey', 'AsOfDate','Scenario']).sum()
            sch_pmt['Metric'] = 'ScheduledPaymentMade'
            sch_pmt.set_index('Metric', append=True, inplace=True)
            scenario_final = scenario_final.append(sch_pmt)
            
            #TotalpaymentMade
            scenario_final.sort_index(inplace=True)
            scenario_final.loc[idx[:, :, scenario, 'TotalPaymentMade'], :] = scenario_final.loc[idx[:, :, scenario, ['TotalPaymentMade', 'PrincipalFullPrepayment']], :].groupby(['BatchKey', 'AsOfDate','Scenario']).sum()
            
            #total prepayment
            total_prepay = scenario_final.loc[idx[:, :, scenario, ['PrincipalFullPrepayment', 'PrincipalPartialPrepayment']], :].groupby(['BatchKey', 'AsOfDate','Scenario']).sum()
            total_prepay['Metric'] = 'PrincipalTotalPrepayment'
            total_prepay.set_index('Metric', append=True, inplace=True)
            scenario_final = scenario_final.append(total_prepay)
            
            #gross cash
            gross_cash = scenario_final.loc[idx[:, :, scenario, ['TotalPaymentMade', 'PrincipalFullPrepayment', 'PostChargeOffCollections']], :].groupby(['BatchKey', 'AsOfDate','Scenario']).sum()
            gross_cash['Metric'] = 'GrossCash'
            gross_cash.set_index('Metric', append=True, inplace=True)
            scenario_final = scenario_final.append(gross_cash)
            
        if len(self.cf_output)==0:
            self.cf_output = scenario_final
        else:
            #self.cf_output.drop(scenario, axis=0, level='Scenario', errors='ignore', inplace=True)
            self.cf_output = self.cf_output.append(scenario_final)
            self.cf_output.sort_index(inplace=True)
        #return combined_metric_list
        #self.scenario_final = scenario_final
        #self.balance_by_status = balance_by_status
    
    def return_metric_list(self):
        return list(self.cf_output.index.unique('Metric'))
    
    def select_plot_metric(self, metric):
        metric_plot = self.cf_output.xs(metric,level='Metric')
        metric_plot = metric_plot.groupby(['AsOfDate','Scenario'], axis=0).sum() #axis=0, skipna=False)
        metric_plot = metric_plot.unstack()['Value']
        return metric_plot
    
    def annualize_rate(self, metric):
        bom_balance = self.select_plot_metric('BOM_PrincipalBalance')
        monthly_rate = np.divide(metric, bom_balance)
        annual_rate =  1 - (1-monthly_rate) ** 12
        return annual_rate
    
    def create_plot(self, metric, begin_period=0, end_period=48, rate=False, cumulative=False):
        
        metric_plot = self.select_plot_metric(metric)
        line_color = [self.line_colors.get(x) for x in metric_plot.columns]
        line_style = [self.line_style.get(x, '--') for x in metric_plot.columns]
        
        if rate:
            metric_plot = self.annualize_rate(metric_plot)
            
        if cumulative:
            for col in metric_plot.columns:
                if col not in ['Actuals', 'Original Pricing']:
                    try:
                        nan_records = metric_plot[col].isnull()
                        max_index = metric_plot[nan_records].index.max()
                        actual_total = metric_plot[nan_records]['Actuals'].sum()
                        metric_plot.loc[max_index][col]=actual_total
                    except:
                        pass
            metric_plot = metric_plot.cumsum()
                
        metric_plot = metric_plot.iloc[begin_period:end_period]
        metric_plot.plot.line(color=line_color, style=line_style)
        
    def create_table(self, metric, begin_period=0, end_period=48, rate=False, cumulative=False):
        metric_plot = self.select_plot_metric(metric)

        if rate:
            metric_plot = self.annualize_rate(metric_plot)
            
        if cumulative:
            for col in metric_plot.columns:
                if col not in ['Actuals', 'Original Pricing']:
                    try:
                        nan_records = metric_plot[col].isnull()
                        max_index = metric_plot[nan_records].index.max()
                        actual_total = metric_plot[nan_records]['Actuals'].sum()
                        metric_plot.loc[max_index][col]=actual_total
                    except:
                        pass
            metric_plot = metric_plot.cumsum()
                
        metric_plot = metric_plot.iloc[begin_period:end_period]
        
        return metric_plot.astype(float)
        
    
    
    def create_avp(self, batch_keys = []):
        """
        Creates an excel AVP to visualize all results

        Returns
        -------
        Excel Doc

        """
        if len(batch_keys)>0:
            batch_array = self.cf_output.index.get_level_values('BatchKey').values 
            batch_mask = np.in1d(batch_array, [504, 522])
            avp_data = self.cf_output[batch_mask].copy()
        else:
            avp_data = self.cf_output.copy()

    
        #open copy of workbook
        
        avp_path = pkg_resources.path(templates, 'AvP Template.xlsx')
        with avp_path as path:
            avp_workbook = xl.load_workbook(path.resolve())
            
        #save_path = io.BytesIO(xl.writer.excel.save_virtual_workbook(avp_workbook))
        #save_path = os.path.join(sys.path[0], '\Documents\AvP.xlsx')
        
        save_path = tempfile.NamedTemporaryFile(suffix='.xlsx', dir=self.temp_folder, delete=False) 
        self.temp_files.append(save_path)
        
        #create writer to add data to workbook
        with pd.ExcelWriter(save_path.name, engine='openpyxl') as writer:
            # Save workbook as base (ensures all tabs are included)
            writer.book = avp_workbook
            writer.sheets = dict((ws.title, ws) for ws in avp_workbook.worksheets)
        
            # copy data into workbook
            avp_data.reset_index().to_excel(writer,'Data', index = False)
        
            # Save and close the file
            writer.save()
            writer.close()
            try:
                avp_workbook.close()
            except:
                pass
            
        #open the workbook
        #os.startfile(save_path.read())
        #with save_path as f:
        os.startfile(save_path.name)
        save_path.close()
        #f.unlink(f.name)
        
        #tmpf.close()
        #os.unlink(save_path.name)
        
        #return save_path
        
    def delete_temp_files(self):
        for file in self.temp_files:
            try:
                file.close()
            except:
                pass
            os.unlink(file)
            
        