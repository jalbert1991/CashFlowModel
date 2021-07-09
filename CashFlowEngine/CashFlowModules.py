# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 10:01:37 2020

@author: jalbert
"""

import pandas as pd
import numpy as np
import datetime as date
from dateutil.relativedelta import *
import re
from dataclasses import dataclass
from collections import OrderedDict
from abc import ABC, abstractmethod, abstractproperty
  
import psutil
import os
import gc
import multiprocessing as mp

#################################################################################
#                           CF Modules
#################################################################################

class CashFlowEngine(object):
    """
    Product Class - the Builder returns a completed Cash Flow Mediator
    Acts as a mediator for the cash flow modules. All data pushes and pulls are 
    sent through the mediator. This way we don't have to directly connect all of the 
    individual Classes
    
        pass self into classes on __init__. this registers the mediator.
    """
    
    def __init__(self):
        self._model_config = {}
        self._cf_modules = OrderedDict()
        self._cf_data = {}
        self.cf_input_data = {}
        self.model_state_output={}
    
    def set_model_config(self, model_type, data_tape, rate_curves, model_config=None):
        if model_config is None:
            self._model_config={}
        else:
            self._model_config=model_config
        
        self._model_config['model_type'] = model_type # 1 for cdr/cpr; 2 for roll rates; 3 for monte carlo        
        self._model_config['raw_tape'] = pd.merge(data_tape.set_index('AccountKey'), rate_curves.segment_account_map.drop('BatchKey',axis=1), left_index=True, right_index=True, sort=True) #data tape input
        self._model_config['data_tape'] = self._model_config['raw_tape'].copy()
        #self._model_config['rate_curves'] = rate_curves.reset_index().set_index(['curve_type','segment_id','period']).sort_index() #rate curves input
        self._model_config['rate_curves'] = rate_curves.data_rate_curves.reset_index().set_index(['curve_type','segment_id','period']).sort_index() #rate curves input
        
        self._model_config['num_accounts'] = len(data_tape)
        #self._model_config['num_account_groups'] = len(data_tape)
        #self.cf_model_data['projection_periods']=301
        
        if not self._model_config.get('account_status_list'):
            self._model_config['account_status_list'] = {
                    0:'deferment',
                    1:'current',
                    2:'1-29 dpd', 
                    3:'30-59 dpd', 
                    4:'60-89 dpd', 
                    5:'90-119 dpd', 
                    6:'120-149 dpd', 
                    7:'150-179 dpd',
                    8:'default',
                    9:'forbearance',
                    10:'mod',
                    11:'bk',
                    12:'prepay',
                    13:'not yet purchased',
                    14:'not in repay'
                }
            self._model_config['account_status_active'] = np.array([1,1,1,1,1,1,1,1,0,1,1,0,0,0,1], dtype='float32')
        self._model_config['num_status'] = len(self._model_config['account_status_list'])
        
        #process raw data tape
        self._model_config['acct_grouping_cols'] = self._model_config.get('acct_grouping_cols')
        #print(self._model_config['acct_grouping_cols'])
        self._model_config['group_accounts'] = self._model_config.get('group_accounts',True)
        self.process_raw_tape(self._model_config['group_accounts'])
        
        #add placeholders if not imported
        self._model_config['repay_begin_mth'] = self._model_config.get('repay_begin_mth',1) #for accounts not yet in repay (deferment or FF where account not yet purchased)
        self._model_config['int_rates'] = self._model_config.get('int_rates') # int rate input
        self._model_config['rate_compounding'] = self._model_config.get('rate_compounding', 'monthly')
        self._model_config['pmt_matrix'] = self._model_config.get('pmt_matrix') #payment matrix input
        self._model_config['int_accrue_matrix'] = self._model_config.get('int_accrue_matrix')
        self._model_config['int_cap_matrix'] = self._model_config.get('int_cap_matrix')
        self._model_config['default_amort_type'] = self._model_config.get('default_amort_type', 'monthly')
        self._model_config['amort_formula'] = self._model_config.get('amort_formula')
        self._model_config['amort_timing'] = self._model_config.get('amort_timing')
        #self._model_config['agg_functions'] = self._model_config.get('agg_functions')
        
        self._cf_metrics=['as_of_date','month_of_year','months_on_book','rem_term',
                          'bom_final_trans','eom_final_trans','bom_upb','bom_int','bom_units',
                          'upb_trans','int_trans','units_trans','eom_upb','eom_int','eom_units','rate',
                          'int_accrue','sch_pmt','amort_rule_used' , 'sch_pmt_trans', 'pmt_made','ipmt','ppmt',
                          'curtail','int_cap', 'cum_charge_off', 'recovery']
                            #,'cur_trans_rates'
        
    def set_account_groups(self, group_accounts=False):
        
        data_tape=self._model_config['raw_tape'].copy()
        
        if not group_accounts:
            self._model_config['data_tape'] = self._model_config['raw_tape'].copy()
            self._model_config['data_tape']['rec_cnt'] = 1
        
        else:
            #if group_accounts is set to True, group accounts as much as possible
            #if grouping rules not provided create default
            if not self._model_config['acct_grouping_cols']:
                
                key_cols = ['DealID','BatchKey','BatchAcquisitionDate','AsOfDate','MonthsOnBook','RemainingTerm',
                            'FirstPaymentDueDate','InterestRateType','InterestRateIndex','InterestRateChangeFrequency',
                            'MinInterestRate','MaxInterestRate','AccountStatusCode','PromoType','PromoEndDate','PromoEndMOB',
                            'PromoTerm', 'OriginationTerm']

                sum_cols = ['OriginationBalance','PurchaseBalance','PromoTargetBalance','BOM_PrincipalBalance','InterestBalance',
                            'TotalPrincipalBalance','TotalBalance','ScheduledPaymentAmount','MinimumPaymentDue','sch_prin_pmt',
                            'sch_int_pmt','tot_sch_pmt','prepay_partial', 'prepay_full', 'tot_prin_pmt','postchargeoffcollections',
                            'tot_pmt'
                        ]
                weight_avg_cols = ['InterestRate','InterestMargin','OriginationCreditScore']
                
                self._model_config['acct_grouping_cols'] = {
                        'key_cols': key_cols,
                        'sum_cols': sum_cols,
                        'weight_avg_cols': weight_avg_cols
                        }
                
            #remove cols that are all null
            key_cols = self._model_config['acct_grouping_cols']['key_cols']
            for col in reversed(key_cols):
                if pd.isnull(data_tape[col]).all():
                    self._model_config['acct_grouping_cols']['key_cols'].remove(col)

            #add and segment colums found
            for col in data_tape.columns:
                if col.startswith('segment_') and col not in key_cols:
                    self._model_config['acct_grouping_cols']['key_cols'].extend([col])
                    
            #self._model_config['acct_grouping_cols']['key_cols']=key_cols
            tape_grouped = data_tape.groupby(self._model_config['acct_grouping_cols']['key_cols']).apply(self.assign_agg_functions)
            self._model_config['data_tape'] = tape_grouped.reset_index()
        
    def assign_agg_functions(self, x):
        
        agg_dict = {}
        
        #count number of loans being aggregated in each group
        agg_dict['rec_cnt'] = x.count()
        
        #sum fields
        for col in self._model_config['acct_grouping_cols']['sum_cols']:
            agg_dict[col] = x[col].sum()
        
        #weighted Average fields
        for col in self._model_config['acct_grouping_cols']['weight_avg_cols']:
            agg_dict[col] = (x[col] * x['BOM_PrincipalBalance']).sum()/x['BOM_PrincipalBalance'].sum()
        
        return pd.Series(agg_dict, index=list(agg_dict.keys()))
    
    
    
    #@staticmethod
    #def agg_functions(agg_dict):
    #    agg_dict = self._model_config['agg_functions']
    #    pd.Series(agg_dict, index=list(agg_dict.keys()))
        
    def process_raw_tape(self, grouping = False):
        """
        Converts input data tape into arrays for use in the model
        two types of arrays
            Static Fields are 1d array (1 row per account). This is for account level attributes that don't change based on status
            Status Fields are 2d array. (Account x Status) This is mostly for balances. this array places balance in the appropriate status
        
        optional Grouping parameter
            'Account' will take every 
            'segments' will group accounts to the highest level possible. This reduces the number of records that need to be processed
                and makes the model speed up significantly. with the downside of loosing account level projections.
        """
        #set grouping level
        self.set_account_groups(grouping)
        self._model_config['num_accounts'] = len(self._model_config['data_tape'])
        
            
        #total valid field list 
        columns = self._model_config['data_tape'].columns
        status_fields = ['BOM_PrincipalBalance','InterestBalance', 'ScheduledPaymentAmount', 'PromoTargetBalance', 'OriginationBalance']
        static_fields = [c for c in columns if c not in status_fields]
        
        #create Static Arrays
        static_data = self._model_config['data_tape'][static_fields].to_dict(orient='list')
        for k in static_data.keys():
            #try to force dtype to float
            field_list = ['InterestRate']
            if k in field_list:
                self.cf_input_data[k] = np.array(static_data[k], dtype='float32')
            else:
                self.cf_input_data[k] = np.array(static_data[k])
            
        #create Status Arrays
        for col in self._model_config['data_tape'][status_fields].columns:
            zero_array = np.zeros((self._model_config['num_accounts'], self._model_config['num_status']), dtype='float32')
            for a in range(self._model_config['num_accounts']):
                zero_array[a, int(self.cf_input_data['AccountStatusCode'][a])]=self._model_config['data_tape'][col].iloc[a]
            self.cf_input_data[col] = np.nan_to_num(zero_array) #fill Nan with 0
        
        
    def add_module(self, name, instance):
        self._cf_modules[name] = instance
    
    def request_data(self, request_type, field_nm):
        """
        Some Cash Flow Modules require data created/maintained in another class. 
        i.e CalcTime stores calendar months, as of dates, remaining term. etc.
        Modules are operating under a mediator pattern so they do not directly interact.
        This function passes a data pull request from one class to another. 
        =======================
        Parameters:
        request_type: int
            1 = raw input array from the data tape
            2 = current attributes from another module
                Type 2 will search through all cash flow modules until it finds the correct variable to return
        
        Returns:
            various: ndarray, int, string 
        """
        
        if request_type==1:
            return self.cf_input_data[field_nm]
        elif request_type==2:
            for key, module in self._cf_modules.items():
                if field_nm in module.cf_model_data.keys():
                    return module.cf_model_data[field_nm]
                else:
                    next
                
            raise ValueError('Variable {} not found in any CF module'.format(field_nm))
    
    def send_output(self, operator, target, output_array):
        """
        add or subtract data from the monthly transition arrays stored in the CalcBalance Class. 
        after a calculation if the balance if effected, will send the changes to the Balance Class
        =======================
        Parameters:
        operator: str
            valid options 'add' or 'sub'
        """
        #validate target array
        if target not in ['int','upb','units']:
            raise ValueError('{} is not a valid target. options are "upb" or "int".'.format(target))
        #update transition matrix for desired attribute
        self._cf_modules['balance'].update_transition_balance(operator, target, output_array)
       
    def generator(self):
        i=1
        
        #begin cash flow engine
        while i <= self._model_config['projection_periods']:
            
            #reset output dict
            #combined_output={}
            
            #run model
            for key, module in self._cf_modules.items():
                module.run_module()
            #combined_output = self.extract_model_state()
            
            #yield combined_output
            yield self.extract_model_state(initial=False)
            
            #del combined_output
            
            #gc.collect()
            
            #reset eom and iterate forward
            for key, module in self._cf_modules.items():
                module.run_eom_reset()
            
            i+=1
            
    def extract_model_state(self, initial=False):
        
        for key, module in self._cf_modules.items():
                for data_key, data in module.cf_model_data.items():
                    if data_key in self._cf_metrics:
                        self.model_state_output[data_key]=data
        
        if initial:
            for key, value in self.model_state_output.items():
                #extend shape out to full projection periods to preallocate memory
                extended_shape = (self._model_config['projection_periods']+1,) + value.shape
                self._cf_data[key] = np.full(extended_shape, 0, dtype=value.dtype)
                self._cf_data[key][0] = value
            
    def run_generator(self):
        """
        execute generator
        """  
        #store init values at t=0
        #model_state=self.extract_model_state()
        self.extract_model_state(initial=True)
        
        cf_engine = self.generator()
        for engine_output in cf_engine:
            proj_month = self._cf_modules['time'].cf_model_data['projection_month']
            print(proj_month, self.mem_usage())
            for key, value in self.model_state_output.items():
                np.copyto(self._cf_data[key][proj_month], value)
        
        #swap axis to put account first
        for key, value in self._cf_data.items():
            self._cf_data[key] = np.swapaxes(value, 0, 1)
            
    def run_model(self):
        """
        spawn subprocess to run model calculations
        """
        proc = mp.Process(target=self.run_generator())
        proc.start()
        proc.terminate()
        proc.join()
        
    
    def mem_usage(self):
        psutil.virtual_memory()[2]
        pid=os.getpid()
        py = psutil.Process(pid)
        memoryuse=py.memory_info()[0]/2.**30
        return memoryuse
    
class CashFlowBaseModule(ABC):
    """
    Parent class for all cash flow calculation modules
    Provides mediator functonality and data transfer functions
    """

    def __init__(self, mediator):
        self._mediator = mediator
    
    def request_data(self, request_type, field_nm):
        """
        Request for data that gets sent up to the mediator
        """
        return self._mediator.request_data(request_type, field_nm)
    
    def request_all(self, field_list):
        """
        Some Cash Flow Modules require data created/maintained in another class. 
        i.e CalcTime stores calendar months, as of dates, remaining term. etc.
        Modules are operating under a mediator pattern so they do not directly interact.
        This function passes a data pull request from one class to another. 
        
        First attempts request type #2:
            type 2 = current attributes from another module
                Type 2 will search through all cash flow modules until it finds the correct variable to return
        if no match is found, wil search through the processed data tape for input fields
            type 1 = raw input array from the data tape
        =======================
        Parameters:
        field list : list
            list of text names to search for
        
        Returns:
            various: ndarray, int, string 
        """
        data={}
        for field in field_list:
            try:
                data[field]=self.request_data(2,field)
            except:
                try:
                    data[field]=self.request_data(1,field)
                except:
                    data[field]=self._mediator._model_config[field]

        return data
    
    def update_cf_inputs(self):
        self.cf_input_data = self.request_all(self._cf_input_fields)
    
    def send_output(self, operator, target, output):
        self._mediator.send_output(operator, target, output)
    
    @abstractmethod
    def run_module(self):
        #placeholder method to be overwritten by subclass
        pass   
    
    @abstractmethod
    def run_eom_reset(self):
        #placeholder method; reset base values necessary to run next month
        pass
            
    
###############################
# CF Modules

class CalcTime(CashFlowBaseModule):
    """
    Class to hold all of the various time metrics during model run
    run_calc method rolls forward all values to the next month
    
    """
    def __init__(self, mediator):
        super().__init__(mediator)
        
        #initialize fields at T0
        self.cf_model_data = {}
        self.cf_model_data['projection_month'] = 0
        #self.cf_model_data['as_of_date'] = np.array([date.datetime.strptime(d,"%Y-%m-%d").date() for d in self.request_data(1,'AsOfDate')])
        self.cf_model_data['as_of_date'] = np.array(self.request_data(1, 'AsOfDate'))
        self.cf_model_data['month_of_year'] = np.array([d.month for d in self.cf_model_data['as_of_date']])
        self.cf_model_data['months_on_book'] = self.request_data(1,'MonthsOnBook')
        self.cf_model_data['rem_term'] = self.request_data(1,'OriginationTerm') - self.request_data(1,'MonthsOnBook')
        
        self._mediator._model_config['projection_periods']=max(self.cf_model_data['rem_term'])
        
    def run_module(self):
        self.cf_model_data['projection_month']+=1
        self.cf_model_data['as_of_date']+=relativedelta(months=+1, day=31) #move cal month forward
        self.cf_model_data['month_of_year'] = np.array([d.month for d in self.cf_model_data['as_of_date']]) #reassign the current month number
        self.cf_model_data['months_on_book'] += 1 #move MOB forward
        self.cf_model_data['rem_term'] -= 1 #move rem term back
        
    def run_eom_reset(self):
        pass
        
class CalcRollRates(CashFlowBaseModule):
    """
    Generates 4D Transition Array [time x ] from account data and CDR/CPR/Roll Rate. 
    Then fills in any missing rate values to convert PD curves into transition matrix

    apply forward fill and back fill across all "to status" to fill holes
    ie. loaded curve is automatically set to "1 to 8". ffill and bfill copy values to all other "from" status in the same vintage
    so 0 to 8, 1 to 8, 2 to 8.... 12 to 8 will all have the same rate
    =======================
    Parameters:
    self
    
    Returns:
    Array: ndarray (4 dimensions)
        4d array into self.cf_model_data[rate_arrays][transition]
    
    """
    
    def __init__(self, mediator): # model_type, rate_data, data_tape, account_status_list
        super().__init__(mediator)
        #'model_type','rate_curves','data_tape', 'account_status_list','num_accounts', 'num_status', projection_periods
        self._cf_input_fields = ['months_on_book', 'repay_begin_mth', 'account_status_list', 'num_status','num_accounts','projection_periods']
        self.cf_input_data = self.request_all(self._cf_input_fields)
        self.max_mob_range = self.cf_input_data['projection_periods']+max(self.cf_input_data['months_on_book'])+1
        self.neg_mob = 0
        if np.amin(self.cf_input_data['months_on_book'])<0:
            self.neg_mob=np.amin(self.cf_input_data['months_on_book'])
        
        self.cf_model_data = {}
        self.cf_model_data['rate_arrays'] = {} 
        
        if self._mediator._model_config['model_type']==1: #CDR/CPR
            
            #set up input data 
            #columns=[x for x in self.cf_input_data['data_tape'].keys() if 'segment_' in x]
            columns=[x for x in self._mediator._model_config['data_tape'].keys() if 'segment_'  in x]
            account_data = self._mediator._model_config['data_tape'][columns].copy()
            account_data.rename(columns={x:x.replace('segment_','') for x in columns}, inplace=True)
            
            #############################################
            # Process CDR/CPR
            """
            Preprocessing tags each account (or account group) with appropriate Segment id.   
            For Proportional Hazard models this is two curves, CDR and CPR. Model calculations are 
            roll rate driven so we need to convert these two curves into a transition matrix. 
            Essentially, CDR is converted to the Current to Default roll and CPR is converted to the
            Current to Prepay roll. 
            """
            
            account_segments = account_data[['default','prepay']].copy()
            
            #switch account status desc with keys and rename columns
            account_status_list_swap = {v:k for k,v in self.cf_input_data['account_status_list'].items()}
            account_segments.rename(columns=account_status_list_swap, inplace=True)
            
            #populate all other status columns ## CDR=8, CPR=12, added columns will be 0
            account_segments.columns = account_segments.columns.astype(str)
            account_segments = account_segments.assign(**{str(col):(account_segments[str(col)] if str(col) in account_segments.columns.values else 0) for col in list(self.cf_input_data['account_status_list'].keys())})
            account_segments.columns = account_segments.columns.astype(int)
            account_segments.sort_index(axis=1, inplace=True)
            
            #map rate curves onto account list
            segments_array = self.map_rate_arrays(account_segments, self._mediator._model_config['rate_curves'], output_type=1)

            #reshape into desired form
            #array_reshape = segments_array.reshape(self.cf_input_data['num_accounts'], self.cf_input_data['num_status'], self.cf_input_data['num_status'], self.cf_input_data['projection_periods'])
            array_reshape = segments_array.reshape(self.cf_input_data['num_accounts'], self.cf_input_data['num_status'], self.cf_input_data['num_status'], (self.max_mob_range-self.neg_mob))
            array_reshape = np.moveaxis(array_reshape, 3,1)
            array_reshape = array_reshape.swapaxes(2,3)
            
            #update values in Array
            #fill any nan
            array_reshape = np.nan_to_num(array_reshape, copy=False)
    
            #freeze from_status 8 and 12 (once you go into, you cannot come out)
            array_reshape[:,:,[8,12],:]=0
            
            #where from_status = to_status (same status) set to 1-sum(row)
            status=list(self.cf_input_data['account_status_list'].keys())
            array_reshape[:,:,status, status] = 1-np.sum(array_reshape, axis=3)

            #find maximum MOB in status 13 and generate range list to shift UPB at the appropriate time
            try:
                max_mob_13 = max(self._mediator._model_config['data_tape'][self._mediator._model_config['data_tape']['AccountStatusCode']==13]['MonthsOnBook'])
            except:
                max_mob_13 = None
                
            #overwrite transition for 13 to 1 at MOB = 0 by default; transition Not In Repay balance into Current to start model
            if max_mob_13 is not None:
                
                try:
                    mob_range = list(range(self.cf_input_data['repay_begin_mth'], max_mob_13+2))
                except:
                    mob_range=[0]
                
                array_reshape[:,mob_range,13,:]=0
                array_reshape[:,mob_range,13,1]=1
                
            #store final array
            self.cf_model_data['rate_arrays']['transition'] = array_reshape 

            #############################################
            # Process Curtailment
            account_segments = account_data[['curtail']].copy()
            segments_array = self.map_rate_arrays(account_segments, self._mediator._model_config['rate_curves'], output_type=2)
    
            #fill any nan
            segments_array = np.nan_to_num(segments_array, copy=False)
                
            #store array
            self.cf_model_data['rate_arrays']['curtail'] = segments_array
        
        elif self._mediator._model_config['model_type']==2: #roll rate model
            pass
        
        elif self._mediator._model_config['model_type']==3: #monte carlo
            pass
            
        #final transition sums roll rates to final status each month. used if payments are configured to "Scale"
        self.cf_model_data['bom_final_trans'] = np.minimum(self.request_data(1,'BOM_PrincipalBalance'),1.0) ##start unit transition at 100% of current status
        self.cf_model_data['count_array'] = np.arange(len(self.cf_model_data['rate_arrays']['transition']))
        
        #initialize first month rates
        self.cf_model_data['cur_trans_rates']=self.cf_model_data['rate_arrays']['transition'][self.cf_model_data['count_array'], self.cf_input_data['months_on_book']]
        self.cf_model_data['eom_final_trans']=np.sum(self.cf_model_data['cur_trans_rates'], axis=1)
        #np.sum(self.cf_model_data['cur_trans_rates'], axis=1, out=self.cf_model_data['eom_final_trans'])
        #self.run_module()
        
    def map_rate_arrays(self, account_segments, rate_data, output_type):
        """
        Takes segment ID by account and maps in entire rate curve onto that account 
        
        =======================
        Parameters:
        account_segments: dataframe
            dataframe with account_ids as key and segment_ids as columns
        rate_data: dataframe
            dataframe with monthly rates by segment id
        output_type: int (1 or 2)
            1 = Transition matrix, returns a 13x13 matrix for each account, each month
            2 = single value, returns a single value for each account, each month
        """
        #pad tail zeros
        rate_data_fill = rate_data.groupby('segment_id').apply(self.fill_missing_vintage)
        #add tail records for neg MOB (negative index will go to end of array)
        rate_data_fill = rate_data_fill.groupby('segment_id').apply(self.fill_neg_mob)
        
        #convert curves to lists
        rate_dict = {l: rate_data_fill.xs(l, level=0)['rate'].values.tolist() for l in rate_data_fill.index.levels[0]}
        #zeros = [0]*self.cf_input_data['projection_periods']
        zeros = [0]*(self.max_mob_range-self.neg_mob)
        rate_dict.update({0:zeros})
            
        #flatten account segment list and map rate curves
        account_segments_flat = account_segments.to_numpy().flatten()
        array_map = map(rate_dict.get, account_segments_flat)
        segments_mapped = list(array_map)
        
        #convert to array. fastest option to create zero array and fill values
        if output_type==1:
            output_shape=(len(segments_mapped), self.cf_input_data['num_status'], (self.max_mob_range-self.neg_mob))
        else:
            output_shape=(len(segments_mapped), (self.max_mob_range-self.neg_mob))
        segments_array = np.zeros(output_shape, dtype='float32')
        for i, v in enumerate(segments_mapped):
            segments_array[i] = v
            
        return segments_array
    
    def fill_missing_vintage(self, x):
        """
        All Rate Curves must all be the same length. ie a 120 month term curve may only extend 120 months 
        but if there are 240 loans the model will break after we get past month 240
        This function extends each array to the max length and does a "forward fill" to continue 
        the last number forward. This extends the arrays by filling the last value available
        """
        idx = range(self.max_mob_range)
        x.reset_index(level=['curve_type','segment_id'], drop=True, inplace=True)
        return x.reindex(idx, method='ffill')
    
    def fill_neg_mob(self, x):
        """
        when backtesting FF deals there will be negative MOB until the asofdate gets to the batch 
        acquisition date. The model reads Negative MOB as a negative index. so a MOB of -2 will lookup up
        rates from the second to last item in the array. 
        
        to account for this behavior this function adds additional records to the end of the array. 
        these records will have a value of "0". that way when the transition matrix gets generated 
        all balance will stay in the same status. when the model advances to MOB=0 the lookups will
        function normally.
        """
        idx = range(self.max_mob_range-self.neg_mob)
        x.reset_index(level=['segment_id'], drop=True, inplace=True)
        return x.reindex(idx, fill_value=0)
    
    def run_module(self):
        """
        extracts the current month transition rates for each account 
        into a separate variable. convienent to do here once since several modules use these rates
        """
        self.update_cf_inputs()
        self.cf_model_data['cur_trans_rates']=self.cf_model_data['rate_arrays']['transition'][self.cf_model_data['count_array'], self.cf_input_data['months_on_book']]
        np.sum(self.cf_model_data['cur_trans_rates'], axis=1, out=self.cf_model_data['eom_final_trans'])
        
    def run_eom_reset(self):
        self.cf_model_data['bom_final_trans']=self.cf_model_data['eom_final_trans']#.copy()
        
class CalcBalance(CashFlowBaseModule):
    """
    Class responsible for balances. BOM, EOM and transitions between status
    """
    def __init__(self, mediator):
        super().__init__(mediator)
        
        #self._cf_input_fields = ['num_accounts','projection_periods','num_status']
        self._cf_input_fields = ['BOM_PrincipalBalance', 'InterestBalance','cur_trans_rates','num_accounts','num_status']
        self.cf_input_data = self.request_all(self._cf_input_fields)
        self.cf_model_data = {}
        
        #bom values
        self.cf_model_data['bom_upb'] = self.cf_input_data['BOM_PrincipalBalance'] #self.request_data(1,'BOM_PrincipalBalance')
        self.cf_model_data['bom_int'] = self.cf_input_data['InterestBalance'] #self.request_data(1,'InterestBalance')
        #change this to a unit count based on account number
        self.cf_model_data['bom_units'] = np.minimum(self.cf_model_data['bom_upb'],1.0) ##start unit transition at 100% of current status
        
        #Balance Transition
        self.cf_model_data['upb_trans'] = np.zeros((self._mediator._model_config['num_accounts'], self._mediator._model_config['num_status'], self._mediator._model_config['num_status']), dtype='float32')
        self.cf_model_data['int_trans'] = np.zeros((self._mediator._model_config['num_accounts'], self._mediator._model_config['num_status'], self._mediator._model_config['num_status']), dtype='float32')
        self.cf_model_data['units_trans'] = np.zeros((self._mediator._model_config['num_accounts'], self._mediator._model_config['num_status'], self._mediator._model_config['num_status']), dtype='float32')
        
        self.cf_model_data['upb_trans_agg'] = np.zeros((self._mediator._model_config['num_accounts'], self._mediator._model_config['num_status'], self._mediator._model_config['num_status']), dtype='float32')
        self.cf_model_data['int_trans_agg'] = np.zeros((self._mediator._model_config['num_accounts'], self._mediator._model_config['num_status'], self._mediator._model_config['num_status']), dtype='float32')
        self.cf_model_data['units_trans_agg'] = np.zeros((self._mediator._model_config['num_accounts'], self._mediator._model_config['num_status'], self._mediator._model_config['num_status']), dtype='float32')
        
        #eom values
        self.cf_model_data['eom_upb'] = np.zeros_like(self.cf_model_data['bom_upb'], dtype='float32')
        self.cf_model_data['eom_int'] = np.zeros_like(self.cf_model_data['bom_int'], dtype='float32')
        self.cf_model_data['eom_units'] = np.zeros_like(self.cf_model_data['bom_units'], dtype='float32')
    
    def update_transition_balance(self, operator, target, output_array):
        """
        add or subtract data from the monthly transition arrays 
        after a calculation if the balance needs to be updated, will send the changes to the Balance Class
        =======================
        Parameters:
        operator: str
            valid options 'add' or 'sub'
        """
        #validate target array
        if target not in ['int','upb', 'units']:
            raise ValueError('{} is not a valid target. options are "upb" or "int".'.format())
        
        try:
            np.sum(output_array, axis=1, out=self.trans_bal)
        except:
            self.trans_bal = np.sum(output_array, axis=1)
        
        target_trans = target+'_trans_agg'
        target_eom = 'eom_'+target
        
        if operator=='add':
            #update transition array
            np.add(self.cf_model_data[target_trans], output_array, self.cf_model_data[target_trans])
            #update total EOM balance
            np.add(self.cf_model_data[target_eom], self.trans_bal, self.cf_model_data[target_eom])
        elif operator=='sub':
            #update transition array
            np.subtract(self.cf_model_data[target_trans], output_array, self.cf_model_data[target_trans])
            #update total EOM balance
            np.subtract(self.cf_model_data[target_eom], self.trans_bal, self.cf_model_data[target_eom])
    
    def run_module(self):
        
        #get current month transition rates
        transition_rates=self.request_data(2,'cur_trans_rates')
        
        #apply transition rates to beginning of month balances
        #self.cf_model_data['upb_trans'] += self.cf_model_data['bom_upb'][:,:, np.newaxis] * transition_rates
        #self.cf_model_data['int_trans'] += self.cf_model_data['bom_int'][:,:, np.newaxis] * transition_rates
        #self.cf_model_data['units_trans'] += self.cf_model_data['bom_units'][:,:, np.newaxis] * transition_rates
        
        #numpy operators to save memory
        np.multiply(self.cf_model_data['bom_upb'][:,:, np.newaxis], transition_rates, out=self.cf_model_data['upb_trans'])
        np.multiply(self.cf_model_data['bom_int'][:,:, np.newaxis], transition_rates, out=self.cf_model_data['int_trans'])
        np.multiply(self.cf_model_data['bom_units'][:,:, np.newaxis], transition_rates, out=self.cf_model_data['units_trans'])
        
        #send values to eom
        #kind of redundant since its stored in this class, but this is the interface all other 
        #classes use. so we are keeping it
        self.send_output('add', 'upb', self.cf_model_data['upb_trans'])
        self.send_output('add', 'int', self.cf_model_data['int_trans'])
        self.send_output('add', 'units', self.cf_model_data['units_trans'])
        
    def run_eom_reset(self):
        #set bom balance = eom balance
        #self.cf_model_data['bom_upb'] = self.cf_model_data['eom_upb'].copy()
        #self.cf_model_data['bom_int'] = self.cf_model_data['eom_int'].copy()
        #self.cf_model_data['bom_units'] = self.cf_model_data['eom_units'].copy()
        
        np.copyto(self.cf_model_data['bom_upb'], self.cf_model_data['eom_upb'])
        np.copyto(self.cf_model_data['bom_int'], self.cf_model_data['eom_int'])
        np.copyto(self.cf_model_data['bom_units'], self.cf_model_data['eom_units'])
        
        #reset Balance Transition
        self.cf_model_data['upb_trans'].fill(0)
        self.cf_model_data['int_trans'].fill(0)
        self.cf_model_data['units_trans'].fill(0)
        self.cf_model_data['upb_trans_agg'].fill(0)
        self.cf_model_data['int_trans_agg'].fill(0)
        self.cf_model_data['units_trans_agg'].fill(0)
        
        #reset eom values
        self.cf_model_data['eom_upb'].fill(0)
        self.cf_model_data['eom_int'].fill(0)
        self.cf_model_data['eom_units'].fill(0)
        
class CalcInterest(CashFlowBaseModule):
    """
    Class to hold Interest rate information and reset rate when necessary
    """
    def __init__(self, mediator, int_accrue_matrix=None, compound_type=None):
        super().__init__(mediator)
        
        #initialize fields at T0
        self._cf_input_fields=['InterestRate','InterestRateType','InterestRateIndex','InterestMargin',
                                'InterestRateChangeFrequency','MaxInterestRate','MinInterestRate',
                                'as_of_date', 'months_on_book']
        #'num_status', 'num_accounts', 'account_status_list'
        self.cf_input_data = self.request_all(self._cf_input_fields)
        self.cf_model_data = {}
        
        #raw inputs from data tape
        if not compound_type:
            self._compound_type = 'Monthly'
        else:
            self._compound_type = compound_type
        #self.cf_model_data['rate_type'] = self.request_data(1,'InterestRateType')
        #self.cf_model_data['rate_index'] = self.request_data(1,'InterestRateIndex')
        #self.cf_model_data['rate_change_freq'] = self.request_data(1,'InterestRateChangeFrequency')
        #self.cf_model_data['rate'] = self.request_data(1,'InterestRate')
        self.cf_model_data['rate'] = self.cf_input_data['InterestRate']
        self.cf_model_data['int_accrue'] = np.zeros((self._mediator._model_config['num_accounts'], self._mediator._model_config['num_status'], self._mediator._model_config['num_status']), dtype='float32')
        #self.cf_model_data['rate_margin'] = self.request_data(1,'InterestMargin')
        #self.cf_model_data['rate_max'] = self.request_data(1,'MaxInterestRate')
        #self.cf_model_data['rate_min'] = self.request_data(1,'MinInterestRate')
        
        #import interest rate index
        
        #default fields (editable by user)
        #self.status_accrue_active = np.array([1,1,1,1,1,1,1,1,0,1,1,0,0])
        
        if int_accrue_matrix:
            self.int_accrue_matrix = int_accrue_matrix
        else:
            self.int_accrue_matrix = np.ones((self._mediator._model_config['num_status'], self._mediator._model_config['num_status']), dtype='float32')
            self.create_int_matrix() 
            
    def create_int_matrix(self):
        """
        Generates interest accrue matrix
        All Start at 100% except for 8 to 8 and 12 to 12. 
        if Default/prepay then 0% interest accrue
        """
        #from status zero payment
        self.int_accrue_matrix[[8,12],[8,12]] = 0
    
    def update_active_matrix(self, active_matrix):
        self.int_accrue_matrix = active_matrix
        
    def calc_int_rate(self):
        cond_list = [self.cf_input_data['months_on_book']<0, self.cf_input_data['InterestRateType']=='Variable', self.cf_input_data['InterestRateType']=='Fixed']
        choice_list = [np.zeros((self.cf_input_data['InterestMargin'].shape)) , self.cf_input_data['InterestMargin'], self.cf_input_data['InterestRate']] #variable should be margin+index
        rate = np.select(cond_list, choice_list)
        
        #compound calculation
        if self._compound_type == 'monthly':
            rate = rate/12
        elif self._compound_type == 'daily':
            #rate/365 * days in month
            rate = (rate/365)* np.array([m.day for m in self.request_data(2,'as_of_date')])
        elif self._compound_type == 'continuous':
            pass
        
        self.cf_model_data['rate'] = np.array(rate, dtype='float32')
    
    def calc_int_accrued(self):
        
        #apply rate to bom upb (using transition data)
        self.cf_model_data['int_accrue'] = self.request_data(2,'upb_trans_agg') * self.cf_model_data['rate'][:,np.newaxis, np.newaxis] * self.int_accrue_matrix
        
    def run_module(self):
        
        self.calc_int_rate()
        self.calc_int_accrued()
        
        #update balances
        #add accrued interest to totals
        self.send_output('add', 'int', self.cf_model_data['int_accrue'])
        
    def run_eom_reset(self):
        #self.cf_model_data['int_accrue'] = np.zeros_like(self.cf_model_data['int_accrue'], dtype='float32')
        self.cf_model_data['int_accrue'].fill(0)
        
class CalcPayments(CashFlowBaseModule):
    """
    class responsible for Payment information
        Scheduled Payments
        split between PPMT, IPMT, Curtailment
        
    """
    def __init__(self, mediator, pmt_matrix=None):
        super().__init__(mediator)
        
        self._cf_input_fields = ['as_of_date', 'month_of_year','months_on_book','rem_term', 
                                 'rate','bom_upb', 'int_trans_agg', 'upb_trans_agg', 'units_trans', 
                                 'bom_final_trans','cur_trans_rates', 'num_status', 'account_status_active',
                                 'num_accounts']
        self.cf_input_data = self.request_all(self._cf_input_fields)
        self.cf_model_data = {}
        
        #initialize cf model fields
        self.cf_model_data['sch_pmt'] = self.request_data(1,'ScheduledPaymentAmount')
        self.cf_model_data['pmt_made'] = np.zeros((self.cf_input_data['num_accounts'], self.cf_input_data['num_status'], self.cf_input_data['num_status']), dtype='float32') #np.zeros_like(self.cf_model_data['sch_pmt'])
        self.cf_model_data['ipmt'] = np.zeros((self.cf_input_data['num_accounts'], self.cf_input_data['num_status'], self.cf_input_data['num_status']), dtype='float32') #np.zeros_like(self.cf_model_data['sch_pmt'])
        self.cf_model_data['ppmt'] = np.zeros((self.cf_input_data['num_accounts'], self.cf_input_data['num_status'], self.cf_input_data['num_status']), dtype='float32') #np.zeros_like(self.cf_model_data['sch_pmt'])
        self.cf_model_data['curtail'] = np.zeros((self.cf_input_data['num_accounts'], self.cf_input_data['num_status'], self.cf_input_data['num_status']), dtype='float32') #np.zeros_like(self.cf_model_data['sch_pmt'])
        
        self.cf_model_data['sch_ppmt'] = np.zeros((self.cf_input_data['num_accounts'], self.cf_input_data['num_status'], self.cf_input_data['num_status']), dtype='float32') #np.zeros_like(self.cf_model_data['sch_pmt'])
        self.cf_model_data['pmt_remain'] = np.zeros((self.cf_input_data['num_accounts'], self.cf_input_data['num_status'], self.cf_input_data['num_status']), dtype='float32') #np.zeros_like(self.cf_model_data['sch_pmt'])
        
        self.cf_model_data['sch_pmt_trans'] = np.zeros_like(self.cf_input_data['upb_trans_agg'])
        
        #user input fields
        if pmt_matrix:
            self.pmt_matrix = pmt_matrix
        else:
            self.pmt_matrix = np.zeros((self.cf_input_data['num_status'], self.cf_input_data['num_status']), dtype='float32')
            self.create_payment_matrix() 
            
        #self.amort_calc_type = 'monthly'
        self.amort_formula = {
            0:"np.pmt(cf_input_data['rate'][:,np.newaxis], cf_input_data['rem_term'][:,np.newaxis], cf_input_data['bom_upb'])",
            98:"np.zeros_like(cf_input_data['bom_upb'])",
            99:"np.pmt(cf_input_data['rate'][:,np.newaxis], cf_input_data['rem_term'][:,np.newaxis], cf_input_data['bom_upb'])"
        }
        self.amort_timing = {
            0:"np.isnan(np.sum(cf_model_data['sch_pmt'], axis=1))[:,np.newaxis]", #if scheduled payment is null
            98:"(np.sum(cf_input_data['bom_upb'] * cf_input_data['account_status_active'], axis=1) == 0)[:,np.newaxis]", #if active UPB ==0 just set to 0
            99:"" #if no other rule triggered
        }

        #set default behavior
        self.change_amort_calc_type(self._mediator._model_config['default_amort_type'])
        #update field list for payment formulas
        self.add_cf_input_fields()

        #run calc to set initial values
        self.run_module()

    def add_cf_input_fields(self):
        #fina all columns referenced in formulas
        field_list = self._cf_input_fields
        new_list = []
        
        #add fields in 
        for key, value in self.amort_formula.items():
            field_list.extend(re.findall('\[.*?\]', value))
            
        for key, value in self.amort_timing.items():
            field_list.extend(re.findall('\[.*?\]', value))
            
        #format field names
        for col in np.unique(field_list):
            new_list.append(col.replace("['",'').replace("']",''))
        try:
            new_list.remove('[:,np.newaxis]')
        except:
            pass
        
        #remove any references to fields maintained in this class
        for col in new_list:
            if col in self.cf_model_data.keys():
                new_list.remove(col)
        
        self._cf_input_fields = new_list
        
        #update inputs
        self.cf_input_data = self.request_all(self._cf_input_fields)
        
    def update_cf_inputs(self):
        self.cf_input_data = self.request_all(self._cf_input_fields)
    
    def create_payment_matrix(self):
        """
        Generates payment matrix
        if FromStatus=ToStatus = 100% payment
        if Roll forward = 0% payment
        if Cure = 200% payment
        if Default/Forbearance/BK then 0% payment
        if prepay then 100% payment (one normal payment, then all remaining principal)
        """
        #fill diagonals
        #same status = 100% pmt
        np.fill_diagonal(self.pmt_matrix, 1.0)
        #Delinquent cure = 200% pmt
        for x in range(1, self.cf_input_data['num_status']):
            np.fill_diagonal(self.pmt_matrix[x:8,:], 1.0+x)
        
        #from status zero payment
        self.pmt_matrix[[0,8,11,13],:] = 0
        #to status zero payment
        self.pmt_matrix[:,[0,8,9,11,13]] = 0
        #to payoff make one normal payment
        self.pmt_matrix[:,12]=1
    
    def change_amort_calc_type(self, ream_type):
        """
        
        """
        if ream_type=='monthly':
            self.amort_formula[99] = "np.pmt(cf_input_data['rate'][:,np.newaxis], cf_input_data['rem_term'][:,np.newaxis], cf_input_data['bom_upb'])"
        elif ream_type =='scale':
            self.amort_formula[99] = "cf_model_data['sch_pmt']*cf_input_data['bom_final_trans']"
            
        self.amort_calc_type=ream_type
        
        #update key array
        self.create_amort_key_array()

    def add_new_amort_rule(self, time_rule, custom_amort_formula = None):
        """
        create rules to tell model when to recalulate amortization. 
            -time_rule is the trigger for the reamortization. This can use [months_on_book], [month_of_year], [projection_month] 
                or possibly [cal_month] (not yet added) if ream happens at one specific point in time.  
            -custom_amort_formula is (optional) the actual reamortization formula for things like custom promotions, or unique programs
        Pass in a timing rule using logic calculations ("==", ">=", "<", etc.) or the Modulus function (%)
            -Modulus only returns the remainder in a division. 
                so [MonthOfYear] % 3 = 0 would reamortize the loan quarterly on each 3rd month (March, June, etc). 
                    [MonthsOnBook] % 12 = 0 would reamortize the loan every 12th month on book

        =======================
        Parameters:
        time_rule: string
            formula for when to reamortize. 
            must use [MonthsOnBook] or [MonthOfYear]
            Must use logical functions or Modulus function
        custom_amort_formula: string
            optional formula for custom amortization function. 
            if blank, will just use standard numpy payment calculation function
                np.pmt(InterestRate, Remaining Term, UPB)
        """
        #create new rule key
        rule_key = sorted(self.amort_formula.keys())[-3]+1
        
        ##################################
        #process time_rule
        #validate inputs
        #valid_cols = ['[MonthsOnBook]', '[MonthOfYear]', '[ProjectionMonth]', '[PromoEndMOB]','[PromoBeginMOB]']
        #col_matches = re.findall('\[.*?\]',time_rule)

        #if len(col_matches)==0:
        #    print('No input columns found. Include brackets on input columns. Valid options are ' + ', '.join(valid_cols))

        #for m in col_matches:
        #    if m not in valid_cols:
        #        print(m +' not a valid column. Options are case sensitive ' + ', '.join(valid_cols))
        #        return
            
        ##timing rule 
        time_rule = self.format_column_inputs(time_rule)
        #Final eval formula result must have 2 axis in order to broadcast later
        if (eval(time_rule, globals(), self.__dict__)).ndim==1:
            time_rule = '(' + time_rule +")[:,np.newaxis]"
        
        ##################################
        #process amort_formula
        #validate inputs
        if custom_amort_formula:
            formula = self.format_column_inputs(custom_amort_formula)
        else:
            formula = "np.pmt(cf_input_data['rate'][:,np.newaxis], cf_input_data['rem_term'][:,np.newaxis], cf_input_data['bom_upb'])"

        #add rules into dicts
        self.amort_timing[rule_key] = time_rule
        self.amort_formula[rule_key] = formula
            
        #update key array
        self.create_amort_key_array()
        
        #update input field values
        self.add_cf_input_fields()
        
    def format_column_inputs(self, full_rule):
        #find referenced columns
        col_matches = re.findall('\[.*?\]',full_rule)
        for col in np.unique(col_matches):
            #add source df
            if col in self.cf_model_data.keys():
                new_axis=''
                if self.cf_model_data[col.replace("['",'').replace("']",'')].ndim==1:
                    new_axis = "[:,np.newaxis]"
                full_rule.replace("[","cf_model_data['").replace(']',"']"+new_axis)
            else:
                new_axis=''
                if self.cf_input_data[col.replace("['",'').replace("']",'')].ndim==1:
                    new_axis = "[:,np.newaxis]"
                full_rule.replace("[","cf_input_data['").replace(']',"']"+new_axis)
        return full_rule
        
    def create_amort_key_array(self):
        self.amort_key_array = [np.full((self.cf_input_data['num_accounts'],self.cf_input_data['num_status']), key) for key in sorted(self.amort_timing.keys()) if key != 99]
    
    ########################################################################
    #               Cash Flow Calculations
    def calc_sch_pmt(self, return_output=0):
        """
        Evaluates all Conditions and Amortization formulas entered
        Then selects first valid option
        amort_calc_type is final option in list. 
            if no other rule is satisfied, what will the model do?
            if "monthly" then standard amortization payment is calculated
            if "scale" then will take prior calculated payment and scale down by amount of balance still active
    
        """
        conditions = self.amort_timing
        selections = self.amort_formula
        #re_am_type = self.re_am_type
        
        #eval conditions and selections to boolean
        cond_eval = [eval(conditions[r], globals(), self.__dict__) for r in sorted(conditions.keys()) if r != 99]
        select_eval = [eval(selections[s], globals(), self.__dict__) for s in sorted(selections.keys()) if s != 99]
        default_eval = eval(selections[99], globals(), self.__dict__) #if no other formula is triggered
        
        self.amort_timing_eval = cond_eval
        self.amort_formula_eval = select_eval
      
        #broadcast to all 13 statuses
        for i in range(len(cond_eval)):
            cond_eval[i] = np.broadcast_to(cond_eval[i], (len(cond_eval[i]),self.cf_input_data['num_status'])) #(len(cond_eval[i]),13)
            
        sch_pmt = np.select(cond_eval, select_eval, default_eval)
        amort_rule_used = np.select(cond_eval, self.amort_key_array, 99)
        
        #return final payment and selected formula
        #return sch_pmt#, pmt_used
        if return_output==1:
            return sch_pmt, amort_rule_used
        else:
            np.absolute(sch_pmt, out=self.cf_model_data['sch_pmt'])
            #self.cf_model_data['sch_pmt_trans'] = np.clip((sch_pmt[:,:, np.newaxis] * self.cf_input_data['cur_trans_rates']), a_min=None, a_max=self.cf_input_data['upb_trans_agg'])
            #np.multiply(sch_pmt[:,:, np.newaxis], self.cf_input_data['cur_trans_rates'], out=self.cf_model_data['sch_pmt_trans'])
            np.multiply(self.cf_model_data['sch_pmt'][:,:, np.newaxis], self.cf_input_data['cur_trans_rates'], out=self.cf_model_data['sch_pmt_trans'])
            np.clip(self.cf_model_data['sch_pmt_trans'], a_min=None, a_max=self.cf_input_data['upb_trans_agg'], out=self.cf_model_data['sch_pmt_trans'])
            self.cf_model_data['amort_rule_used'] = amort_rule_used
    
    def calc_pmt_made(self):
        """
        Scheduled Payment is calculated across the beginning status in each month. 
        This has to be allocated based on transition rates
        then the payment matrix is applied to scale payment amounts
            i.e. a cure is 200% of a normal payment
        """
        #self.cf_model_data['pmt_made'] = self.cf_model_data['sch_pmt'][:,:, np.newaxis] * self.cf_input_data['cur_trans_rates'] * self.pmt_matrix
        #self.cf_model_data['pmt_made'] = np.clip((self.cf_model_data['sch_pmt_trans'] * self.pmt_matrix), a_min=None, a_max=self.cf_input_data['upb_trans_agg'])
        np.multiply(self.cf_model_data['sch_pmt_trans'], self.pmt_matrix, out=self.cf_model_data['pmt_made'])
        np.clip(self.cf_model_data['pmt_made'], a_min=None, a_max=self.cf_input_data['upb_trans_agg'], out=self.cf_model_data['pmt_made'])
        
    def calc_pmt_curtail(self):
        """
        on CDR/CPR models curtailment curves are built as a percent of BOM UPB
        only calculate curtailment for current bucket UPB
        """
        curtail_rate = self.request_data(2,'rate_arrays')['curtail'][np.arange(self.cf_input_data['num_accounts']), self.cf_input_data['months_on_book']]
        try:
            np.multiply(self.cf_input_data['bom_upb'][:,1], curtail_rate, out=self.cf_model_data['curtail_calc'])
        except:
            self.cf_model_data['curtail_calc'] = np.multiply(self.cf_input_data['bom_upb'][:,1], curtail_rate)
        #curtail_amt = self.cf_input_data['bom_upb'][:,1] * curtail_rate
        #np.mutiply(self.cf_input_data['bom_upb'][:,1], curtail_rate, out=curtail_amt)
        #self.cf_model_data['pmt_made'][:,1,1] += curtail_amt
        np.add(self.cf_model_data['pmt_made'][:,1,1], self.cf_model_data['curtail_calc'], out=self.cf_model_data['pmt_made'][:,1,1])
        
    
    def calc_pmt_split(self):
        """
        Split total payments into various buckets
        results in 3 arrays:
            ipmt
            ppmt
            curtail
        we use np.clip() to cap payment at each stage
        """

        np.copyto(self.cf_model_data['pmt_remain'], self.cf_model_data['pmt_made'])
        
        #interest payment
        np.clip(self.cf_model_data['pmt_remain'], a_min=0, a_max=self.cf_input_data['int_trans_agg'], out=self.cf_model_data['ipmt'])
        np.subtract(self.cf_model_data['pmt_remain'], self.cf_model_data['ipmt'], out=self.cf_model_data['pmt_remain'])
        
        #scheduled principal 
        #scheduled payment by transition
        np.subtract(self.cf_model_data['sch_pmt_trans'], self.cf_model_data['ipmt'], out=self.cf_model_data['sch_ppmt'])
        np.clip(self.cf_model_data['pmt_remain'], a_min=0, a_max=self.cf_model_data['sch_ppmt'], out=self.cf_model_data['ppmt'])
        np.subtract(self.cf_model_data['pmt_remain'], self.cf_model_data['ppmt'], out=self.cf_model_data['pmt_remain'])
        
        #anything remaining is curtailment
        np.copyto(self.cf_model_data['curtail'], self.cf_model_data['pmt_remain'])
    
    def run_module(self):
        self.update_cf_inputs()
        self.calc_sch_pmt()
        self.calc_pmt_made()
        self.calc_pmt_curtail()
        self.calc_pmt_split()
        
        #update balances
        #add accrued interest to totals
        self.send_output('sub', 'int', self.cf_model_data['ipmt'])
        self.send_output('sub', 'upb', self.cf_model_data['ppmt'])
        self.send_output('sub', 'upb', self.cf_model_data['curtail'])
        
    def run_eom_reset(self):
        self.cf_model_data['pmt_made'].fill(0)  
        self.cf_model_data['ipmt'].fill(0)
        self.cf_model_data['ppmt'].fill(0) 
        self.cf_model_data['curtail'].fill(0) 
        
class CalcIntCapitalize(CashFlowBaseModule):
    """
    class responsible for Interest Capitalization
    only specific transitions will have capitalizations and/or at specific points
    in time. 
        i.e. annually in january
        
    """
    def __init__(self, mediator, cap_matrix=None, transition_rules = 1, timing_rules=1):
        super().__init__(mediator)
        
        self._cf_input_fields = ['as_of_date', 'month_of_year','months_on_book','rem_term', 'account_status_list', 'account_status_active',
                                 'num_status', 'num_accounts', 'rate','int_trans_agg','units_trans', 'bom_units']
        self.cf_input_data = self.request_all(self._cf_input_fields)
        self.cf_model_data = {}
        self.timing_rules = {}
        
        #initialize cf model fields
        self.cf_model_data['int_cap'] = np.zeros((self.cf_input_data['num_status'], self.cf_input_data['num_status']), dtype='float32')
        
        #user input fields
        if cap_matrix:
            self.cap_matrix = cap_matrix
        else:
            #if matrix is not passed in, build one based on "Transition_rules" and "Timing Rules" flags
            self.cap_matrix = np.zeros((self.cf_input_data['num_status'], self.cf_input_data['num_status']), dtype='float32')
            self.create_capitalize_matrix(transition_rules, timing_rules)
            
    def create_capitalize_matrix(self, transition_rules=0, timing_rules=0):
            
        if transition_rules==0 and timing_rules==0:
            #reset matrix to zero
            self.cap_matrix = np.zeros((self.cf_input_data['num_status'], self.cf_input_data['num_status']), dtype='float32')
        elif transition_rules==0 and timing_rules==1:
            #reset matrix to 1
            self.cap_matrix = np.ones((self.cf_input_data['num_status'], self.cf_input_data['num_status']), dtype='float32')
        elif transition_rules==1:
            #reset matrix to zero
            self.cap_matrix = np.zeros((self.cf_input_data['num_status'], self.cf_input_data['num_status']), dtype='float32')
        
    def calc_timing(self, timing_rules=0):
        
        if timing_rules==0:
            pass
        
    def add_new_timing_rule(self, time_rule):
        """
        create rules to tell model when to capitalize outstanding interest balance. 
        Pass in a timing rule using logic calculations ("==", ">=", "<", etc.) or the Modulus function (%)
            -Modulus only returns the remainder in a division. 
                so [MonthOfYear] % 3 = 0 would capitalize the interest quarterly on each 3rd month (March, June, etc). 
                    [MonthsOnBook] % 12 = 0 would capitalize interest every 12th month on book

        =======================
        Parameters:
        time_rule: string
            formula for when to reamortize. 
            must use [MonthsOnBook] or [MonthOfYear]
            Must use logical functions or Modulus function
        """
        #create new rule key
        if len(self.timing_rules)==0:
            rule_key=1
        else:
            rule_key = sorted(self.timing_rules.keys())[-1]+1
        
        ##################################
        #process time_rule
        #validate inputs
        #valid_cols = ['[MonthsOnBook]', '[MonthOfYear]', '[ProjectionMonth]', '[PromoEndMOB]','[PromoBeginMOB]']
        #col_matches = re.findall('\[.*?\]',time_rule)

        #if len(col_matches)==0:
        #    print('No input columns found. Include brackets on input columns. Valid options are ' + ', '.join(valid_cols))

        #for m in col_matches:
        #    if m not in valid_cols:
        #        print(m +' not a valid column. Options are case sensitive ' + ', '.join(valid_cols))
        #        return
            
        ##timing rule 
        time_rule = self.format_column_inputs(time_rule)
        #Final eval formula result must have 2 axis in order to broadcast later
        if (eval(time_rule, globals(), self.__dict__)).ndim==1:
            time_rule = '(' + time_rule +")[:,np.newaxis]"
        
        #add rules into dicts
        self.timing_rules[rule_key] = time_rule
        
        #update key array
        self.create_amort_key_array()
        
        #update input field values
        self.add_cf_input_fields()
        
        
    def run_module(self):
        self.cf_model_data['int_cap'] = self.cf_input_data['int_trans_agg'] * self.cap_matrix * self.timing_rules
        
        #send balance changes
        self.send_output('sub', 'int', self.cf_model_data['int_cap'])
        self.send_output('add', 'upb', self.cf_model_data['int_cap'])
        
    def run_eom_reset(self):
        self.cf_model_data['int_cap'] = np.zeros_like(self.cf_input_data['int_cap'], dtype='float32')
        
class CalcRecovery(CashFlowBaseModule):
    """
    """
    def __init__(self, mediator):
        super().__init__(mediator)
    
    def run_module(self):
        pass
    
    def run_eom_reset(self):
        pass
        

        
        
    