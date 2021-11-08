# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 10:01:37 2020

@author: jalbert
"""

import pandas as pd
import numpy as np
import numpy_financial as npf
import datetime as date
from dateutil import relativedelta
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
        self._scenario_name = ''
        self._model_config = {}
        self._cf_modules = OrderedDict()
        self._cf_data = {}
        self.cf_input_data = {}
        self.model_state_output={}
    
    def set_model_config(self, scenario_name, model_type, data_tape, rate_curves, cutoff_date, curve_stress, model_config=None):
        if model_config is None:
            self._model_config={}
        else:
            self._model_config=model_config.copy() #make a copy. do not want to alter the input
        
        self._scenario_name = scenario_name
        self._model_config['cutoff_date'] = cutoff_date
        self._model_config['model_type'] = model_type # 1 for cdr/cpr; 2 for roll rates; 3 for monte carlo        
        self._model_config['rate_curves'] = rate_curves #.return_curve_group() #.reset_index().set_index(['curve_type','segment_id','period']).sort_index() #rate curves input
        self._model_config['group_accounts'] = self._model_config.get('group_accounts', True)
        self._model_config['account_id'] = self._model_config.get('account_id', None)
        
        self._model_config['data_tape'] = data_tape.attach_curve_group(rate_curves, cutoff_date, self._model_config['group_accounts'], self._model_config['account_id'])   #self._model_config['raw_tape'].copy()      
        self._model_config['curve_stress'] = curve_stress
        
        self._model_config['curve_type_info'] = rate_curves.curve_type_info
        self._model_config['num_cohorts'] = len(self._model_config['data_tape'])
        self._model_config['account_status_list'] = data_tape.account_status_list
        self._model_config['account_status_active'] = data_tape.account_status_active
        self._model_config['num_status'] = len(self._model_config['account_status_list'])
        
        #process raw data tape into arrays
        self._model_config['segment_keys'] = {}
        self.process_raw_tape()
        
        self._model_config['projection_periods']=int(min(max(self.cf_input_data['RemainingTerm']), max(self.cf_input_data['OriginationTerm'])))
        
        #add placeholders if not imported
        self._model_config['repay_begin_mth'] = self._model_config.get('repay_begin_mth',0) #for accounts not yet in repay (deferment or FF where account not yet purchased)
        self._model_config['int_rates'] = self._model_config.get('int_rates') # int rate input
        self._model_config['rate_compounding'] = self._model_config.get('rate_compounding', 'monthly')
        self._model_config['pmt_matrix'] = self._model_config.get('pmt_matrix') #payment matrix input
        self._model_config['int_accrue_matrix'] = self._model_config.get('int_accrue_matrix')
        self._model_config['int_cap_matrix'] = self._model_config.get('int_cap_matrix')
        self._model_config['default_amort_type'] = self._model_config.get('default_amort_type', 'monthly')
        self._model_config['amort_formula'] = self._model_config.get('amort_formula')
        self._model_config['amort_timing'] = self._model_config.get('amort_timing')
        #self._model_config['agg_functions'] = self._model_config.get('agg_functions')
        
        self._cf_metrics = ['AsOfDate','month_of_year','MonthsOnBook','RemainingTerm', 'MonthsToAcquisition', 'CalendarMonth',
                          'bom_final_trans','eom_final_trans','BOM_PrincipalBalance','bom_int','bom_units',
                          'upb_trans', 'upb_trans_agg','int_trans', 'int_trans_agg','units_trans','TotalPrincipalBalance','InterestBalance','eom_units','InterestRate', 'int_rate_calc', 'rate_change_trigger',
                          'int_accrue', 'ScheduledPaymentCalc', 'ScheduledPaymentAmount','amort_rule_used' , 'sch_pmt_trans', 'TotalPaymentMade','InterestPayment','ContractualPrincipalPayment',
                          'PrincipalPartialPrepayment', 'curtail_calc', 'int_cap', 'PostChargeOffCollections']
                            #,'cur_trans_rates'
        self._cf_init_metrics = ['hist_default','hist_recovery']
    
    def process_raw_tape(self):
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
        self._model_config['num_cohorts'] = len(self._model_config['data_tape'])
            
        #total valid field list 
        columns = list(self._model_config['data_tape'].columns)
        
        #create segment_key arrays
        for s in self._model_config['rate_curves'].segment_types:
            if s in columns:
                self._model_config['segment_keys'][s] = self._model_config['data_tape'][s].values.astype(int)
                columns.remove(s)
        
        status_fields = ['BOM_PrincipalBalance', 'TotalPrincipalBalance', 'InterestBalance', 'ScheduledPaymentAmount']#, 'PromoTargetBalance', 'OriginationBalance']
        static_fields = [c for c in columns if c not in status_fields]
        
        #create Static Arrays
        static_data = self._model_config['data_tape'][static_fields].to_dict(orient='list')
        for k in static_data.keys():
            #try to force dtype to float
            field_list = ['InterestRate', 'InterestMargin', 'MaxInterestRate','MinInterestRate'
                          , 'PromoTargetBalance', 'OriginationBalance']
            if k in field_list:
                self.cf_input_data[k] = np.array(static_data[k], dtype='float32')
            else:
                self.cf_input_data[k] = np.array(static_data[k])

        #create Status Arrays
        for col in self._model_config['data_tape'][status_fields].columns:
            zero_array = np.zeros((self._model_config['num_cohorts'], self._model_config['num_status']), dtype='float32')
            for a in range(self._model_config['num_cohorts']):
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
            return self.cf_input_data[field_nm].copy()
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
                        
            #run model
            for key, module in self._cf_modules.items():
                module.run_module()
            
            #yield combined_output
            yield self.extract_model_state(initial=False)
                        
            i+=1
    
    def roll_rate_generator(self, step=6, ):
        i = 1
        while i <= self._model_config['projection_periods']:
            
            #run model
            for key, module in self._cf_modules.items():
                module.run_module()
                
            #yield output
            yield self.extract_model_state(initial=False)
            
            i+=1
        
    
    def extract_model_state(self, initial=False, final=False):
        
        for key, module in self._cf_modules.items():
                for data_key, data in module.cf_model_data.items():
                    #metrics extracted each month of model
                    if data_key in self._cf_metrics:
                        self.model_state_output[data_key] = data
                    #metrics extracted once at end
                    if final and data_key in self._cf_init_metrics:
                        self._cf_data[data_key] = data
                    
        if initial:
            for key, value in self.model_state_output.items():
                if key in self._cf_metrics:
                    #extend shape out to full projection periods to preallocate memory
                    extended_shape = (self._model_config['projection_periods']+1,) + value.shape
                    #print(key)
                    #self._cf_data[key] = np.full(extended_shape, 0, dtype=(value.dtype if value.dtype==np.object else np.float32)) #value.dtype)
                    self._cf_data[key] = np.zeros(extended_shape, dtype=(value.dtype if value.dtype==np.object else np.float32)) #value.dtype)
                    #just set up shape. do not want to extract outputs until month 1 of model
                    self._cf_data[key][0] = np.nan #value
            
    def run_generator(self):
        """
        execute generator
        """  
        #store init values at t=0
        self.extract_model_state(initial=True)
        #for key, value in self.model_state_output.items():
        #        np.copyto(self._cf_data[key][0], value)
        #reset eom for first month of model
        for key, module in self._cf_modules.items():
            module.run_eom_reset()
        
        total_months = self._model_config['projection_periods']
        
        cf_engine = self.generator()
        
        with np.errstate(divide='ignore', invalid='ignore'): #surpress divide by zero errors if exist
            for engine_output in cf_engine:
                proj_month = int(self._cf_modules['time'].cf_model_data['ProjectionMonth'])
                #print(proj_month, self.mem_usage())
                self.progress_bar(proj_month, total_months, prefix=(self._scenario_name + ' - Running Model - '))
                
                #extract month output
                for key, value in self.model_state_output.items():
                    np.copyto(self._cf_data[key][proj_month], value)
                    
                #reset eom then iterate forward
                for key, module in self._cf_modules.items():
                    module.run_eom_reset()
                
            #swap axis to put account first
            for key, value in self._cf_data.items():
                self._cf_data[key] = np.swapaxes(value, 0, 1)
            
        #at end of loop clear progress bar
        print('\r'+' '*120, end='\r')
        
        #grab any one off metrics left
        self.extract_model_state(final=True)
            
    def run_model(self):
        """
        spawn subprocess to run model calculations
        """
        proc = mp.Process(target=self.run_generator())
        proc.start()
        proc.terminate()
        proc.join()
        
        gc.collect()
    
    def mem_usage(self):
        psutil.virtual_memory()[2]
        pid=os.getpid()
        py = psutil.Process(pid)
        memoryuse=py.memory_info()[0]/2.**30
        return memoryuse
    
    def progress_bar(self, progress, total, prefix = '', suffix = '', decimals = 1, length = 50, fill = '█', printEnd = "\r"):
        """
        Call in a loop to create terminal progress bar
        
        ==============================
        Parameters:
            progress: int
                current iteration
            total: int 
                total iterations
            prefix: str
                Optional prefix string
            suffix: str
                Optional suffix string
            decimals: int
                Optional positive number of decimals in percent complete
            length: int
                Optional character length of bar
            fill: str
                Optional bar fill character
            printEnd: str
                Optional end character (e.g. "\r", "\r\n")
        """
        # Progress Bar Printing Function
        percent = ("{0:." + str(decimals) + "f}").format(100 * (progress / float(total)))
        filledLength = int(length * progress // total)
        bar = fill * filledLength + '-' * (length - filledLength)
        print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    
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
        
        self._cf_input_fields = ['AsOfDate', 'MonthsOnBook', 'RemainingTerm', 'MonthsToAcquisition', 'RemainingTerm']
        self.cf_input_data = self.request_all(self._cf_input_fields)
        
        #initialize fields at T0
        self.cf_model_data = {}
        self.cf_model_data['ProjectionMonth'] = 0
        self.cf_model_data['AsOfDate'] = np.array(self.cf_input_data['AsOfDate'])
        self.cf_model_data['month_of_year'] = np.array([d.month for d in self.cf_model_data['AsOfDate']])
        self.cf_model_data['MonthsOnBook'] = self.cf_input_data['MonthsOnBook'].astype(int)
        self.cf_model_data['RemainingTerm'] = self.cf_input_data['RemainingTerm'].astype(int)
        self.cf_model_data['CalendarMonth'] = np.zeros_like(self.cf_model_data['RemainingTerm'])
        self.cf_model_data['MonthsToAcquisition'] = self.cf_input_data['MonthsToAcquisition'].astype(int)
              
        
    def run_module(self):
        self.cf_model_data['ProjectionMonth']+=1
        self.cf_model_data['CalendarMonth']+=1
        self.cf_model_data['AsOfDate'] += relativedelta.relativedelta(months=+1, day=31) #move cal month forward
        self.cf_model_data['month_of_year'] = np.array([d.month for d in self.cf_model_data['AsOfDate']]) #reassign the current month number
        self.cf_model_data['MonthsOnBook'] += 1 #move MOB forward
        self.cf_model_data['RemainingTerm'] -= 1 #reduce rem term
        self.cf_model_data['MonthsToAcquisition'] -= 1
        
    def run_eom_reset(self):
        pass 


class CalcPromo(CashFlowBaseModule):
    """
    Class responsible for Promo calculations and timing.
    
    Promo can span Interest Rates, Balances, or Payment adjustments
    
    PromoType: "Interest Only", "No Interest (0%) w/out Payments", "No Interest (0%) with Payments", "Same As Cash", "No Promo"
    """
    
    def __init__(self, mediator):
        super().__init__(mediator)
        
        self._cf_input_fields = ['PromoType', 'PromoStartMonth', 'PromoEndMonth', 'PromoTerm', 'PromoTargetBalance'
                                 ,'PromoAmortTerm' ,'PromoBalloonDate', 'PromoBalloonFlag', 'OriginationVintage'
                                 ,'MonthsOnBook', 'num_cohorts', 'num_status'
                                 ]
        
        self.cf_input_data = self.request_all(self._cf_input_fields)
        self.cf_model_data = {}
        
        #balloon flag triggers
        BalloonDate = pd.to_datetime(self.cf_model_data['PromoBalloonDate'].fillna('1900-01-01').values, format="%Y-%m-%d")
        OriginationDate = pd.to_datetime(self.cf_model_data['OriginationVintage'].values)
        
        BalloonMOB = BalloonDate.to_period('M').astype(int) - OriginationDate.to_period('M').astype(int)
        self.cf_model_data['BalloonMOB'] = BalloonMOB.values
        self.cf_model_data['PromoBalloonFlag'] = self.cf_input_data['PromoBalloonFlag']
        
        #promo payment
        #self.cf_model_data['PromoPaymentFlag'] = np.zeros((self.cf_input_fields['num_cohorts']), dtype='float32')
        self.cf_model_data['PromoPaymentFlag'] =  np.in1d(self.cf_input_data['PromoType'], ['Interest Only', 'No Interest (0%) w/out Payments'])
        #promo interest
        #self.cf_model_data['PromoInterestFlag'] = np.zeros((self.cf_input_fields['num_cohorts']), dtype='float32')
        self.cf_model_data['PromoInterestFlag'] =  np.in1d(self.cf_input_data['PromoType'], ['No Interest (0%) w/out Payments', 'No Interest (0%) with Payments'])
        #promo balloon
        self.cf_model_data['PromoBalloonFlag'] = self.cf_input_data['PromoBalloonFlag']
        
        
    def calc_balloon_status(self):
        pass
    
    def calc_promo_status(self):
        """
        tag each promo type as payment mod and/or interest mod
        """
        
        #payment mod active in the current month
        
        
        #interest mod active in the current month
        np.in1d(batch_array, [504, 522])
        self.cf_input_data['PromoType'] = np.logical_and(self.cf_input)
        
        
        pass
        
    def run_module(self):
        pass
    
    def run_eom_reset(self):
        pass
        
class CalcRateCurves(CashFlowBaseModule):
    """
    Class to create and manage rate curve sets and isolate single month matrices for the current model run month
    
    ----------------------------- Roll Rates -------------------------------------------
    Generates 4D Transition Array (curveid x time x from_status x to_status) from account data and CDR/CPR/Roll Rate. 
    Then fills in any missing rate values to convert PD curves into transition matrix

    apply forward fill and back fill across all "to status" to fill holes
    ie. loaded curve is automatically set to "1 to 8". ffill and bfill copy values to all other "from" status in the same vintage
    so 0 to 8, 1 to 8, 2 to 8.... 12 to 8 will all have the same rate
    
    ----------------------------- Others --------------------------------------------
    Stores simple 2d matrix (curve id x time)
    
    =======================
    Parameters:
    self
    
    Returns:
    Array: ndarray (4 dimensions)
        4d array into self.cf_model_data[rate_arrays][transition]
        (curve_id x time x from_status x to_status)
    
    """
    
    def __init__(self, mediator): # model_type, rate_data, data_tape, account_status_list
        super().__init__(mediator)
        #'model_type','rate_curves','data_tape', 'account_status_list','num_cohorts', 'num_status', projection_periods
        self._cf_input_fields = ['MonthsOnBook', 'CalendarMonth', 'repay_begin_mth', 'account_status_list'
                                 ,'num_status','num_cohorts','projection_periods','MonthsToAcquisition']
        self.cf_input_data = self.request_all(self._cf_input_fields)
        #self.max_mob_range = self.cf_input_data['projection_periods']+max(self.cf_input_data['MonthsOnBook'])+1
        #self.neg_mob = 0
        #if np.amin(self.cf_input_data['MonthsOnBook'])<0:
        #    self.neg_mob=np.amin(self.cf_input_data['MonthsOnBook'])
        
        self.cf_model_data = {}
        self.cf_model_data['rate_arrays'] = {} 
        self.cf_model_data['stress_arrays'] = {}
        self.cf_model_data['cur_month_rates'] = {}
        self.segment_keys = self._mediator._model_config['segment_keys']
        self.curve_type_info = self._mediator._model_config['curve_type_info']
        
    
        #############################################
        # Process CDR/CPR/Rollrates
        """
        Preprocessing tags each account (or account group) with appropriate Segment id.   
        For Proportional Hazard models this is two curves, CDR and CPR. Model calculations are 
        roll rate driven so we need to convert these two curves into a transition matrix. 
        Essentially, CDR is converted to the Current to Default roll and CPR is converted to the
        Current to Prepay roll. 
        """
        
        #convert curves to raw matrix
        array_reshape = self._mediator._model_config['rate_curves'].return_transition_matrix()
        #fill any nan
        array_reshape = np.nan_to_num(array_reshape, copy=False)

        #freeze from_status 8 and 12 and 13 (once you go into, you cannot come out)
        array_reshape[:,:,[8,12,13],:]=0
        
        #where from_status = to_status (same status) set to 1-sum(row)
        status=list(self.cf_input_data['account_status_list'].keys())
        array_reshape[:,:,status, status] = 1-np.sum(array_reshape, axis=3)
        
        self.cf_model_data['rate_arrays']['rollrate'] = array_reshape 
        
        #############################################
        # Process All other types of curves, curtail, recovery, etc
        for curve_type in self._mediator._model_config['curve_type_info'].keys():
            if curve_type not in ['default', 'prepay', 'rollrate']:
                segment_array = self._mediator._model_config['rate_curves'].return_rate_matrix(curve_type)
                segment_array = np.nan_to_num(segment_array, copy=False)
                self.cf_model_data['rate_arrays'][curve_type] = segment_array
            
        #roll rate model
        #elif self._mediator._model_config['model_type']==2: #roll rate model
            #process Roll Rates
            
        #    pass
        
        #elif self._mediator._model_config['model_type']==3: #monte carlo
        #    pass
            
            
        #final transition sums roll rates to final status each month. used if payments are configured to "Scale"
        self.cf_model_data['bom_final_trans'] = np.minimum(self.request_data(1,'BOM_PrincipalBalance'),1.0) ##start unit transition at 100% of current status
        self.cf_model_data['count_array'] = np.arange(len(self.cf_model_data['rate_arrays']['rollrate']))
        
        ########################### Apply Stress Curves #############################        
        if self._mediator._model_config['curve_stress']:
            
            ####################################################
            ## generate stress arrays
            rr_length = self.cf_model_data['rate_arrays']['rollrate'].shape
            stress_array = self._mediator._model_config['curve_stress'].return_stress_array(['default', 'prepay'], rr_length[1])
            self.cf_model_data['stress_arrays'] = {**self.cf_model_data['stress_arrays'], **stress_array}
            
            #all other types
            for key in self.cf_model_data['rate_arrays'].keys():
                if key not in ['default', 'prepay']:
                    array_length = self.cf_model_data['rate_arrays'][key].shape
                    stress_array = self._mediator._model_config['curve_stress'].return_stress_array([key], array_length[1])
                    self.cf_model_data['stress_arrays'] = {**self.cf_model_data['stress_arrays'], **stress_array}
                    
            """
            ####################################################
            #Apply stress curves
            
            #Default mask - apply stress to forward rolls only
            default_mask = np.zeros(shape=[self.cf_input_data['num_status'],self.cf_input_data['num_status']], dtype=bool)
            #forward rolls only
            from_adjust = np.array([1,2,3,4,5,6], dtype=int)
            to_adjust = np.array([2,3,4,5,6,7], dtype=int)
            default_mask[from_adjust, to_adjust]=1
            default_mask[:, 8] = 1
            
            #prepay mask
            prepay_mask = np.zeros(shape=[self.cf_input_data['num_status'],self.cf_input_data['num_status']], dtype=bool)
            prepay_mask[:,12] = 1
            
            combined_mask = np.logical_or(default_mask, prepay_mask)
            
            #apply all stress curves available
            for curve_type, stress in self.cf_model_data['stress_arrays'].items():
                if curve_type=='default':
                    default_stress = self.cf_model_data['stress_arrays']['default']
                    #apply stress to forward rolls
                    np.multiply(self.cf_model_data['rate_arrays']['rollrate'][:, :, :, :], default_stress[np.newaxis, :, np.newaxis, np.newaxis], where=default_mask[np.newaxis, np.newaxis, :, :], out=self.cf_model_data['rate_arrays']['rollrate'])
                    np.clip(self.cf_model_data['rate_arrays']['rollrate'], a_min=0.0, a_max=1.0, out=self.cf_model_data['rate_arrays']['rollrate'])
                    
                elif curve_type=='prepay':
                    prepay_stress = self.cf_model_data['stress_arrays']['prepay']
                    np.multiply(self.cf_model_data['rate_arrays']['rollrate'][:, :, :, :], prepay_stress[np.newaxis, :, np.newaxis, np.newaxis], where=prepay_mask[np.newaxis, np.newaxis, :, :], out=self.cf_model_data['rate_arrays']['rollrate'])
                    np.clip(self.cf_model_data['rate_arrays']['rollrate'], a_min=0.0, a_max=1.0, out=self.cf_model_data['rate_arrays']['rollrate'])
                    
                else: #curve_type=='curtail':      
                    np.multiply(self.cf_model_data['rate_arrays'][curve_type], self.cf_model_data['stress_arrays'][curve_type][np.newaxis, :], out=self.cf_model_data['rate_arrays'][curve_type])
                    np.clip(self.cf_model_data['rate_arrays'][curve_type], a_min=0.0, a_max=1.0, out=self.cf_model_data['rate_arrays'][curve_type])
                
            #Normalize Roll Rates Non-Forward rolls so array ties out to 100%
            rr_variance = np.sum(self.cf_model_data['rate_arrays']['rollrate'][:, :, :, :], axis=3) - 1
            #tilde (~) is a python operator to invert the input. The combined mask is a boolean array. adding the tilde will invert each value (True -> False, False -> True)
            non_fwd_roll_total = np.sum(self.cf_model_data['rate_arrays']['rollrate'][:, :, :, :], where=~combined_mask[np.newaxis, np.newaxis, :, :], axis=3)
            non_fwd_roll_scale = np.divide(self.cf_model_data['rate_arrays']['rollrate'][:, :, :, :], non_fwd_roll_total[:, :, np.newaxis], where=~combined_mask)
            variance_alloc = np.multiply(non_fwd_roll_scale, rr_variance[:, :, np.newaxis, :])
            np.subtract(self.cf_model_data['rate_arrays']['rollrate'], variance_alloc, out=self.cf_model_data['rate_arrays']['rollrate'])
            np.around(self.cf_model_data['rate_arrays']['rollrate'], decimals=6, out=self.cf_model_data['rate_arrays']['rollrate'])
        """
            
        ###################### initialize first month rates #############################
        self.run_module()     
    
    def create_rr_mask(self):
        default_mask = np.zeros(shape=[self.cf_input_data['num_status'],self.cf_input_data['num_status']], dtype=bool)
        #forward rolls only
        from_adjust = np.array([1,2,3,4,5,6], dtype=int)
        to_adjust = np.array([2,3,4,5,6,7], dtype=int)
        default_mask[from_adjust, to_adjust]=1
        default_mask[:, 8] = 1
        
        #prepay mask
        prepay_mask = np.zeros(shape=[self.cf_input_data['num_status'],self.cf_input_data['num_status']], dtype=bool)
        prepay_mask[:,12] = 1
        
        combined_mask = np.logical_or(default_mask, prepay_mask)
        
        return default_mask, prepay_mask, combined_mask
    
    def apply_stress_curves(self, projection_month):
        
        ####################################################
        #Apply stress curves
        
        #Default mask - apply stress to forward rolls only
        default_mask = np.zeros(shape=[self.cf_input_data['num_status'],self.cf_input_data['num_status']], dtype=bool)
        #forward rolls only
        from_adjust = np.array([1,2,3,4,5,6], dtype=int)
        to_adjust = np.array([2,3,4,5,6,7], dtype=int)
        default_mask[from_adjust, to_adjust]=1
        default_mask[:, 8] = 1
        
        #prepay mask
        prepay_mask = np.zeros(shape=[self.cf_input_data['num_status'],self.cf_input_data['num_status']], dtype=bool)
        prepay_mask[:,12] = 1
        
        combined_mask = np.logical_or(default_mask, prepay_mask)
        
        #apply all stress curves available
        for curve_type, stress in self.cf_model_data['stress_arrays'].items():
            if curve_type=='default':
                default_stress = self.cf_model_data['stress_arrays']['default'][projection_month]
                #apply stress to forward rolls
                np.multiply(self.cf_model_data['cur_month_rates']['rollrate'][:, :, :], default_stress, where=default_mask[np.newaxis, :, :], out=self.cf_model_data['cur_month_rates']['rollrate'])
                np.clip(self.cf_model_data['cur_month_rates']['rollrate'], a_min=0.0, a_max=1.0, out=self.cf_model_data['cur_month_rates']['rollrate'])
                
            elif curve_type=='prepay':
                prepay_stress = self.cf_model_data['stress_arrays']['prepay'][projection_month]
                np.multiply(self.cf_model_data['cur_month_rates']['rollrate'][:, :, :], prepay_stress, where=prepay_mask[np.newaxis, :, :], out=self.cf_model_data['cur_month_rates']['rollrate'])
                np.clip(self.cf_model_data['cur_month_rates']['rollrate'], a_min=0.0, a_max=1.0, out=self.cf_model_data['cur_month_rates']['rollrate'])
                
            else: #curve_type=='curtail':      
                np.multiply(self.cf_model_data['cur_month_rates'][curve_type], self.cf_model_data['stress_arrays'][curve_type][projection_month], out=self.cf_model_data['cur_month_rates'][curve_type])
                np.clip(self.cf_model_data['cur_month_rates'][curve_type], a_min=0.0, a_max=1.0, out=self.cf_model_data['cur_month_rates'][curve_type])
         
        
        #Normalize Roll Rates Non-Forward rolls so array ties out to 100%
        rr_variance = np.sum(self.cf_model_data['cur_month_rates']['rollrate'][:, :, :], axis=2) - 1
        #tilde (~) is a python operator to invert the input. The combined mask is a boolean array. adding the tilde will invert each value (True -> False, False -> True)
        non_fwd_roll_total = np.sum(self.cf_model_data['cur_month_rates']['rollrate'][:, :, :], where=~combined_mask[np.newaxis, :, :], axis=2)
        #calc the % spread among remaining rolls
        non_fwd_roll_scale = np.divide(self.cf_model_data['cur_month_rates']['rollrate'][:, :, :], non_fwd_roll_total[:, np.newaxis, :], where=~combined_mask[np.newaxis, :, :])
        #ensure stressed rolls are zeroed out in scale array
        mask_full = np.broadcast_to(combined_mask, non_fwd_roll_scale.shape)
        non_fwd_roll_scale = np.where(mask_full, 0, non_fwd_roll_scale)
        #plug any nans
        non_fwd_roll_scale = np.nan_to_num(non_fwd_roll_scale)
        variance_alloc = np.multiply(non_fwd_roll_scale, rr_variance[:, np.newaxis, :])
        #variance_alloc = np.multiply(non_fwd_roll_total, rr_variance[:, np.newaxis, :])
        np.subtract(self.cf_model_data['cur_month_rates']['rollrate'], variance_alloc, out=self.cf_model_data['cur_month_rates']['rollrate'])
        np.around(self.cf_model_data['cur_month_rates']['rollrate'], decimals=6, out=self.cf_model_data['cur_month_rates']['rollrate'])
        
        
    
    def get_curve_lookup(self, curve_type):
        """
        finds curve type from lookup array and returns correct matrix
        """
        if self.curve_type_info[curve_type][0]=='MOB':
            lookup_array = self.cf_input_data['MonthsOnBook']
        else:
            lookup_array = self.cf_input_data['CalendarMonth']
            
        return lookup_array
            
    def run_module(self):
        """
        extracts the current month transition rates for each account 
        into a separate variable. several modules use these rates
        """
        self.update_cf_inputs()
        
        #mob_lookup = self.cf_input_data['MonthsOnBook']
        #cal_lookup = self.cf_input_data['CalendarMonth'] #do we need to subtract 1 from this?
        
        ############################## Roll Rates ###################################
        #roll rates are all shifted based on MOB. so we lookup RR based on Cal month ##NOT ANYMORE, DEPRICATED
        #self.cf_model_data['cur_month_rates']['rollrate']=self.cf_model_data['rate_arrays']['rollrate'][self.segment_keys['rollrate'],self.cf_input_data['MonthsOnBook']]
        
        ##select mob or cal based on curve type
        try:
            lookup_array = self.get_curve_lookup('rollrate')
        except:
            lookup_array = self.get_curve_lookup('default')
        self.cf_model_data['cur_month_rates']['rollrate']=self.cf_model_data['rate_arrays']['rollrate'][self.segment_keys['rollrate'],lookup_array]
        
        # for accounts pre acquisition update rolls to keep balance in 13 until the correct month
        pre_acq_shift = np.where(self.cf_input_data['MonthsToAcquisition']==0)
        self.cf_model_data['cur_month_rates']['rollrate'][pre_acq_shift,13,:] = 0 #at acquisition month "from 13" set to zero
        self.cf_model_data['cur_month_rates']['rollrate'][pre_acq_shift,13,1] = 1 #at acquisition month "from 13 to 1" set to 1
            
        
        if 'eom_final_trans' in self.cf_model_data:
            np.multiply(self.cf_model_data['bom_final_trans'][:,:,np.newaxis], self.cf_model_data['cur_month_rates']['rollrate'], out=self.cf_model_data['eom_final_trans'])
        else:
            self.cf_model_data['eom_final_trans'] = self.cf_model_data['bom_final_trans'][:,:,np.newaxis] * self.cf_model_data['cur_month_rates']['rollrate']
        ############################## Others #######################################
        
        #recovery return entire curve. 
        #if recovery curve not loaded replace with zero
        if 'recovery' in self.cf_model_data['rate_arrays'].keys():
            self.cf_model_data['cur_month_rates']['recovery'] = self.cf_model_data['rate_arrays']['recovery'][self.segment_keys['recovery']]
        else:
            self.cf_model_data['cur_month_rates']['recovey'] = np.zeros(shape=(len(self.segment_keys), 100))
        
        #any others grab current month
        #for s in ['curtail']:
        for s in self.cf_model_data['rate_arrays'].keys() - ['rollrate', 'recovery']:
            #lookup_array = mob_lookup if self.curve_type_info[s][0] == 'MOB' else cal_lookup
            lookup_array = self.get_curve_lookup(s)
            #print(s)
            #print(lookup_array)
            self.cf_model_data['cur_month_rates'][s] = self.cf_model_data['rate_arrays'][s][self.segment_keys[s],lookup_array]
            
        #############################
        #apply stress arrays
        if self._mediator._model_config['curve_stress']:
            self.apply_stress_curves(self.cf_input_data['CalendarMonth'][0])
        
        
    def run_eom_reset(self):
        np.sum(self.cf_model_data['eom_final_trans'], axis=1, out=self.cf_model_data['bom_final_trans'])       
        
class CalcRollRateConvert(CashFlowBaseModule):
    """
    Class to convert CDR/CPR models into roll rates necessary for PD calculations
    """
    def __init__(self, mediator):
        super().__init__(mediator)
        
        self._cf_input_fields = ['BOM_PrincipalBalance', 'TotalPrincipalBalance', 'num_status']
        self.cf_input_data = self.request_all(self._cf_input_fields)
        self.cf_model_data = {}
        
        #create base roll rate matrix
        
        
        
class CalcBalance(CashFlowBaseModule):
    """
    Class responsible for balances. BOM, EOM and transitions between status
    """
    def __init__(self, mediator):
        super().__init__(mediator)
        
        self._cf_input_fields = ['BOM_PrincipalBalance','TotalPrincipalBalance', 'InterestBalance','cur_month_rates','num_cohorts','num_status']
        self.cf_input_data = self.request_all(self._cf_input_fields)
        self.cf_model_data = {}
        
        #bom values
        self.cf_model_data['BOM_PrincipalBalance'] = self.cf_input_data['BOM_PrincipalBalance'] #self.request_data(1,'BOM_PrincipalBalance')
        self.cf_model_data['bom_int'] = self.cf_input_data['InterestBalance'] #self.request_data(1,'InterestBalance')
        #change this to a unit count based on account number
        self.cf_model_data['bom_units'] = np.minimum(self.cf_model_data['BOM_PrincipalBalance'],1.0) ##start unit transition at 100% of current status
        
        #Balance Transition
        self.cf_model_data['upb_trans'] = np.zeros((self._mediator._model_config['num_cohorts'], self._mediator._model_config['num_status'], self._mediator._model_config['num_status']), dtype='float32')
        self.cf_model_data['int_trans'] = np.zeros((self._mediator._model_config['num_cohorts'], self._mediator._model_config['num_status'], self._mediator._model_config['num_status']), dtype='float32')
        self.cf_model_data['units_trans'] = np.zeros((self._mediator._model_config['num_cohorts'], self._mediator._model_config['num_status'], self._mediator._model_config['num_status']), dtype='float32')
        
        self.cf_model_data['upb_trans_agg'] = np.zeros((self._mediator._model_config['num_cohorts'], self._mediator._model_config['num_status'], self._mediator._model_config['num_status']), dtype='float32')
        self.cf_model_data['int_trans_agg'] = np.zeros((self._mediator._model_config['num_cohorts'], self._mediator._model_config['num_status'], self._mediator._model_config['num_status']), dtype='float32')
        self.cf_model_data['units_trans_agg'] = np.zeros((self._mediator._model_config['num_cohorts'], self._mediator._model_config['num_status'], self._mediator._model_config['num_status']), dtype='float32')
        
        #eom values
        #self.cf_model_data['TotalPrincipalBalance'] = np.zeros_like(self.cf_model_data['BOM_PrincipalBalance'], dtype='float32')
        self.cf_model_data['TotalPrincipalBalance'] = self.cf_input_data['TotalPrincipalBalance']
        self.cf_model_data['InterestBalance'] = np.zeros_like(self.cf_model_data['bom_int'], dtype='float32')
        self.cf_model_data['eom_units'] = np.zeros_like(self.cf_model_data['bom_units'], dtype='float32')
        
        #calculate month 0 values
        #np.multiply(self.cf_model_data['BOM_PrincipalBalance'][:,:, np.newaxis], self.cf_input_data['cur_month_rates']['rollrate'], out=self.cf_model_data['upb_trans'])
        #self.update_transition_balance('add', 'upb', self.cf_model_data['upb_trans'])
        
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
        if target=='upb':
            target_eom = 'TotalPrincipalBalance'
        elif target=='int':
            target_eom = 'InterestBalance'
        else:
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
        self.update_cf_inputs()
        #numpy operators to save memory
        np.multiply(self.cf_model_data['BOM_PrincipalBalance'][:,:, np.newaxis], self.cf_input_data['cur_month_rates']['rollrate'], out=self.cf_model_data['upb_trans'])
        np.multiply(self.cf_model_data['bom_int'][:,:, np.newaxis], self.cf_input_data['cur_month_rates']['rollrate'], out=self.cf_model_data['int_trans'])
        np.multiply(self.cf_model_data['bom_units'][:,:, np.newaxis], self.cf_input_data['cur_month_rates']['rollrate'], out=self.cf_model_data['units_trans'])
        
        #send values to eom
        #kind of redundant since its stored in this class, but this is the interface all other 
        #classes use. so we are keeping it
        self.send_output('add', 'upb', self.cf_model_data['upb_trans'])
        self.send_output('add', 'int', self.cf_model_data['int_trans'])
        self.send_output('add', 'units', self.cf_model_data['units_trans'])
        
    def run_eom_reset(self):
        #set bom balance = eom balance        
        np.copyto(self.cf_model_data['BOM_PrincipalBalance'], self.cf_model_data['TotalPrincipalBalance'])
        np.copyto(self.cf_model_data['bom_int'], self.cf_model_data['InterestBalance'])
        np.copyto(self.cf_model_data['bom_units'], self.cf_model_data['eom_units'])
        
        #reset Balance Transition
        self.cf_model_data['upb_trans'].fill(0)
        self.cf_model_data['int_trans'].fill(0)
        self.cf_model_data['units_trans'].fill(0)
        self.cf_model_data['upb_trans_agg'].fill(0)
        self.cf_model_data['int_trans_agg'].fill(0)
        self.cf_model_data['units_trans_agg'].fill(0)
        
        #reset eom values
        self.cf_model_data['TotalPrincipalBalance'].fill(0)
        self.cf_model_data['InterestBalance'].fill(0)
        self.cf_model_data['eom_units'].fill(0)


class CalcInterest(CashFlowBaseModule):
    """
    Class to hold Interest rate information and reset rate when necessary
    """
    def __init__(self, mediator, int_accrue_matrix=None, compound_type=None):
        super().__init__(mediator)
        
        #initialize fields at T0
        self._cf_input_fields=['AsOfDate', 'MonthsOnBook', 'cur_month_rates', 'upb_trans_agg']
        #'num_status', 'num_cohorts', 'account_status_list'
        self.cf_input_data = self.request_all(self._cf_input_fields)
        self.cf_model_data = {}
        
        #raw inputs from data tape
        if not compound_type:
            self._compound_type = 'Monthly'
        else:
            self._compound_type = compound_type
            
        self.fixed_rate = self.request_data(1,'InterestRate')
        
        self.cf_model_data['InterestRateType'] = self.request_data(1,'InterestRateType')
        self.cf_model_data['InterestRateIndex'] = self.request_data(1,'InterestRateIndex')
        self.cf_model_data['InterestMargin'] = self.request_data(1,'InterestMargin')
        self.cf_model_data['InterestRateChangeFrequency'] = self.request_data(1,'InterestRateChangeFrequency')
        self.cf_model_data['MaxInterestRate'] = self.request_data(1,'MaxInterestRate')
        self.cf_model_data['MinInterestRate'] = self.request_data(1,'MinInterestRate')
        
        self.cf_model_data['InterestRate'] = self.cf_model_data['InterestMargin'].astype(np.float32)
        self.cf_model_data['int_rate_calc'] = np.zeros_like(self.cf_model_data['InterestRate'])
        self.cf_model_data['int_accrue'] = np.zeros((self._mediator._model_config['num_cohorts'], self._mediator._model_config['num_status'], self._mediator._model_config['num_status']), dtype='float32')

        
        
        #default fields (editable by user)
        #self.status_accrue_active = np.array([1,1,1,1,1,1,1,1,0,1,1,0,0])
        
        if int_accrue_matrix:
            self.int_accrue_matrix = int_accrue_matrix
        else:
            self.create_int_matrix() 
        
        self.calc_int_rate()
            
    def create_int_matrix(self):
        """
        Generates interest accrue matrix. matrix of statuses on where to accrue interest or not.
        All Start at 100% except for 8, 12, and 13. (default, prepay, and Not yet purchased) 
        if Default/prepay then 0% interest accrue
        """
        self.int_accrue_matrix = np.ones((self._mediator._model_config['num_status'], self._mediator._model_config['num_status']), dtype='float32')
        #from status zero payment
        self.int_accrue_matrix[[8,12,13],[8,12,13]] = 0
        
    def update_active_matrix(self, active_matrix):
        self.int_accrue_matrix = active_matrix
        
    def calc_int_rate(self):
        
        #choose variable or fixed
        cond_list = [self.cf_model_data['InterestRateType']=='Variable', self.cf_model_data['InterestRateType']=='Fixed'] #self.cf_input_data['MonthsOnBook']<0,
        choice_list = [self.cf_model_data['InterestMargin'], self.fixed_rate] #variable should be margin+index #np.zeros((self.cf_input_data['InterestMargin'].shape)) ,
        rate = np.select(cond_list, choice_list)
        
        #add index
        np.add(rate, self.cf_input_data['cur_month_rates']['index'], out=self.cf_model_data['int_rate_calc'])
        
        #upper and lower bounds
        np.clip(self.cf_model_data['int_rate_calc'], self.cf_model_data['MinInterestRate'], self.cf_model_data['MaxInterestRate'], out=self.cf_model_data['int_rate_calc'])
        
        #compound calculation
        if self._compound_type == 'monthly':
            rate = self.cf_model_data['int_rate_calc']/12
        elif self._compound_type == 'daily':
            #rate/365 * days in month
            rate = (self.cf_model_data['int_rate_calc']/365)* np.array([m.day for m in self.request_data(2,'AsOfDate')])
        elif self._compound_type == 'continuous':
            pass
        
    
        #if interest change month choose new calc, else keep old value
        #assume MOB is from origination. in that case just modulo on MOB. 
        #mob%3==0
        self.cf_model_data['rate_change_trigger'] = self.cf_input_data['MonthsOnBook'] % self.cf_model_data['InterestRateChangeFrequency'] == 0
        self.cf_model_data['InterestRate'] = np.where(self.cf_model_data['rate_change_trigger'], rate, self.cf_model_data['InterestRate'])
        
        #self.cf_model_data['InterestRate'] = np.array(rate, dtype='float32')
    
    def calc_int_accrued(self):
        
        #apply rate to bom upb (using transition data)
        #self.cf_model_data['int_accrue'] = self.request_data(2,'upb_trans_agg') * self.cf_model_data['InterestRate'][:,np.newaxis, np.newaxis] * self.int_accrue_matrix
        np.multiply(self.cf_input_data['upb_trans_agg'], self.cf_model_data['InterestRate'][:, np.newaxis, np.newaxis], out=self.cf_model_data['int_accrue'])
        np.multiply(self.cf_model_data['int_accrue'], self.int_accrue_matrix, out=self.cf_model_data['int_accrue'])
        
    def run_module(self):
        self.update_cf_inputs()
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
        
    inputs:
        amort_formula - dict with amortization formulas, ordered by priority
        amort_timing - dict with amortization timing, ordered by priority
        pmt_matrix - 2d array
        default_amort_type - monthly or scale, recalculate payment every single month, or 
        
    """
    def __init__(self, mediator, pmt_matrix=None):
        super().__init__(mediator)
        
        self._cf_input_fields = ['AsOfDate', 'ProjectionMonth', 'CalendarMonth', 'month_of_year','MonthsOnBook'
                                 ,'RemainingTerm', 'InterestRate','BOM_PrincipalBalance', 'int_trans_agg', 'upb_trans_agg'
                                 ,'units_trans', 'bom_final_trans','cur_month_rates', 'num_status', 'account_status_active',
                                 'num_cohorts']
        self.cf_input_data = self.request_all(self._cf_input_fields)
        self.cf_model_data = {}
        
        #initialize cf model fields
        self.cf_model_data['ScheduledPaymentAmount'] = self.request_data(1,'ScheduledPaymentAmount') #from inputs
        self.cf_model_data['ScheduledPaymentCalc'] = self.request_data(1,'ScheduledPaymentAmount') # np.zeros((self.cf_input_data['num_cohorts'], self.cf_input_data['num_status']), dtype='float32') #last calculated payment value
        self.cf_model_data['TotalPaymentMade'] = np.zeros((self.cf_input_data['num_cohorts'], self.cf_input_data['num_status'], self.cf_input_data['num_status']), dtype='float32') #np.zeros_like(self.cf_model_data['sch_pmt'])
        self.cf_model_data['InterestPayment'] = np.zeros((self.cf_input_data['num_cohorts'], self.cf_input_data['num_status'], self.cf_input_data['num_status']), dtype='float32') #np.zeros_like(self.cf_model_data['sch_pmt'])
        self.cf_model_data['ContractualPrincipalPayment'] = np.zeros((self.cf_input_data['num_cohorts'], self.cf_input_data['num_status'], self.cf_input_data['num_status']), dtype='float32') #np.zeros_like(self.cf_model_data['sch_pmt'])
        self.cf_model_data['PrincipalPartialPrepayment'] = np.zeros((self.cf_input_data['num_cohorts'], self.cf_input_data['num_status'], self.cf_input_data['num_status']), dtype='float32') #np.zeros_like(self.cf_model_data['sch_pmt'])
        self.cf_model_data['curtail_calc'] = np.zeros((self.cf_input_data['num_cohorts'], self.cf_input_data['num_status'], self.cf_input_data['num_status']), dtype='float32') #np.zeros_like(self.cf_model_data['sch_pmt'])
        
        self.cf_model_data['sch_ppmt'] = np.zeros((self.cf_input_data['num_cohorts'], self.cf_input_data['num_status'], self.cf_input_data['num_status']), dtype='float32') #np.zeros_like(self.cf_model_data['sch_pmt'])
        self.cf_model_data['pmt_remain'] = np.zeros((self.cf_input_data['num_cohorts'], self.cf_input_data['num_status'], self.cf_input_data['num_status']), dtype='float32') #np.zeros_like(self.cf_model_data['sch_pmt'])
        
        self.cf_model_data['sch_pmt_trans'] = np.zeros_like(self.cf_input_data['upb_trans_agg'])
        
        #user input fields
        if pmt_matrix:
            self.pmt_matrix = pmt_matrix
        else:
            self.pmt_matrix = np.zeros((self.cf_input_data['num_status'], self.cf_input_data['num_status']), dtype='float32')
            self.create_payment_matrix() 
            
        #self.amort_calc_type = 'monthly'
        self.amort_formula = {
            #0:"np.pmt(cf_input_data['InterestRate'][:,np.newaxis], cf_input_data['RemainingTerm'][:,np.newaxis], cf_input_data['BOM_PrincipalBalance'])",
            #1:"np.zeros_like(cf_input_data['BOM_PrincipalBalance'])",
            1:"np.zeros_like(cf_input_data['BOM_PrincipalBalance'])",
            99:"npf.pmt(cf_input_data['InterestRate'][:,np.newaxis], cf_input_data['RemainingTerm'][:,np.newaxis], cf_input_data['BOM_PrincipalBalance'])"
        }
        self.amort_timing = {
            #0:"np.isnan(np.sum(cf_model_data['ScheduledPaymentAmount'], axis=1))[:,np.newaxis]", #if scheduled payment is null
            #98:"(np.sum(cf_input_data['BOM_PrincipalBalance'] * cf_input_data['account_status_active'], axis=1) == 0)[:,np.newaxis]", #if active UPB ==0 just set to 0
            #1:"(cf_input_data['BOM_PrincipalBalance'] * cf_input_data['account_status_active']) == 0", #if account_status is inactive (default, prepayment, etc.), set payment to zero, Balance will roll into these statuses and remain there permanently. 
            1: "cf_input_data['BOM_PrincipalBalance'] == 0", #if BOM balance is zero set payment to zero
            99:"" #if no other rule triggered
        }

        #set base payment behavior
        self.change_amort_calc_type(self._mediator._model_config['default_amort_type'])
        #add custom amort rules
        if self._mediator._model_config['amort_timing']:
            for rule in self._mediator._model_config['amort_timing'].keys():
                timing = self._mediator._model_config['amort_timing'][rule]
                amort_formula = self._mediator._model_config['amort_formula'][rule]
                amort_formula = None if amort_formula=='' else amort_formula
                self.add_new_amort_rule(timing, amort_formula)
        
        #update field list for payment formulas
        #self.add_cf_input_fields()

        #run calc to set initial values
        self.run_module()

    def add_cf_input_fields(self, rule):
        #find all columns referenced in formulas
        field_list = self._cf_input_fields
        new_list = []
        
        field_list.extend(re.findall('\[.*?\]', rule))

        #format field names
        for col in np.unique(field_list):
            new_list.append(col.replace("['",'').replace("']",''))
        try:
            new_list.remove('[:,np.newaxis]')
        except:
            pass
        
        #remove any references to fields already maintained in this class
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
        if prepay then 0% payment (all remaining principal is prepayment)
        """
        #fill diagonals
        #same status = 100% pmt
        np.fill_diagonal(self.pmt_matrix, 1.0)
        #Delinquent cure = 200% pmt
        for x in range(1, self.cf_input_data['num_status']):
            np.fill_diagonal(self.pmt_matrix[x:8,:], 1.0+x)
        
        #to payoff make one normal payment
        self.pmt_matrix[:,12]=0
        #from status zero payment
        self.pmt_matrix[[0,8,11,12,13],:] = 0
        #to status zero payment
        self.pmt_matrix[:,[0,8,9,11,13]] = 0
        #accounts enter repay
        self.pmt_matrix[[13,14], 1] = 1
        
    
    def change_amort_calc_type(self, ream_type):
        """
        
        """
        if ream_type=='monthly':
            self.amort_formula[99] = "npf.pmt(cf_input_data['InterestRate'][:,np.newaxis], cf_input_data['RemainingTerm'][:,np.newaxis], cf_input_data['BOM_PrincipalBalance'])"
        elif ream_type =='scale':
            #take the scheduled payment calculated last month and scale down based on roll rates
            self.amort_formula[99] = "cf_model_data['ScheduledPaymentAmount']" 
            
        self.amort_calc_type=ream_type
        
        #update key array
        self.create_amort_key_array()

    def add_new_amort_rule(self, time_rule, custom_amort_formula = None):
        """
        create rules to tell model when to recalulate amortization. 
            -time_rule is the trigger for the reamortization. This can use [MonthsOnBook], [month_of_year], [ProjectionMonth] 
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
                npf.pmt(InterestRate, Remaining Term, UPB)
        """
        #create new rule key
        rule_key = sorted(self.amort_formula.keys())[-2]+1
        
        ##################################
        #process time_rule
            
        self.add_cf_input_fields(time_rule)
        time_rule = self.format_column_inputs(time_rule)

        #Final eval formula result must have 2 axis in order to broadcast later
        if (eval(time_rule, globals(), self.__dict__)).ndim==1:
            time_rule = '(' + time_rule +")[:,np.newaxis]"
        
        ##################################
        #process amort_formula
        #validate inputs
        if custom_amort_formula:
            self.add_cf_input_fields(custom_amort_formula)
            formula = self.format_column_inputs(custom_amort_formula)
        else:
            formula = "npf.pmt(cf_input_data['InterestRate'][:,np.newaxis], cf_input_data['RemainingTerm'][:,np.newaxis], cf_input_data['BOM_PrincipalBalance'])"

        #add rules into dicts
        self.amort_timing[rule_key] = time_rule
        self.amort_formula[rule_key] = formula
            
        #update key array
        self.create_amort_key_array()
        
    def format_column_inputs(self, full_rule):
        #find referenced columns
        col_matches = re.findall('\[.*?\]',full_rule)
        for col in np.unique(col_matches):
            #add source df
            if col in self.cf_model_data.keys():
                new_axis=''
                #if the input is an integer leave as is.
                if isinstance(self.cf_input_data[col.replace("['",'').replace("']",'')], int):
                    new_axis=''
                #if input is a 1D array expand to 2D 
                elif self.cf_model_data[col.replace("['",'').replace("']",'')].ndim==1:
                    new_axis = "[:,np.newaxis]"
            else:
                new_axis=''
                if isinstance(self.cf_input_data[col.replace("['",'').replace("']",'')], int):
                    new_axis=''
                elif self.cf_input_data[col.replace("['",'').replace("']",'')].ndim==1:
                    new_axis = "[:,np.newaxis]"
            
            col_replace = col.replace("[","cf_input_data[").replace("]","]"+new_axis)
            #replace final column reference
            full_rule = full_rule.replace(col,col_replace)
            
        return full_rule
        
    def create_amort_key_array(self):
        self.amort_key_array = [np.full((self.cf_input_data['num_cohorts'],self.cf_input_data['num_status']), key) for key in sorted(self.amort_timing.keys()) if key != 99]
    
    ########################################################################
    #               Cash Flow Calculations
    def calc_sch_pmt(self):
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
      
        #broadcast to all statuses
        for i in range(len(cond_eval)):
            cond_eval[i] = np.broadcast_to(cond_eval[i], (len(cond_eval[i]),self.cf_input_data['num_status'])) #(len(cond_eval[i]),13)
        
        #sch_pmt_calc = np.select(cond_eval, select_eval, self.cf_model.data['ScheduledPaymentCalc'])
        sch_pmt = np.select(cond_eval, select_eval, default_eval)
        amort_rule_used = np.select(cond_eval, self.amort_key_array, 99)
        
        #return final payment and selected formula
        np.absolute(sch_pmt, out=self.cf_model_data['ScheduledPaymentCalc']) #final payment to be used, includes default amort function
        #multiply by active statuses only
        np.multiply(self.cf_model_data['ScheduledPaymentCalc'][:,:, np.newaxis], self.cf_input_data['cur_month_rates']['rollrate'], out=self.cf_model_data['sch_pmt_trans'])
        if self.cf_input_data['ProjectionMonth']>0: #first month of model is just to initalize many metrics. upb trans does not exist at that point
            np.clip(self.cf_model_data['sch_pmt_trans'], a_min=None, a_max=self.cf_input_data['upb_trans_agg'], out=self.cf_model_data['sch_pmt_trans'])
        self.cf_model_data['amort_rule_used'] = amort_rule_used
        
        #sum scheduled payment trans into Final EOM status
        np.sum(self.cf_model_data['sch_pmt_trans'], axis=1, out=self.cf_model_data['ScheduledPaymentAmount'])
        
    def calc_pmt_made(self):
        """
        Scheduled Payment is calculated across the beginning status in each month. 
        This has to be allocated based on transition rates
        then the payment matrix is applied to scale payment amounts
            i.e. a cure is 200% of a normal payment
        """
        np.multiply(self.cf_model_data['sch_pmt_trans'], self.pmt_matrix, out=self.cf_model_data['TotalPaymentMade'])
        np.clip(self.cf_model_data['TotalPaymentMade'], a_min=None, a_max=self.cf_input_data['upb_trans_agg'], out=self.cf_model_data['TotalPaymentMade'])
        
    def calc_pmt_curtail(self):
        """
        on CDR/CPR models curtailment curves are built as a percent of BOM UPB
        only calculate curtailment for current bucket UPB
        
        only if curtail rate is found in model
        """
        #if self.cf_input_data['cur_month_rates'].get('curtail'):
        if 'curtail' in self.cf_input_data['cur_month_rates']:
            curtail_rate = self.cf_input_data['cur_month_rates']['curtail']
            #try:
                #np.multiply(self.cf_input_data['BOM_PrincipalBalance'][:,1], curtail_rate, out=self.cf_model_data['curtail_calc'])
                
            #except:
            #    self.cf_model_data['curtail_calc'] = np.multiply(self.cf_input_data['BOM_PrincipalBalance'][:,1], curtail_rate)
            
            #only apply curtailment to active loan balance in paying statuses
            curtail_mask = np.zeros(shape=[self.cf_input_data['num_status'],self.cf_input_data['num_status']], dtype=bool)
            curtail_mask[:, [1, 2, 3, 4, 5, 6, 7]]=1
            np.multiply(self.cf_input_data['upb_trans_agg'], curtail_rate[:, np.newaxis, np.newaxis], where=curtail_mask[np.newaxis, :, :],  out=self.cf_model_data['curtail_calc'])
            
            #np.add(self.cf_model_data['TotalPaymentMade'][:,1,1], self.cf_model_data['curtail_calc'], out=self.cf_model_data['TotalPaymentMade'][:,1,1])
            np.add(self.cf_model_data['TotalPaymentMade'], self.cf_model_data['curtail_calc'], out=self.cf_model_data['TotalPaymentMade'])
    
    def calc_pmt_split(self):
        """
        Split total payments into various buckets
        results in 3 arrays:
            ipmt
            ppmt
            curtail
        we use np.clip() to cap payment at each stage
        """

        np.copyto(self.cf_model_data['pmt_remain'], self.cf_model_data['TotalPaymentMade'])
        
        #interest payment
        np.clip(self.cf_model_data['pmt_remain'], a_min=0, a_max=self.cf_input_data['int_trans_agg'], out=self.cf_model_data['InterestPayment'])
        np.subtract(self.cf_model_data['pmt_remain'], self.cf_model_data['InterestPayment'], out=self.cf_model_data['pmt_remain'])
        
        #scheduled principal 
        #scheduled payment by transition
        np.subtract(self.cf_model_data['sch_pmt_trans'], self.cf_model_data['InterestPayment'], out=self.cf_model_data['sch_ppmt'])
        np.clip(self.cf_model_data['pmt_remain'], a_min=0, a_max=self.cf_model_data['sch_ppmt'], out=self.cf_model_data['ContractualPrincipalPayment'])
        np.subtract(self.cf_model_data['pmt_remain'], self.cf_model_data['ContractualPrincipalPayment'], out=self.cf_model_data['pmt_remain'])
        
        #anything remaining is curtailment
        np.copyto(self.cf_model_data['PrincipalPartialPrepayment'], self.cf_model_data['pmt_remain'])
    
    def run_module(self):
        self.update_cf_inputs()
        self.calc_sch_pmt()
        self.calc_pmt_made()
        self.calc_pmt_curtail()
        self.calc_pmt_split()
        
        #update balances
        #add accrued interest to totals
        self.send_output('sub', 'int', self.cf_model_data['InterestPayment'])
        self.send_output('sub', 'upb', self.cf_model_data['ContractualPrincipalPayment'])
        self.send_output('sub', 'upb', self.cf_model_data['PrincipalPartialPrepayment'])
        
    def run_eom_reset(self):
        self.cf_model_data['TotalPaymentMade'].fill(0)  
        self.cf_model_data['InterestPayment'].fill(0)
        self.cf_model_data['ContractualPrincipalPayment'].fill(0) 
        self.cf_model_data['PrincipalPartialPrepayment'].fill(0) 
        self.cf_model_data['curtail_calc'].fill(0)
        
class CalcIntCapitalize(CashFlowBaseModule):
    """
    class responsible for Interest Capitalization
    only specific transitions will have capitalizations and/or at specific points
    in time. 
        i.e. annually in january
        
    """
    def __init__(self, mediator, cap_matrix=None, transition_rules = 1, timing_rules=1):
        super().__init__(mediator)
        
        self._cf_input_fields = ['AsOfDate', 'month_of_year','MonthsOnBook','RemainingTerm', 'account_status_list', 'account_status_active',
                                 'num_status', 'num_cohorts', 'InterestRate','int_trans_agg','units_trans', 'bom_units']
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
        self.update_cf_inputs()
        self.cf_model_data['int_cap'] = self.cf_input_data['int_trans_agg'] * self.cap_matrix * self.timing_rules
        
        #send balance changes
        self.send_output('sub', 'int', self.cf_model_data['int_cap'])
        self.send_output('add', 'upb', self.cf_model_data['int_cap'])
        
    def run_eom_reset(self):
        self.cf_model_data['int_cap'] = np.zeros_like(self.cf_input_data['int_cap'], dtype='float32')
        
class CalcRecovery(CashFlowBaseModule):
    """
    Calculates Recoveries based on Hitorical Defaults, Projected Defaults and Recovery Curves
    
    """
    def __init__(self, mediator):
        super().__init__(mediator)
        
        self._cf_input_fields = ['AsOfDate', 'ProjectionMonth', 'CalendarMonth', 'month_of_year','MonthsOnBook'
                                 ,'RemainingTerm', 'upb_trans_agg'
                                 ,'units_trans', 'cur_month_rates', 'num_status', 'account_status_active',
                                 'num_cohorts', 'segment_keys']
        self.cf_input_data = self.request_all(self._cf_input_fields)
        self.cf_model_data = {}
        
        #initalize with Historical Chargeoffs
        self.recovery_curves = self.cf_input_data['cur_month_rates']['recovery']
        self.cf_model_data['hist_default'] = self._mediator._model_config['rate_curves'].return_historical_defaults()
        self.cf_model_data['hist_recovery'] = self.shift_historical_defaults(self.cf_model_data['hist_default'])
        
        #initialize empty arrays 
        self.cf_model_data['total_defaults'] = np.zeros((self.cf_input_data['num_cohorts']))
        self.cf_model_data['PostChargeOffCollections'] = np.zeros((self.cf_input_data['num_cohorts'], self.recovery_curves.shape[1]))
    
    def shift_historical_defaults(self, hist_df):
        
        curve_lookup = hist_df.index.get_level_values('recovery_curve').values
        default_amount = hist_df['ChargeOffAmount'].astype('float32').values 
        recovery_amount = default_amount[:, np.newaxis] * self.recovery_curves[curve_lookup]
        projection_shift = hist_df['ProjectionMonth'].values.copy()
        projection_month = hist_df['ProjectionMonth'].values.copy()
        
        #shift by negative time
        rows, column_indices = np.ogrid[:recovery_amount.shape[0], :recovery_amount.shape[1]]
        projection_shift[projection_shift<0] += recovery_amount.shape[1]
        column_indices = column_indices - projection_shift[:, np.newaxis]
        result = recovery_amount[rows, column_indices]
        
        #set shifted values to zero
        shift_ix = -projection_month[:,np.newaxis] + np.arange(recovery_amount.shape[1])
        mask = ~(shift_ix<recovery_amount.shape[1])
        result[mask] = np.nan
        return result
        
    def run_module(self):
        self.update_cf_inputs()
        #aggregate default amounts
        #sum up all balance entering into 8. [exclude 8 to 8]
        from_index = np.arange(0,self.cf_input_data['num_status'])
        from_index = np.delete(from_index, 8)

        np.sum(self.cf_input_data['upb_trans_agg'][:, from_index, 8], axis=1, out=self.cf_model_data['total_defaults'])
        np.multiply(self.cf_model_data['total_defaults'][:, np.newaxis], self.cf_input_data['cur_month_rates']['recovery'], out=self.cf_model_data['PostChargeOffCollections'])
        
        #pad for current projection month
        #np.pad()
        #unsure if this is necessary yet. might be able to do this in the eval module
        
        
    def run_eom_reset(self):
        self.cf_model_data['PostChargeOffCollections'].fill(0)
        self.cf_model_data['total_defaults'].fill(0)
        

        
        
    