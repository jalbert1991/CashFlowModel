# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 22:24:57 2019

@author: jalbert
"""

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import sqlalchemy as sa
import pandas as pd
import numpy as np
import time
import datetime as date
from dateutil.relativedelta import relativedelta
import warnings
import itertools
import json
import gc
import json
from contextlib import contextmanager
import sys, os
#import openpyxl as xl
#import os
#import pathlib
#import io
#import re
#import tkinter as tkr
#from tkinter import ttk
#import PySimpleGUI as sg


#from dateutil.relativedelta import *
#from dataclasses import dataclass

#import project modules
import CashFlowEngine.Models as CF_Models
import DataPrep.DataMover as DataPrep
import GUI.DataPrepGUI as GUI
import EvalResults.EvalResults as Evaluate

#set default settings
pd.options.display.float_format = '{:20,.2f}'.format

    
class CashFlowModel():
    """
    wrapper for cash flow model and all supporting methods
    """
    
    #define class attribute for sql engine
    sql_engine_model = DataPrep.SQLEngine(server_name='srv-produs-dw', db_name='FRA')
    
    def __init__(self, model_id, model_name, deal_ids, batch_keys, asset_class, uw_id=None, uw_month=None, uw_type=None): #asofdate
        """
        initialize the Cash Flow Model and store model level parameters
        """
        
        self.model_id = model_id
        self.model_name = model_name
        self.deal_ids = deal_ids
        self.batch_keys = batch_keys
        self.asset_class = asset_class
        #self.is_template = template
        self.model_template_key = None
        self.model_template_name = None
        self.uw_id = uw_id
        self.uw_month = uw_month
        self.uw_type = uw_type
        
        self.cf_output_status = pd.DataFrame()
        self.cf_output = pd.DataFrame()
        
        # model inputs/scenarios
        self.data_tape = None 
        self.prior_uw_projections = None
        self.rate_curve_groups={}
        self.model_configs={}        
        self.cf_scenarios = {}
        
        self.data_prep = DataPrep.DataPrep()
        self.gui_data_prep = GUI.DataPrepMain(self.data_prep)
        self.model_builder = CF_Models.ModelBuilder()
        model_details = (self.model_name, self.model_id, self.deal_ids, self.batch_keys, self.uw_type, self.uw_month)
        self.eval = Evaluate.AVP(self.cf_scenarios, model_details)
        
    @classmethod
    def load_model(cls, model_name, uw_month=None, scenario_list=[]):
        """
        Loads parameters and configuration from a prior model
        
        Parameters
        ============
        Model_name: str
            name of model to download
        uw_month: int
            the specific underwrite to download (YYYMM)
        hist_data: bool
            if True, will import all history from DR. if false, will only import single month
        scenario_list: list
            optional:
            list of scenarios to download
        """
        
        if uw_month:
            uw_month_filter = " and uw_month='" + str(uw_month) + "'"
        else:
            uw_month_filter = ''
        
        sql = """
            	SELECT top 1 m.model_key,
                   m.model_name,
                   m.asset_class,
                   m.deal_ids,
                   m.batch_keys,
                   m.data_tape_source,
    			   u.uw_key,
    			   u.uw_month,
    			   u.uw_type,
                   m.create_ts,
                   m.create_user
    		FROM fra.cf_model.model m
    			left JOIN fra.cf_model.model_uw u
    				ON u.model_key = m.model_key
            where 1=1 
                and m.model_name='{}' 
                {}
            order by uw_month desc
            """.format(model_name, uw_month_filter)
        
        model_attr = cls.sql_engine_model.execute(sql, output=True)
        model_attr = model_attr.squeeze() 
        
        #create new model object
        loaded_model = cls(int(model_attr['model_key']), model_attr['model_name'], json.loads(model_attr['deal_ids']), json.loads(model_attr['batch_keys']), model_attr['asset_class'], model_attr['uw_key'], model_attr['uw_month'], model_attr['uw_type'])
        if model_attr['data_tape_source']=='cdw':
            loaded_model.import_data_tape_sql(source='cdw', save_source=False)
        elif not model_attr['data_tape_source']:
            pass
        elif model_attr['data_tape_source']!='excel':
            loaded_model.import_data_tape_sql(source='sql', query=model_attr['data_tape_source'], save_source=False)
        #Download CF Scenarios
        loaded_model.download_cf_scenario(model_attr['model_name'], uw_month, scenario_list, update_curve_map=False, save_scenario=False)
        
        return loaded_model
        
    @classmethod
    def create_template(cls, asset_class):
        """
        creates/updates a template scenario for an asset class

        Parameters
        ======================
        asset_class : str
            Asset class that corresponds to the CAAM. Must match exactly to be picked up by the model 
            on creating new models

        """
        
        model_name= asset_class + ' template'
        deal_ids = []
        batch_keys = []
        uw_month=None
        uw_type='template'
        
        #check if any templates already exist
        #if this asset class already exists download the most recent version
        sql="""
            SELECT TOP 1 model_key
                ,model_name
            	,scenario_key
                ,data_tape_source
            	,seq_order
            FROM fra.cf_model.vw_scenario
            WHERE 1=1
            	AND uw_type='template'
            	AND asset_class='{}'
            ORDER BY seq_order
        """.format(asset_class)
        
        model_attr = cls.sql_engine_model.execute(sql, output=True)
        
        if len(model_attr)==0:
        
            #create new model record
            sql = """
                    insert into FRA.cf_model.model
                    values('{}', '{}', '{}', '{}', null, getdate(), user_name(), {}); 
                    commit
                    select @@identity
                    """.format(model_name, asset_class, [], [], 0)
  
            #create model record
            model_id = cls.sql_engine_model.execute(sql, output=True).iloc[0][0]
            
            #create new UW record
            sql = """
                    insert into fra.cf_model.model_uw
                    values({}, '{}', '{}');
                    commit
                    select @@identity
                """.format(model_id, uw_month, uw_type)
            
            uw_id = cls.sql_engine_model.execute(sql, output=True).iloc[0][0]
        
            new_model = cls(int(model_id), model_name, deal_ids, batch_keys, asset_class, int(uw_id), uw_month, uw_type) #asofdate
            new_model.uw_id = uw_id
        
            print('New Template Created')
        
        else:
            
            print('Downloading Most Recent Template')
            #raise Exception('This model name already exists. Please try another name')
            #if this asset class already exists download the most recent version
            sql="""
                SELECT TOP 1 model_key
                    ,model_name
                    ,uw_key
                	,scenario_key
                    ,data_tape_source
                	,seq_order
                FROM fra.cf_model.vw_scenario
                WHERE 1=1
                	AND uw_type='template'
                	AND asset_class='{}'
                ORDER BY seq_order
            """.format(asset_class)
            
            model_attr = cls.sql_engine_model.execute(sql, output=True)
            
            #create python model object
            new_model = cls(int(model_attr['model_key']), model_attr['model_name'], deal_ids, batch_keys, asset_class, int(model_attr['uw_key']), uw_month,  uw_type)
            
            #download latest scenario
            if len(model_attr)>0:
                model_attr = model_attr.squeeze() 
                new_model.model_template_key = model_attr['model_key']
                new_model.model_template_name = model_attr['model_name']
                #import data tape
                if model_attr['data_tape_source']=='cdw':
                    new_model.import_data_tape_sql(source='cdw', save_source=True)
                elif model_attr['data_tape_source']!='excel':
                    new_model.import_data_tape_sql(source='sql', query=model_attr['data_tape_source'], save_source=True)
                #import latest scenario
                #new_model.download_model_template()
                #Download CF Scenarios
                new_model.download_cf_scenario(model_attr['model_name'], uw_month, scenario_list=[], update_curve_map=False, save_scenario=False)
            
        return new_model
    
    @classmethod
    def new_model(cls, model_name, deal_ids, batch_keys, asset_class, uw_month=None, uw_type=None): # cutoff_date=None,
        """
        create a new model instance. 
        optional:load default configuration from asset class template
        
        Parameters
        ================
        model_name: str
            name for new model
        deal_ids: list
            id for entire deal(s) as in edw_prod.cdw.batch
        batch_keys: list
            id for batch(es) as in edw_prod.cdw.batch
        model_name: str
            user entered name for this model
        asset_class: str
            asset type. different model construction based on different assets
        uw_month: int   
            month the underwrite takes place (YYYMM)
        uw_type: str
            type of uw we are creating. 
            Pricing, RUW, Test
        """
        
        if not uw_month:
            raise Exception('Please Enter a uw_month in this format: YYYYMM')
            
        if uw_type=='test':
            test_model = 1
        else:
            test_model = 0
        
        #create new model record
        sql = """
                insert into FRA.cf_model.model
                values('{}', '{}', '{}', '{}', null, getdate(), user_name(), {}); 
                commit
                select @@identity
                """.format(model_name, asset_class, deal_ids, batch_keys, test_model)
        
        try:

            model_id = cls.sql_engine_model.execute(sql, output=True).iloc[0][0]
            
        except:
            raise Exception('This model name already exists. Please try another name')
        
        #create new UW record
        sql = """
                insert into fra.cf_model.model_uw
                values({}, '{}', '{}');
                commit
                select @@identity
            """.format(model_id, uw_month, uw_type)
        
        uw_id = cls.sql_engine_model.execute(sql, output=True).iloc[0][0]
        
        new_model = cls(int(model_id), model_name, deal_ids, batch_keys, asset_class, int(uw_id), uw_month, uw_type) #asofdate
        new_model.uw_id = uw_id
        
        
        """
        #create place holder scenario
        sql_cmd = "exec fra.cf_model.usp_upload_scenario_config ?, ?, ?, ?, ?, ?, ?, ?, ?"
        params = [model_id, uw_month, uw_type, 'Base Case', None, None, None, None, None]
        cls.sql_engine_model.execute(sql_cmd, params)
        """
        
        #download model_template key        
        sql="""
            SELECT TOP 1 model_key
                ,model_name
            	,scenario_key
                ,data_tape_source
            	,seq_order
            FROM fra.cf_model.vw_scenario
            WHERE 1=1
            	AND uw_type='template'
            	AND asset_class='{}'
            ORDER BY seq_order
        """.format(asset_class)
        
        model_attr = cls.sql_engine_model.execute(sql, output=True)
        
        #if no template found, skip download
        if len(model_attr)>0:
            model_attr = model_attr.squeeze() 
            new_model.model_template_key = model_attr['model_key']
            new_model.model_template_name = model_attr['model_name']
            #import data tape
            if model_attr['data_tape_source']=='cdw':
                new_model.import_data_tape_sql(source='cdw', save_source=True)
            elif model_attr['data_tape_source']!='excel':
                new_model.import_data_tape_sql(source='sql', query=model_attr['data_tape_source'], save_source=True)
            #import model template data
            new_model.download_model_template()
            
        return new_model

        
    @classmethod
    def ruw_model(cls, model_name, uw_month, uw_type='ruw', scenario_list=[]):
        """
        Create a refresh for an existing model. download all prior parameters and config in order to create new scenarios. 
        
        Parameters
        ==========================
        model_name: str
            name for new model
        uw_month: int
            the month of the current UW (YYYMM)
        uw_type: str
            underwrite type
            (pricing, ruw, refresh)
        scenario_list: list
            list of scenario names to download from source model
        """

        #download prior model setup
        sql = """
            select model_key
                ,model_name
            	,asset_class
            	,deal_ids
            	,batch_keys
                ,data_tape_source
            from fra.cf_model.model
            where 1=1
            	and model_name='{}'
            """.format(model_name)
        
        model_attr = cls.sql_engine_model.execute(sql, output=True)
        model_attr = model_attr.squeeze() 
        
        """
        #create new model record
        sql = 
                insert into FRA.cf_model.model
                values('{}', {}, '{}', '{}', '{}', getdate(), user_name()); 
                commit
                select @@identity
                .format(model_name, model_attr['asset_class'], json.loads(model_attr['deal_ids']), json.loads(model_attr['batch_keys']), model_attr['data_tape_source'])
        
        model_id = cls.sql_engine_model.execute(sql, output=True).iloc[0][0]
        """
        
        #create new uw record
        sql = """
                insert into fra.cf_model.model_uw
                values({}, '{}', '{}');
                commit
                select @@identity
            """.format(model_attr['model_key'], uw_month, uw_type)
        
        uw_id = cls.sql_engine_model.execute(sql, output=True).iloc[0][0]
        
        #create new model object
        loaded_model = cls(int(model_attr['model_key']), model_name, json.loads(model_attr['deal_ids']), json.loads(model_attr['batch_keys']), model_attr['asset_class'], int(uw_id), uw_month, uw_type)
        loaded_model.uw_id = uw_id
        
        #parse out new refresh cutoff
        #if uw_type == 'refresh':
        #month_end_date = date.date(int(str(uw_month)[0:4]), int(str(uw_month)[4:6]), 1) + relativedelta(day=31)
        #new_cutoff = month_end_date.strftime("%Y-%m-%d")
        
        if model_attr['data_tape_source']=='cdw':
            #if uw_type=='refresh' and not data_tape:
            #    cutoff_date=new_cutoff
            #else: 
            #    cutoff_date = None
            loaded_model.import_data_tape_sql(source='cdw', save_source=False)
        elif not model_attr['data_tape_source']:
            pass
        else:
            loaded_model.import_data_tape_sql(source='sql', query=model_attr['data_tape_source'],  save_source=False)
        #Download CF Scenarios from old model
        loaded_model.download_cf_scenario(model_name, scenario_list = scenario_list, update_curve_map=False, refresh_date=None)
        

        return loaded_model
    
    @classmethod
    def refresh_model(cls, model_name, uw_month, data_tape=True):
        """
        Creates a new scenario for a monthly update. this is not considered a RUW. 
        Just a quick update to the model for a 
        
        Parameters
        ----------
        model_name : str
            name of the model to download.
        uw_month : int
            month of current refresh.
        data_tape: bool
            if true will download entire performance history. 
            if false will only download the single month needed to run the refresh
        
        Returns
        -------
        CashFlowModel object

        """

        #download prior model setup
        sql = """
            select model_key
                ,model_name
            	,asset_class
            	,deal_ids
            	,batch_keys
                ,data_tape_source
            from fra.cf_model.model
            where 1=1
            	and model_name='{}'
            """.format(model_name)
        
        model_attr = cls.sql_engine_model.execute(sql, output=True)
        model_attr = model_attr.squeeze() 
                
        #create new uw record
        sql = """
                insert into fra.cf_model.model_uw
                values({}, '{}', '{}');
                commit
                select @@identity
            """.format(model_attr['model_key'], uw_month, 'refresh')
        
        uw_id = cls.sql_engine_model.execute(sql, output=True).iloc[0][0]
        
        #create new model object
        loaded_model = cls(int(model_attr['model_key']), model_name, json.loads(model_attr['deal_ids']), json.loads(model_attr['batch_keys']), model_attr['asset_class'], int(uw_id), uw_month, 'refresh')
        loaded_model.uw_id = uw_id
        
        #parse out new refresh cutoff (yyyymm to yyyy-mm-dd)
        month_end_date = date.date(int(str(uw_month)[0:4]), int(str(uw_month)[4:6]), 1) + relativedelta(months=-1, day=31)
        new_cutoff = month_end_date.strftime("%Y-%m-%d")
        
        #download data tape  
        cutoff_date = new_cutoff if not data_tape else None
        if model_attr['data_tape_source']=='cdw':
            loaded_model.import_data_tape_sql(source='cdw', save_source=False, asofdate=cutoff_date)
        elif not model_attr['data_tape_source']:
            pass
        else:
            loaded_model.import_data_tape_sql(source='sql', query=model_attr['data_tape_source'], save_source=False, asofdate=cutoff_date)
        
        #Download CF Scenarios from old model
        loaded_model.download_cf_scenario(model_name, scenario_list = [], update_curve_map=False, refresh_date=new_cutoff)
        
        return loaded_model
        
    def import_data_tape_sql(self, source='cdw', query='', save_source=True, asofdate=None):
        if source=='cdw':
            print('Importing Data Tape')
            asset_class = self.asset_class #if self.model_type==1 else None
            self.data_tape = self.data_prep.import_data_tape_cdw(self.deal_ids, self.batch_keys, asset_class, asofdate)
            self.eval.output_actuals(self.data_tape)
            print('Importing Prior Projections')
            self.prior_uw_projections = self.data_prep.import_projections_cdw(self.deal_ids, self.batch_keys, projection_level='deal')
            self.eval.output_hist_proj(self.prior_uw_projections)
            
            if save_source:
                self.data_prep.save_data_tape_source(model_name=self.model_name, dt_source=source)
            
        elif source=='sql':
            print('Importing Data Tape')
            self.data_tape = self.data_prep.import_data_tape_query(query)  
            print('Importing Prior Projections')
            self.prior_uw_projections = self.data_prep.import_projections_cdw(self.deal_ids, self.batch_keys, projection_level='deal')
            self.eval.output_hist_proj(self.prior_uw_projections)
            if save_source:
                self.data_prep.save_data_tape_source(model_name=self.model_name, dt_source=query)
            
        #if segments already exist, run segment logic
        for curve_group_name, curve_group in self.rate_curve_groups.items():
            curve_group.data_tape = self.data_tape
            for segment_type, segment_set in curve_group.segments.items():
                segment_set.create_segment(**segment_set.segment_input)
                curve_group.map_segments_to_curves(segment_type)
                #process account map
                curve_group.create_account_map()
            
    def import_data_tape_excel(self, file_path, ws_name, ws_range):
        ###########################################
        # load from excel
        self.data_tape = self.data_prep.import_data_tape_excel(file_path, ws_name, ws_range)
        self.eval.output_actuals(self.data_tape)
        
        self.data_prep.save_data_tape_source(model_name=self.model_name, dt_source='excel')
        
    def create_curve_group(self, curve_group_name, curve_group_key=None):
        """
        Creates a new curve group and generates a unique ID to store in Database
        
        Parameters
        =================================
        curve_group_name: str
            name for this curve group
        curve_group_key: int
            optional curve group key. 
                if provided, will use that key
                if not provided will create a new key
        duplicate_curve_group: str
            optional curve group name
            if provided will create a copy of an existing curve group instead of creating a blank one
            use this when you want to run scenarios with differnet curve adjustments
        """
        
        if not curve_group_key:
            sql = """insert into FRA.cf_model.curve_group
                    values('{}', getdate(), user_name());
                    commit
                    select @@identity
                    """.format(curve_group_name)
            
            curve_group_key = self.sql_engine_model.execute(sql, output=True).iloc[0][0]
        
        #curve_group = DataPrep.rate_curves(curve_group_key, curve_group_name, self.data_prep.data_tape)
        self.rate_curve_groups[curve_group_name] = DataPrep.CurveGroup(curve_group_key, curve_group_name, self.data_tape)
        #self.data_prep.rate_curve_groups[curve_group_name]=curve_group
        self.curve_group_in_focus = curve_group_name
        
    def download_index_projections(self, cutoff_date, projection_date = None):
        """
        Downloads interest rate index projections and adds to the dictionary of available curves. 
        will only download once for the model to use as needed. This function will be triggered when a
        scenario is run. 
        
        Parameters
        ===========================================
        cutoff_date: date
            cutoff date of scenario being run
        projection_date: date
            Optional; provide if a specific projection date is needed.
            if None, will default to closest projections to cutoff_date
        

        """
        
        #check if this cutoff date/projection date combo is already downloaded
        if (projection_date, cutoff_date) not in self.data_prep.index_projections:
            try:
                self.data_prep.import_index_projections(projection_date=projection_date, cutoff_date=cutoff_date)
            except:
                print('Warning: Error downloading Index Projections. No future rates have been downloaded.')
                
    def copy_curves(self, curve_group_name, source_curve_group, curve_type=['all']):
        """
        Copies curves from a different curve_group

        Parameters
        ===========================================
        curve_group_name : str
            destination curve group
        source_curve_group : str
            source curve group
        curve_type : list, optional
            list of curve group types to copy over. The default is [].

        """
        #create new curve group
        self.create_curve_group(curve_group_name)
        
        #identify curve group ids 
        #curve_group_key = self.rate_curve_groups[source_curve_group].curve_group_key
        
        #run curve import process. will recreate the curves/segments and set up a new map in server
        for curve in curve_type:
            self.import_rate_curves_sql(curve_group_name, source_curve_group_name=source_curve_group, curve_type=curve)
        
          
    def import_rate_curves_sql(self, curve_group_name, source_curve_group_name=None, model_name=None, uw_month=None, scenario_name=None, curve_type='all', curve_sub_type='all', update_curve_map=True):
        """
        Imports data from an existing curve group stored in the server. 
        must provide either
            1) curve group name
            2) OR Model Name + scenario name

        Parameters
        ----------
        curve_group_name : str
            the curve group name to load curve data into.
        source_curve_group_name : str, optional
            Name of curve group to download. The default is None.
        model_name : str, optional
            name of source model. The default is None.
        scenario_name : str, optional
            name of source scenario name. The default is None.
        curve_type : str, optional
            Curve type to download. can be 1 specific type or "all" will download all curves in this group
            The default is 'all'.
        curve_sub_type : str, optional
             Curve type to download. can be 1 specific type or "all" will download all sub curve types in this group. 
             The default is 'all'.
        update_curve_map : bool, optional
            Whether to map this curve group to this model or just download as is. The default is True.

        Returns
        -------
        None.

        """
        #if curve group does not exist create it
        if not self.rate_curve_groups.get(curve_group_name):
            self.create_curve_group(curve_group_name)
        
        #import and load curves, segments, and maps
        self.data_prep.import_rate_curves_sql(self.rate_curve_groups[curve_group_name], source_curve_group_name, model_name, uw_month, scenario_name, curve_type, curve_sub_type, update_curve_map)
        
        if curve_type == 'all':
            self.rate_curve_groups[curve_group_name].update_all_mappings()
        else:
            self.rate_curve_groups[curve_group_name].map_segments_to_curves(curve_type)
        
        if self.data_tape:
            try:
                self.rate_curve_groups[curve_group_name].create_account_map()
            except:
                print("Account Map could not be created.")
        
    def import_rate_curves_excel(self, curve_group_name, curve_type, curve_sub_type, file_path, ws_name, ws_range, key_cols=['Key'], period_type='MOB', pivot=False):
        ###########################################
        # load from excel
        self.data_prep.import_rate_curves_excel(self.rate_curve_groups[curve_group_name], curve_type, curve_sub_type, file_path, ws_name, ws_range, key_cols=key_cols, pivot=True)
        self.rate_curve_groups[curve_group_name].map_segments_to_curves(curve_type)
        if self.data_tape:
            self.rate_curve_groups[curve_group_name].create_account_map()
        ###########################################
        # upload curves into SQL
        self.upload_rate_curves(curve_group_name, curve_type, curve_sub_type, period_type, 'Excel Custom Curves')
        """
        curve_group_key = self.rate_curve_groups[curve_group].curve_group_key
        rate_json = self.rate_curve_groups[curve_group].curves[(curve_type, curve_sub_type)].curve_json()
        #look for corresponding segment if exists
        try:
            segment_key = self.rate_curve_groups[curve_group].segments[curve_type].segment_key
        except:
            segment_key=None
        
        sql_cmd = "exec fra.cf_model.usp_upload_curve_rates ?, ?, ?, ?, ?, ?, ?"
        params = [curve_group_key, curve_type, curve_sub_type, period_type, 'Excel Custom Curves', rate_json, segment_key]
        self.sql_engine_model.execute(sql_cmd, params)
        """
        
    def upload_rate_curves(self, curve_group, curve_type, curve_sub_type, period_type, curve_source='Excel Custom Curves'):
        """
        Upload curves into SQL
        
        Parameters
        ===========================
        curve_group: str
            name of curve group 
        curve_type: str
            curve type to load
        curve_sub_type: str
            curve sub type to load
        period_type: str
            MOB or Calendar
        curve_source: str
            description of source of curves
        
        """
        curve_group_key = self.rate_curve_groups[curve_group].curve_group_key
        rate_json = self.rate_curve_groups[curve_group].curves[(curve_type, curve_sub_type)].curve_json()
        #look for corresponding segment if exists
        try:
            segment_key = self.rate_curve_groups[curve_group].segments[curve_type].segment_key
        except:
            segment_key=None
        
        sql_cmd = "exec fra.cf_model.usp_upload_curve_rates ?, ?, ?, ?, ?, ?, ?"
        params = [curve_group_key, curve_type, curve_sub_type, period_type, curve_source, rate_json, segment_key]
        self.sql_engine_model.execute(sql_cmd, params)
    
    def import_rate_curves(self, use_gui=False, curve_type=None, file_path='', ws_name='', ws_range='C3:NF363', key_cols=['Key'], key_rename=[]):
        if use_gui:
            self.gui_data_prep.import_rate_curves.import_curves_main()
        else:
            #self.data_prep.import_rate_curves.import_curves_main(self, curve_type, file_path, ws_name, ws_range='C3:NF363', period_type='month_on_book', key_cols=['Key'], key_rename=[])
            pass
        
    def create_segment(self, use_gui=False, curve_group_name=None, segment_type=None, **kwargs):
        
        if not curve_group_name:
            curve_group_name=self.curve_group_in_focus
        
        curve_group = self.rate_curve_groups[curve_group_name]
        
        if use_gui:
            pass
        else:
            curve_group.add_segment(segment_type, **kwargs)
            #upload segment data
            if segment_type != 'index':
                # for index projections do not run upload
                segment_json = curve_group.segments[segment_type].generate_segment_json()
                #run upload
                segment_key = self.upload_segments(curve_group.curve_group_key, segment_type, json.dumps(segment_json))
                curve_group.segments[segment_type].segment_key = int(segment_key.iloc[0][0])
            else:
                curve_group.segments[segment_type].segment_key = -1
            
            self.rate_curve_groups[curve_group_name].map_segments_to_curves(segment_type)
            
    def upload_segments(self, curve_group_key, segment_type, segment_json):
        """
        Upload Segment definition into SQL

        Parameters
        ----------
        curve_group_key : TYPE
            DESCRIPTION.
        segment_type : TYPE
            DESCRIPTION.
        segment_json : TYPE
            DESCRIPTION.

        Returns
        -------
        segment_key : int
            

        """
            
        sql_cmd = "exec fra.cf_model.usp_upload_segments ?, ?, ?"
        params = [curve_group_key, segment_type, segment_json]
        
        segment_key = self.sql_engine_model.execute(sql_cmd, params, output=True)
        
        return segment_key
        
    def map_curves(self, use_gui=False, curve_group_name=None, manual_map=None):
        curve_group = self.rate_curve_groups[curve_group_name]
        
        if not curve_group_name:
            curve_group_name=self.curve_group_in_focus
        
        if use_gui:
            self.gui_data_prep.map_segments.map_segments_main(curve_group)
            curve_group.update_all_mappings()
        else:
            curve_group.update_all_mappings(manual_map)
            
        #upload updated map to server
        segment_curve_map = curve_group.segment_curve_map.to_json(orient='records')
        
        segment_keys=[]
        for k, v in self.rate_curve_groups[curve_group_name].segments.items():
            segment_keys.append({'segment_type':k, 'segment_key':v.segment_key})
        segment_keys = json.dumps(segment_keys)
        
        sql_cmd = "exec cf_model.usp_upload_segment_map ?, ?, ?"
        params = [curve_group.curve_group_key, segment_keys, segment_curve_map]
        
        self.sql_engine_model.execute(sql_cmd, params, output=False)
        
    def create_model_config(self, model_config_name, config_type=1, config_dict={}):
        """
        Creates a new model configuration. A dict with model parameters must be passed in manually
        
        Parameters
        ====================
        model_config_name: str
            the name for the new configuration
        config_type: int
            type of model we are building
            (1: Hazard, 2: Roll Rate, 3: Monte Carlo)
        config_dict: dict
            dict containing model config parameters
        """
        
        self.model_configs[model_config_name] = self.data_prep.create_model_config(model_config_name, config_type, config_dict)
        
        
    def download_model_config(self, model_config_name, version=None):
        
        self.data_prep.download_model_config(model_config_name, version)
    
    def create_recovery_curve(self, curve_group_name, *args):
        """
        Creates a recovery timing curve with single month payoff. Can use when a recovery curve isn't available'
        
        Parameters
        =====================
        args: tuples
            tuples of (collection_month, collection_percent) 
            will loop through each tuple as breakpoints to create recovery rate and scale values inbetween each point
        
        """
        
        recovery_df_list = []
        
        #if list is length 1 apply stress to all months
        if len(args)==1:
            for arg in args:
                recovery_array = np.zeros(300)
                recovery_array[arg[0]] = arg[1]
                recovery_time= [x for x in range(0, 300)]
                recovery_df_list.append(pd.DataFrame({'period':recovery_time, 'rate':recovery_array}, columns=['period', 'rate']))
        
        else:
            #iterate 2 items at once
            for period_begin, period_end in zip(args[:-1],args[1:]):               
                recovery_array = np.linspace(period_begin[1], period_end[1], (period_end[0]-period_begin[0]))
                recovery_time = [x for x in range(period_begin[0], period_end[0])]
                recovery_df_list.append(pd.DataFrame({'period':recovery_time, 'rate':recovery_array}, columns=['period', 'rate']))
        
        recovery_df = pd.concat(recovery_df_list)
        #add other columns
        recovery_df['curve_type']  = 'recovery'
        recovery_df['curve_sub_type'] = 'base'
        recovery_df['curve_id'] = 'all'
        recovery_df['curve_name'] = 1
        recovery_df['from_status'] = -1
        recovery_df['to_status'] = -1
        
        recovery_df.set_index(['curve_type', 'curve_sub_type', 'curve_id', 'period'], inplace=True)
        recovery_df = recovery_df.astype({'rate':'float32'})
        
        curve_group = self.rate_curve_groups[curve_group_name]
        curve_group.add_curve(recovery_df, 'recovery', 'base', 'MOB', 'Custom Curve')
        self.rate_curve_groups[curve_group_name].map_segments_to_curves('recovery')
        
        self.upload_rate_curves(curve_group_name, 'recovery', 'base', 'MOB', 'Custom Curves')
        
    def return_curve_map(self, curve_group_name):
        """
        returns the current segment/curve map. can use this dict to manually map any missing matches

        Parameters
        ----------
        curve_group_name : str
            curve_group_name to extract.

        Returns
        -------
        dict of dict
        {curve_type={segment_name: curve_name}}

        """
        curve_map = self.rate_curve_groups[curve_group_name].segment_curve_map
        curve_map_dict = curve_map.groupby('segment_type')[['segment_name','curve_id']].apply(lambda x: pd.Series(x.curve_id.values, index=x.segment_name).to_dict()).to_dict()
        return curve_map_dict
    
    def return_cohort_strats(self, asofdate, curve_group_name):
        """
        calculates aggregated metrics by cohort for a specific cutoff date
        
        Parameters
        =======================================
        asofdate : date
            cutoff date to use for the calculation
        curve_group_name: str, optional
            the curve type to use
        """
        
        #set high level grouping columns
        key_cols = ['DealID','BatchKey','BatchAcquisitionDate','AsOfDate']
        sum_cols = ['OriginationBalance','PurchaseBalance','BOM_PrincipalBalance','InterestBalance',
                    'TotalPrincipalBalance', 'InterestBearingPrincipalBalance', 'DeferredPrincipalBalance','ScheduledPaymentAmount'
                ]
        weight_avg_cols = ['MonthsOnBook', 'RemainingTerm', 'InterestRate','OriginationCreditScore', 'OriginationTerm']
                
        group_cols = { #self.account_grouping_cols
                'key_cols': key_cols,
                'sum_cols': sum_cols,
                'weight_avg_cols': weight_avg_cols
                }
        
        rate_curves = self.rate_curve_groups[curve_group_name]
        
        summary_metrics = self.data_tape.attach_curve_group(rate_curves, asofdate, group_accounts=True, grouping_cols=group_cols)
        #add curve ids into table
        
        for curve_type in rate_curves.segment_types:
            #if curve_type in summary_metrics.columns:
            if curve_type in rate_curves.segments:    
                #add segments
                segment_type_recs = rate_curves.segments[curve_type].segment_rules_combined['rule_name_combined'].reset_index(drop=False)
                segment_type_dict = dict(zip(segment_type_recs['index'], segment_type_recs['rule_name_combined']))
                summary_metrics[curve_type+'_segment'] = summary_metrics[curve_type+'_segment'].map(segment_type_dict)
                
                curve_type_recs = rate_curves.transition_keys.loc[rate_curves.transition_keys['curve_type']==curve_type, ['curve_key', 'curve_id']]
                curve_type_dict = dict(zip(curve_type_recs['curve_key'], curve_type_recs['curve_id']))
                summary_metrics[curve_type+'_curve'] = summary_metrics[curve_type].map(curve_type_dict)
                #drop original curve ids
                summary_metrics.drop(rate_curves.segment_types, axis=1, errors='ignore', inplace=True)
                
        return summary_metrics
        #return group_cols
        
    def create_curve_stress(self, stress_dict={}):
        """
        converts input instructions into stresses by month
        
        Parameters
        =================
        stress_dict: dict
        
        
        """
        return DataPrep.CurveStress(stress_dict)
    
    def create_cf_scenario(self, scenario_name, cutoff_date='max', curve_group=None, curve_stress=None, index_projection_date = None, model_config=None, save_scenario=True): 
        """
        Creates a new CF Scenario based on user input
        """
        #default to current cutoffdate
        #cutoff_date = self.data_prep.data_tape.max_date if cutoff_date is None else date.datetime.strptime(cutoff_date,"%Y-%m-%d").date()
        
        #if this is a model template then allow "Max"/"Min" keyword. else replace with date.
        #if cutoff_date=='max' and self.model_type!=1:
        if cutoff_date=='max' and self.uw_type!='template':
            if self.data_tape:
                cutoff_date = self.data_tape.max_date
            else:
                today = date.date.today()
                cutoff_date = today.replace(day=1) - date.timedelta(days=1)
                cutoff_date = cutoff_date.strftime('%Y-%m-%d')
        #elif cutoff_date=='min' and self.model_type!=1:
        elif cutoff_date=='min' and self.uw_type!='template':
            if self.data_tape:
                cutoff_date = self.data_tape.min_date
            else:
                cutoff_date = '2010-01-31'
        

        try:
            del self.cf_scenarios[scenario_name]
            gc.collect()
        except:
            pass
        
        # check input parameters for none. if only one option exists use that input
        if not curve_group:
            if len(self.rate_curve_groups)==1:
                curve_group = next(iter(self.rate_curve_groups))
            else:
                raise Exception('There is more than 1 available Curve Group. Enter selection for this parameter.')
                
        if not model_config:
            if len(self.model_configs)==1:
                model_config = next(iter(self.model_configs))
            elif len(self.model_configs)==0:
                model_config={}
            else:
                raise Exception('There is more than 1 available Model Configuration. Enter selection for this parameter.')
                        
        #download index projections
        if cutoff_date:
            self.download_index_projections(cutoff_date, index_projection_date)
                
        model_params = [cutoff_date, curve_group, curve_stress, index_projection_date, model_config]
        
        self.cf_scenarios[scenario_name]=(model_params, None)
        if save_scenario:
            self.save_model(scenario_name)
        print('CF Scenario Created\n    Scenario Name: {}\n    Curve Group: {}\n    Cutoff Date: {}\n'.format(scenario_name, curve_group, cutoff_date))
        
    def download_cf_scenario(self, model_name, uw_month=None, scenario_list=[], update_curve_map=True, refresh_date=None, lock_curve_group=False, save_scenario=True):
        """
        Downloads a cash flow scenario configuration stored in the server. 
        Includes curve sets and segment definitions
        
        Parameters
        =====================
        model_name: int
            Model name to pull from
        uw_month: str
            YYYYMM - reunderwrite scenario group to extract
        scenario_list: list
            scenarios we want to download from the selected Model
        update_curve_map: bool
            if true, will register the curve group to this new model. 
            Use false when downloading an existing model
            use True when downloading scenarios created in prior models
        refresh_date: date
            if a refresh date is provided will take all assumptions from source model and create
            a scenario with updated asofdate
        """
        
        #if no UW was provided use most recent
        if not uw_month:
            sql="""
                SELECT top 1 uw_month
                    , scenario_key
                from fra.cf_model.vw_scenario
                	where 1=1
                		and model_name='{}'
                order by uw_month desc
                    """.format(model_name)
            model_attr = self.sql_engine_model.execute(sql, output=True)
            
            #if no scenarios found kill process
            if len(model_attr)==0:
                #raise Exception('There is no prior UW for this model. No scenarios were downloaded')
                print('There are no UWs found for this Model, No Scenarios have been downloaded')
                return
            else:
                model_attr= model_attr.squeeze()
            uw_month = model_attr['uw_month']
             
        if self.uw_type=='template' or not uw_month:
            uw_month_filter = ''
        else:
            uw_month_filter = " and uw_month='" + str(uw_month) + "'"
        
        scenario_list = ", ".join(["'{}'".format(value) for value in scenario_list])
        
        if scenario_list:
            scenario_list = ' and scenario_name in ({})'.format(scenario_list)
        elif refresh_date is not None:
            scenario_list = ' and scenario_loaded=1'
        
        sql = """
                select model_key
                     , model_name
                     , asset_class
                     , deal_ids
                     , batch_keys
                     , uw_month
                     , uw_type
                     , scenario_key
                     , scenario_name
                     , uw_scenario_name
                     , model_config_key
            		 , config_name
            		 , config_type_key
            		 , config_json
                     , cutoff_dt
                     , index_proj_dt
                     , curve_stress
                     , scenario_loaded
                     , curve_group_key
                     , curve_group_name
            	from fra.cf_model.vw_scenario
            	where 1=1
            		--and seq_order = 1
            		and model_name='{}'
                    {}
            		{}
                """.format(model_name, uw_month_filter, scenario_list)
        
        model_attr = self.sql_engine_model.execute(sql, output=True)
        
        #create curve group objects and download curve sets only if not already in model
        download_keys = model_attr['curve_group_key'].unique()
        download_key_dict = {}
        
        existing_keys = {int(curve_group.curve_group_key):key for key, curve_group in self.rate_curve_groups.items()}
        
        for key in download_keys:
            if key in existing_keys.keys(): 
                download_key_dict[key] = existing_keys[key]
            else:
                curve_group_attr = model_attr[model_attr['curve_group_key']==key].iloc[0].squeeze()
                self.create_curve_group(curve_group_attr['curve_group_name'], int(key))
                self.import_rate_curves_sql(curve_group_attr['curve_group_name'], None, curve_group_attr['model_name'], int(curve_group_attr['uw_month']), curve_group_attr['scenario_name'], curve_type='all', curve_sub_type='all', update_curve_map=update_curve_map)
                download_key_dict[key]=curve_group_attr['curve_group_name']
                if lock_curve_group:
                    self.rate_curve_groups[curve_group_attr['curve_group_name']].lock_curve_group()
                
        #create model config objects
        for key in model_attr['model_config_key'].unique():
            config_attr = model_attr[model_attr['model_config_key']==key].iloc[0].squeeze()
            config_dict = json.loads(config_attr['config_json'])
            new_config = DataPrep.ModelConfig(config_attr['model_config_key'], config_attr['config_name'], config_attr['config_type_key'], config_dict)
            self.model_configs[config_attr['config_name']]=new_config
        
        #create scenarios
        for ix, row in model_attr.iterrows():
            scenario_name = row['scenario_name']
            if refresh_date is not None and row['scenario_name']!='Backtest':
                #cutoff_date = 'max' 
                new_cutoff_date = refresh_date
                old_cutoff_date = row['cutoff_dt']
            else:
                new_cutoff_date = row['cutoff_dt']
                if new_cutoff_date:
                    new_cutoff_date = str(new_cutoff_date)
            #scenario_type = row['scenario_type']
            index_proj_date = row['index_proj_dt']
            curve_group_key = row['curve_group_key']
            curve_group_name = download_key_dict[curve_group_key]
            #create stress if available
            if row['curve_stress']:
                curve_stress = json.loads(row['curve_stress'])
                #convert list of lists into tuples
                curve_stress = {k:[tuple(i) for i in v] for (k, v) in curve_stress.items()}
                
                #if a refresh shift the stress forward
                if refresh_date is not None:
                    #for key, value in loaded_model.cf_scenarios.items():
                    #unpack parameters
                    #old_cutoff, curve_group_name, curve_stress, index_proj_date, model_config = value[0]
                    if old_cutoff_date.lower() != 'backtest':
                        #calculate number of months since RUW
                        new_cutoff = refresh_date.strftime("%Y-%m-%d")
                        old_cutoff_format = date.datetime.strptime(old_cutoff_date, '%Y-%m-%d').date()
                        num_months = (new_cutoff.year - old_cutoff_format.year) * 12 + (new_cutoff.month - old_cutoff_format.month)
                        #shift stress if exists
                        stress_object = self.create_curve_stress(curve_stress)
                        curve_stress = stress_object.shift_stress(num_months)                
            else:
                curve_stress = None
            model_config = row['config_name']
            
            self.create_cf_scenario(scenario_name, new_cutoff_date, curve_group_name, curve_stress, index_proj_date, model_config, save_scenario=save_scenario)
    
    def download_model_template(self):
        """
        Downloads Model Template Inputs, including:
            Rate Curve Groups/Segment Definitions
            Model Configuration
        
        This method will not automatically create scenarios. Just download the building blocks
        """
        
        self.download_cf_scenario(self.model_template_name, scenario_list=['Base Case'], update_curve_map=False, refresh_date=None, lock_curve_group=True)
        
    def run_cash_flows(self, scenarios=[], del_detail_on_complete=True, auto_run=True):
        """
        Runs cash flow scenarios and generates output evaluation.
        
        Parameters
        =================
        scenarios: list
            optional list of scenario names. if left blank will run all scenarios
        """
        
        #rerun actual output if a single account was run prior
        if self.eval.single_account==True:
            self.eval.single_account=False
            self.eval.output_actuals(self.data_tape)
        
        if len(scenarios)==0:
            scenarios = list(self.cf_scenarios.keys())
        
        
        #for model in self.cf_scenarios.keys():
        #if len(scenarios)==0 or model in scenarios:
        for model in scenarios:    
            if model not in self.cf_scenarios:
                print(f'\r{model} - Failed to Initialize')
                print(f'     There is no scenario called "{model}"')
                print('     Validate scenarios name and try again')
                continue
            
            print(f'{model} - Initializing Model', end='\r')
            #create model object
            model_params = self.cf_scenarios[model][0]
            cutoff_date, curve_group_name, curve_stress, index_projection_date, config_name = self.cf_scenarios[model][0]
            rate_curves = self.rate_curve_groups[curve_group_name]
            config_dict = self.model_configs[config_name].config_dict
            model_type = self.model_configs[config_name].config_type
            
            
            if not any(curve in rate_curves.curve_type_info for curve in ['default','prepay', 'rollrate']):
                print('f\r{model} - Failed to Initialize')
                print('     Curve Group must contain CDR/CPR or Roll Rate Curves')
                print('     Add the needed curves and try again')
                # on this exception skip to next iteration
                continue
                
            if curve_stress:
                scenario_stress = self.create_curve_stress(curve_stress)
            else:
                scenario_stress = None
                
            #set cutoff date for max or min
            if cutoff_date=='max':
                cutoff_date = self.data_tape.max_date
            elif cutoff_date=='min':
                cutoff_date = self.data_tape.min_date
            else:
                cutoff_date = cutoff_date
                
            #remove existing model object
            self.cf_scenarios[model] = None
            self.cf_scenarios[model]=(model_params, None)
            gc.collect()
            
            #add index projections
            #self.download_index_projections(cutoff_date, index_projection_date)
            index_rates_df = self.data_prep.index_projections[(index_projection_date,cutoff_date)].data_rate_curves
            rate_curves.add_curve(index_rates_df, 'index', 'base', 'Calendar', 'Rate Projections Table')
            with self.suppress_stdout():
                self.create_segment(use_gui=False, curve_group_name=curve_group_name, segment_type='index', InterestRateIndex=[])
                rate_curves.create_account_map()
                
            #set account id to none
            config_dict['account_id']=None
            
            #save model configuration
            self.save_model(model)
            
            with np.errstate(divide='ignore', invalid='ignore'):
                new_model = self.model_builder.build_new_model(model, model_type, self.data_tape, rate_curves, cutoff_date, scenario_stress, config_dict)
                self.cf_scenarios[model]=(model_params, new_model)
                if auto_run:
                    self.cf_scenarios[model][1].run_model()
                
                    print(f'\r{model} - Evaluating Output', end='\r')
                    self.eval.output_proj_all(model)
            
            if del_detail_on_complete:
                del self.cf_scenarios[model][1]._cf_data
                del self.cf_scenarios[model][1]._model_config
            gc.collect()
            
            print(f'\r{model} - Complete            ')
        
            
                
    def run_single_account(self, scenario, account_id, auto_run=True):
        """
        Runs a single account through a model scenario for QA/validation purposes

        Parameters
        ----------
        scenario : str
            the scenario name to run.
        account_id : int
            account id to run through the model.

        """
        
        print(f'{scenario} - {account_id} - Initializing Model', end='\r')
        #create model object
        model_params = self.cf_scenarios[scenario][0]
        cutoff_date, curve_group_name, curve_stress, index_projection_date, config_name = self.cf_scenarios[scenario][0]
        rate_curves = self.rate_curve_groups[curve_group_name]
        config_dict = self.model_configs[config_name].config_dict.copy()
        model_type = self.model_configs[config_name].config_type
        if curve_stress:
            scenario_stress = self.create_curve_stress(curve_stress)
        else:
            scenario_stress = None
            
        #set cutoff date for max or min
        if cutoff_date=='max':
            cutoff_date = self.data_tape.max_date
        elif cutoff_date=='min':
            cutoff_date = self.data_tape.min_date
        else:
            cutoff_date = cutoff_date
                        
        #remove existing model object
        self.cf_scenarios[scenario] = None
        self.cf_scenarios[scenario]=(model_params, None)
        gc.collect()
        
        #add index projections
        #self.download_index_projections(cutoff_date, index_projection_date)
        index_rates_df = self.data_prep.index_projections[(index_projection_date,cutoff_date)].data_rate_curves
        rate_curves.add_curve(index_rates_df, 'index', 'base', 'Calendar', 'Rate Projections Table')
        with self.suppress_stdout():
            self.create_segment(use_gui=False, curve_group_name=curve_group_name, segment_type='index', InterestRateIndex=[])
            rate_curves.create_account_map()
                    
        #update config for single account scenario
        config_dict['account_id']=account_id
        
        with np.errstate(divide='ignore', invalid='ignore'):
            new_model = self.model_builder.build_new_model(scenario, model_type, self.data_tape, rate_curves, cutoff_date, scenario_stress, config_dict)
            self.cf_scenarios[scenario]=(model_params, new_model)
            if auto_run:
                self.cf_scenarios[scenario][1].run_model()
            
                print(f'\r{scenario} - {account_id} - Evaluating Output', end='\r')
                self.eval.output_actuals(self.data_tape, account_id=account_id)
                self.eval.output_proj_all(scenario)
        
        #if del_detail_on_complete:
        #    del self.cf_scenarios[scenario][1]._cf_data
        #    del self.cf_scenarios[scenario][1]._model_config
        gc.collect()
        
        print(f'\r{scenario} - {account_id} - Complete            ')
        
    
    def run_pd_curves(self, scenario):
        
        print(f'{scenario} - Initializing PD Model', end='\r')
        #create model object
        model_params = self.cf_scenarios[scenario][0]
        cutoff_date, curve_group_name, curve_stress, config_name = self.cf_scenarios[scenario][0]
        rate_curves = self.rate_curve_groups[curve_group_name]
        config_dict = self.model_configs[config_name].config_dict
        model_type = self.model_configs[config_name].config_type
        if curve_stress:
            scenario_stress = self.create_curve_stress(curve_stress)
        else:
            scenario_stress = None
            
        #set cutoff date for max or min
        if cutoff_date=='max':
            cutoff_date = self.data_tape.max_date
        elif cutoff_date=='min':
            cutoff_date = self.data_tape.min_date
        else:
            cutoff_date = cutoff_date
            
        #create model object
        
    @contextmanager
    #supress print statements when necessary
    def suppress_stdout(self):
        with open(os.devnull, "w") as devnull:
            old_stdout = sys.stdout
            sys.stdout = devnull
            try:  
                yield
            finally:
                sys.stdout = old_stdout
    
    """
    def generate_segment_json(self, curve_group, segment_type):
        segment_config = json.loads('{}')
        segment = self.data_prep.rate_curve_groups[curve_group]
        #segment rules
        segment_rules_upload=segment.segment_rules[segment.segment_rules['segment_type']==segment_type][['segment_type','column','column_order','min_value','max_value','rule_name']]
        segment_rules_upload.reset_index(inplace=True)
        segment_rules_upload.rename(columns={'index':'segment_rule_id','column':'column_name'}, inplace=True)
        segment_rules_upload = segment_rules_upload.to_json(orient='records')
        
        #segment combined
        segments_upload = segment.segment_rules_combined[segment.segment_rules_combined['segment_type']==segment_type].drop(columns=['rule_eval_combined', 'rule_name_combined', 'column_combined','curve_id','curve_key']).copy()
        segments_upload.index.set_names('segment_id', inplace=True)
        segments_upload.reset_index(inplace=True)
        segments_upload_pivot = pd.melt(segments_upload, id_vars=['segment_id', 'segment_type'], var_name='column_name', value_name='segment_rule_id') #'column_combined','rule_name_combined'
        segments_upload_pivot.dropna(inplace=True)
        segments_upload_pivot['segment_rule_id'] = segments_upload_pivot['segment_rule_id'].astype('int32')
        segments_upload = segments_upload_pivot[['segment_id','segment_rule_id']].to_json(orient='records')
        
        #return segment_rules_upload, segments_upload
        segment_config.update({'segment_input':segment.segment_input[segment_type]})
        segment_config.update({'segment_rules':json.loads(segment_rules_upload)})
        segment_config.update({'segment_rules_combined':json.loads(segments_upload)})
        segment_config.update({'segment_curve_map':json.loads(segment.segment_curve_map[segment.segment_curve_map['segment_type']==segment_type].to_json(orient='records'))})
        
        return segment_config
    """
    #def generate_curves_json(self):
        
    def set_final_scenario(self, scenario):
        """
        Tags a sceanrio as final. This is the scenario that will be loaded to finance and ops once an underwrite is complete

        Parameters
        ----------
        scenario : str
            name of the scenario to be locked.

        """
        #check that scenario exists
        if scenario not in self.cf_scenarios:
            print(f'"{scenario}" is not an existing scenario. Check the scenario names and try again')
            return
        
        sql_cmd = "exec fra.cf_model.usp_set_final_scenario ?, ?, ?"
        params = [int(self.model_id), int(self.uw_id), scenario]
        self.sql_engine_model.execute(sql_cmd, params)
        
        print(f'"{scenario}" successfully set as final scenario')
        
        
    def save_model(self, scenario):
        
        #self.segment_config = json.loads('{}')
        #self.model_config=json.loads('{}')
        #self.model_config = {}
        
        ######################################
        #           Config Variables
        ######################################

        #model configuration
        #exclude_list = ['rate_curves','data_tape', 'cutoff_date','num_cohorts','num_status','int_rates','segment_keys', 'projection_periods']
        #self.model_config = {key: v for key, v in self.cf_scenarios[scenario]._model_config.items() if key not in exclude_list}
        #add final matrixes
        #for key, module in self.cf_scenarios[scenario]._cf_modules.items():
        #    matrix_list = [k for k in module.__dict__.keys() if k.endswith('matrix')]
        #    for x in matrix_list:
        #        self.model_config[x] = module.__dict__[x].tolist()
        
        
        
        #clean up numpy data types
        #for k, v in self.model_config.items():
        #    if type(v)==np.ndarray:
        #        self.model_config[k] = v.tolist()
                
        #print(self.model_config.keys())
        
        ######################################
        #           insert into sql table
        ######################################
        
        cutoff_date, curve_group_name, curve_stress, index_projection_date, config_name = self.cf_scenarios[scenario][0]
        
        config_key = int(self.model_configs[config_name].model_config_id)
        curve_group_key = int(self.rate_curve_groups[curve_group_name].curve_group_key)
        if curve_stress is not None:
            curve_stress = json.dumps(curve_stress)
        
        sql_cmd = "exec fra.cf_model.usp_upload_scenario_config ?, ?, ?, ?, ?, ?, ?, ?"
        params = [int(self.model_id), int(self.uw_id), scenario, curve_group_key, curve_stress, cutoff_date, index_projection_date, config_key]
        self.sql_engine_model.execute(sql_cmd, params)
        
