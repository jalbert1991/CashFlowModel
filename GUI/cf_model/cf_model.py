# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 21:17:40 2020

@author: jalbert
"""

import DataPrep.DataMover as DataPrep
import CashFlowModelMainV2 as cf_main

#create data prep instance to import base lookup data
sql_engine = DataPrep.SQLEngine()

#import batch and deal ids
sql_cmd = """
    SELECT DISTINCT DealName
		,DealID
		,DealAcquisitionDate
		,BatchName
		,BatchKey
		,BatchAcquisitionDate
	FROM edw_prod.cdw.Batch
	WHERE 1=1
		AND dealid<>0
	ORDER BY DealAcquisitionDate
    """
deal_batch = sql_engine.execute(sql_cmd, output=True)

#import RUW/scenario list


#model_dict
cf_models={}

def unique_deal_ids():
    
    deal_ids = deal_batch[['DealName', 'DealID']].drop_duplicates()
    deal_ids['label'] = deal_ids['DealID'].astype(str) + ' - ' + deal_ids['DealName']
    deal_ids.rename(columns={'DealID':'value'}, inplace=True)
    deal_id_dict = deal_ids[['label', 'value']].to_dict(orient='records')
    return deal_id_dict
    
def unique_batch_keys():
    
    batch_keys = deal_batch[['BatchName', 'BatchKey']].drop_duplicates()
    batch_keys['label'] = batch_keys['BatchKey'].astype(str) + ' - ' + batch_keys['BatchName']
    batch_keys.rename(columns={'BatchKey':'value'}, inplace=True)
    batch_key_dict = batch_keys[['label' ,'value']].to_dict(orient='records')
    return batch_key_dict
    
#create model
def create_model(create_type, model_name, deal_ids, batch_keys, model_type, asset_class):
    
    if create_type=='new':
        new_model = cf_main.CashFlowModel.new_model(model_name, deal_ids, batch_keys, model_type, asset_class)
        
    elif create_type=='refresh':
        pass
    return new_model
