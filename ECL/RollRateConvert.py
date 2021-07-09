# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 18:28:22 2021

@author: jalbert
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize, Bounds

class RollRateConvert(object):
    """
    Class to generate Calendar Month roll rates. These roll rates can then be used to 
    calculate PD curves for ECL
    """
    def __init__(self, proj_months=300, co_days=120, variance_step=6):#, scenario):
        
        self.proj_months = proj_months
        self.co_status = int(co_days/30)+1
        self.variance_step = variance_step
        
        #self.base_rolls_forward = [1.000, 0.0175, 0.150, 0.300, 0.550, 0.650, 0.750, 0.800]
        #self.base_rolls_cure = [0.000, 0.000, 0.500, 0.250, 0.150, 0.100, 0.050, 0.030]
        #self.base_rolls_po = [0.000, 0.0175, 0.0150, 0.0055, 0.0050, 0.0030, 0.0020, 0.0010]
        
        self.base_rolls = np.array([ #initial values to start matrix
            [1.000, 0.0175, 0.150, 0.300, 0.550, 0.650, 0.750, 0.800], #forward roll
            [0.000, 0.000, 0.500, 0.250, 0.150, 0.100, 0.050, 0.030], #cure
            [0.000, 0.0175, 0.0150, 0.0055, 0.0050, 0.0030, 0.0020, 0.0010] #po 
            ])
        
        #initalize base roll matrix
        self.full_roll_matrix = np.zeros([self.proj_months,15,15])
        self.full_drift_matrix = np.ones([self.proj_months,15,15])
        self.base_rolls = self.base_rolls[:, :self.co_status]
        #for all rolls greater than co status = 100%
        #self.base_rolls[0, self.co_status:] = 1
        #self.base_rolls[1, self.co_status:] = 0
        #self.base_rolls[2, self.co_status:] = 0
        
        self.create_mask()
        #self.generate_roll_matrix(self.base_rolls[0], self.base_rolls[1][1:], self.base_rolls[2])
        #self.generate_roll_matrix(self.base_rolls)    
        
        self.generate_drift_matrix()
        self.generate_full_rolls(self.base_rolls, self.forward_drift_matrix, self.cure_drift_matrix, self.po_drift_matrix)
        
    def generate_full_rolls(self, base_rolls, forward_drift, cure_drift, po_drift):
        """
        Creates a full roll rate matrix based on base rolls, and drift matrixes
        
        1) base array inputs into rr format (axis 0 sum to 1))
        2) forward/cure/po drift into rr format (1 = 100% = no change)
        3) apply drift matrix in loop to prior month rr

        Parameters
        ----------
        base_rolls : TYPE
            DESCRIPTION.
        forward_drift : TYPE
            DESCRIPTION.
        cure_drift : TYPE
            DESCRIPTION.
        po_drift : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        #create base rolls
        np.absolute(base_rolls, out=base_rolls)
        self.generate_roll_matrix(base_rolls[0], base_rolls[1][1:], base_rolls[2])        
        #apply shift
        self.format_drift_matrix(forward_drift, cure_drift, po_drift)        
        for x in range(1, self.proj_months):
            np.multiply(self.full_roll_matrix[x-1], self.full_drift_matrix[x], out=self.full_roll_matrix[x])        
        #adjust back to 100%
        #tie_out = 1-np.sum(self.full_roll_matrix, axis=2)
        tie_out = 1-np.sum(self.full_roll_matrix, axis=2, where=self.combined_mask[np.newaxis, :, :])
        from_status = list(range(15))
        self.full_roll_matrix[:, from_status, from_status] = tie_out
            
        
    def create_mask(self):
        self.forward_mask = np.zeros([15,15], dtype=bool)
        self.cure_mask = np.zeros([15,15], dtype=bool)
        self.po_mask = np.zeros([15,15], dtype=bool)
        self.co_mask = np.zeros([15,15], dtype=bool)
        
        for x in range(self.co_status):
        #for x in range(8):
            #forward_roll
            self.forward_mask[x, x+1] = 1
            #cure
            if x>0:
                self.cure_mask[x, x-1] = 1
            #po
            self.po_mask[x, 12] = 1
        
        self.co_mask[:, 8] = 1
        
            
        self.combined_mask = np.logical_or(self.forward_mask, self.cure_mask)
        self.combined_mask = np.logical_or(self.combined_mask, self.po_mask)
        self.combined_mask = np.logical_or(self.combined_mask, self.co_mask)
        
        #return self.combined_mask
        
    def format_drift_matrix(self, forward_drift, cure_drift, po_drift):
        """
        Converts input matrices into RR format for easy np multiply

        Parameters
        ----------
        forward_drift : np 2d array
            array with forward drift multipliers.
        cure_drift : np 2d array
            DESCRIPTION.
        po_drift : TYPE
            DESCRIPTION.

        """
        #spread shift across step months
        forward_drift = np.repeat(forward_drift, repeats = self.variance_step, axis=0)
        cure_drift = np.repeat(cure_drift, repeats = self.variance_step, axis=0)
        po_drift = np.repeat(po_drift, repeats = self.variance_step, axis=0)
        #assign matrices
        self.full_drift_matrix[:, self.forward_mask] = forward_drift
        self.full_drift_matrix[:, self.cure_mask] = cure_drift
        self.full_drift_matrix[:, self.po_mask] = po_drift
            
    def generate_roll_matrix(self, base_roll_forward, base_roll_cure, base_roll_po):
        """
        Sets up the base roll rate matrix
        """
        
        #create empty matrix
        base_rolls = np.zeros([15, 15])
        
        base_rolls[self.forward_mask] = base_roll_forward
        base_rolls[self.cure_mask] = base_roll_cure
        base_rolls[self.po_mask] = base_roll_po
        
        #hard code 1 to 0 to 0%
        base_rolls[1, 0] = 0
        base_rolls[0, 12] = 0
        #any balance greater than co_status just move to default
        base_rolls[self.co_status:9, 8] = 1
        
        #from status tie out to 100%
        tie_out = 1-np.sum(base_rolls, axis=1)
        from_status = list(range(15))
        base_rolls[from_status, from_status] = tie_out
        
        self.full_roll_matrix[0] = base_rolls
        
    def generate_drift_matrix(self):
        self.forward_drift_matrix = np.ones([int(self.proj_months/self.variance_step), self.co_status])
        self.cure_drift_matrix = np.ones([int(self.proj_months/self.variance_step), self.co_status-1])
        self.po_drift_matrix = np.ones([int(self.proj_months/self.variance_step), 1])
        
        self.rr_inputs_flatten()
        
    def calibrate_rolls(self, scenario):
        """
        process scenario outputs to generate base CF inputs and target metrics.

        Parameters
        ----------
        scenario : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.scenario_metrics = {}
        #calculate target metrics
        self.scenario_metrics['AsOfDate'] = scenario._cf_data['AsOfDate'][0]
        self.scenario_metrics['bom_upb'] = np.sum(scenario._cf_data['BOM_PrincipalBalance'], axis=0)
        self.scenario_metrics['eom_upb'] = np.sum(scenario._cf_data['upb_trans_agg'], axis=0)
        self.scenario_metrics['eom_upb_sum'] = np.sum(self.scenario_metrics['eom_upb'], axis=1)
        
        co_cum = np.nan_to_num(self.scenario_metrics['eom_upb_sum'][:, 8])
        self.scenario_metrics['co_target'] = self.calc_cum_step(co_cum)
        po_cum = np.nan_to_num(self.scenario_metrics['eom_upb_sum'][:, 12])
        self.scenario_metrics['po_target'] = self.calc_cum_step(po_cum)
        
        
        
        #add partial prepay
        curtail = scenario._cf_data['PrincipalPartialPrepayment']
        curtail_sum = self.eom_status_sum(curtail)
        curtail_sum = np.sum(curtail_sum, axis=1)
        
        self.scenario_metrics['po_target'] = self.scenario_metrics['po_target']+np.nan_to_num(curtail_sum)
        
        #################################################
        #create cash flow metrics
        #sum begin upb
        self.cf_metrics = {}
        self.cf_metrics['bom_upb'] = np.zeros_like(self.scenario_metrics['bom_upb'])
        self.cf_metrics['eom_upb'] = np.zeros_like(self.scenario_metrics['eom_upb'])
        self.cf_metrics['eom_upb_sum'] = np.zeros_like(self.scenario_metrics['eom_upb_sum'])
        
        self.cf_metrics['bom_upb'][0] = np.sum(scenario.cf_input_data['BOM_PrincipalBalance'], axis=0)
        self.cf_metrics['bom_upb'][0][[8, 12, 13]] = 0
        
        #future acquisitions (forward flows only, will be zero for others)
        self.cf_metrics['future_acquisition'] = np.nan_to_num(self.scenario_metrics['eom_upb'][:, 13, 1])
        self.cf_metrics['ppmt'] = np.nan_to_num(np.sum(scenario._cf_data['ContractualPrincipalPayment'], axis=0))
        
        #run goal seek
        #create constraints
        #cons = {'type':'ineq', 'fun': self.constraint_rr_positive}
        #cons = {'type':'eq', 'fun': self.cons_base_roll_tie_out}
        #create bounds
        lbound = np.zeros(self.goal_seek_input.shape)
        ubound = np.zeros(self.goal_seek_input.shape)
        
        lbound[:self.base_rolls.flatten().shape[0]] = 0.000
        ubound[:self.base_rolls.flatten().shape[0]] = 1.000
        lbound[self.base_rolls.flatten().shape[0]:] = 0.500
        ubound[self.base_rolls.flatten().shape[0]:] = 1.500
        bound = Bounds(lbound, ubound)
        
        self.goal_seek_result = minimize(self.goal_seek_rr, self.goal_seek_input, options={'disp': True}, bounds=bound, method='L-BFGS-B') # constraints=cons method='SLSQP', #'Nelder-Mead'
        
    def rr_inputs_flatten(self):
        base_rolls = self.base_rolls.flatten()
        forward_drift = self.forward_drift_matrix.flatten()
        cure_drift = self.cure_drift_matrix.flatten()
        po_drift = self.po_drift_matrix.flatten()
        
        self.goal_seek_input = np.hstack([base_rolls, forward_drift, cure_drift, po_drift])
        
    def rr_inputs_reshape(self, rr_input):
        
        base_shape = self.base_rolls.shape
        forward_shape = self.forward_drift_matrix.shape
        cure_shape = self.cure_drift_matrix.shape
        po_shape = self.po_drift_matrix.shape
        
        #base_roll = rr_input[:np.prod(base_shape)]
        #forward_roll = rr_input[np.prod(base_shape):np.prod(forward_shape)]
        #cure_roll = rr_input[np.prod(forward_shape):np.prod(cure_shape)]
        #po_roll = rr_input[np.prod(cure_shape):]
        base_split = np.prod(base_shape)
        forward_split = np.prod(forward_shape) + base_split
        cure_split = np.prod(cure_shape) + forward_split
        input_reshape = np.array_split(rr_input, [base_split, forward_split, cure_split])
        
        #return input_reshape
        self.base_rolls = np.reshape(input_reshape[0], base_shape)
        self.forward_drift_matrix = np.reshape(input_reshape[1], forward_shape)
        self.cure_drift_matrix = np.reshape(input_reshape[2], cure_shape)
        self.po_drift_matrix = np.reshape(input_reshape[3], po_shape)
        
        
        
    def calc_cum_step(self, metric):
        """
        Converts cumulative totals into monthly step values

        Parameters
        ----------
        metric : array
            array of cumulatively increasing values

        Returns
        -------
        array
        
        """
        
        metric_output = metric - np.pad(metric, (1,0), mode='constant')[:-1]
        return metric_output
    
    def cons_base_roll_tie_out(self, rr_inputs):
        """
        validate that base roll input ties out to 100%

        Parameters
        ----------
        rr_inputs : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.rr_inputs_reshape(rr_inputs)
        self.generate_full_rolls(self.base_rolls, self.forward_drift_matrix, self.cure_drift_matrix, self.po_drift_matrix)
        #output formatted to equal 0
        return np.sum(np.sum(abs(self.full_roll_matrix[0]), axis=1))-15
    
    def constraint_rr_tie_out(self, roll_rates):
        """
        Validate that input rr tie out to 100%

        Parameters
        ----------
        roll_rates : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        np.sum(self.full_roll_matrix, axis=2)
        
    def constraint_rr_positive(self, rr_inputs):
        """
        Validate that all rr are greater than zero

        Parameters
        ----------
        roll_rates : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.rr_inputs_reshape(rr_inputs)
        self.generate_full_rolls(self.base_rolls, self.forward_drift_matrix, self.cure_drift_matrix, self.po_drift_matrix)
        
        return self.full_roll_matrix
        
    
    def goal_seek_rr(self, rr_inputs):
        """
        Objective function to goal seek the roll rates

        Parameters
        ----------
        rr_inputs : tuple
            tuple of inputs needed to calculate new scenario roll rates
            (base_rolls, forward_drift, cure_drift, po_drift)
            self.base_rolls, self.forward_drift_matrix, self.cure_drift_matrix, self.po_drift_matrix
            
        Returns
        -------
        None.

        """
        #reshape flattened vector into separate input arrays
        self.rr_inputs_reshape(rr_inputs)
        #self.generate_full_rolls(rr_inputs[0], rr_inputs[1], rr_inputs[2], rr_inputs[3])
        self.generate_full_rolls(self.base_rolls, self.forward_drift_matrix, self.cure_drift_matrix, self.po_drift_matrix)
        co_var, po_var = self.calc_cash_flows(self.full_roll_matrix)
        return co_var + po_var
        
    def calc_cash_flows(self, roll_rates):
        
        for x in range(min(self.proj_months, self.cf_metrics['eom_upb'].shape[0])):
            if x>0:
                self.cf_metrics['bom_upb'][x] = self.cf_metrics['eom_upb_sum'][x-1]
            #process rolls
            np.multiply(self.cf_metrics['bom_upb'][x][:, np.newaxis], roll_rates[x], out=self.cf_metrics['eom_upb'][x])
            #adjust upb for payments and new acquisition
            np.subtract(self.cf_metrics['eom_upb'][x], self.cf_metrics['ppmt'][x], out=self.cf_metrics['eom_upb'][x])
            np.clip(self.cf_metrics['eom_upb'][x], a_min=0, a_max=None, out=self.cf_metrics['eom_upb'][x])
            #np.add(self.cf_metrics['eom_upb'][x][13, 1], self.cf_metrics['future_acquisition'][x], out=self.cf_metrics['eom_upb'][x, 13,1])
            self.cf_metrics['eom_upb'][x, 13,1] = self.cf_metrics['future_acquisition'][x]
            #sum eom balance 
            np.sum(self.cf_metrics['eom_upb'][x], axis=0, out=self.cf_metrics['eom_upb_sum'][x])
            
        #calc variance
        self.co_output = self.calc_cum_step(self.cf_metrics['eom_upb_sum'][:, 8])
        self.po_output = self.calc_cum_step(self.cf_metrics['eom_upb_sum'][:, 12])
            
        self.co_var = abs(self.co_output-self.scenario_metrics['co_target'])
        self.po_var = abs(self.po_output-self.scenario_metrics['po_target'])
        
        return sum(self.co_var), sum(self.po_var)
            
        
        
        
    def eom_status_sum(self, metric):
        #sum accounts
        metric_sum = np.sum(metric, axis=0)
        #sum begin status
        metric_sum = np.sum(metric_sum, axis=2)
        return metric_sum
        
        
    """
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
    
    
    
         
    rr_test = RollRateConvert()
    
    
    rr_test.forward_drift_matrix
    rr_test.full_roll_matrix
    
    pd.DataFrame(rr_test.cure_mask[:, :6])
    rr_test.full_roll_matrix
    rr_test.full_drift_matrix.shape
    
    rr_test.forward_mask
    
    np.sum(rr_test.full_roll_matrix[0], axis=1)
    
    rr_test.full_drift_matrix[:, rr_test.forward_mask] = 0
    rr_test.full_drift_matrix
    
    rr_test.base_rolls[:,:5+1]
    rr_test.base_rolls[:, :rr_test.co_status]
    
    rr_test.drift_matrix
    rr_test.create_mask()
    rr_test.forward_mask
    rr_test.cure_mask
    
    rr_test.
    rr_test.forward_drift_matrix
    rr_test.base_rolls[0][:5]
    rr_test.forward_mask
    rr_test.base_roll_matrix
    
    int(300/rr_test.variance_step)
    
    #      0     1      2      3      4      5      6      7      8(CO)  9      10     11     12(po) 13     14
        [0.000, 1.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
        [0.000, 0.965, 0.0175, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.0175, 0.000, 0.000],
        [0.000, 0.500, 0.335, 0.150, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.015, 0.000, 0.000],
        [0.000, 0.000, 0.250, 0.4445, 0.300, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.0055, 0.000, 0.000],
        [0.000, 0.000, 0.000, 0.150, 0.2995, 0.550, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.0005, 0.000, 0.000],
        [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 1.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
        [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 1.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
        [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 1.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
        [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 1.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
        [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 1.000, 0.000, 0.000, 0.000, 0.000, 0.000],
        [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 1.000, 0.000, 0.000, 0.000, 0.000],
        [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 1.000, 0.000, 0.000, 0.000],
        [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 1.000, 0.000, 0.000],
        [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 1.000, 0.000],
        [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 1.000],
        ])
    
   """
   