# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 09:30:41 2020

@author: jalbert
"""

import CashFlowEngine.CashFlowModulesV2 as cf_modules
from abc import ABC, abstractmethod, abstractproperty

class ModelBuilder(object):
    """
    Acts as a director to build various types of models
    """
    
    def __init__(self): 
        self._builder=None
        self.model_list={
                'hazard':CreateModelHazard(),
                'roll_rate':CreateModelRollRate(),
                'monte_carlo':CreateModelMonteCarlo()
                }
    
    #def reset(self):
    #    """
    #    Creates an empty CashFlowEngine objects
    #    """
    #    self._builder=None
    #    #
        
    
    @property
    def builder(self):
        """
        returns the final model and resets to be ready for a new object
        """
        model_out = self._builder.model
        self._builder=None
        for key, m in self.model_list.items():
            m.reset()
        
        return model_out
    
    @builder.setter
    def builder(self, builder):
        self._builder=builder
        
    def set_builder(self, model_type):
        
        #self.reset()
        
        if model_type==1:
            self.builder = self.model_list['hazard']
        elif model_type==2:
            self.builder= self.model_list['roll_rate']
        elif model_type==3:
            self.builder= self.model_list['monte_carlo']
            
        #self._builder.reset()
    
    def validate_module(self, module, model_config):
        """
        Check if module is included in config. if not included exits without initalizing
        """
        module_name = module.__name__
        
        if not model_config.get('modules'):
            module()
        elif model_config['modules'].get(module_name.split('_', 1)[-1], True):
            module()
            
    
    def build_new_model(self, scenario_name, model_type, data_tape, data_rate_curves, cutoff_date, curve_stress=None, model_config=None):
        """
        create new model based on input options
        """
        self.set_builder(model_type)
        
        self._builder.set_model_config(scenario_name, model_type, data_tape, data_rate_curves, cutoff_date, curve_stress, model_config)
        
        self.validate_module(self._builder.add_time, model_config)
        self.validate_module(self._builder.add_rate_curves, model_config)
        self.validate_module(self._builder.add_balance, model_config)
        self.validate_module(self._builder.add_interest, model_config)
        self.validate_module(self._builder.add_payments, model_config)
        self.validate_module(self._builder.add_recovery, model_config)
               
        return self.builder
    
    def build_template_model(self):
        pass
    
    def build_blank_model(self, scenario_name, model_type, data_tape, data_rate_curves, cutoff_date, curve_stress=None, model_config=None):
        """
        Create empty model for user to manually build a new configuration
        """
        self.set_builder(model_type)
        
        self._builder.set_model_config(scenario_name, model_type, data_tape, data_rate_curves, cutoff_date, curve_stress, model_config)        
        
        return self.builder

class EngineBuilder(ABC):
    """
    Base Builder Class
    Interface for the individual models
    Defines placeholder methods to create each cash flow module
    """
    
    #def reset(self):
    #    """
    #    Creates an empty CashFlowEngine object
    #    """
    #    self._model = cf_modules.CashFlowEngine()    
    
    @abstractmethod
    def set_model_config(self):
        pass
    
    #@abstractmethod
    #def add_input_data(self):
    #    pass
    
    @abstractmethod
    def add_time(self):
        pass
    
    @abstractmethod
    def add_rate_curves(self):
        pass
    
    @abstractmethod
    def add_balance(self):
        pass
    
    @abstractmethod
    def add_interest(self):
        pass
    
    @abstractmethod
    def add_payments(self):
        pass
    
    @abstractmethod
    def add_capitalization(self):
        pass
    
    @abstractmethod
    def add_recovery(self):
        pass
        
class CreateModelHazard(EngineBuilder):
    """
    Concrete class to create CDR/CPR model
    """
    def __init__(self):
        self.reset()
        
    def reset(self):
        """
        Creates an empty CashFlowEngine object
        """
        self._model = None
        self._model = cf_modules.CashFlowEngine()
    
    @property
    def model(self):
        """
        returns the final model and resets to be ready for a new object
        """
        model = self._model
        self.reset()
        return model
    
    @model.setter
    def model(self, model):
        self._model=model

    def set_model_config(self, scenario_name, model_type, data_tape, rate_curves, cutoff_date, curve_stress=None, model_config=None):
        self._model.set_model_config(scenario_name, model_type, data_tape, rate_curves, cutoff_date, curve_stress, model_config)
    
    def add_time(self):
        time = cf_modules.CalcTime(self._model)
        self._model.add_module('time', time)

    def add_rate_curves(self): #, model_type, rate_data, data_tape, account_status_list
        rate_curves = cf_modules.CalcRateCurves(self._model)
        self._model.add_module('rate_curves', rate_curves)
        
    def add_balance(self):
        balance = cf_modules.CalcBalance(self._model)
        self._model.add_module('balance',balance)
    
    def add_interest(self, int_accrue_matrix=None, compound_type='monthly'):
        interest = cf_modules.CalcInterest(self._model, int_accrue_matrix=None, compound_type=compound_type)
        self._model.add_module('interest',interest)
    
    def add_payments(self, pmt_matrix=None, amort_type='scale'):
        payments = cf_modules.CalcPayments(self._model, pmt_matrix=None)
        self._model.add_module('payments',payments)
    
    def add_capitalization(self, cap_matrix=None):
        capitalize = cf_modules.CalcCapitalization(self._model, cap_matrix=None, )
        self._model.add_module('capitalize', capitalize)
    
    def add_collateral(self):
        pass
    
    def add_recovery(self):
        recovery = cf_modules.CalcRecovery(self._model)
        self._model.add_module('recovery', recovery)
        
    
    
class CreateModelRollRate(EngineBuilder):
    """
    Concrete class to create RollRate model
    """
    def __init__(self):
        self.reset()
    
    def reset(self):
        """
        Creates an empty CashFlowEngine objects
        """
        self.model = cf_modules.CashFlowEngine()
    
    @property
    def model(self):
        """
        returns the final model and resets to be ready for a new object
        """
        model = self._model
        self.reset()
        return model
    
    @model.setter
    def model(self, model):
        self._model=model
    
    def set_model_config(self, scenario_name, model_type, data_tape, rate_curves, cutoff_date, curve_stress=None, model_config=None):
        self._model.set_model_config(scenario_name, model_type, data_tape, rate_curves, cutoff_date, curve_stress, model_config)
    
    #def add_input_data(self, model_type, data_tape, rate_curves, int_rates=None, pmt_matrix=None):
    #    input_data = cf_modules.InputData(self._model, data_tape, rate_curves, int_rates=None, pmt_matrix=None)
    #    self._model.add_module('input_data',input_data)
    
    def add_time(self):
        time = cf_modules.CalcTime(self._model)
        self._model.add_module('time', time)

    def add_rate_curves(self): #, model_type, rate_data, data_tape, account_status_list
        rate_curves = cf_modules.CalcRateCurves(self._model)
        self._model.add_module('rate_curves', rate_curves)
    
    def add_balance(self):
        balance = cf_modules.CalcBalance(self._model)
        self._model.add_module('balance',balance)
    
    def add_interest(self, int_accrue_matrix=None, compound_type='monthly'):
        interest = cf_modules.CalcInterest(self._model, int_accrue_matrix=None, compound_type=compound_type)
        self._model.add_module('interest',interest)
    
    def add_payments(self, pmt_matrix=None, amort_type='scale'):
        payments = cf_modules.CalcPayments(self._model, pmt_matrix=None, amort_type='scale')
        self._model.add_module('payments',payments)
    
    def add_capitalization(self, cap_matrix=None):
        pass
    
    def add_recovery(self):
        pass
    
class CreateModelMonteCarlo(EngineBuilder):
    """
    Concrete Class to create monte carlo model
    """
    def __init__(self):
        self.reset()
        
    def reset(self):
        """
        Creates an empty CashFlowEngine objects
        """
        self.model = cf_modules.CashFlowEngine()
    
    @property
    def model(self):
        """
        returns the final model and resets to be ready for a new object
        """
        model = self._model
        self.reset()
        return model
    
    @model.setter
    def model(self, model):
        self._model=model
        
    def set_model_config(self, scenario_name, model_type, data_tape, rate_curves, cutoff_date, curve_stress=None, model_config=None):
        self._model.set_model_config(scenario_name, model_type, data_tape, rate_curves, cutoff_date, curve_stress, model_config)

    def add_time(self):
        pass
    
    def add_rate_curves(self):
        pass
    
    def add_balance(self):
        pass
    
    def add_interest(self):
        pass
    
    def add_payments(self):
        pass
    
    def add_capitalization(self):
        pass
    
    def add_recovery(self):
        pass
    
    
    