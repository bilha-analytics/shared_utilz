'''
author: bg
goal: 
type: 
refactor: class
'''
# import sys
# sys.path.append("../../../shared") 
import zlogger

import pickle
import re

'''
Training Handler: Listener for training is completed iff done async
Expectation:
 - notifyTrainingIsCompleted with accuracy level, numper of epocs and duration
'''
class ZModelTrainingHandler():
    def training_completed(self, accuracy_achieved, n_epochs, time_taken): 
        raise NotImplementedError

'''
Expectations:
 - init: Initialize model resources 
 - load: Load model from file 
 - dump: save model to file 
 - accuracy: get accuracy of model 
 - predict: get predicted value for a given observation or set of observations
 - train: train model using given dataset 
 - training_handler: listener for when training is done if done async << TODO: allow multiple listners; one for now
'''
class ZModel():
    def __init__(self, name=None):
        self.name = name 


    
    #################### APPLY: PREDICT ######################
    '''
    Load unserialized trained model for reuse
    Filename = <app_name>.zmd
    '''
    def load(self, fpath=None):        
        self.name = getClassName() if fpath is None else re.search('(.*)\.zmd', fpath)[1]
        fpath = self.getModelFPath() if fpath is None else fpath 
        
        try:
            with open( fpath, "rb") as fd:
                self.model = pickle.load( fd) 
                zlogger.log("{}.model.load".format(self.__class__), "Model loaded from file successfully")                
        except:
            zlogger.logError("{}.model.load".format(self.__class__), "Pickle to File - {}".format(fpath) ) 

    '''
    Serialize and save to file a trained model for reuse 
    Filename = <app_name>.zmd 
    '''
    def dump(self, fpath=None): 
        fpath = self.model_fpath if fpath is None else fpath
        try:
            with open( fpath, "wb") as fd:
                pickle.dump( self.model, fd)  
                zlogger.log("{}.model.dump".format(self.__class__), "Model saved to file successfully")
        except:
            zlogger.logError("{}.model.dump".format(self.__class__), "Pickle to File - {}".format(fpath) ) 


    def predict(self, input_text):
        return input_text.upper() 
    

    #################### CREATE: BUILD & TRAIN ###################### << not liking below signatures. TODO:
    '''
    ''' 
    def build(self, **kwargz):
        pass

    def train(self, **kwargz):
        pass 

    def validate(self, **kwargz):
        pass 


    ################## HELPERS #################
    def getModelFPath(self):
        nm = self.getClassName() if self.name is None else self.name 
        return "{}.zmd".format( nm ) 

    def getClassName(self):
        return re.search( '<class.*\.(.*)\'>.*', str(self.__class__) )[1]
    
    
    '''
    '''
    def setTrainingHandler(self, training_handler): 
        self.training_handler = training_handler

    '''
    '''
    def updateTrainingHandler(self):
        if self.training_handler is not None:
            self.training_handler.training_completed(self.accuracy, self.training_epochs, self.training_time ) 

    '''
    '''
    def __str__(self):
        return "{} {} with model file '{}'".format( self.__class__, self.name, self.model_fpath)