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

import zdataset 

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
        self.model = None 
        self.model_fpath = None
        self.preprocessor = None 
        self.persist = {
            'name': self.name , 
            'model' : self.model, 
            } 

    
    #################### APPLY: PREDICT ######################
    '''
    Load unserialized trained model for reuse
    Filename = <app_name>.zmd
    '''
    def loadDump(self, fpath=None):  
        fpath = self.getModelFPath(fpath)  

        def unpackPersist():
            if self.persist is not None:
                for k, v in self.persist.items():
                    setattr(self, k, v) 
        
        try:
            with open( fpath, "rb") as fd:
                self.persist = pickle.load( fd) 
                zlogger.log("{}.model.load".format(self.__class__), "Model loaded from file successfully")                
        except:
            zlogger.logError("{}.model.load".format(self.__class__), "Pickle to File - {}".format(fpath) ) 

        unpackPersist()        
        zlogger.log("{}.model.load".format(self.__class__), "Persist unpacked successfully")                

    '''
    Serialize and save to file a trained model for reuse 
    Filename = <app_name>.zmd 
    '''
    def dumpSave(self, fpath=None): 
        fpath = self.getModelFPath(fpath) 
        
        try:           
            with open( fpath, "wb") as fd:
                pickle.dump( self.persist, fd)  
                zlogger.log("{}.model.dump".format(self.__class__), "Model saved to file successfully")
        except:
            zlogger.logError("{}.model.dump".format(self.__class__), "Pickle to File - {}".format(fpath) ) 


    '''
    ''' 
    def preprocessText(self, input_text): 
        clean_text = input_text

        if self.preprocessor is None:
            self.preprocessor = {
                'cleanup_and_lemmatize' : {
                    'remove_stopwordz' : True, 
                    'remove_numberz': True,
                }, 
                # 'doPredictEncode' : { 
                #     'context' : zdataset.getVocabList(input_text, remove_stopwordz=True, remove_numberz=True), 
                #     'enc_type' : zdataset.ZENC_ONEHOT,
                # },
            }

        for fxn, argz in self.preprocessor.items():
            do_fxn = getattr(zdataset, fxn)
            # print( do_fxn ) 
            # print( argz ) 
            clean_text = do_fxn( clean_text, **argz ) 
        
        return clean_text
    
    '''
    '''     
    def predict(self, input_text):
        clean_text = self.preprocessText(input_text)        
        if isinstance(clean_text, np.ndarray):
            clean_text = list( clean_text)
        return "TEMP: {}".format( "__".join(clean_text).upper()  ) 
    

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
    def getModelFPath(self, fpath=None): 
        if self.model_fpath is None:
            self.model_fpath = self.getClassName() if self.name is None else self.name 
        
        self.model_fpath = "{}.zmd".format( fpath ) if fpath is not None else self.model_fpath 

        return self.model_fpath 

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
        return "{} {} with model file '{}'".format( self.__class__, self.name, self.getModelFPath() )