'''
author: bg
goal: 
type: 
refactor: class
'''
import zlogger
import zdataset
from zmodel  import ZModel 


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk 
import numpy as np 


class ZCosineSimilarity(ZModel):  

    #################### CREATE: BUILD & TRAIN ######################
    '''
    ''' 
    def build(self, tfidf_context, tfidf_matrix, train_ylabz_idxs):
        self.model = tfidf_context 
        self.trained_matrix = tfidf_matrix 
        self.train_ylabz_idxs = train_ylabz_idxs 


    def train(self, **kwargz):
        pass 

    '''
    Requires same length and numerics in np.array lists to compute correctly
        return % correctly classified 
    '''
    def validate(self, text_list, ylabelz_list): 
        def getPredictedYlabzIdxs(pred_x_idx):
            return self.train_ylabz_idxs[ pred_x_idx]

        # zlogger.log('cosine.validate', "y.IN: {}".format( repr(ylabelz_list) )  )
        predicted_list = np.array( [ self.predict( [rec] ) for rec in text_list ]  ) 
        predicted_list = np.array([ getPredictedYlabzIdxs(x) for x in predicted_list ] ) 
        return np.array( predicted_list == np.array(ylabelz_list) ).mean() , predicted_list  



    
    #################### APPLY: PREDICT ######################
    '''
    requires that input_text has been preprocessed in same way as training text
    returns index of most similar doc
    '''
    def predict(self, clean_input_text):
        # zlogger.log('cosine.predict', "IN: {}".format( repr(clean_input_text ) )  )
        input_vec = self.model.transform( clean_input_text )        
        valz = cosine_similarity( input_vec, self.trained_matrix )  
        idx = valz.argsort()[0][-2] 
        
        # zlogger.log('cosine.predict', "ANS: {}".format( idx ) )  

        # flatz = valz.flatten()
        # flatz.sort()
        # resp = flatz[-2]
        # if resp == 0: ## TODO threshold it 
        #     idx =  None

        return idx 