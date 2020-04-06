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
    def build(self, tfidf_context):
        self.model = tfidf_context 


    def train(self, **kwargz):
        pass 

    '''
    Requires same length and numerics in np.array lists to compute correctly
        return % correctly classified 
    '''
    def validate(self, text_list, ylabelz_list): 
        predicted_list = np.array( [ self.predict(rec) for rec in text_list ]  ) 
        return np.array( predicted_list == np.array(ylabelz_list) ) , predicted_list  



    
    #################### APPLY: PREDICT ######################
    '''
    requires that input_text has been preprocessed in same way as training text
    returns index of most similar doc
    '''
    def predict(self, clean_input_text):
        zlogger.log('cosine.predict', "IN: {}".format(clean_input_text ) ) 
        input_vec = self.model.transform( clean_input_text )        
        valz = cosine_similarity( input_vec, self.model )  
        idx = valz.argsort()[0][-2] 
        
        zlogger.log('cosine.predict', "ANS: {}".format( idx ) )  

        flatz = valz.flatten()
        flatz.sort()
        resp = flatz[-2]
        if resp == 0: ## TODO threshold it 
            return None
        else:
            return idx 
    