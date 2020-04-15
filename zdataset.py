'''
author: bg
goal: 
type: 
refactor: class, encoders and selectors decouple
'''
# import sys
# sys.path.append("../../../shared") 
import zlogger 
import zdata_source

from collections.abc import Iterable
import string, re 
import math
import os
import numpy as np 
import pandas as pd 

import nltk
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


ZENC_ONEHOT = 1
ZENC_COUNT = 2
ZENC_TFIDF = 3
ZENC_EMBED = 4

ZLEVEL_CHAR = 1
ZLEVEL_WORD = 2
ZLEVEL_SUBWORDS = 3 #phonemes 
ZLEVEL_PHRASES = 4 #n-gram, POS, 
ZLEVEL_SENT = 5


'''
Preprocessing, encoding and filtering helpers 
TODO: word Vs char Vs sentense level << this needs thought!! see all the silly print lines; not even using logger :/
'''

'''
Convert text to lower + remove punctuations 
Input: a list of texts 
return: a list of tokens 
'''
def cleanup_case_and_puncts(text): 
    punctuations = dict( (ord(p), None) for p in string.punctuation )
    # print("IN>>> {}".format( repr(text) ) ) 
    tokenz = " ".join( text ) if not isinstance(text, str) else text
    # print("PUNKT>>> {}".format( repr(tokenz) ) ) 
    tokenz =  nltk.word_tokenize( tokenz.lower().translate( punctuations) ) 
    # print("PUNKT.FIN>>> {}".format( repr(tokenz) ) ) 
    return  tokenz 
'''
Remove stop words 
Input: 
    text: a list of texts tokens 
    stop_wordz: default is nltk.english 
return: a list of tokens 
'''
def cleanup_stopwords(text, stop_wordz=None):    
    stopz = stopwords.words('english') if stop_wordz is None else stop_wordz 
    stopz = set( stopz )     
    # print("\n\n>>>>> CLEAN and stop_words contains covid = {} ".format(
    #         'covid' in stopz if stopz is not None else 'Is None'
    #     ))
    # print("stopz {}".format( repr(stopz) ) ) 
    tokenz = " ".join( text ) if not isinstance(text, str) else text
    # print("STOPZ>>> {}".format( repr(tokenz) ) ) 
    tokenz = [w for w in nltk.word_tokenize( tokenz ) if w.lower().strip() not in stopz ] 
    # print("STOPZ.FIN>>> {}".format( repr(tokenz) ) ) 
    return  tokenz

'''
Remove numerics 
Input: a list of texts tokens 
return: a list of tokens 
'''
def cleanup_numbers(text):    
    # print("IN_NUMZ>>> {}".format( repr(text) ) ) 
    tokenz = " ".join( text ) if not isinstance(text, str) else text    
    tokenz = re.sub( '\d+', "", tokenz ) 
    tokenz = nltk.word_tokenize( tokenz )
    # print("NUMZ.FIN>>> {}".format( repr(tokenz) ) ) 
    return tokenz 

'''
Clean up text: 
Input: 
    text: a list of texts tokens , 
        :   what operations
return: a list of tokens 
'''
def cleanup_text(text, remove_stopwordz=False, stop_wordz=None, remove_numberz=False):
    tokenz =  cleanup_case_and_puncts( text ) 
    if remove_stopwordz:
        tokenz = cleanup_stopwords( tokenz, stop_wordz=stop_wordz) 
    if remove_numberz:
        tokenz = cleanup_numbers( tokenz ) 
    return tokenz
   
'''
Use WordNetLemmatizer english dict

Input:  
Output:  
'''
def lemmatizeTokens(text, remove_stopwordz=False, stop_wordz=None, remove_numberz=False):
    tokenz = cleanup_text( text,  
        remove_stopwordz=remove_stopwordz, stop_wordz=stop_wordz, 
        remove_numberz=remove_numberz) 
    # print("LEMMAZ.IN>>> {}".format( repr(tokenz) ) ) 

    lemmatizer = nltk.stem.WordNetLemmatizer() 
    tokenz = [ lemmatizer.lemmatize( token )  for token in tokenz ] 
    # print("LEMMAZ.FIN>>> {}".format( repr(tokenz) ) ) 

    return tokenz 


'''
TODO: review 
Input:  
Output:  
'''
def cleanup_and_lemmatize(text, remove_stopwordz=False, stop_wordz=None, 
                            remove_numberz=False, lemmatized=True, unique=False ): 

    # print("\n\n>>>>> STARTING and stop_words contains covid = {} ".format(
    #         'covid' in stop_wordz if stop_wordz is not None else 'Is None'
    #     ))
    result = lemmatizeTokens( 
                text, 
                remove_stopwordz=remove_stopwordz, stop_wordz=stop_wordz, 
                remove_numberz=remove_numberz
            ) if lemmatized else cleanup_text( 
                text, 
                remove_stopwordz=remove_stopwordz, stop_wordz=stop_wordz, 
                remove_numberz=remove_numberz)

    if unique:
        result = set(result )

    # print("<<<<< DONE CLEAN UP <<<< {}".format( repr(result) ) ) 

    return result 
'''
word tokens without punct, lemmatize, get unique set

Input:  
Output:  
'''
def getVocabList(text, remove_stopwordz=False, stop_wordz=None, 
                remove_numberz=False, lemmatized=False):
    vocab = cleanup_and_lemmatize( text, 
        remove_stopwordz=remove_stopwordz, stop_wordz=stop_wordz, 
        remove_numberz=remove_numberz, lemmatized=lemmatized) 
    return sorted( set( vocab ) ) 

'''
Assumes preprocessing already done for now. TODO: reconcile with char Vs word Vs sentence level
Input:  
Output: (vocab_vec, matrix) Vocab used and matrix per sentence encoded. 
'''
def onehotEncode(text_list, ngram_range=None):
    vocab = sorted( set( list( np.array(text_list).flatten() ) ) )  
    vect = onehotVectorizer(vocab, text_list)
    return vocab, vect

def onehotVectorizer(vocab, text):
    vect = []
    for ln in text:
        tmp = np.zeros( len(vocab) ) 
        for i, v in enumerate(vocab):
            if v in ln.lower() :
                tmp[i] = 1
        vect.append( tmp )
    return vect

'''
Assumes preprocessing already done for now. TODO: reconcile with char Vs word Vs sentence level
TODO: other CountVectorize params
Input:  
Output: (count_vectorizer, res_matrix) Vectorizer used and matrix per sentence encoded. 
'''
def countEncode(text_list, ngram_range=(1,1), **kwargz ):
    cv = CountVectorizer( ngram_range=ngram_range ) 
    res_matrix = cv.fit_transform( text_list ) #.toarray().atype('float32')

    return cv, res_matrix 


'''
Assumes preprocessing already done for now. TODO: reconcile with char Vs word Vs sentence level
TODO: Other TfidfVectorizer params e.g min-df
Input:  
Output: (count_vectorizer, res_matrix) Vectorizer used and matrix per sentence encoded. 
'''
def tfidfEncode(text_list, ngram_range=(1,1), **kwargz):  
    cv = TfidfVectorizer( ngram_range=ngram_range ) 

    res_matrix = cv.fit_transform( text_list ) #.toarray().atype('float32')

    return cv, res_matrix 


'''
Assumes preprocessing already done for now. TODO: reconcile with char Vs word Vs sentence level
Input:  
Output: (count_vectorizer, res_matrix) Vectorizer used and matrix per sentence encoded. 
'''
def embeddingEncode(text_list, ngram_range=(1,1), **kwargz):  
    raise NotImplementedError 


'''
TODO:
'''
def doTrainEncode(text_list, enc_type=1000, ngram_range=(1,1), **argz):
#     params = ['remove_stopwordz', 'stop_wordz', 'remove_numberz','lemmatized', 'ngram_range' ]
#     for p in params:
#         if not p in argz.keys():
#             argz[p] = None 

    encoder = dict_encoders.get(enc_type, tfidfEncode) 
    
    return encoder( text_list, ngram_range=ngram_range, **argz)  ## encoder/context and encoded matrix  

'''
TODO: include preprocessing
'''
def doPredictEncode(text, context=None, enc_type=ZENC_TFIDF):
    result = None 

    if enc_type == ZENC_ONEHOT:
        result = onehotVectorizer(context, np.array(text) )

    elif enc_type == ZENC_COUNT or enc_type == ZENC_TFIDF:
        result = context.transform( text )        
    else:
        pass 

    return result

'''
TODO: np.array Vs list consistent use 
'''
def splitTrainTest(clean_data, test_prop=0.2): 
    the_data = np.array( clean_data ) 
    
    zlogger.log('splitTrainTest', "Provided data size = {}\n{}".format( len(the_data), the_data[0] ) ) 

    n_recs = len( the_data ) 
    n_test = math.trunc( 0.2*n_recs ) 
    # shuffle 
    np.random.shuffle( the_data )
    # split 
    train_data, test_data = the_data[:(n_recs-n_test)], the_data[(n_recs-n_test): ]
    #TODO: should we flatten and when

    return list(train_data), list(test_data )



###########################################################
'''
'''

dict_encoders = {
    ZENC_ONEHOT : onehotEncode,
    ZENC_COUNT : countEncode, 
    ZENC_TFIDF : tfidfEncode,
    ZENC_EMBED : embeddingEncode
} # default = tfidf 

dict_selectors = {

}



# class ZEncoder():
#     def __init__(self, ptype, pconfig):
#         self.prep_type = etype 
#         self.prep_cofig = econfig 
#         self.persist = {
#             'encoder_type' : self.encoder_type, 
#             'encoder_config' : self.encoder_cofig, 
#         }
    



'''
TODO: abstract class, iterable, own and encap a data
'''
class ZDataset():
    ## TODO: overloaded constructor 
    ## TODO: expand multiple data in one return e.g. (train, test) or (faq_db, train)
    
    def initz(self):         
        self.stop_wordz = None
        self.clean_data = None
        self.y_labelz = None 
        self.x_index = None 
        self.paramz = None
        self.context = None
        self.encoded_matrix = None 
        self.data = None         
        self.class_categories = None 
        self.preprocessor = None
        ## TODO: save paramz @ dump clean_data
        self.paramz = {
            'remove_stopwordz' : True,
            'stop_wordz' : None, 
            'remove_numberz' : True, 
            'lemmatized' : True, 
            'unique' : False
        }
        self.enc_type = ZENC_TFIDF


    def initFromSeq(self, data):
        self.initz()
        self.data = np.array(data )
        self.updateXIndex()
        self.updateYIndex()

    def initFromResource(self, data_path, data_type): 
        self.initz()
        self.load( data_path, data_type)

    ####TODO: ownership 
    def getStopWords(self, new_stop_words=None):
        if self.stop_wordz is None:
            self.stop_wordz = stopwords.words('english')      #[] #TODO
        if new_stop_words is not None:
            self.stop_wordz.extend( new_stop_words) 
        return self.stop_wordz 

    def load(self, data_path, data_type):
        self.data_path = data_path
        self.data_type = data_type 
        self.data = np.array( data_source.readFrom(data_path, data_type)  )  
        self.updateXIndex()
        self.updateYIndex()

   
    def dumpSave(self, data_path=None, data_type=None):
        dpath = self.data_path if data_path is None else data_path 
        dtype = self.data_type if data_type is None else data_type
        
        filez = self.getDumpSaveItems()

        for ext, db in filez.items():
            tf = "{}.{}".format(dpath, ext) 
            if db is not None: 
                zdata_source.writeTo( db, tf, dtype=dtype) 
    
    def getDumpSaveItems(self):
        return {
            'xftz' : self.data if self.clean_data is None else self.clean_data , 
            'xidx' : self.x_index, 
            'ylbz' : self.y_labelz, 
            'argz' : self.paramz, 
            'ctx' : self.context,
            'encm' : self.encoded_matrix,
            'stpz' : self.stop_wordz,
        }


    def dumpLoad(self, data_path=None, data_type=None):
        self.initz()
        ##TODO: reconcile 
        self.data_path = data_path 

        dpath = self.data_path if data_path is None else data_path 
        dtype = self.data_type if data_type is None else data_type

        filez = self.getDumpLoadItems() 

        for ext, db in filez.items():
            tf = "{}.{}".format(dpath, ext) 
            if os.path.exists( tf ): 
                setattr( self, db, zdata_source.readFrom( tf, dtype=dtype)  ) 
                zlogger.log('zdataset.dumpLoad', "Loaded {} of size {}".format( tf, len(getattr(self, db) ) ) ) 
            else:
                zlogger.log('zdataset.dumpLoad', "Not Found: {}".format( tf) ) 
        
        self.data = self.clean_data 
        self.updateXIndex()
        self.updateYIndex()
        # zlogger.log('zdataset.dumLoad', "FINISHED: clean data size {}".format( len(self.data) ) ) 

    def getDumpLoadItems(self):
        return {
            'xftz' : "clean_data",
            'xidx' : 'x_index', 
            'ylbz' : 'y_labelz', 
            'argz' : 'paramz', 
            'ctx' : 'context',
            'encm' : 'encoded_matrix',
            'stpz' : 'stop_wordz', 
        }

    
    ## TODO: ensure consistency with clean_data and original data
    def updateXIndex(self):
        self.x_index = { i : x for i, x in enumerate(self.clean_data) } if self.clean_data is not None else None 

    def updateYIndex(self):
        self.class_categories = set(self.y_labelz ) if self.y_labelz is not None else None 
    '''
    Can be called before or after preprocessing TODO: set as option 
    Return:
        train <ZDataset>: 
        test <ZDataset>

    TODO: abstract class instantiate subclasses reflection 

    '''
    def splitIntoTrainTest(self, test_prop=0.2):
        db = self.clean_data if self.clean_data is not None else self.data

        train, test = splitTrainTest(db, test_prop) 
        
        self.train_data = ZDataset()
        self.train_data.initFromSeq(train)
        self.train_data.clean_data = self.train_data.data

        self.test_data = ZDataset()
        self.test_data.initFromSeq(test)
        self.test_data.clean_data = self.test_data.data

        return self.train_data, self.test_data

     
    '''
    Calculate (n_observations / avg_words_per_observation ) t_ratio
        If above t_ratio < 1500, tokenize as n-grams and use a simple multi-layer perceptron (MLP) model to classify
        Else Sequence RNN e.g. SepCNN 
    '''
    def thresholdRatio(self):        
        n_observations = len( self.clean_data ) 
        avg_words_per_observation = [ len( nltk.word_tokenize(rec) )  for rec in self.clean_data]
        avg_words_per_observation = np.array( avg_words_per_observation ).mean() 

        return n_observations / avg_words_per_observation


    def getPredictedAtIndex(self, y_index): 
        return self.y_labelz[ y_index ] if self.y_labelz is not None else None 
    
    def getYLabelIndex(self, y_text): 
        # return np.where( self.y_labelz == y_text )  if self.y_labelz is not None else None 
        return list( self.class_categories ).index(y_text )  if self.class_categories is not None else None 

    def ylabelzAsInts(self, val_ylabz = None): 
        val_ylabz = self.y_labelz if val_ylabz is None else val_ylabz 
        # zlogger.log('zdataset.yInts', "{} from cats of {}".format(val_ylabz, repr(self.class_categories)) )
        return [ self.getYLabelIndex(y) for y in val_ylabz ]


    def getNumberClasses(self):
        return len( set( list(self.class_categories) ) ) 

    '''
    Things done
        - to lower case 
        - remove punctuations 
        - remove stop words 
        - remove escape characters 
        - remove numbers 
        - lemmatize 
    TODO: bag of words vs sequences
    Efficient array op @ vectorize

    TODO:
    end state: 
        Opt 1: a string/sentence of tokens <== let vectorizer decide
        Opt 2: self.clean_data represents each observation as a list of tokens 
    '''
    def preprocess(self, remove_stopwordz=False, stop_wordz=None, 
                    remove_numberz=False, lemmatized=True, unique=False): 
        self.paramz = {
            'remove_stopwordz' : remove_stopwordz,
            'stop_wordz' : stop_wordz, 
            'remove_numberz' : remove_numberz, 
            'lemmatized' : lemmatized, 
            'unique' : unique
        }
        # tmp = np.vectorize(cleanup_and_lemmatize)( # 
        self.clean_data  =  np.array( [ 
                " ".join( cleanup_and_lemmatize(
                        rec, 
                        remove_stopwordz=remove_stopwordz, stop_wordz=stop_wordz, 
                        remove_numberz=remove_numberz, lemmatized=lemmatized) )
                for rec in self.data ] ) 
                
        if unique: #use vocab within a record 
            self.clean_data = np.array([
                " ".join( sorted( set( rec ) )   ) 
                        for rec in self.clean_data
            ])
        
        if self.preprocessor is None:
            self.preprocessor = { }
        self.preprocessor['cleanup_and_lemmatize'] = self.paramz   

        self.updateXIndex()

    '''
    @TODO: use same method as preprocess;; reuse
    '''
    def preprocessPredict(self, input_text_list): 
        cleaned_input_text  =  np.array( [ 
                " ".join( cleanup_and_lemmatize(
                        rec, 
                        remove_stopwordz = self.paramz['remove_stopwordz'], 
                        stop_wordz = self.paramz['stop_wordz'], 
                        remove_numberz = self.paramz['remove_numberz'] , 
                        lemmatized = self.paramz['lemmatized'] ) )
                for rec in input_text_list ] ) 
                
        if self.paramz['unique']: #use vocab within a record 
            cleaned_input_text = np.array([
                " ".join( sorted( set( rec ) )   ) 
                        for rec in cleaned_input_text
            ])

        return cleaned_input_text 

    '''
    get vocab across entire data set
    TODO: fix recomputes each time
    ''' 
    def getVocab(self):
        dat = [
                " ".join( sorted( set( nltk.word_tokenize(rec) ) ) ) 
                        for rec in self.clean_data
            ] 
        # print("DSET.VOCAB.JOIN>>> {}".format( repr(dat) ) ) 
        
        dat = nltk.word_tokenize( " ".join(dat) )         
        # print("DSET.VOCAB.TOKZ>>> {}".format( repr(dat) ) ) 

        vocab = sorted( set( dat) )
        # print("DSET.VOCAB.FIN>>> {}".format( repr(vocab) ) ) 

        return  vocab  

    ### TODO: major fix, too tired to make sense of it but there's a silly redundancy in here
    def getWords(self):
        dat = [
                " ".join( sorted( set( nltk.word_tokenize(rec) ) ) ) 
                        for rec in self.clean_data
            ] 
        dat = nltk.word_tokenize( " ".join(dat) )  

        # print("DSET.WORDS.FIN>>> {}".format( repr(dat) ) )         

        return  dat 


    ###------ TODO: OWNERSHIP @ ENCODERS, SELECTORS, 
    '''
    TODO: expand options and train Vs test/predict
    '''
    def encodeTrain(self,  enc_type=ZENC_TFIDF, ngramz=(1,1) ):       
        self.enc_type = enc_type 
        self.context, self.encoded_matrix = doTrainEncode( self.clean_data,  enc_type=enc_type, ngram_range=ngramz ) 
        
        if self.preprocessor is None:
            self.preprocessor = { } 
        self.preprocessor['doPredictEncode'] = {'enc_type': enc_type, 'context': self.context, }  

        return self.context, self.encoded_matrix 

    def encodePredict(self, text_list):
        cleaned_text = self.preprocessPredict(text_list) 
        return doPredictEncode(  cleaned_text, self.context,  self.enc_type )  
   

'''
'''
class ZGsheetFaqDataSet( ZDataset ):    
    ## todo: SUPER 
    def initz(self):
        ZDataset.initz(self)
        self.phrases_db = None
        self.faq_db = None 


    def load(self, data_path, data_type):
        self.data_path = data_path
        self.data_type = data_type 
        # zdata_source. returns dicts with class_cat as key         
        self.faq_db, self.phrases_db = zdata_source.readFrom(data_path, data_type)  
        self.data = np.array( [ k for k in self.phrases_db.keys() ] ) 
        self.y_labelz = np.array( [ k for k in self.phrases_db.values() ] )  
        self.updateXIndex()
        self.updateYIndex()


    ##TODO: super call
    def getDumpSaveItems(self ):
        tmp = ZDataset.getDumpSaveItems( self) 
        tmp2 = {
            'faqdb' : self.faq_db,
            'phrdb' : self.phrases_db,
        }
        return { **tmp , **tmp2}


    def getDumpLoadItems(self ): 
        tmp = ZDataset.getDumpLoadItems(self)
        tmp2 = {
            'faqdb' : 'faq_db',
            'phrdb' : 'phrases_db',
        }
        return { **tmp , **tmp2}

    def getPredictedAtIndex(self, y_index): 
        if y_index is None:
            return None, None
        # zlogger.log( 'zdataset.get_y-at', "IN: {}".format(y_index ) )
        class_cat = self.y_labelz[ y_index ] 
        # zlogger.log( 'zdataset.get_y-at', "CAT: {}".format( class_cat ) ) 
        return class_cat, self.faq_db.get( class_cat , None )  

###########################################################

if __name__ == "__main__":
    src = "dataset.main"
    zlogger.log(src, ">>>>> STARTING\n")
    
    st = "The quick brown fox jumped over the lazy dogs. This is an account of a lost dog. His name was Jazzy and he had 7 bones. Hey there! Okay, bye." 
    # st = nltk.sent_tokenize( st )

    ds = ["The quick brown fox", "He had 7 bones"] 

    ps = "The brown bones" # predict text 
    
    tokz = lemmatizeTokens(st)
    print( "Tokens len: {}\n{}\n".format( len(tokz), tokz) )
    

    dset = ZDataset()
    dset.initFromSeq( ds ) 
    dset.preprocess()     
    print( "Tokens len: {}\n{}\n".format( len(dset.clean_data), dset.clean_data ) ) 

    vocab, matrix = dset.encodeTrain(enc_type=ZENC_ONEHOT)
    matrix = np.array(matrix)
    zlogger.log(src+".OnehotEncode", "Vocab = {} \nMatrix = {}".format(len(vocab), matrix.shape) )
    print( "Context: {}\n Matrix: {}\n".format(vocab, matrix))
    res = dset.encodePredict(ps)
    zlogger.log(src+".OnehotEncodePredict", "Input = {} Encoding = {}".format(ps, res) )

    vocab, matrix = dset.encodeTrain(enc_type=ZENC_COUNT)
    matrix = np.array(matrix)
    zlogger.log(src+".CountEncode", "Vocab = {} \nMatrix = {}".format(vocab, matrix.shape) )
    print( "Context: {}\n Matrix: {}\n".format(vocab, matrix))
    res = dset.encodePredict(ps)
    zlogger.log(src+".CountEncodePredict", "Input = {} Encoding = {}".format(ps, res) )


    vocab, matrix = dset.encodeTrain(enc_type=ZENC_TFIDF)
    matrix = np.array(matrix)
    zlogger.log(src+".TfidfEncode", "Vocab = {} \nMatrix = {}".format(vocab, matrix.shape) )
    print( "Context: {}\n Matrix: {}\n".format(vocab, matrix))
    res = dset.encodePredict(ps)
    zlogger.log(src+".TfidfEncodePredict", "Input = {} Encoding = {}".format(ps, res) )

    zlogger.log(src, "=== From GSheet ===")
    faq_path = [ ('1EuvcPe9WXSQTsmSqhq0LWJG4xz2ZRQ1FEdnQ_LQ-_Ks', 'FAQ responses!A1:G1000'), ('1EuvcPe9WXSQTsmSqhq0LWJG4xz2ZRQ1FEdnQ_LQ-_Ks', 'Classify_Phrases!A1:G1000')]
    faq_typ = zdata_source.zGSHEET_FAQ


    dset = ZGsheetFaqDataSet()
    dset.initFromResource(faq_path, faq_typ)  
    dset.preprocess()     
    dlen =  len(dset.clean_data) 
    print( "Tokens len: {}\n{}\n".format(dlen, dset.clean_data[: min(dlen, 10) - 1 ] ) ) 

    vocab, matrix = dset.encodeTrain(enc_type=ZENC_TFIDF)
    zlogger.log(src+".TfidfEncode", "Vocab = {} \nMatrix = {}".format(vocab, matrix.shape) )
    print( "Context: {}\n Matrix: {}\n".format(vocab, matrix))
    res = dset.encodePredict(ps)
    zlogger.log(src+".TfidfEncodePredict", "Input = {} Encoding = {}".format(ps, res) )

    k = list(dset.faq_db.keys())[0]
    v = dset.faq_db[ k ] 
    zlogger.log(src+".DictFaq",  "{} : {}".format( k, v   ) )
    k = list(dset.phrases_db.keys())[0]
    v = dset.phrases_db[ k ] 
    zlogger.log(src+".Phrases",  "{} : {}".format( k, v   ) )
    # dset.category_to_index()
    # zlogger.log(src+".DictFaq", "{} \nItems: {}".format(len(dset.faq_db), dset.faq_db ) )
    


    zlogger.log(src, "FINISHED <<<<<")
