'''
author: bg
goal: Logic at chat and 
type: util, factory
refactor: class, pre-set responses, 
'''
# import sys
# sys.path.append("../../../shared") 
import zlogger

from termcolor import colored 
import random


from zdataset import cleanup_and_lemmatize
from zmodel import ZModel


## AVAILABLE LEARNING MODELS 
## TODO: refactor 
MODEL_TFIDF = 1 # is default?? 

AVAILABLE_MODELZ = {
}



'''

Expectations
  - Load ES model to be used. Can choose type 
  - Receive user input and fetch appropriate response. Has defaults 
'''


class ZBotLogicFlow():

    ## PRESET REPONSES FOR CERTAIN CATEGORIES TODO: place constants 
    GREETINGZ_INPUT = ["hi", "hello", "greetings", "sasa", "mambo", 'hey', "niaje", "vipi", "salut", "what's up", "are you there"]
    GREETINGZ_RESPONSE = [ "hi", "hey", "hello", "how may i help you today", "what can i do for you", "nice to hear from you", "how are you"]

    EXIT_INPUT = ["bye", "later", "baadaye", "quit", 'laterz']
    EXIT_RESPONSE = ["bye", 'later', 'talk again soon', 'baadaye', 'great chatting with you']

    THANKS_INPUT = ['thanks', 'sounds good', 'asante', 'shukurani', 'shukran']
    THANKS_RESPONSE = ["you're welcome", "glad to be of help", "anytime",  'happy to be of assistance']

    RCODE_KNOWN_RESPONSE = 200
    RCODE_LEARNT_RESPONSE = 210
    RCODE_EXIT_RESPONSE = -99
    '''
    Assumes that model has already been trained 
    '''
    def loadModel(self, mtype, mpath):
        self.model_type = mtype
        mclass = AVAILABLE_MODELZ.get(mtype, ZModel )  
        self.model = mclass() 
        self.model.load(mpath) 

    def getResponse(self, user_input_text): 
        response = None
        rcode = self.RCODE_KNOWN_RESPONSE

        key_words = cleanup_and_lemmatize( user_input_text ) 
        
        was_que = True

        for word in key_words:
            if word in self.GREETINGZ_INPUT:
                response = random.choice( self.GREETINGZ_RESPONSE) 
                was_que = False
                break
            elif word in self.THANKS_INPUT:
                response = random.choice( self.THANKS_RESPONSE )
                was_que = False
                break
            elif word in self.EXIT_INPUT:
                response = random.choice( self.THANKS_RESPONSE )+". "+random.choice( self.EXIT_RESPONSE )
                rcode = self.RCODE_EXIT_RESPONSE 
                return response, rcode 

        if was_que:
            response = self.model.predict( user_input_text ) 
            rcode = self.RCODE_LEARNT_RESPONSE 

        return response, rcode 


    
if __name__ == "__main__":    
    zlogger.log("botLogic.main", "Starting")

    bot = ZBotLogicFlow()
    bot.loadModel( MODEL_TFIDF, "TFIDF_ChatBot.zmd") 

    while( 1 ):
        user_input = input( colored("Talk to me: ", "yellow") )
        prompt = colored( ">>>: ", "green") 

        response, rcode = bot.getResponse( user_input ) 

        print( "{} {}\n".format(prompt, "I don't understand. Try that again" if response is None else response )  ) 
        
        if ( rcode == -99) :
            break 
    
    zlogger.log("botLogic.main", "Finished")
    

