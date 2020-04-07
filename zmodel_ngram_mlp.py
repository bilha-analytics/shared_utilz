'''
author: bg
goal: 
type: 
refactor: class
'''
import zlogger
import zdataset
from zmodel  import ZModel 

import tensorflow as tf 
from tensorflow import keras

import nltk 
import numpy as np 

class NgramMLP(ZModel): 
    
    #################### CREATE: BUILD & TRAIN ######################
    '''
    arch 
        - embedd : from scratch if large, else reuse 
        - 
    ''' 
    def build(self, encoder_context, encoded_matrix, ylabelz, 
                t_ratio, # threshold @  bow Vs seq approach 
                vocab_size, 
                n_classez=2, # bi Vs multi-class @ output layer config 
                n_hidden_layerz = 2, 
                n_hidden_unitz = 64, 
                dropout_rate = 0.2,
                learning_rate = 1e-3
            ): 
        def build_output_layer(n_classez):
            actv_fn, out_unitz = 'sigmoid', 1
            if n_classez  != 2:
                actv_fn, out_unitz = 'softmax', n_classez
            return keras.layers.Dense(activation=actv_fn, units=out_unitz) 

        ### 1. save inflow. expected default = TFIDF 
        self.encoder = encoder_context 
        self.trained_matrix = encoded_matrix 
        self.ylabelz = ylabelz 
        input_shapez = self.trained_matrix.shape[1:]
        # zlogger.log( 'MLP.build' , "input_shapez = {}".format(input_shapez ) ) 

        ### 2. build model << TODO: t_ratio ngram MLP Vs SepCNN
        self.model = keras.models.Sequential() 
        # a. inputs setup etc: flatten 
        # self.model.add( keras.layers.Embedding(vocab_size, n_hidden_unitz) )  ## TODO: reuse
        # self.model.add( keras.layers.Flatten())
        self.model.add( keras.layers.Dropout(rate=dropout_rate , input_shape=input_shapez) ) 
        # b. hidden layers
        for _ in range( n_hidden_layerz - 1 ):
            self.model.add( keras.layers.Dense(
                    units=n_hidden_unitz, 
                    activation = 'relu'
                    ))
            self.model.add(keras.layers.Dropout(
                rate = dropout_rate
            ))
        # c. output layers
        self.model.add( build_output_layer(n_classez) )  

        ### 3. Compile
        loss = 'binary_crossentropy' if n_classez == 2 else 'sparse_categorical_crossentropy'
        optimizer = keras.optimizers.Adam(lr=learning_rate)
        self.model.compile( optimizer=optimizer, loss=loss, metrics=['acc'] ) 


    def train(self,  
                validation_x_y_pair, 
                epochs=10,
                batch_size = 128,
            ):
        ### 1. create callback for early stopping on validation loss if not loss decrease in two consecutive tries
        callbacks = [
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)
        ]
        ### 2. setup data + selectors etc 
        #.astype('float32').toarray().astype('float32')         
        v_train_x = self.trained_matrix.astype('float32').toarray().astype('float32') 
        train_y = np.asarray( self.ylabelz  ).astype('float32') 
        
        val_x, val_y = validation_x_y_pair
        val_x = val_x.astype('float32').toarray().astype('float32')
        val_y = np.asarray( val_y  ).astype('float32')
        
        ### 3. train and validate 
        train_history = self.model.fit(
            v_train_x,
            train_y, 
            epochs = epochs,
            callbacks=callbacks,
            validation_data=(val_x, val_y), 
            verbose=0, # log once per epoch = 2       
            batch_size=batch_size
        )

        train_history = train_history.history 

        # zlogger.log( 'MLP.train', "FINISHED: Epochs {}. Train acc = {} Validation: Accuracy = {} Loss = {}".format(
        #     epochs ,  train_history['acc'][-1], 
        #     train_history['val_acc'][-1], train_history['val_loss'][-1]
        # ))
        return train_history 

    '''
    Requires same length and numerics in np.array lists to compute correctly
        return % correctly classified 
    '''
    def validate(self, text_list, ylabelz_list): 
        # zlogger.log('mlp.validate', "y.IN: {}".format( repr(ylabelz_list) )  )
        predicted_list = np.array( [ self.predict( [rec] ) for rec in text_list ]  ) 
        
        return np.array( predicted_list == np.array(ylabelz_list) ).mean() , predicted_list  



    
    #################### APPLY: PREDICT ######################
    '''
    requires that input_text has been preprocessed in same way as training text
    returns index of most similar doc
    '''
    def predict(self, clean_encoded_text):
        idx = None 
        zlogger.log('mlp.predict', "IN: {}".format( repr(clean_encoded_text ) )  )

        idx = self.model.predict( clean_encoded_text )

        zlogger.log('mlp.predict', "ANS: {}".format( idx ) )  
        
        return idx 