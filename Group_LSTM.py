#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import keras
#from keras import Sequential
#from keras.layers import Dense
#from keras.layers import LSTM
#import keras.backend as K
import matplotlib.pyplot as plt
import pandas as pd
#from keras.callbacks import EarlyStopping
import datetime

class GroupAnalysis:
    def __init__(self, dataset):
        self.dataset = dataset
        self.model = None
        self.model_filename = None
        self.symbols = self.dataset.symbols
        self.index = self.dataset.index
        self.default_weights = None
        self.weights = {}
        self.history = {}
    
    def save_model(self):
        self.model.save(self.model_filename)
    
    def load_model(self):
        self.model.load(self.model_filename)
    
    def build_model(self, input_shape):
        """
        inputs to LSTM must be shape:
                samples, time steps, and features.
        model.add take input_shape=time_steps, features
        """
        model = keras.Sequential()
        model.add(keras.layers.LSTM(64, return_sequences=True, input_shape=input_shape, activation='tanh'))
        model.add(keras.layers.Dense(64))
        model.add(keras.layers.LSTM(64, return_sequences=False))
#        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(1))# , activation='selu')

#        model = keras.Sequential()
#        model.add(keras.layers.LSTM(16, return_sequences=True, input_shape=input_shape, activation='tanh'))
#        model.add(keras.layers.LSTM(16, return_sequences=True, activation='tanh'))
#        model.add(keras.layers.LSTM(16, return_sequences=False, activation='tanh'))
#        model.add(keras.layers.Dense(1))# , activation='selu')
        loss = tf.keras.losses.MeanSquaredError()
        metrics=[tf.keras.metrics.MeanAbsoluteError()]
        
        model.compile(loss=loss, optimizer='adam', metrics=metrics)
        
        self.default_weights = model.get_weights()
        self.model = model
        return model

    def reset_weights(self):
        session = keras.backend.get_session()
        for layer in self.model.layers: 
            if hasattr(layer, 'kernel.initializer'): 
                layer.kernel.initializer.run(session=session)
            if hasattr(layer, 'bias.initializer'):
                layer.bias.initializer.run(session=session)  
            
    def train(self, inputs, labels, epochs=10):
        history = self.model.fit(inputs, labels, epochs=epochs, shuffle=False)
        return history
    
    def predict(self, inputs):
        prediction = self.model.predict(inputs)
        return prediction.flatten()
        
    def eval_from_dataset(self, input_data, label_data):
        p = {}
        for sym in self.symbols:
            self.reset_weights()
            self.model.set_weights(self.weights[sym])
            p[sym] = self.model.evaluate(input_data[sym], label_data[sym])
        return p

    def train_from_dataset(self,input_data, label_data, epochs=15, patience=None):
        for sym in self.symbols:
            self.reset_weights()
            self.model.set_weights(self.default_weights)
            callback1 = keras.callbacks.EarlyStopping(monitor='mean_squared_error', patience=epochs, restore_best_weights=True)
#            logdir="logs/fit/" + sym + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
#            callback2 = keras.callbacks.TensorBoard(log_dir=logdir)
            history = self.model.fit(input_data[sym],
                                     label_data[sym],
                                     epochs=epochs,
                                     shuffle=False,
                                     callbacks=[callback1])
            self.history[sym] = history.history
            self.weights[sym] = self.model.get_weights()
        return self.history
    
    def val_fit(self, training_inputs, training_labels, validation_inputs, validation_labels, epochs=15):
        for sym in self.symbols:
            self.reset_weights()
            self.model.set_weights(self.default_weights)
            history = self.model.fit(training_inputs[sym],
                                     training_labels[sym],
                                     epochs=epochs,
                                     shuffle=False,
                                     batch_size=2,
                                     validation_data=(validation_inputs[sym], validation_labels[sym]))
            self.history[sym] = history.history
            self.weights[sym] = self.model.get_weights()
        return self.history
    
    def predict_group(self, inputs):
        predictions = {}
        for sym in self.symbols:
            self.reset_weights()
            self.model.set_weights(self.weights[sym])
            p = self.predict(inputs[sym])
            predictions[sym] = p.flatten()
        return predictions
    
    def descale_predictions(self, predictions):
        descaled_predictions = {}
        for sym in self.symbols:
            descaled_predictions[sym] = self.dataset.scalers[sym]['label'].inverse_transform(predictions[sym])
        return descaled_predictions

    def results_df(self, evaluation, descaled_predictions, labels):
        results = {}
        for sym in self.symbols:
            results[sym] = {}
            results[sym]['training_loss'] = self.history[sym]['loss'][-1]
            results[sym]['training_error'] = self.history[sym]['mean_absolute_error'][-1]            
            results[sym]['eval_loss'] = evaluation[sym][0]
            results[sym]['eval_error'] = evaluation[sym][1]            
            results[sym]['last_prediction'] = descaled_predictions[sym][-1]            
        results_df = pd.DataFrame.from_dict(results, orient='index')
        results_df = results_df.sort_values(by=['last_prediction'])
        return results_df





