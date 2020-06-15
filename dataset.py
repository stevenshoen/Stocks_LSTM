#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#from iexfinance.stocks import get_historical_data
#import datetime
#from iexfinance.stocks import Stock
import numpy as np

class Dataset:
    def __init__(self, data,
                 index=None,
                 symbols=[],
                 funcs=None,
                 label_func=None):
        self.symbols = symbols
        self.data = data
        self.index = index
        self.inputs = {}
        self.labels = {}
        self.scalers = {}
        self.funcs = funcs
        self.label_func = label_func
    
    def create_labels(self):
        labels = {}
        for sym in self.symbols:
            print('making labels for ', sym)
            labels[sym] = {}
            for dt in list(self.data[sym]):
                labels[sym][dt] = self.label_func(self, sym, dt)
        return labels
    
    def create_inputs(self):
        inputs = {}
        for sym in list(self.data):
            print('making inputs for ', sym)
            inputs[sym] = {}
            for dt in list(self.data[sym]):
                new_ = {}
                for fun in self.funcs:
                    res = self.funcs[fun](self, sym, dt)
                    if not np.isnan(res):
                        new_[fun] = res
                if len(new_) == len(self.funcs):
                    inputs[sym][dt] = new_
                else:
                    print(' --dropping input for ', dt, ' -- ')
                    print(len(new_), ' not equal to ', len(self.funcs))
                    print('got inputs:')
                    print(new_)
                    
        return inputs    
    
    def run(self):
        inputs = self.create_inputs()
        labels = self.create_labels()
        self.fit_scalers(inputs, labels)        
        scaled_inputs = self.scale_inputs(inputs)
        scaled_labels = self.scale_labels(labels)        
        return inputs, labels, scaled_inputs, scaled_labels

    def split_data(self, inputs, labels, ratio=0.8):
        cut = int(len(inputs[self.index]) * ratio)
        train_dts = sorted(list(inputs[self.index]))[:cut]
        val_dts = list(inputs[self.index])[cut:]
        train_inputs = {sym:{dt:inputs[sym][dt] for dt in train_dts if dt in list(inputs[sym])} for sym in self.symbols}
        train_labels = {sym:{dt:labels[sym][dt] for dt in train_dts if dt in list(labels[sym])} for sym in self.symbols}
        val_inputs = {sym:{dt:inputs[sym][dt] for dt in val_dts if dt in list(inputs[sym])} for sym in self.symbols}
        val_labels = {sym:{dt:labels[sym][dt] for dt in val_dts if dt in list(labels[sym])} for sym in self.symbols}                
        return train_inputs, train_labels, val_inputs, val_labels
    
    def fit_scalers(self, inputs, labels):
        scalers = {}
        for sym in self.symbols:
            scalers[sym] = {}
            for fun in self.funcs:
                scalers[sym][fun] = Scaler(-1, 1)
                scalers[sym][fun].fit([inputs[sym][dt][fun] for dt in list(inputs[sym]) if fun in list(inputs[sym][dt])])
            scalers[sym]['label'] = Scaler(-1, 1)
            scalers[sym]['label'].fit([labels[sym][dt] for dt in list(labels[sym])])
        self.scalers = scalers
    
    def scale_inputs(self, inputs):
        scaled_inputs = {}
        for sym in self.symbols:
            scaled_inputs[sym] = {}
            for dt in list(inputs[sym]):
                scaled_inputs[sym][dt] = {}
                for fun in list(inputs[sym][dt]):
                    res = self.scalers[sym][fun].transform(inputs[sym][dt][fun])
                    scaled_inputs[sym][dt][fun] = res
        return scaled_inputs
    
    def scale_labels(self, labels):
        scaled_labels = {}
        for sym in self.symbols:
            scaled_labels[sym] = {}
            for dt in list(labels[sym]):
                scaled_labels[sym][dt] = self.scalers[sym]['label'].transform(labels[sym][dt])
        return scaled_labels
    
    def stack_inputs(self, scaled_inputs, scaled_labels, STACK_DAYS=4):
        input_dts = list(scaled_inputs[self.index])
        label_dts = list(scaled_labels[self.index])
        dts = sorted(list(set(input_dts) & set(label_dts)))
        days = len(dts)
        offset = days%STACK_DAYS
        num_samples = days//STACK_DAYS
        stacked_dts = [dts[offset + (i * STACK_DAYS): offset + ((i + 1) * STACK_DAYS)] for i in range(num_samples)]
        stacked_input_nps = {}
        stacked_label_nps = {}
        for sym in self.symbols:
            input_sample_np = []
            label_sample_np = []
            for sample in range(num_samples):
                input_step_np = []        
                label_step_np = []
                for dt in stacked_dts[sample]:
                    if dt in list(scaled_inputs[sym]):
                        input_step_np.append(np.array([scaled_inputs[sym][dt][fun] for fun in list(self.funcs)]))
                    else:
                        print('missing input ', sym, ' on day ', dt)
                        print(' --skipping-- ')
                    if dt in list(scaled_labels[sym]):
                        label_step_np.append(np.array([scaled_labels[sym][dt]]))
                    else:
                        print('missing label ', sym, ' on day ', dt)
                        print(' --skipping-- ')
                input_sample_np.append(np.array(input_step_np))
                label_sample_np.append(label_step_np[-1])
            stacked_input_nps[sym] = np.array(input_sample_np)
            stacked_label_nps[sym] = np.array(label_sample_np)
        return stacked_input_nps, stacked_label_nps, stacked_dts     
    
    def stack_dense_inputs(self, scaled_inputs, scaled_labels, STACK_DAYS=4):
        input_dts = list(scaled_inputs[self.index])
        label_dts = list(scaled_labels[self.index])
        dts = sorted(list(set(input_dts) & set(label_dts)))
        days = len(dts)
        stacked_dts = []
        stacked_input_nps = {}
        stacked_label_nps = {}
        for sym in self.symbols:
            input_sample_np = []
            label_sample_np = []
            for sample in range(STACK_DAYS, days + 1):
                input_step_np = []        
                label_step_np = []
                dts_range = dts[sample - STACK_DAYS: sample]
                for dt in dts_range:
                    input_step_np.append(np.array([scaled_inputs[sym][dt][fun] for fun in list(self.funcs)]))
                    label_step_np.append(np.array([scaled_labels[sym][dt]]))
                    stacked_dts.append(dts_range[-1])    
                input_sample_np.append(np.array(input_step_np))
                label_sample_np.append(label_step_np[-1])                
            stacked_input_nps[sym] = np.array(input_sample_np)
            stacked_label_nps[sym] = np.array(label_sample_np)
        return stacked_input_nps, stacked_label_nps, np.unique(stacked_dts)

class Scaler:
    def __init__(self, min_, max_):
        self.min = min_
        self.max = max_
        self.fit_min = None
        self.fit_max = None
        
    def fit(self, x):
        self.fit_min = min(x)
        self.fit_max = max(x)
        return self.fit_min, self.fit_max
    
    def transform(self, x):
        std = (np.array(x) - self.fit_min) / (self.fit_max - self.fit_min)
        x_scaled = std * (self.max - self.min) + self.min
        return x_scaled
    
    def inverse_transform(self, x_scaled):
        std = (np.array(x_scaled) - self.min) / (self.max - self.min)
        x = std * (self.fit_max - self.fit_min) + self.fit_min
        return x
