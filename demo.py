#!/usr/bin/env python3
# -*- coding: utf-8 -*-
new = False
update = False
imports = True

if imports:
    
    from iexcredentials import credentials
    from data_manager import DataManager
    from Group_LSTM import GroupAnalysis
    from dataset import Dataset
    from functions import close_label, atr, percent_change, index_change, volume_change
    from functions import am_atr, am_price, am_index, am_volume
    from plots import plot_errorloss, asset_plot, histogram, results_df
#    import tensorboard
#!rm -rf ./logs/

#credentials()

symbols = ['SPY', 'AAPL', 'MSFT', 'TSLA']
index = 'SPY'

dm = DataManager('LIVEFORTUNE', symbols, trail_days=250)


if new:
    data = dm.build_data()
    dm.save_data(data)
else:
    data = dm.load_data()
    if update:
        data = dm.update(data)
        dm.save_data(data)

print('data loaded ...')
      
funcs = {
        'atr': atr,
        'percent_change': percent_change,
        'index_change': index_change,
        'volume_change': volume_change,
        'am_atr': am_atr,
        'am_price': am_price,
        'am_index': am_index,
        'am_volume': am_volume
        }

# build a dataset object from the data
ds = Dataset(data, index=index, symbols=symbols, funcs=funcs, label_func=close_label)
print('dataset built ...')

# build a GA around it
ga = GroupAnalysis(ds)

# create an input vector and label for each day and for each symbol
# also normalizes both and stores the normalization parameters
inputs, labels, scaled_inputs, scaled_labels = ga.dataset.run()

# slice off the last 20 percent for validation
train_inputs, train_labels, val_inputs, val_labels = ga.dataset.split_data(scaled_inputs, scaled_labels, ratio=0.8)

#stack training and validation data
# for this implementation LSTM will non-overlapping samples
# each having a set number of time steps (STACK_DAYS)
# the input data is to have shape: (samples, time_steps, functions)
# the label data is to have shape: (samples, 1, 1) because the label
# for each sample will a scalar value, in this case, the percent
# in price change for that asset between open and close on that day

stacked_train_inputs, stacked_train_labels, stacked_train_dts = ga.dataset.stack_inputs(train_inputs, train_labels, STACK_DAYS=3)

# validation data can overlap without modifying the model
# so it is stacked using dense stack, the associated dates are returned seperately
stacked_val_inputs, stacked_val_labels, stacked_val_dts = ga.dataset.stack_dense_inputs(val_inputs, val_labels, STACK_DAYS=3)

# build and train model
print('building model')
print('input data shape:', stacked_train_inputs[ga.index].shape)
print('label data shape:', stacked_train_labels[ga.index].shape)
ga.build_model((stacked_train_inputs[ga.index].shape[1], stacked_train_inputs[ga.index].shape[2]))


# train on set
history = ga.train_from_dataset(stacked_train_inputs,
                                stacked_train_labels,
                                epochs=250, patience = 100)
#                                epochs=250, callbacks=[callback])
plot_errorloss(ga)

# evaluate the model over the validation set
# for evaluation of monitored metrics
evaluation = ga.eval_from_dataset(stacked_val_inputs, stacked_val_labels)

# also getting predictions over the same set
# for comparisons though in practice this would
# be a 'live' set of data aside from the validation set

predictions = ga.predict_group(stacked_val_inputs)
# descale with the same scaling parameters
descaled_predictions = ga.descale_predictions(predictions)


# see results
res = results_df(ga, evaluation, descaled_predictions, labels)
print(res)

highest_pred = res.index[-1]

asset_plot(ga.dataset, highest_pred, labels, val_labels, stacked_val_dts, descaled_predictions)

histogram(ga.dataset, highest_pred, labels, val_labels, stacked_val_dts, descaled_predictions)
