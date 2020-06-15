#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt

def histogram(ds, sym, labels, validation_labels, stacked_val_dts, descaled_predictions):    
    comparison = {}
    last_dt = stacked_val_dts[-1]
    for i, dt in enumerate(stacked_val_dts):
        comparison[dt] = {}
        comparison[dt]['predicted'] = descaled_predictions[sym][i]
        if dt != last_dt:
            comparison[dt]['actual'] = labels[sym][dt]
            comparison[dt]['abs_error'] = descaled_predictions[sym][i] - labels[sym][dt]
    comparison_df = pd.DataFrame.from_dict(comparison, orient='index')
    fig, (histl_ax, histp_ax, histe_ax) = plt.subplots(3, 1, sharex='col', sharey='col')   
    
    histl_ax.set_title(sym + ' Actual Label Distribution')
    histl_ax.hist([labels[sym][dt] for dt in list(validation_labels[sym])])
    histl_ax.set_ylabel('count')
    
    histp_ax.set_title(sym + ' Prediction Distribution')
    histp_ax.hist(descaled_predictions[sym])       
    histp_ax.set_ylabel('count')
    
    histe_ax.set_title(sym + ' Absolute Error Distribution')
    histe_ax.hist(comparison_df['abs_error'].values)     
    histe_ax.set_ylabel('count')
    histe_ax.set_xlabel('% change')
    
def asset_plot(ds, sym, labels, validation_labels, stacked_val_dts, descaled_predictions):  
    comparison = {}
    prices = {}
    last_dt = stacked_val_dts[-1]
    for i, dt in enumerate(stacked_val_dts):
        comparison[dt] = {}
        comparison[dt]['predicted'] = descaled_predictions[sym][i]
        if dt != last_dt:
            comparison[dt]['actual'] = labels[sym][dt]
            comparison[dt]['abs_error'] = descaled_predictions[sym][i] - labels[sym][dt]
        prices[dt] = {'high':max([ds.data[sym][dt][t]['high'] for t in list(ds.data[sym][dt])]),
                    'low':min([ds.data[sym][dt][t]['low'] for t in list(ds.data[sym][dt])]),
                    'close':ds.data[sym][dt][list(ds.data[sym][dt])[-1]]['close'],
                    'open':ds.data[sym][dt][list(ds.data[sym][dt])[0]]['open']}
    comparison_df = pd.DataFrame.from_dict(comparison, orient='index')
    price_df = pd.DataFrame.from_dict(prices, orient='index')    
    
    fig, (price_ax, comp_ax, err_ax) = plt.subplots(3, 1, sharex='col')   
    comp_ax.set_title(sym + ' Predicted vs. Actual')
    comp_ax.set_ylabel('% change')
    comparison_df.drop(columns=['abs_error']).plot(ax=comp_ax)
    comp_ax.legend(*comp_ax.get_legend_handles_labels(), loc='center left')
    
    err_ax.set_title(sym + ' Prediction Error')
    err_ax.set_ylabel('% change')
    comparison_df['abs_error'].plot(ax=err_ax, sharex=comp_ax)
    err_ax.legend(*err_ax.get_legend_handles_labels(), loc='center left')
    
    price_ax.set_title(sym + ' Prices')
    price_ax.set_ylabel('USD price')
    price_df.plot(ax=price_ax, sharex=comp_ax, grid=True)
    price_ax.legend(*price_ax.get_legend_handles_labels(), loc='center left')
    
    plt.xticks(rotation=45)
    plt.xlim(stacked_val_dts[0], stacked_val_dts[-1])    

def plot_errorloss(ga):
    h = ga.history
    for sym in ga.symbols:
        loss = h[sym]['loss']
        mse = h[sym]['mean_absolute_error']
        epochs = range(len(loss))
        plt.figure('Model Loss')
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.plot(epochs, loss, label=sym)
        plt.figure('Model Error')
        plt.xlabel('epochs')
        plt.ylabel('mean squared error')
        plt.plot(epochs, mse, label=sym)
    plt.figure('Model Loss')
    plt.legend()
    plt.figure('Model Error')
    plt.legend()

def results_df(ga, evaluation, descaled_predictions, labels):
    results = {}
    for sym in ga.symbols:
        results[sym] = {}
        results[sym]['training_loss'] = ga.history[sym]['loss'][-1]
        results[sym]['training_mserror'] = ga.history[sym]['mean_absolute_error'][-1]            
        results[sym]['eval_loss'] = evaluation[sym][0]
        results[sym]['eval_mserror'] = evaluation[sym][1]            
        results[sym]['last_prediction'] = descaled_predictions[sym][-1]            
    results_df = pd.DataFrame.from_dict(results, orient='index')
    results_df = results_df.sort_values(by=['last_prediction'])
    return results_df

    