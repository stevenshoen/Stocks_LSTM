#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
#import datetime

def close_label(ds, sym, dt):
    if len(list(ds.data[sym][dt])) >= 5:
        o = np.average([ds.data[sym][dt][t]['open'] for t in list(ds.data[sym][dt])[:5]])
        c = ds.data[sym][dt][list(ds.data[sym][dt])[-1]]['close']
        if np.isnan(c):
            c = o
    else:
        return np.nan
    return (c / o) - 1.0
    
def day_atr(o, h, l, c):
    return max([(h-l), abs(h-c), abs(l-c), abs(o-c)]) / o
    
def am_atr(ds, sym, dt):
    if dt in ds.data[sym]:
        if len(list(ds.data[sym][dt])) >= 61:
            h = max([ds.data[sym][dt][t]['high'] for t in list(ds.data[sym][dt])[:60]])
            l = min([ds.data[sym][dt][t]['low']for t in list(ds.data[sym][dt])[:60]])
            o = ds.data[sym][dt][list(ds.data[sym][dt])[0]]['open']
            c = ds.data[sym][dt][list(ds.data[sym][dt])[60]]['close']
            return day_atr(o, h, l, c)
    return np.nan

def atr(ds, sym, dt):
    yest_i = sorted(list(ds.data[sym])).index(dt) - 1
    if yest_i >= 0:
        yest_dt = list(ds.data[sym])[yest_i]
        if dt in ds.data[sym] and yest_dt in ds.data[sym]:
            open_ = ds.data[sym][dt][list(ds.data[sym][dt])[0]]['open']
            high_ = max([ds.data[sym][yest_dt][t]['high'] for t in list(ds.data[sym][yest_dt])])
            low_ = min([ds.data[sym][yest_dt][t]['low'] for t in list(ds.data[sym][yest_dt])])
            close_ = ds.data[sym][yest_dt][list(ds.data[sym][yest_dt])[-1]]['close']
            return day_atr(open_, high_, low_, close_)
    return np.nan
    
def am_price(ds, sym, dt):
    if dt in ds.data[sym]:
        if len(list(ds.data[sym][dt])) >= 61:
            o = ds.data[sym][dt][list(ds.data[sym][dt])[0]]['open']
            c = ds.data[sym][dt][list(ds.data[sym][dt])[60]]['close']
        return c / o - 1.0
    return np.nan

def am_index(ds, sym, dt):
    if dt in ds.data[ds.index]:
        if len(list(ds.data[ds.index][dt])) >= 61:
            o = ds.data[ds.index][dt][list(ds.data[ds.index][dt])[0]]['open']
            c = ds.data[ds.index][dt][list(ds.data[ds.index][dt])[60]]['close']
        return c / o - 1.0
    return np.nan

def am_volume(ds, sym, dt):
    if dt in ds.data[ds.index]:
        if len(list(ds.data[ds.index][dt])) >= 61:
            vol = np.sum([ds.data[sym][dt][t]['volume'] for t in list(ds.data[sym][dt])[:60]])
            av_vol = np.average([np.sum([ds.data[sym][dt][t]['volume'] for t in list(ds.data[sym][dt]) if 'volume' in list(ds.data[sym][dt][t])]) for dt in list(ds.data[sym])])
            return vol/av_vol
    return np.nan

def volume_change(ds, sym, dt):
    yest_i = sorted(list(ds.data[sym])).index(dt) - 1
    if yest_i >= 0:
        yest_dt = list(ds.data[sym])[yest_i]
        if dt in ds.data[sym] and yest_dt in ds.data[sym]:
            av_vol = np.average([np.sum([ds.data[sym][dt][t]['volume'] for t in list(ds.data[sym][dt]) if 'volume' in list(ds.data[sym][dt][t])]) for dt in list(ds.data[sym])])
        yest_vol = np.sum([ds.data[sym][yest_dt][t]['volume'] for t in list(ds.data[sym][yest_dt])])
        return yest_vol / av_vol - 1.0
    return np.nan
    
def percent_change(ds, sym, dt):
    yest_i = sorted(list(ds.data[sym])).index(dt) - 1
    if yest_i >= 0:
        yest_dt = list(ds.data[sym])[yest_i]
        if dt in ds.data[sym] and yest_dt in ds.data[sym]:
            o = ds.data[sym][dt][list(ds.data[sym][dt])[0]]['open']
            yest_o = ds.data[sym][yest_dt][list(ds.data[sym][yest_dt])[0]]['open']
            return o / yest_o - 1.0
    return np.nan

def index_change(ds, sym, dt):
    yest_i = sorted(list(ds.data[ds.index])).index(dt) - 1
    if yest_i >= 0:
        yest_dt = list(ds.data[ds.index])[yest_i]
        if dt in ds.data[ds.index] and yest_dt in ds.data[ds.index]:
            o = ds.data[ds.index][dt][list(ds.data[ds.index][dt])[0]]['open']
            yest_o = ds.data[ds.index][yest_dt][list(ds.data[ds.index][yest_dt])[0]]['open']
            return o / yest_o - 1.0
    return np.nan
