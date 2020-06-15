#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import datetime
import pickle as pk
from iexfinance.stocks import get_historical_intraday
from iexcredentials import credentials, sandbox
import numpy as np
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar

credentials()

"""
build / manage pik files for dataset

        intra_data is a dict
            keys are syms
            vals are dicts
                
                keys are dts
                vals are dicts
                
                    keys are pd datetime
                    vals are dicts
                    
                        keys are funcd
                        vals are floats
"""
def populate_data_pickle(data, filename):
    with open(filename, 'wb+') as f:
        pk.dump(data, f)   

def data_from_pickle(filename):
    try:
        with open(filename, 'rb+') as f:
            symbols = pk.load(f)
    except Exception as e:
        symbols = {}
        print(e)
    return symbols

class DataManager:
    def __init__(self, name, symbols, trail_days=60):
        self.name = name
        self.path = 'data/'
        self.filename = name + '.pik'
        self.symbols = symbols
        self.data = None
        self.file_list = {}
        self.trail_days = trail_days
        self.fill_max = 10
        self.bad_dts = {}
        
    def dt_range(self):
        today = datetime.datetime.now().date()
        today = datetime.datetime(today.year, today.month, today.day)
        
        d = datetime.timedelta(days=1)
        cal = calendar()
        holidays = [x.to_pydatetime() for x in cal.holidays((today - d * self.trail_days), today)]
        dt_range_ = sorted([(today - d * i) for i in range(self.trail_days)])
        dt_range_ = [dt for dt in dt_range_ if datetime.date.weekday(dt) < 5 and dt not in holidays]
        return dt_range_
    
    def verify(self, data):
        columns = ['open', 'high', 'low', 'close', 'volume']
        bad_dts = {sym:[] for sym in self.symbols}
        for dt in self.dt_range():
            for sym in self.symbols:
        
                if dt in list(data[sym]):
                    res = data[sym][dt]
                    if len(res) >=61:
                        t = list(res)[0]
                        if all([col in data[sym][dt][t] for col in columns]):
                            if all([not np.isnan(data[sym][dt][t][col]) for col in columns]):
                                continue
                            else:
                                print('nan fail-', sym, dt, t)
                        else:
                            print('num values fail-', sym, dt, t)
                    else:
                        print('dt length fail-', sym, dt)
                else:
                    print('dt fail-', sym, dt)
                bad_dts[sym].append(dt)
        self.bad_dts = bad_dts
        return bad_dts
                    
    def build_data(self):
        keep_cols = ['open', 'high', 'low', 'close', 'volume']
        data = {}
        for sym in self.symbols:
            data[sym] = {}
            for dt in self.dt_range():
                retry = True
                while retry:
                    try:
                        ret = get_historical_intraday(sym, dt, output_format='pandas')
                        retry = False
                    except Exception as e:
                        print('data download error for ', sym, dt)
                        print(e)
                        retry = bool(input('proceed?'))
                if len(ret) >= 65:
                    for c in ret.columns:
                        if c not in keep_cols:
                            ret = ret.drop(c, axis=1)
                    ret =  ret.fillna(method='bfill')
                    print('received data for ', sym, ' - ', dt)
                    data[sym][dt] = ret.to_dict(orient='index')    
                else:
                    print('missing values for ', sym, ' on ', dt)
                    print('got back:', ret)
        return data
    
    def update_data(self, data):
        keep_cols = ['open', 'high', 'low', 'close', 'volume']
        today = datetime.datetime.now().date()
        today = datetime.datetime(today.year, today.month, today.day)
        d = datetime.timedelta(days=1)
        for sym in list(data):
            cur_dt = list(data[sym])[-1] + d
            while cur_dt <= today:
                if datetime.date.weekday(cur_dt) < 5:
                    ret = get_historical_intraday(sym, cur_dt, output_format='pandas')
                    if len(ret) >= 65:
                        for c in ret.columns:
                            if c not in keep_cols:
                                ret = ret.drop(c, axis=1)
                        ret =  ret.fillna(method='bfill')
                        ret = ret.to_dict(orient='index')
                        data[sym][cur_dt] = ret
                        print(sym, ' updated for ', cur_dt)
                else:
                    print('missing values for ', sym, ' on ', cur_dt)
                    print('got back:', ret)
                cur_dt += d
        return data
            
    def save_data(self, data):
        populate_data_pickle(data, self.path + self.filename)
        
    def load_data(self):
        data = data_from_pickle(self.path + self.filename)
        return data

        
