from copy import * 
from symtable import Symbol
from numpy import source
import pandas as pd
import datetime
import os
from bunch import *
from my_persian_date import *
from my_utils import *
import persian
from matplotlib.backends.backend_pdf import PdfPages
from bidi.algorithm import get_display
import arabic_reshaper 

import matplotlib.pyplot as plt
import matplotlib
from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.ticker as ticker

import matplotlib.dates as mpdates
import mplfinance as mpf
import matplotlib.dates as mpl_dates
matplotlib.use('agg')
from my_constants import *

import pandas_ta as ta

_SYMBOL = 'Symbol'
_OPEN = 'Open'
_CLOSE = 'Close'
_FINAL = 'FINALPRICE'
_HIGH = 'High'
_LOW = 'Low'
_VOL = 'Volume'
_VWAP = 'vwap'
_VWAP_RATIO = 'vwap_ratio'
_YESTERDAYPRICE = 'YESTERDAYPRICE'
_VALUE = 'TRADEVALUE'
_TRADECOUNT = 'TRADECOUNT'
_DATE = 'Date'
_PERSIAN_DATE = 'persian_date'
_PERSIAN_PREDICTION_DATE = 'prediction_date'
_PERCENT_CHANGE_SUFFIX = '_percent'

#'<SYMBOL>': _SYMBOL, '<DATE>': _DATE,  '<PERSIANDATE>': _PERSIAN_DATE
def change_tse_fieldnames(df):
    dic = {'<TICKER>': _SYMBOL, '<DATE>': _DATE,  '<OPEN>': _OPEN, '<HIGH>': _HIGH,
           '<LOW>': _LOW, '<CLOSE>': _CLOSE,  '<FINALPRICE>': _FINAL,  '<VOL>': _VOL, '<TRADECOUNT>': _TRADECOUNT,
           '< VALUE>': _VALUE, '<YESTERDAYPRICE>': _YESTERDAYPRICE}
    df.rename(dic, inplace=True, axis=1)

    return df

def change_metatrader_fieldnames(df):
    dic = {'time': _DATE, 'open': _OPEN, 'high': _HIGH, 'low': _LOW, 'close': _CLOSE,
           'real_volume': _VOL}
    df.rename(dic, inplace=True, axis=1)

    return df

def get_all_symbols():
    df = pd.read_excel('Symbol_Info_active_company_symbs.xlsx', engine='openpyxl')
    return df

def get_symbols_price_list(update_data=True, from_date=None, symb_save_path='D:/MTraderPython/StockData/'):

    if from_date is None:
        from_date = datetime(2019, 1, 1, 1)

    fn = 'Symbol_Info_active_company_symbs.xlsx'
    symbol_list_name, symbol_list_data = es.download_stock_data(symb_save_path, [fn], from_date, update_data)
    df_list = []
    symbol_df_list = []
    for i in range(len(symbol_list_name)):
        sym_list = symbol_list_name[i]
        sym_data_list = symbol_list_data[i]
        for j in range(len(sym_list)):
            symbol = sym_list[j]
            symbol_data_file = sym_data_list[j]
            symbol_df = pd.read_csv(symbol_data_file)

            df_list.append(df)
            symbol_df_list.append(symbol)

    return df_list, symbol_df_list

def screen_symbols(screener_func, update_data, from_date, symb_save_path):
    stock_data_list, symbol_df_list = get_symbols_price_list(update_data, from_date, symb_save_path)
    screen_out_df_list = []
    screen_out_symbol_list = []
    for i in range(len(symbol_df_list)):
        symbol = symbol_df_list[i]
        st_df = stock_data_list[i]
        if screener_func(symbol, st_df):
            screen_out_df_list.append(st_df)
            screen_out_symbol_list.append(symbol)
    return screen_out_df_list, screen_out_symbol_list

def screener_func_all(symbol, st_df):
    return True

def screener_func_my_watchlist(symbol, st_df):
    mywatchlist = pd.read_csv('my_watchlist.csv')

    return True

def filter_func_my_watchlist(symbol):
    return True
    mywatchlist = pd.read_excel('my_watchlist.xlsx', engine='openpyxl')
    filt = sum(1*(mywatchlist['نماد']==symbol))==1
    return filt

def my_fmt(x, y):
    x = x / 10
    return '{:5.0f}'.format(x)

def is_unimodal(df, col):
    new_col = 'difference'
    df[new_col] = df[col].diff()
    df['temp'] = (df[new_col].shift(1) <= 3) & (df[new_col] >= -3)
    modal = sum(df['temp'])
    if modal == 1:
        return True
    return False

def make_monotonic(df, cols=None):
    if cols is None:
        cols = df.columns

    df1 = df.copy()[cols]

    while True:
        mon_inc = (df1.diff().fillna(0) >= 0).all(axis=1)
        if mon_inc.all():
            break
        df1 = df1[mon_inc]
    return df1


def change_stock_to_percentage(symbol_df):
    df = symbol_df
    df[_HIGH] = df[_HIGH].div(df[_OPEN])
    df[_HIGH] = df[_HIGH].subtract(1) * 100

    df[_LOW] = df[_LOW].div(df[_OPEN])
    df[_LOW] = df[_LOW].subtract(1) * 100

    df[_CLOSE] = df[_CLOSE].div(df[_OPEN])
    df[_CLOSE] = df[_CLOSE].subtract(1) * 100

    unnamed_cols = [c for c in df.columns if c.find('Unnamed') != -1]
    try:
        df.drop(unnamed_cols, axis=1, inplace=True)
    except:
        pass
    return df

n_days_before = 360
n_weeks_before = 52
min_max_price_range_mode = 2     # n_weeks_before weeks before

mva_n_days = [5,  20, 50, 100, 150]

def draw_candle_stick_mpf(stockdata, symbol, sample_level='lower'):

    data = stockdata.get_df(sample_level)
    stochrsi_data = stockdata.get_indicator_data(sample_level)
    reshapedtext = arabic_reshaper.reshape(symbol)
    titlestr = get_display(reshapedtext)
    # titlestr = symbol[::-1] # reverse string
    if sample_level == 'higher':
        df = data.loc[:, [_DATE, _OPEN, _HIGH, _LOW, _CLOSE, _VOL]]  # use tse closing price
    else:
        df = data.loc[:, [_DATE, _OPEN, _HIGH, _LOW, _CLOSE, _VOL, _VWAP]]  # use tse closing price


    df[_DATE] = pd.to_datetime(df[_DATE])
    df.set_index(_DATE, inplace=True)
    stochrsi_plot = mpf.make_addplot(stochrsi_data, panel=1, ylabel='STCHRSI_'+sample_level)
    draw_pbv = True
    if draw_pbv and sample_level == 'lower':
        apdict = mpf.make_addplot(df[_VWAP])
        kwargs = dict(type='ohlc', style='mike', show_nontrading=False, returnfig=True, closefig=True,
                      tight_layout=True,
                      volume=True, title=titlestr, figratio=(38, 19), figscale=0.85,
                      ylabel='Price', datetime_format='%d-%m', xrotation=20, mav=(3, 12, 20),
                      addplot=[stochrsi_plot, apdict])
        fig, axlist = mpf.plot(df, **kwargs)
        bucket_size = 0.0050 * max(df[_CLOSE])
        volprofile = df[_VOL].groupby(df[_CLOSE].apply(lambda x: bucket_size * round(x / bucket_size, 0))).sum()

        mc = mpf.make_marketcolors(base_mpf_style='yahoo')
        s = mpf.make_mpf_style(base_mpf_style='nightclouds', marketcolors=mc)

        vpax = fig.add_axes(axlist[0].get_position())
        vpax.set_axis_off()
        vpax.set_xlim(right=1.2 * max(volprofile.values))
        vpax.barh(volprofile.keys().values, volprofile.values, height=0.75 * bucket_size, align='center', color='purple',
                  alpha=0.45)
    else:
        kwargs = dict(type='ohlc', style='mike', show_nontrading=False, returnfig=True, closefig=True,
                      tight_layout=True,
                      volume=True, title=titlestr, figratio=(38, 19), figscale=0.85,
                      ylabel='Price', datetime_format='%d-%m', xrotation=20, mav=(3, 12, 20),
                      addplot=[stochrsi_plot])
        fig, axlist = mpf.plot(df, **kwargs)

    ax1 = axlist[0]
    #ax1.set_yscale('log')
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(my_fmt))
    ax1.yaxis.set_minor_formatter(ticker.FuncFormatter(my_fmt))

    return fig, None

def draw_candle_stick(data, symbol):
    plt.style.use('ggplot')
    ohlc = data.loc[:, [_DATE, _OPEN, _HIGH, _LOW, _CLOSE, _VOL]]
    ohlc[_DATE] = pd.to_datetime(ohlc[_DATE])
    # apply map function
    ohlc[_DATE] = ohlc[_DATE].map(mpdates.date2num)
    #
    # ohlc.set_index('Date', inplace=True)
    #ohlc['Date'] = ohlc['Date'].map(mpdates.date2num)
    #ohlc['Date'] = ohlc['Date'].apply(mpl_dates.date2num)

    kwargs = dict(type='candle', mav=(2, 4, 6), volume=True, figratio=(11, 8), figscale=0.85)
    #mpf.plot(ohlc, **kwargs, style='classic')
    #
    # creating Subplots
    fig, ax = plt.subplots()

    # plotting the data
    candlestick_ohlc(ax, ohlc.values, width=0.6,
                     colorup='green', colordown='red',
                     alpha=0.8, no_xgaps = True, style='mike')

    # allow grid
    ax.grid(True)

    # # Setting labels
    # ax.set_xlabel('Date')
    # ax.set_ylabel('Price')

    # setting title
    plt.title(symbol)

    # Formatting Date
    date_format = mpdates.DateFormatter('%d-%m-%Y')
    ax.xaxis.set_major_formatter(date_format)
    fig.autofmt_xdate()

    fig.tight_layout()

    return fig, plt

class TargetPredictor:
    def __init__(self):
        pass
    def regressor(self, df, date_field, x_fields, y_field):
        pass
    def predict(self, df, date_field, x_fields):
        pass

## field types:
_DATE_TYPE = 1
_SYMBOL_TYPE = 2
_OHLC_TYPE = 3
_VOLUME_TYPE = 4
_TRADE_VALUE_TYPE = 5

_RANGE_UNKNOWN = 0
_RANGE_0_1     = 1
_RANGE_M1_P1   = 2
_RANGE_0_100   = 3
_RANGE_M100_P100 = 4
class Field:
    def __init__(self, name, computed=False, field_type=_OHLC_TYPE, numeric=True, range=_RANGE_UNKNOWN, 
                       symbol=None, next_day=0):
        self.fieldname = name
        self.computed = computed
        self.original_name = name
        self.field_type = field_type
        self.symbol = symbol
        self.numeric = numeric
        self.range = range
        self.next_day = next_day

class FieldList:
    def __init__(self):
        self.field_list = []

    def add_standard_column_fields(self, columns):
        for c in columns:
            self.add_field_standard_column_name(c)

    def add_field_standard_column_name(self, c):
        if c==_SYMBOL:
            f = Field(name=_SYMBOL, computed=False, field_type=False,numeric=False,
                    symbol=self.symbol, next_day=0)
        elif c==_DATE:
            f = Field(name=_DATE, computed=False, field_type=False,numeric=False,
                    symbol=self.symbol,next_day=0)
        elif c==_OPEN or c==_CLOSE or c==_LOW or c==_HIGH:
            f = Field(name=c, computed=False,field_type=True, numeric=True,
                    symbol=self.symbol,next_day=0)
        elif c==_FINAL:
            f = Field(name=c, computed=False, field_type=False, numeric=True,
                    symbol=self.symbol, next_day=0)
        elif c==_VOL:
            f = Field(name=c, computed=False, field_type=False, numeric=True,
                    symbol=self.symbol, next_day=0)
        elif c==_TRADECOUNT:
            f = Field(name=c, computed=False, field_type=False, numeric=True,
                    symbol=self.symbol, next_day=0)
        elif c==_VALUE:
            f = Field(name=c, computed=False, field_type=False, numeric=True,
                    symbol=self.symbol, next_day=0)
        elif c==_YESTERDAYPRICE:
            f = Field(name=c, computed=False, field_type=False, numeric=True,
                    symbol=self.symbol, next_day=0)

        self.add_field(f)

    def add_field(self, field_to_add: Field):
        self.field_list.append(field_to_add)

    def set_field_name(self, original_name, new_name):
        for f in self.field_list:
            if f.original == original_name:
                f.fieldname = new_name
                return 
    
class Indicators:
    def __init__(self, to_add_indicators_list=None):
        self.to_add_indicators_list = to_add_indicators_list
        self.param_bunch_def = Bunch()
        self.param_bunch_def['STOCHRSI'] = Bunch(length=14, rsi_length=14, k=3, d=3, diff_period=5, eps_threshold=0.01)
        self.param_bunch_def['MACD'] = Bunch(fast=12, slow=26, signal=9, diff_period=2, eps_threshold=0.01)
        self.param_bunch_def['OBV'] = Bunch(diff_period=2, eps_threshold=1.0)
        self.param_bunch_def['EMA'] = Bunch(length=9, diff_period=2, eps_threshold=1.0)
        self.param_bunch_def['BBAND'] = Bunch(length=20, diff_period=2, eps_threshold=0.01)
        self.param_bunch_def['EOM'] = Bunch(length=14, divisor=10000, drift=3, diff_period=2, eps_threshold=0.01)
        self.param_bunch_def['CMF'] = Bunch(length=14, diff_period=2, eps_threshold=0.01)
        self.param_bunch_def['ADX'] = Bunch(length=14, smoothinglen=14, diff_period=2, eps_threshold=0.01)
        self.param_bunch_def['VWAP'] = Bunch(from_date=datetime.now()-timedelta(days=530), div_by_price=True, div_price='Close', diff_period=2, eps_threshold=0.001)

    def add_compute_indicator(self, df, indic_bunch):
        if indic_bunch['name'] == 'STOCHRSI':
            if indic_bunch['default_param']:
                indic_param = self.param_bunch_def['STOCHRSI']
            else:
                indic_param = indic_bunch.param
            df, cols = self.add_stochrsi(df, indic_param)
        elif indic_bunch['name'] == 'MACD':
            if indic_bunch['default_param']:
                indic_param = self.param_bunch_def['MACD']
            else:
                indic_param = indic_bunch.param
            df, cols = self.add_macd(df, indic_param)
        elif indic_bunch['name'] == 'OBV':
            if indic_bunch['default_param']:
                indic_param = self.param_bunch_def['OBV']
            else:
                indic_param = indic_bunch.param
            df, cols = self.add_obv(df, indic_param)
        elif indic_bunch['name'] == 'EMA':
            if indic_bunch['default_param']:
                indic_param = self.param_bunch_def['EMA']
            else:
                indic_param = indic_bunch.param
            df, cols = self.add_ema(df, indic_param)
        elif indic_bunch['name'] == 'BBAND':
            if indic_bunch['default_param']:
                indic_param = self.param_bunch_def['BBAND']
            else:
                indic_param = indic_bunch.param
            df, cols = self.add_bband(df, indic_param)
        elif indic_bunch['name'] == 'EOM':
            if indic_bunch['default_param']:
                indic_param = self.param_bunch_def['EOM']
            else:
                indic_param = indic_bunch.param
            df, cols = self.add_eom(df, indic_param)
        elif indic_bunch['name'] == 'CMF':
            if indic_bunch['default_param']:
                indic_param = self.param_bunch_def['CMF']
            else:
                indic_param = indic_bunch.param
            df, cols = self.add_cmf(df, indic_param)
        elif indic_bunch['name'] == 'ADX':
            if indic_bunch['default_param']:
                indic_param = self.param_bunch_def['ADX']
            else:
                indic_param = indic_bunch.param
            df, cols = self.add_adx(df, indic_param)
        elif indic_bunch['name'] == 'VWAP':
            if indic_bunch['default_param']:
                indic_param = self.param_bunch_def['VWAP']
            else:
                indic_param = indic_bunch.param
            df, cols = self.add_vwap(df, indic_param)

        if indic_bunch['comp_diff']:
            period = indic_param['diff_period']
            eps_threshold = indic_param['eps_threshold']
            df, dif_cols = self.add_diff_indics(df, cols, period, eps_threshold)

        if indic_bunch['diff_only']:
            df.drop(cols, inplace=True)
            cols = dif_cols
        else:
            if indic_bunch['comp_diff']:
                cols = cols + dif_cols
        return df, cols

    def add_diff_indics(self, df, col, period, eps_threshold):
        dif_cols = [c+'_+0-' for c in col]
        diff_d = df[col].diff(period)
        df[dif_cols] = 1*(diff_d > eps_threshold) + -1*(diff_d < -eps_threshold)
        return df, dif_cols

    def add_stochrsi(self, df, indic_param=None):
        if indic_param is None:
            indic_param = self.param_bunch_def['STOCHRSI']
        length = indic_param['length']
        rsi_length = indic_param['rsi_length']
        k = indic_param['k']
        d = indic_param['d']

        dt = ta.stochrsi(df[_CLOSE], length=length, rsi_length=rsi_length, k=k, d=d)

        df['momentum_stoch_rsi_d'] = dt['STOCHRSId_14_14_3_3']
        df['momentum_stoch_rsi_k'] = dt['STOCHRSIk_14_14_3_3']
        columns = ['momentum_stoch_rsi_d', 'momentum_stoch_rsi_k']
        return df, columns

    def add_macd(self, df, indic_param):
        fast = indic_param['fast']
        slow = indic_param['slow']
        signal = indic_param['signal']
        dt = ta.macd(df[_CLOSE], fast=fast, slow=slow, signal=signal)
        df['macd'] = dt['MACD_12_26_9']
        df['macd_h'] = dt['MACDh_12_26_9']
        df['macd_s'] = dt['MACDs_12_26_9']
        cols = ['macd', 'macd_h', 'macd_s']

        return df, cols

    def add_obv(self, df, indic_param):
        dt = ta.obv(df[_CLOSE], df[_VOL])
        df['obv'] = dt

        cols = ['obv']
        return df, cols

    def add_ema(self, df, indic_param):
        length = indic_param['length']
        dt = ta.ema(df[_CLOSE], length=length)
        df['ema'] = dt
        cols = ['ema']
        return df, cols

    def add_bband(self, df, indic_param):
        length = indic_param['length']
        #std = indic_param['std']
        dt = ta.bbands(df[_CLOSE], length)
        df['BBL'] = dt['BBL_20_2.0']
        df['BBM'] = dt['BBM_20_2.0']
        df['BBU'] = dt['BBU_20_2.0']
        df['BBB'] = dt['BBB_20_2.0']
        df['BBP'] = dt['BBP_20_2.0']
        cols = ['BBL', 'BBM', 'BBU', 'BBB', 'BBP']
        return df, cols

    def add_eom(self, df, indic_param):
        length = indic_param['length']
        divisor = indic_param['divisor']
        drift = indic_param['drift']
        dt = ta.eom(df[_HIGH], df[_LOW], df[_CLOSE], df[_VOL], length, divisor)
        df['eom'] = dt
        cols = ['eom']
        return df, cols

    def add_cmf(self, df, indic_param):
        length = indic_param['length']
        dt = ta.cmf(df[_HIGH], df[_LOW], df[_CLOSE], df[_VOL], df[_OPEN], length)
        df['cmf'] = dt
        cols = ['cmf']
        return df, cols

    def add_adx(self, df, indic_param):
        length = indic_param['length']
        smoothing = indic_param['smoothinglen']

        dt = ta.adx(df[_HIGH], df[_LOW], df[_CLOSE], length, smoothing)
        df['adx'] = dt['ADX_14']
        df['dmn'] = dt['DMN_14']
        df['dmp'] = dt['DMN_14']
        cols = ['adx', 'dmn', 'dmp']
        return df, cols

    def comp_vwap(self, df, from_date, div_b_price, div_price='Close'):
        df_t = df[df[_DATE] >= from_date]
        p = (df_t[_CLOSE] + df[_HIGH] + df[_LOW]).div(3).values
        v = df_t[_VOL].values
        df_t = df_t.assign(vwap=(p * v).cumsum() / v.cumsum())
        if div_b_price:
            assert div_price != _CLOSE or div_price != _HIGH or div_price != _LOW or div_price != _OPEN, 'error undefined value'
            df_t[_VWAP_RATIO] = df_t[_VWAP].div(df_t[div_price])
            return df_t, [_VWAP, _VWAP_RATIO]
        else:
            return df_t, [_VWAP]

    def add_vwap(self, df, indic_param):

        from_date = indic_param['from_date']
        div_b_price = indic_param['div_by_price']
        div_price = indic_param['div_price']
        df_t, cols = self.comp_vwap(df, from_date, div_b_price, div_price)
        df_t = df_t[['Date']+cols]
        df = pd.merge(df, df_t, how='right', on=_DATE)

        return df, cols

    def add_volatil(self, df):
        volat = ta.add_volatility_ta(df, high=_HIGH, close=_CLOSE, fillna=True)
        return volat

    def add_tredn(self, df):
        trend = ta.add_trend_ta(df, high=_HIGH, low=_LOW, close=_CLOSE, fillna=True)
        return trend

    def add_vol(self, df):
        vol = ta.add_volume_ta(df, high=_HIGH, low=_LOW, close=_CLOSE, volume=_VOL, fillna=True)

    def add_others(self, df):
        others = ta.add_others_ta(df, close=_CLOSE)
        return others

class StockData:
    def __init__(self, data_source_folder) -> None:
       sym_industry_file = data_source_folder + '/complete_sym_indust.xlsx'
       symbol_filename_list = data_source_folder+'symbol_data_file.xlsx'
       df_ind_file = None

    def _get_symbol_filename(self, symbol):
        symbol_list = [symbol]
        selected_df = StockData.df_ind_file.loc[StockData.df_ind_file[_SYMBOL].isin(symbol_list)]
        filename, sym, industry = self._get_filename_symbol_industry(selected_df, 0)
        return filename

    def _get_filename_symbol_industry(self, df, i):
        filename = df.iloc[i, df.columns.get_loc('File')]
        symbol = df.iloc[i, df.columns.get_loc('Symbol')]
        industry = df.iloc[i, df.columns.get_loc('Industry')]
        return filename, symbol, industry

    def __init__(self, symbol, from_date=None, datasource='TSE'):
        StockData.df_ind_file = pd.read_excel(StockData.symbol_filename_list, engine="openpyxl")
        self.filename = self._get_symbol_filename(symbol)
        self.symbol = symbol
        try:
            self.df = pd.read_csv(self.filename, encoding='utf-16')
        except:
            try:
                self.df = pd.read_csv(self.filename, encoding='utf8')
            except:
                self.df = None
                AssertionError(self)
                
        self.is_empty = True
        self.field_list = []
        if self.df.empty:
            self.highertimeframe_df = None
            self.highertimeframe_time = '7' # it means a week
            self.seq_stats_df = None
            self.mva_cols = []
            self.datasource = datasource
            self.indic = Indicators()
            self.indic_bunch_list = []
            self.ind_cols = None
            self.add_indicator_diff = True
            return
        self.is_empty = False
        if datasource == 'TSE':
            self.change_tse_fieldnames()
            # self.df['نماد'] = self.symbol
            if self.symbol is None:
                self.symbol = self.df[_SYMBOL].iloc[0]
                self.symbol = self.symbol.replace('-ت', '').strip()
                self.symbol = persian.convert_ar_characters(self.symbol)

            self.df[_DATE] = self.df[_DATE].astype(str)
            self.df[_PERSIAN_DATE] = self.df.apply(lambda x: convert_to_jdate_list(x.Date, dateformat='%Y%m%d'), axis=1)
            self.df[_DATE] = pd.to_datetime(self.df['Date'], format='%Y%m%d')
        elif datasource == 'METATRADER':
            self.change_metatrade_fieldnames()
            self.symbol = None
            try:
                self.df[_PERSIAN_DATE] = self.df.apply(lambda x: convert_to_jdate_list(x.Date, ''), axis=1)
            except:
                pass
            self.df.drop(['tick_volume', 'spread'], inplace=True, axis=1)
            self.df[_DATE] = pd.to_datetime(self.df['Date'], format='%Y-%m-%d')

        self.highertimeframe_df = None
        self.highertimeframe_time = '7' # it means a week
        self.seq_stats_df = None
        self.mva_cols = []
        self.datasource = datasource
        self.indic = Indicators()
        self.indic_bunch_list = []
        self.ind_cols = None
        self.add_indicator_diff = True

        # if last_year:
        #     self.remove_old_data()
        if from_date is None:
            tod = datetime.now()
            d = timedelta(days=10000)
            from_date = tod - d
        self.df = self.df[self.df[_DATE] >= from_date]
        self.vwap_from_date = from_date

        self.date_field = _DATE
        self.symbol_field = _SYMBOL
        self.persian_date_field = _PERSIAN_DATE
        self.close_price_field = _CLOSE
        self.open_price_field = _OPEN
        self.high_price_field = _HIGH
        self.low_price_field = _LOW
        self.final_price_field = _FINAL
        self.vol_field = _VOL
        # self.vwap_field = _VWAP
        # self.vwap_ratio_field = _VWAP_RATIO
                        #            self.vwap_field,
                        #    self.vwap_ratio_field,
                        #    self.trade_value_field,
        self.yesterday_price_field = _YESTERDAYPRICE
        self.trade_value_field = _VALUE 
        self.trade_count_field = _TRADECOUNT 
        self.field_list = [self.date_field, 
                           self.symbol_field,
                           self.persian_date_field,
                           self.close_price_field,
                           self.open_price_field,
                           self.high_price_field,
                           self.low_price_field,
                           self.final_price_field,
                           self.vol_field,
                           self.yesterday_price_field,
                           self.trade_count_field]
        self.date_fields_list = [self.date_field, self.persian_date_field]

        # if add_percentage:
        #     self.df = self.df.loc[:,~self.df.columns.duplicated()].copy()
        #     self.add_percentage_prices_from_open()

        self.keep_only_field_list_cols()

    def add_suffix_to_fields_list(self, suffix, fields_list):
        if fields_list is None:
            fields_list = self.field_list
        try:
            fields_list.remove(self.date_fields_list)
        except:
            pass
        dic = {}
        for f in fields_list:
            dic[f] = f + suffix

        self.df.rename(columns=dic, inplace=True)
        #self.field_list = fields_list.append(self.date_fields_list)

    def keep_only_field_list_cols(self):
        list = [c for c in self.field_list if  c in self.df.columns]
        self.df = self.df[list]

    def remove_unwanted_cols(self):
        unnamed_cols = [c for c in self.df.columns if c.find('Unnamed') != -1]
        col_del = unnamed_cols

        for col in col_del:
            try:
                self.df.drop(col, axis=1, inplace=True)
            except:
                pass

    def get_date_field(self, on_persian_date=True):
        if on_persian_date:
            return self.persian_date_field
        return self.date_field

    def add_now_next_day_field(self, from_date=None, next_day=1, on_persian_date=True):
        if on_persian_date:
            persiandate = PersianDate()
            if from_date is None:
                start_persian_date = START_PERSIAN_DATE
            date_df = persiandate.get_working_and_future_date()
            self.df = pd.merge(self.df, date_df, on=self.persian_date_field, how='inner')
            added_field = _PERSIAN_PREDICTION_DATE
            return added_field

    def get_symbol(self):
        return self.symbol

    def convert_symbol_names_to_persian(self):
        for i in range(len(self.df.index)):
            self.df['Symbol'].iloc[i] = persian.convert_ar_characters(self.df['Symbol'].iloc[i])

    def is_empty(self):
        return len(self.df.index) == 0

    def get_df_from_date(self, from_dt):
        return self.df.loc[self.df[_DATE] >= from_dt, :]

    def get_df_from_previous_days(self, n_days):
        from_dt = PersianDate().get_previous_date(n_days=n_days)
        return self.get_df_from_date(from_dt=from_dt)

    def remove_old_data(self, from_dt=None):
        if from_dt is None:
            from_dt = PersianDate().get_previous_date(n_weeks_before)

        self.df = self.get_df_from_date(from_dt)

    def create_field_list(self, columns):
        self.field_list = FieldList()
        self.field_list.add_standard_column_fields(columns)
                    
    def change_tse_fieldnames(self):
        self.df = change_tse_fieldnames(self.df)
        columns = self.df.columns
        self.create_field_list(columns)

    def change_metatrade_fieldnames(self):
        AssertionError 
        self.df = change_metatrader_fieldnames(self.df)

    def compute_indicators(self, ind_list=None, two_level=False):
        ind_bunch = Bunch()
        ind_bunch['name'] = 'STOCHRSI'
        ind_bunch['default_param'] = True
        ind_bunch['comp_diff'] = True
        ind_bunch['diff_only'] = False
        self.indic_bunch_list.append(ind_bunch)
        ind_bunch = Bunch()
        ind_bunch['name'] = 'MACD'
        ind_bunch['default_param'] = True
        ind_bunch['comp_diff'] = True
        ind_bunch['diff_only'] = False
        self.indic_bunch_list.append(ind_bunch)
        ind_bunch = Bunch()
        ind_bunch['name'] = 'OBV'
        ind_bunch['default_param'] = True
        ind_bunch['comp_diff'] = True
        ind_bunch['diff_only'] = False
        self.indic_bunch_list.append(ind_bunch)
        ind_bunch = Bunch()
        ind_bunch['name'] = 'EMA'
        ind_bunch['default_param'] = True
        ind_bunch['comp_diff'] = True
        ind_bunch['diff_only'] = False
        self.indic_bunch_list.append(ind_bunch)
        ind_bunch = Bunch()
        ind_bunch['name'] = 'BBAND'
        ind_bunch['default_param'] = True
        ind_bunch['comp_diff'] = True
        ind_bunch['diff_only'] = False
        self.indic_bunch_list.append(ind_bunch)
        ind_bunch = Bunch()
        ind_bunch['name'] = 'EOM'
        ind_bunch['default_param'] = True
        ind_bunch['comp_diff'] = True
        ind_bunch['diff_only'] = False
        self.indic_bunch_list.append(ind_bunch)
        ind_bunch = Bunch()
        ind_bunch['name'] = 'CMF'
        ind_bunch['default_param'] = True
        ind_bunch['comp_diff'] = True
        ind_bunch['diff_only'] = False
        self.indic_bunch_list.append(ind_bunch)
        ind_bunch = Bunch()
        ind_bunch['name'] = 'ADX'
        ind_bunch['default_param'] = True
        ind_bunch['comp_diff'] = True
        ind_bunch['diff_only'] = False
        self.indic_bunch_list.append(ind_bunch)
        ind_bunch = Bunch()
        ind_bunch['name'] = 'VWAP'
        ind_bunch['default_param'] = True
        ind_bunch['comp_diff'] = True
        ind_bunch['diff_only'] = False
        self.indic_bunch_list.append(ind_bunch)

        ind_cols = []
        for indic_bunch in self.indic_bunch_list:
            if (ind_list is None) or (indic_bunch['name'] in ind_list):
                self.df, cols = self.indic.add_compute_indicator(self.df, indic_bunch=indic_bunch)
                # the following has errors
                two_level = False
                if two_level:
                    self.highertimeframe_df, colp = self.indic.add_compute_indicator(self.highertimeframe_df, indic_bunch=indic_bunch)
                ind_cols = ind_cols + cols
        self.ind_cols = ind_cols
        # return self.df[self.ind_cols]
    def get_technical_data(self, ind_list=None, two_level=False):
        self.compute_indicators(ind_list, two_level)
        cols = [self.date_field, self.symbol_field, self.persian_date_field, self.close_price_field, 
                self.low_price_field, self.high_price_field] + self.ind_cols
        # self.df = self.df.reindex(columns=cols)
        return self.df[cols], cols

    def get_indicator_data(self, timeframe_level='lower', indic_name_list=None):
        if indic_name_list is None:
            indic_name_list = ['momentum_stoch_rsi_d', 'momentum_stoch_rsi_k']
        if timeframe_level == 'higher':
            return self.highertimeframe_df[indic_name_list]
        return self.df[indic_name_list]

    def comp_df_weekly_timeframe(self):
        ohlc_dict = {
            _OPEN: 'first',
            _HIGH: 'max',
            _LOW: 'min',
            _CLOSE: 'last',
            _VOL: 'sum',
            _SYMBOL: 'first'
        }
        self.highertimeframe_df = self.df.resample('W', closed='left', label='left', on='Date').agg(ohlc_dict)
        self.highertimeframe_df[_DATE] = self.highertimeframe_df.index
        self.highertimeframe_df.dropna(inplace=True)

    def get_df_weekly_timeframe(self):
        if self.highertimeframe_df is None:
            self.comp_df_weekly_timeframe()
        return self.highertimeframe_df

    def get_df(self, sample_level='lower'):
        if sample_level == 'higher':
            return self.get_df_weekly_timeframe()
        return self.df

    def add_percentage_prices_from_open(self, add_to_field_list=True):
        new_field = _HIGH+_PERCENT_CHANGE_SUFFIX
        self.df[new_field] = self.df[_HIGH].div(self.df[_OPEN])
        self.df[new_field] = self.df[new_field].subtract(1) * 100
    
        if add_to_field_list:
            self.field_list.append(new_field)
        new_field = _LOW+_PERCENT_CHANGE_SUFFIX
        self.df[new_field] = self.df[_LOW].div(self.df[_OPEN])
        self.df[new_field] = self.df[_LOW+_PERCENT_CHANGE_SUFFIX].subtract(1) * 100
        if add_to_field_list:
            self.field_list.append(new_field)
        new_field = _CLOSE+_PERCENT_CHANGE_SUFFIX
        self.df[new_field] = self.df[_CLOSE].div(self.df[_OPEN])
        self.df[new_field] = self.df[new_field].subtract(1) * 100
        if add_to_field_list:
            self.field_list.append(new_field)
    # dic = {'<TICKER>': _SYMBOL, '<DATE>': _DATE,  '<OPEN>': _OPEN, '<HIGH>': _HIGH,
    #        '<LOW>': _LOW, '<CLOSE>': _CLOSE,  '<FINALPRICE>': _FINAL,  '<VOL>': _VOL, '<TRADECOUNT>': _TRADECOUNT,
    #        '< VALUE>': _VALUE, '<YESTERDAYPRICE>': _YESTERDAYPRICE}
    def min_price_from(self, from_n_days_before=n_days_before, col_price=_CLOSE):
        from_dt = PersianDate().get_previous_date(n_days=from_n_days_before)
        min_price = self.df.loc[self.df[_DATE] >= from_dt, col_price].min(skipna=True)
        return min_price

    def min_price_week(self, from_n_weeks_before=n_weeks_before, col_price=_CLOSE):
        n_d = from_n_weeks_before*7
        return self.min_price_from(from_n_days_before=n_d, col_price=col_price)

    def min_price(self, col_price=_CLOSE):
        if min_max_price_range_mode == 1:
            return self.min_price_from(col_price=col_price)
        else:
            return self.min_price_week(col_price=col_price)

    def max_price_from(self, from_n_days_before=n_days_before, col_price=_CLOSE):
        from_dt = PersianDate().get_previous_date(n_days=from_n_days_before)
        max_price = self.df.loc[self.df[_DATE] >= from_dt, col_price].max(skipna=True)
        return max_price

    def max_price_week(self, from_n_weeks_before=n_weeks_before, col_price=_CLOSE):
        n_d = from_n_weeks_before*7
        return self.max_price_from(from_n_days_before=n_d, col_price=col_price)

    def max_price(self, col_price=_CLOSE):
        if min_max_price_range_mode == 1:
            return self.max_price_from(col_price=col_price)
        else:
            return self.max_price_week(col_price=col_price)

    def is_it_back_to_max(self, from_n_weeks_before=None, col_price=_CLOSE, percent_range=(75, 115), recent_days=5):
        if from_n_weeks_before is None:
            from_n_weeks_before = n_weeks_before

        mx_price = self.max_price_week(from_n_weeks_before=from_n_weeks_before, col_price=col_price)
        r_df = self.df.tail(recent_days)
        cond = (r_df[col_price] >= mx_price*percent_range[0]/100) & (r_df[col_price] <= mx_price*percent_range[1]/100)
        return any(cond)

    def is_it_back_to_min(self, from_n_weeks_before=None, col_price=_CLOSE, percent_range=(75, 115), recent_days=5):
        if from_n_weeks_before is None:
            from_n_weeks_before = n_weeks_before

        mx_price = self.min_price_week(from_n_weeks_before=from_n_weeks_before, col_price=col_price)
        r_df = self.df.tail(recent_days)
        cond = (r_df[col_price] >= mx_price*percent_range[0]/100) & (r_df[col_price] <= mx_price*percent_range[1]/100)
        return any(cond)

    def mva_col(self, col, n_day):
        return self.df[col].rolling(window=n_day).mean()

    def add_mva_col(self, n_day, col=_CLOSE):
        col_name = col+'_MA{}'.format(n_day)
        self.df[col_name] = self.mva_col(col, n_day)
        return col_name

    def add_mva(self, mva_day_list=mva_n_days, col_list=[_CLOSE]):
        self.mva_cols = []
        #mva_day_list = mva_day_list.sort()
        for col in col_list:
            for d in mva_day_list:
                mva_col_name = self.add_mva_col(n_day=d, col=col)
                self.mva_cols.append((col, mva_col_name))

    def is_above_mva(self, mva_col_pair):
        a = self.df[mva_col_pair[0]] > self.df[mva_col_pair[1]]
        return a

    def add_mva_col_is_above(self):   
        for mvc in self.mva_cols:
            self.df[mvc[0]+'_>_'+mvc[1]] = self.is_above_mva(mvc)

    def add_mva_and_is_above(self, mva_day_list=mva_n_days, col_list=[_CLOSE]):
        self.add_mva(mva_day_list=mva_day_list, col_list=col_list)
        self.add_mva_col_is_above()

    def approx_larger_than(self, col_tuple, percent=95, n_previous_days=30):
        df = self.get_df_from_previous_days(n_previous_days)
        a = df[col_tuple[0]] >= df[col_tuple[1]]
        l = len(a)
        if sum(a) >= percent*l/100:
            return True
        return False

    def mva_short_gt_long_term(self, n_previous_days=30, col=_CLOSE, mva_day_list=mva_n_days, col_list=[_CLOSE], percent=95):
        if self.mva_cols is None or len(self.mva_cols) == 0:
            self.add_mva(mva_day_list=mva_n_days, col_list=[_CLOSE])
        mvc = self.mva_cols[0]
        yes = self.approx_larger_than(mvc, percent, n_previous_days)
        l = len(self.mva_cols)
        i = 0
        while i < l-1 and yes:
            mvc = self.mva_cols[i]
            mva_c1 = mvc[1]
            mva_2 = self.mva_cols[i+1]
            mva_c2 = mva_2[1]
            yes = yes and self.approx_larger_than((mva_c1, mva_c1), percent, n_previous_days)
            i = i + 1
        return yes

    def search_max_best_mva(self, n_day_list, percent_above_it=90):
        # it is the smallest mva days which the price is nearly always above it.
        # or the largest mva days in which the price is closes to it.
        return

    def consecutive_stats(self):
        if self.is_empty():
            return None
        c_df = self.df.copy()
        c_df[_DATE] = pd.to_datetime(c_df[_DATE])
        c_df.sort_index(inplace=True)
        c_df.reset_index(inplace=True)
        c_df['cp'] = c_df[_CLOSE+_PERCENT_CHANGE_SUFFIX] > 0
        c_df_bool = c_df['cp'] != c_df['cp'].shift()
        dfcumsum = c_df_bool.cumsum()
        groups = c_df.groupby(dfcumsum)

        group_counts = groups.agg({_DATE: ['count', 'min', 'max'],
                                  _CLOSE+_PERCENT_CHANGE_SUFFIX: ['min', 'max', 'median', 'sum'],
                                  })

        group_counts.columns = ['_'.join(col).strip() for col in group_counts.columns.values]
        #groupCounts.to_excel('test3.xlsx')
        self.seq_stats_df = group_counts

    def add_momentum(self, time_frame):
        if time_frame == 'lower':
            self.df, col = self.indic.add_stochrsi(self.df)
        elif time_frame == 'higher':
            if self.highertimeframe_df is None:
                self.highertimeframe_df = self.get_df_weekly_timeframe()
            self.highertimeframe_df, col = self.indic.add_stochrsi(self.highertimeframe_df)

    def add_inc_dec_zero_change(self, cols):
        for c in cols:
            pass

    def set_multi_timeframe_stochrsi(self):
        self.comp_df_weekly_timeframe()
        self.add_momentum('higher')
        self.add_momentum('lower')

    def is_multi_timeframe_robertminer(self):
        self.set_multi_timeframe_stochrsi()
        df_tail_higher = self.highertimeframe_df.tail(2)
        df_tail = self.df.tail(10)
        is_inc_higher = df_tail_higher['momentum_stoch_rsi_d'].is_monotonic
        over_bought = df_tail_higher['momentum_stoch_rsi_d'].iloc[-1] > 85

        above_d_signal = (df_tail['momentum_stoch_rsi_d'].iloc[-1] > df_tail['momentum_stoch_rsi_k'].iloc[-1])

        is_reversing_1 = (df_tail['momentum_stoch_rsi_d'].iloc[-1] >= df_tail['momentum_stoch_rsi_d'].iloc[-2])\
                       & (df_tail['momentum_stoch_rsi_d'].iloc[-3] > df_tail['momentum_stoch_rsi_d'].iloc[-2])

        is_reversing_2 = (df_tail['momentum_stoch_rsi_d'].iloc[-1] > df_tail['momentum_stoch_rsi_d'].iloc[-2])\
                       & (df_tail['momentum_stoch_rsi_d'].iloc[-2] >= df_tail['momentum_stoch_rsi_d'].iloc[-3])\
                       & (df_tail['momentum_stoch_rsi_d'].iloc[-4] >= df_tail['momentum_stoch_rsi_d'].iloc[-3])

        is_reversing_3 = (df_tail['momentum_stoch_rsi_d'].iloc[-1] >
                          min(df_tail['momentum_stoch_rsi_d'].iloc[-2], df_tail['momentum_stoch_rsi_d'].iloc[-3]))\
                        & (df_tail['momentum_stoch_rsi_d'].iloc[-5] >
                          min(df_tail['momentum_stoch_rsi_d'].iloc[-3], df_tail['momentum_stoch_rsi_d'].iloc[-4]))\

        is_reversing = is_reversing_1 | is_reversing_2 | is_reversing_3
        #is_reversing = is_unimodal(df_tail, 'momentum_stoch_rsi_d')
        return is_inc_higher and is_reversing and not over_bought

class StockJoin:
    def __init__(self):
        self.join_stock_list = []
        self.data_joint_symbol_folder = data_joint_symbol_folder
        self.target_stock_list = []
        self.stockjoin = None
    
    def add_target_stock(self, target_stock:StockData, target_field=[_CLOSE]):
        target_suffix = '_T_' + target_stock.symbol
        target_bunch = Bunch(target_stock=target_stock, target_field=target_field, target_suffix=target_suffix)
        self.target_stock_list.append(target_bunch) 
    
    def add_source_stock(self, source_stock:StockData, next_day=1, field_list=None):
        source_suffix = '_' + str(next_day) + '_' + source_stock.symbol
        source_bunch = Bunch(source_stock=source_stock, next_day=next_day, 
                             field_list=field_list, source_suffix=source_suffix)
        self.join_stock_list.append(source_bunch)

    def merge_on_date(self, on_persian_date=True):
        self.merge_target_stocks_on_date(on_persian_date)
        self.merge_source_stocks_on_date(on_persian_date)

    def merge_target_stocks_on_date(self, on_persian_date=True):
        l = len(self.target_stock_list)
        if l == 0:
            return  
        self.set_stock_to_stockjoin(self.target_stock_list[0])
        for i in range(1, l):
            self._merge_target_stock_on_date_field(i, on_persian_date)

    def merge_source_stocks_on_date(self, on_persian_date=True):
        for i in range(len(self.join_stock_list)):
            self._merge_source_stock_on_date_field(i, on_persian_date)

    def set_stock_to_stockjoin(self, target_bunch):
        stock = target_bunch['target_stock']
        fields = target_bunch['target_field']
        suffix = target_bunch['target_suffix']

        self.stockjoin = stock
        self.stockjoin.add_suffix_to_fields_list(suffix, fields)

    def _merge_target_stock_on_date_field(self, i, on_persian_date):
        assert self.stockjoin is not None, 'Error stock join must be assigned before'
        target_bunch = self.target_stock_list[i]
        t_suffix = target_bunch['target_suffix']
        t_stock = target_bunch['target_stock']
        t_fields = target_bunch['target_field']
        if t_fields is None:
            t_fields = t_stock.field_list
        
        left_on = self.stockjoin.get_date_field(on_persian_date=on_persian_date)
        right_on = t_stock.get_date_field(on_persian_date=on_persian_date)
        t_fields.append(right_on)
        rightdf = self.stockjoin.df[t_fields]
        rightdf = rightdf.loc[:,~rightdf.columns.duplicated()].copy()
        self.stockjoin.df = pd.merge(rightdf, t_stock.df, how='inner', left_on=left_on,
                                     right_on=right_on, suffixes=('',t_suffix))
            
    def _merge_source_stock_on_date_field(self, ind, on_persian_date):
        source_stock_bunch = self.join_stock_list[ind]
        source_stock = source_stock_bunch['source_stock']
        next_day = source_stock_bunch['next_day']
        field_list = source_stock_bunch['field_list']
        source_suffix = source_stock_bunch['source_suffix']
        if field_list is None:
            field_list = source_stock.field_list
        left_on = self.stockjoin.get_date_field(on_persian_date=on_persian_date)
        right_on = source_stock.add_now_next_day_field(None, next_day, on_persian_date)
        field_list.append(right_on)
        sourcedf = source_stock.df[field_list]
        sourcedf = sourcedf.loc[:,~sourcedf.columns.duplicated()].copy()
        self.stockjoin.df = pd.merge(self.stockjoin.df, sourcedf, how='left', 
                        left_on=left_on, right_on = right_on, suffixes=('', source_suffix))
        self.stockjoin.df.drop(right_on, axis=1, inplace=True)

    def save(self):
        filename = self.data_joint_symbol_folder + self.target.symbol + '.csv'
        self.target.df.to_csv(filename)

_PREDICTCLOSELOW_ = 0

class StockJoinView:
    def __init__(self, target_stock_list, source_stock_list, next_day=1, source_field_list=None, on_persian_date=True):
        self.stockjoiner = StockJoin()
        for target_stock in target_stock_list:
            self.stockjoiner.add_target_stock(target_stock)
        for source_stock in source_stock_list:
            self.stockjoiner.add_source_stock(source_stock, next_day, source_field_list)
        self.stockjoiner.merge_on_date(on_persian_date)
        self.predictor_field = None
        self.targe_field = None
        self.new_view_df = None
        source_field_1 = Bunch(field = _CLOSE, mul=1.0)
        source_field_2 = Bunch(field = _LOW, mul=-1.0)
        source_fields = [source_field_1, source_field_2]
        self.source_bunch = []
        source_bunch = Bunch(source_fields=source_fields, field_name='PredictCloseLow')
        self.source_bunch[_PREDICTCLOSELOW_] = source_bunch
        
    def compute_predictor(self,  source_field_bunch, predictor_func=None):
        if predictor_func is None:
            self.stockjoiner.df, self.predictor_field = default_predictor_func(self.stockjoiner.df, source_field_bunch)
        self.stockjoiner.df, self.predictor_field = predictor_func(self.stockjoiner.df, source_field_bunch)
     
    def compute_target(self, target_field_bunch, target_func=None):
        if target_func is None:
            self.stockjoiner.df, self.targe_field = default_target_func(self.stockjoiner.df, target_field_bunch)
        self.stockjoiner.df, self.targe_field = target_func(self.stockjoiner.df, target_field_bunch)

    def get_predictor_target_view(self):
        newfields = self.predictor_field
        newfields.append(self.targe_field)
        self.new_view_df = self.stockjoiner.df[newfields]
        return self.new_view_df


def default_predictor_func(df, field_bunch_list, new_field_name = 'NEW_PREDICTOR'):
    field_0 = field_bunch_list[0].field
    m_0 = field_bunch_list[0].mul
    field_1 = field_bunch_list[1].field
    m_1 = field_bunch_list[1].mul
    df[new_field_name] = df[field_0]*m_0 + df[field_1]*m_1

    return df, new_field_name

def default_target_func(df, field_bunch_list, new_field_name = 'NEW_TARGET'):
    field_0 = field_bunch_list[0].field
    m_0 = field_bunch_list[0].mul
    field_1 = field_bunch_list[1].field
    m_1 = field_bunch_list[1].mul
    df[new_field_name] = df[field_0]*m_0 + df[field_1]*m_1

    return df, new_field_name


class StockDataList:
    def __init__(self, data_source_folder = 'datasource/EasyFilter/'):
        self.sym_industry_file = data_source_folder + '/complete_sym_indust.xlsx'

    def get_symbols_list(self):
        df_ind_list = pd.read_excel(self.sym_industry_file, engine='openpyxl')
        df_ind_list_ls = [persian.convert_ar_characters(c) for c in df_ind_list['Symbol']]
        df_ind_list[_SYMBOL] = df_ind_list_ls
        return df_ind_list

class StockTechnicalDataList(StockDataList):
    def __init__(self, taedil, soud, symb_save_path, data_source_folder='datasource/EasyFilter/'):
        super().__init__()
        self.sm_list = None
        self.symb_save_path = symb_save_path
        self.stock_fn_list = None
        self.taedil = taedil
        self.soud = soud
        self.namad_file_name = True
        self.df_technicals_all_filename = data_source_folder + 'techincals_data_all'
        self.save_technicals_to_excel = True
        self.symbol_filename_list = data_source_folder+'symbol_data_file.xlsx'
        self.symbol_file_exist = False
        self.technical_data_init()

    def get_list_of_symbol_files(self):
        if self.symb_save_path is None:
            self.symb_save_path = '/StockData/'
        if self.taedil:
            symb_save_path = self.symb_save_path + '/Adjusted/'
            file_list = os.listdir(symb_save_path)
        else:
            file_list = os.listdir(self.symb_save_path)
        file_list = pd.DataFrame(file_list, columns=['filename'])
        return file_list

    def technical_data_init(self):
        if self.symb_save_path is None:
            self.symb_save_path = 'D:/MTraderPython/StockData/'
        self.sm_list = self.get_list_of_symbol_files()
        # fn = 'isin_list.csv'
        # self.sm_list = pd.read_csv(fn)
        if self.taedil:
            symb_save_path = self.symb_save_path + '/Adjusted/'
            self.stock_fn_list = [symb_save_path + '/' + c for c in self.sm_list['filename']]
            # if not self.namad_file_name:
            #     if self.soud:
            #         self.stock_fn_list = [symb_save_path + '/' + c.replace('.csv', '-i.csv') for c in
            #                               self.sm_list['filename']]
            #     else:
            #         self.stock_fn_list = [symb_save_path + '/' + c.replace('.csv', '-a.csv') for c in
            #                               self.sm_list['filename']]
            # else:
            #     if self.soud:
            #         self.stock_fn_list = [symb_save_path + '/' + c.replace('.csv', '-ت.csv') for c in
            #                               self.sm_list['filename']]
            #     else:
            #         self.stock_fn_list = [symb_save_path + '/' + c.replace('.csv', '-ا.csv') for c in
            #                               self.sm_list['filename']]
        else:
            self.stock_fn_list = [self.symb_save_path + '/' + c for c in self.sm_list['filename']]

    def compute_technical_data_df(self,  from_date=None):
        df_ind_file = self.get_symbol_filename_df()
        l = len(df_ind_file.index)
        printProgressBar(0, l, prefix='Preparing Technicals:', suffix='Complete', length=50)
        j = 0
        frames = []
        for i in range(len(df_ind_file)):
            try:
                filename = df_ind_file.iloc[i, df_ind_file.columns.get_loc('File')]
                symbol = df_ind_file.iloc[i, df_ind_file.columns.get_loc('Symbol')]
                industry = df_ind_file.iloc[i, df_ind_file.columns.get_loc('Industry')]
                std = StockData(symbol)
                df_std = std.get_technical_data()
                frames.append(df_std)
                title = symbol +':' + industry
            except:
                print('cannot open file:', filename)
            printProgressBar(i + 1, l, prefix='Preparing Technicals:', suffix='Complete', length=50)

        df_technicals = pd.concat(frames)
        df_technicals.to_csv(self.df_technicals_all_filename+'.csv')
        if self.save_technicals_to_excel:
            df_technicals.to_excel(self.df_technicals_all_filename+'.xlsx')
        return df_technicals

    def get_technical_data_df(self, from_date, load_from_previous=False):
        try:
            if load_from_previous:
                df_technicals = pd.read_csv(self.df_technicals_all_filename+'.csv')
            else:
                df_technicals = self.compute_technical_data_df()
        except:
            df_technicals = self.compute_technical_data_df()
        df_technicals.drop(axis=1, columns=[_DATE, _SYMBOL], inplace=True)
        df_technicals[_PERSIAN_DATE] = df_technicals[_PERSIAN_DATE].astype(str)
        return df_technicals

    def stock_data_report(self, pdf_bunch_list, from_date=None):
        df_ind_file = self.get_symbol_filename_df()
        self.screen_stocks_to_pdf(pdf_bunch_list, df_ind_file, from_date)

    def get_symbol_selectedrows_from_file(self, df_ind_file, symbol_list):
        selected_df = df_ind_file.loc[df_ind_file[_SYMBOL].isin(symbol_list)]
        return selected_df

    def get_symbol_by_industry_from_file(self, df_ind_file, industry_list):
        selected_df = df_ind_file.loc[df_ind_file['Industry'].isin(industry_list)]
        return selected_df 
    
    def get_symbol_filename_industry(self, symbol_list):
        df_ind_file = self.get_symbol_filename_df()
        df = self.get_symbol_selectedrows_from_file(df_ind_file, symbol_list)
        file_list = []
        sym_list = []
        industry_list = []
        for i in range(len(df.index)):
            filename, symbol, industry = self.get_filename_symbol_industry(df, i)
            file_list.append(filename)
            sym_list.append(symbol)
            industry_list.append(industry)
        return file_list, sym_list, industry_list

    def get_filename_symbol_industry(self, df, i):
        filename = df.iloc[i, df.columns.get_loc('File')]
        symbol = df.iloc[i, df.columns.get_loc('Symbol')]
        industry = df.iloc[i, df.columns.get_loc('Industry')]
        return filename, symbol, industry

    def stock_join_view(self, target_symbol_list, source_symbol_list, next_day=1):
        target_stock_list = []
        for target_symbol in target_symbol_list:
            stock = StockData(target_symbol)
            target_stock_list.append(stock)
        source_stock_list = []
        for source_symbol in source_symbol_list:
            stock = StockData(source_symbol)
            source_stock_list.append(stock)

        stockview = StockJoinView(target_stock_list, source_stock_list, next_day)


    def test_joinview(self):
        target_symbol_list = ['آریا', 'آبادا'] 
        source_symbol_list = ['آبادا', 'آریا'] 
        self.stock_join_view(target_symbol_list, source_symbol_list, next_day=5)

    def test_join(self):
        df_ind_file = self.get_symbol_filename_df()
        
        # target_s = ['آبادا', 'آریا', 'بتک']
        # source_s = target_s
        industry_list = ['عرضه برق، گاز، بخاروآب گرم']
        target_s = self.get_symbol_by_industry_from_file(df_ind_file, industry_list)
        target_s = target_s[_SYMBOL]
        source_s = target_s
        source_next_day = [3, 5, 7]
        field_list = [_CLOSE, _OPEN]

        self.join_stock_data(target_s, source_s, source_next_day, field_list)

    def join_stock_data(self, target_symbol_list, source_symbol_list, source_next_day_list, field_list):
        df_ind_file = self.get_symbol_filename_df()
        target_selected_df = self.get_symbol_selectedrows_from_file(df_ind_file, target_symbol_list)
        source_selected_df = self.get_symbol_selectedrows_from_file(df_ind_file, source_symbol_list)
        
        l = len(target_selected_df)*len(source_selected_df)
        stepl = len(source_selected_df)
        printProgressBar(0, l, prefix='Progress:', suffix='Complete', length=50)

        stepl = len(source_symbol_list)
        t = 0
        for target_symbol in target_symbol_list:
            # try:
            stockjoiner = StockJoin()

            std = StockData(target_symbol)
            stockjoiner.set_master_df(std)
            # std.set_multi_timeframe_stochrsi()
            # std.get_technical_data(['VWAP'], two_level=True)
            # title = symbol +':' + industry
            
            for source_symbol in source_symbol_list:
                # try:
                next_day = 3
                stock_source = StockData(source_symbol)
                stockjoiner.add_join_stock(stock_source, next_day, field_list)
                # except:
                    # pass
            stockjoiner.merge_stocks_on_date(on_persian_date=True)
            stockjoiner.save()
            # except:
            #     print('cannot open file:', filename)
            t = t +  stepl 
            printProgressBar(t, l, prefix='Progress:', suffix='Complete', length=50)

    def get_filename_of_symbol(self, symbol):
        pass

    def get_technicals_of_symbol_list(self, symbol_list, start_persian_date, end_persian_date):
        pass

    def get_technicals_of_industry_list(self, industry_list, start_persian_date, end_persian_date):
        pass

    def screen_stocks_to_pdf(self,  pdf_bunch_list, df_file_list, from_date=None):
        l = len(self.sm_list)
        printProgressBar(0, l, prefix='Progress:', suffix='Complete', length=50)
        for b in pdf_bunch_list:
            b.pdffile = PdfPages(b.pdf_filename)
        j = 0
        for i in range(len(df_file_list)):
            try:
                filename = df_file_list.iloc[i, df_file_list.columns.get_loc('File')]
                symbol = df_file_list.iloc[i, df_file_list.columns.get_loc('Symbol')]
                industry = df_file_list.iloc[i, df_file_list.columns.get_loc('Industry')]
                std = StockData(symbol)
                std.set_multi_timeframe_stochrsi()
                std.get_technical_data(['VWAP'], two_level=True)
                title = symbol +':' + industry
                for b in pdf_bunch_list:
                    if b.filter_type == 1:
                        b.cond = True
                    elif b.filter_type == 2:
                        b.cond = std.is_multi_timeframe_robertminer()
                    elif b.filter_type == 3:
                        b.cond = std.mva_short_gt_long_term()
                    elif b.filter_type == 0:
                        b.cond = (industry=='شاخص')
                if len(std.df['Open']) > 0:
                    fig, plt = draw_candle_stick_mpf(std, symbol)
                    fig_h, plt_h = draw_candle_stick_mpf(std, title, 'higher')
                    for b in pdf_bunch_list:
                        if b.cond:
                            pdffile = b.pdffile
                            pdffile.savefig(figure=fig_h, bbox_inches='tight')
                            pdffile.savefig(figure=fig, bbox_inches='tight')
                    if not (plt is None):
                        plt.close()
                        plt_h.close()

            except:
                print('cannot open file:', filename)
            printProgressBar(i + 1, l, prefix='Progress:', suffix='Complete', length=50)

        for b in pdf_bunch_list:
            pdffile = b.pdffile
            pdffile.close()

    def scan_to_symbol_files(self):
        fn_list = []
        symbol_list = []
        lt = len(self.stock_fn_list)
        i = 0
        for fn in self.stock_fn_list:
            # try:
            std = StockData(fn, add_percentage=False)
            if std.is_empty:
                continue
            symbol = std.get_symbol()
            fn_list.append(fn)
            symbol_list.append(symbol)
            # except:
            print('cannot open file:', fn)
            i = i + 1
            printProgressBar(i + 1, lt, prefix='Progress:', suffix='Complete', length=50)
        df = pd.DataFrame(zip(symbol_list, fn_list), columns=['Symbol', 'File'])
        df_list = self.get_symbols_list()
        df_t = pd.merge(df_list, df, how='inner', on='Symbol')
        df_t = df_t.sort_values(by=['Symbol'])
        df_t.to_excel(self.symbol_filename_list)
        self.symbol_file_exist = True
        return df_t

    def get_symbol_filename_df(self):
        try:
            df = pd.read_excel(self.symbol_filename_list, engine="openpyxl")
            return df
        except:
            pass
        return self.scan_to_symbol_files()