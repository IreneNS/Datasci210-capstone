
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
import importlib
from statsmodels.regression.rolling import RollingOLS
import statsmodels.api as sm
import statsmodels.formula.api as smf
from pypfopt import (
    EfficientFrontier, 
    risk_models,
    plotting,
)
import cvxpy as cp
import warnings
warnings.filterwarnings('once')
from sklearn.covariance import LedoitWolf
from joblib import Parallel, delayed
import time
print(os.getcwd())
print(os.listdir())
import src.ds210_ml_portf_data_retrieval as data_unit
import openpyxl
import boto3
from datetime import datetime
from io import BytesIO
import gc
# import importlib
# importlib.reload(data_unit)

s3 = boto3.client('s3')

bucket_name = "dsp-public-streamlit"

def rolling_regression_sm(df_y, df_x, rolling_window, min_nobs):
    # note: the input in sm needs to be either np array or pd series, either works under sm regression
    idx1 = df_x.dropna(how='all').index
    idx2 = df_y.dropna(how='all').index
    idx_comm = idx1.intersection(idx2, sort=False)
    x = df_x.reindex(idx_comm)
    y = df_y.reindex(idx_comm)
    
    x = sm.add_constant(x, has_constant='add')
    model_prefit = RollingOLS(y, x, window=rolling_window, min_nobts=min_nobs)
    sm_model = model_prefit.fit() # normal OLS
    # sm_model = model_prefit.fit_regularized(L1_wt=1.0, alpha=0.001) # lasso
    param_reg = sm_model.params
    return x, y, sm_model, model_prefit, param_reg

def regression_sm(df_y, df_x):
    idx1 = df_x.dropna(how='all').index
    idx2 = df_y.dropna(how='all').index
    
    index_comm = idx1.intersection(idx2, sort=False)
    df_Y = df_y.reindex(index_comm)
    df_X = df_x.reindex(index_comm)
    
    x, y = df_X, df_Y
    x = sm.add_constant(x, has_constant='add')
    model_prefit = sm.OLS(y,x)
    sm_model = model_prefit.fit()
    param_reg = sm_model.params
    return x, y, sm_model, model_prefit, param_reg

def optimizer_ef(ann_exp_exc_rtn, ann_cov_matrix, opt_flag, target_risk, 
                 solver_name='CLARABEL', market_neutral=False):
    ef_ob = EfficientFrontier(ann_exp_exc_rtn, ann_cov_matrix)
    ef_ob._solver = solver_name # IN: 'CLARABEL', this solver works for max_sharpe choice, and likely better for 'target_risk' too
    # ef_ob._solver_options ={'ECOS':cp.ECOS}
    # ef_ob._solver = 'ECOS'

    if opt_flag == 'target_risk':
        ef_ob.efficient_risk(target_volatility=target_risk, market_neutral=market_neutral)
    elif opt_flag == 'max_sharpe':
        ef_ob.max_sharpe()

    return ef_ob.clean_weights()

def prep_rank_feature(df, features_col, shift_d):

    rank_features_col_lag = [x+'_rank' for x in features_col]
    rank_feat = df.set_index(['date','permno'])[features_col].groupby('date')\
        .apply(lambda x: (x.rank(pct=True)-0.5).droplevel(0))\
            .rename(columns=dict(zip(features_col, rank_features_col_lag))).reset_index()
    # regroup by stock and do the shift 
    rank_feat = rank_feat.set_index('date').groupby('permno')[rank_features_col_lag]\
        .apply(lambda x: x.shift(shift_d)).reset_index()

    df = pd.merge(df, rank_feat, on=['date','permno'], how='left')
    
    return df, rank_features_col_lag

def mmd_cal(df, return_col_name): # IN mod2
    df_1 = df.copy() 
    df_1['cum_rtn']=(1+df_1[return_col_name]).cumprod()
    df_1['drawdown'] = (df_1['cum_rtn']-df_1['cum_rtn'].cummax())/df_1['cum_rtn'].cummax()
    df_1['max_drawdown'] =  df_1['drawdown'].cummin()
    return df_1['max_drawdown']

def portfolio_performance(input_df, weight_df, portf_name, rebal_freq, mkt_df, 
                          last_win_only=False, vol_scaler_flag=False, scaling_vol_tgt=0.3, plot_show=True): #IN mod
    # rebal_freq: 'D','M','W'
    if rebal_freq =='D':
        ff_n = 2
    elif rebal_freq == 'W':
        ff_n = 7
    elif rebal_freq == 'M':
        ff_n = 31
        
    if last_win_only is True: #IN mod, if it's last_win_only mode (demo mode), only use last year data, and backfill the weights 
       return_df = input_df[input_df['date'].dt.year.isin([2024])].set_index(['date','permno'])['return'].unstack()[weight_df.columns]
       weight_df=weight_df.asfreq('D').ffill(limit=ff_n).reindex(return_df.index)\
        .ffill().bfill().where(~(return_df.isnull()), 0)
       mkt_df = mkt_df.loc[return_df.index,:]
    else: 
        return_df = input_df.set_index(['date','permno'])['return'].unstack()[weight_df.columns]
        weight_df=weight_df.asfreq('D').ffill(limit=ff_n).reindex(return_df.index).where(~(return_df.isnull()), 0)
    portf_rtn = (return_df*weight_df).sum(axis=1)

    if vol_scaler_flag is True:  #IN mod2
        portf_rtn_0 = portf_rtn.copy()
        if last_win_only is True:
            portf_rtn = (portf_rtn_0*(scaling_vol_tgt/np.sqrt(252))/(portf_rtn_0.rolling(60, min_periods=2).std())).fillna(0) #IN mod2
        else: 
            portf_rtn = portf_rtn_0*(scaling_vol_tgt/np.sqrt(252))/(portf_rtn_0.rolling(60).std()) #IN mod2
        scaler_df = portf_rtn.div(portf_rtn_0, axis=0)
        # scaled_mkt = mkt_df['return_sp']*scaling_vol_tgt/mkt_df['return_sp'].rolling(60).std()
        unscaled_mkt = mkt_df['return_sp'].loc[portf_rtn.index[0]:] #IN mod2

    else:
        scaler_df = portf_rtn.div(portf_rtn, axis=0)
        # scaled_mkt = mkt_df['return_sp']*(portf_rtn.rolling(60).std()/mkt_df['return_sp'].rolling(60).std())
        unscaled_mkt = mkt_df['return_sp'].loc[portf_rtn.index[0]:] #IN mod2
    
    fig1, ax1 = plt.subplots(1,2, figsize=(11,3.5))
    portf_mkt_rtn = pd.concat([portf_rtn.rename(portf_name), unscaled_mkt.rename('Unscaled Market')], axis=1)
    portf_mkt_rtn.cumsum().plot(ax=ax1[0])
    ax1[0].set_title(f'Cumulative Return Comparison')
    ax1[0].legend(loc='upper left')
    ax1[0].set_ylabel('Cumulative Return')
    # plt.legend()
    # plt.show()

    if last_win_only is True:
        (portf_mkt_rtn.rolling(60).std()*np.sqrt(252)).plot(
            ax=ax1[1],
            title='Rolling Annual Vol Comparison')
        ax1[1].legend(loc='upper left')
    else:
        (portf_mkt_rtn.rolling(252).std()*np.sqrt(252)).plot(
        ax=ax1[1],
        title='Rolling Annual Vol Comparison')
        ax1[1].legend(loc='upper left')
    ax1[1].set_ylabel('Rolling Annual Vol')

    fig1.suptitle(f'{portf_name} vs (Scaled) S&P500 Cumulative Return and Rolling Vol Comparison')
    plt.subplots_adjust(top=0.85, bottom=0.01, wspace=0.2)
    if plot_show is True: #IN mod
        plt.show()

    svg_buffer = BytesIO()
    fig1.savefig(svg_buffer, format="svg")
    svg_buffer.seek(0)  # Reset buffer position

    # Save figure as Pickle to an in-memory buffer
    pickle_buffer = BytesIO()
    pickle.dump(fig1, pickle_buffer)
    pickle_buffer.seek(0)

    object_name = f'benchmark_portf_{rebal_freq}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    s3.upload_fileobj(svg_buffer, bucket_name, f'outputs/{object_name}.svg')
    s3.upload_fileobj(pickle_buffer, bucket_name, f'outputs/{object_name}.pkl')

    plt.close()

    stats_df = pd.DataFrame(columns=portf_mkt_rtn.columns)
    stats_df.loc['avg_rtn_ann',:] = portf_mkt_rtn.mean()*252
    stats_df.loc['vol_ann',:] = portf_mkt_rtn.std()*np.sqrt(252)
    stats_df.loc['sharpe_ann',:] = stats_df.loc['avg_rtn_ann',:]/stats_df.loc['vol_ann',:]
    stats_df.loc['max_drawdown',portf_name] = mmd_cal(portf_mkt_rtn, portf_name).iloc[-1]
    stats_df.loc['max_drawdown','Unscaled Market'] = mmd_cal(portf_mkt_rtn, 'Unscaled Market').iloc[-1]
    
    if plot_show is True: #IN mod
        print(stats_df)
    
    # portf_rtn = portf_rtn.reset_index()
    # portf_mkt_rtn = portf_mkt_rtn.reset_index()  
    # stats_df = stats_df.reset_index()
    # scaler_df = scaler_df.reset_index()  

    return portf_rtn, portf_mkt_rtn, stats_df, scaler_df, object_name

def check_weights_plots(opt_weight_df):
    # check basic properties for optimization weights:
    np.round(opt_weight_df.sum(axis=1),2).plot(title='Total weights for the portfolio over time', figsize=(5,3))
    plt.show()

    opt_weight_df.count(axis=1).plot(title='Total number of stocks over time (not all have non zero weight)', figsize=(5,3))
    plt.show()

    opt_weight_df[opt_weight_df>0].count(axis=1).plot(title='Total number of stocks with non-zero weight over time', figsize=(5,3))
    plt.show()


class Benchmark_model_DL:
    '''this is the class to build a benchmark model based on 
    standard statistical method based portfolio construction method'''

    '''note: if you will create the data with new checkpoint name, you will
    need to put in WRDS user name, password, and "Y"'''

    def __init__(self, input_data_obj, rebal_freq):
        self.dir = None
        self.checkpoint_name = None
        # self.data_checkpoint_name =''
        self.input_data_obj = input_data_obj
        # IN: note: train_used and test_used should be post feature engineering
        self.train_used = None
        self.test_used = None
        self.sel_features_adj = None
        self.model_output_dic = {}
        self.rebal_freq = rebal_freq
        self.opt_trained = []
        self.opt_tested = []
         
    def __repr__(self):
        return f'\nThis is a statistical method based bencnmark_model, \
                \nThe model checkpoint name of this data is {self.checkpoint_name},\
                \nThe trained opt models are {self.opt_trained},\
                \nThe tested opt models are {self.opt_tested}'

    def set_model_directory(self, model_directory_path):
        #IN: note this directory_path should be where the model is at
        self.dir = model_directory_path
        os.chdir(model_directory_path)
    
    def check_checkpoint_model(self, checkpoint_name):
        if checkpoint_name not in os.listdir():
          return False
        elif f'bm_model_obj_{self.rebal_freq}.pkl' in os.listdir(os.path.join(self.dir, checkpoint_name)):
            self.checkpoint_name = checkpoint_name
            return True
        else:
            return False

    def load_model(self, checkpoint_name):
        if self.check_checkpoint_model(checkpoint_name): ## CHECK TO DO
            with open(f'./{checkpoint_name}/bm_model_obj_{self.rebal_freq}.pkl', 'rb') as f:
                try:
                    bm_model_obj = pickle.load(f)
                except AttributeError:
                    # Handle the module path issue
                    import sys
                    import src.ds210_ml_portf_data_retrieval 
                    # Register the module with the expected name
                    sys.modules['ds210_ml_portf_data_retrieval'] = src.ds210_ml_portf_data_retrieval
                    sys.modules['__main__'].Benchmark_model_DL = Benchmark_model_DL
                    bm_model_obj = pickle.load(f)
            return bm_model_obj  # IN: or should it be just return bm_model_obj?
        else:
            raise KeyError ('Model result with given checkpoint name does not exist in given directory.')

    def feature_engineer(self, input_raw, mkt_input_raw, ticker_list, check_visual=False):
        print ('Start feature engineering..')
        # prepare features
        # sel_features = ['market_equity','book_to_equity','asset_growth','opt_to_book_eq','ret_12m_d']
        sel_features = ['market_equity','book_to_equity','asset_growth','opt_to_book_eq','ret_12m_d', 'ret_3m_d', 'ret_6m_d',\
                        'ret_1m_d', 'ret_60_12']
        sel_tgt = ['return']

        # prepare model used data
        input_used = input_raw[['date','permno','ticker']+sel_features+sel_tgt]

        if len(ticker_list)!=0:
            input_used = input_used[input_used['ticker'].isin(ticker_list)]

        # further adjustment on return based feature by filling the first year data with ret_12_1 (monthly)
        ret_feat_map = {'ret_12m_d':'ret_12_1', 'ret_6m_d':'ret_6_1', 'ret_3m_d':'ret_3_1', 'ret_1m_d':'ret_1_0'}
        for col in ['ret_12m_d', 'ret_1m_d', 'ret_3m_d', 'ret_6m_d']:
            if col in sel_features:
                input_used.loc[input_used['date']<'2000-12-29', col] = \
                    input_used.loc[input_used['date']<'2000-12-29', col]\
                        .fillna(input_raw.loc[input_raw['date']<'2000-12-29', ret_feat_map[col]])

        # prepare a column called previous period excess return for factor return regression later
        # add mkt return to the data
        print (f'The shape of input_used before adding return_sp and excess return is {input_used.shape}')
        input_used = pd.merge(input_used, mkt_input_raw.reset_index()[['Date','return_sp']].rename(columns={'Date':'date'}),
                            on=['date'], how='left')
        input_used['excess_ret'] = input_used['return']-input_used['return_sp']
        print (f'The shape of input_used after adding return_sp and excess return is {input_used.shape}')

        prev_d_exc_ret_df = input_used.set_index('date').groupby('permno')['excess_ret']\
            .apply(lambda x: x.shift()).rename('prev_d_exc_ret').reset_index()

        print (f'The shape of input_used before adding prev_d_exc_ret is {input_used.shape}')
        input_used = pd.merge(input_used, prev_d_exc_ret_df, on=['date','permno'], how='left')
        print (f'The shape of input_used before adding prev_d_exc_ret is {input_used.shape}')

        # add rank_features, with shift
        print (f'The shape of input_used before adding rank_features is {input_used.shape}')
        input_used, sel_features_adj = prep_rank_feature(input_used, sel_features, shift_d=1)
        print (f'The shape of input_used after adding rank_features is {input_used.shape}')
        print (f'The added columns are: {sel_features_adj}')

        if check_visual:
            # Check the distribution of the ranked and shifted signals
            n = len(sel_features_adj)
            n_per_row=3
            if n%n_per_row==0:
                n_rows = n//n_per_row
            else:
                n_rows = n//n_per_row+1
            fig1, ax1 = plt.subplots(n_rows, n_per_row, figsize=(12,4))
            for i, col in enumerate(sel_features_adj):
                input_used[col].to_frame().hist(ax=ax1[i//n_per_row, i%n_per_row])
                ax1[i//n_per_row, i%n_per_row].set_title(f'Histogram of {col}')
            plt.suptitle('Check the distribution of ranked features')
            plt.subplots_adjust(hspace=0.9)
            plt.show()

        return input_used, sel_features_adj

    def train(self, checkpoint_name, ticker_list, opt_flag, target_risk=0.2, verbose=True):
        #IN: rebal_freq = 'D', 'W', or 'M'
        # opt_flag = 'target_risk' or 'max_sharpe'; 
        # when 'target_risk' flag is chosen, target_risk param needs to be provided

        rebal_freq = self.rebal_freq
        self.checkpoint_name = checkpoint_name
        self.opt_trained.append(opt_flag)

        if (self.train_used is None) or (self.sel_features_adj is None):
            train_sp500 = self.input_data_obj.data_dic['train_sp500']
            train_mkt = self.input_data_obj.data_dic['train_mkt']
            train_used, sel_features_adj = \
                self.feature_engineer(train_sp500, train_mkt, ticker_list, check_visual=True)
            self.train_used = train_used
            self.sel_features_adj = sel_features_adj
        else:
            train_used = self.train_used
            sel_features_adj = self.sel_features_adj

        # train the model with rolling regression and opt
        # use dataframes to record rolling regression and optimization results
        # reg_param_df = pd.DataFrame(columns=['const']+sel_features_adj)
        # pred_df = pd.DataFrame()
        # opt_weight_df = pd.DataFrame()
        n_multiple = 252 # IN: number of period per ann
        # rebal_freq = 'M' # IN: 'D', 'W', 'M'
        # opt_flag = 'max_sharpe'  #IN: 'target_risk' or 'max_sharpe'
        # target_risk = 0.2  #IN: this is an example of 20% annual portfolio risk

        rolling_ana_data_used = train_used
        dates_train_daily = rolling_ana_data_used['date'].unique()[20:] 
        # the first available date for features is the first date of the second month of all data
        dates_train_weekly = pd.Series(np.ones(len(dates_train_daily)), \
                                    index=pd.DatetimeIndex(dates_train_daily)).asfreq('W-WED').index
        dates_train_monthly = pd.Series(np.ones(len(dates_train_daily)), \
                                        index=pd.DatetimeIndex(dates_train_daily)).asfreq('ME').index

        if rebal_freq == 'D':
            rebal_dates = dates_train_daily
        elif rebal_freq == 'W':
            rebal_dates = dates_train_weekly
        elif rebal_freq == 'M':
            rebal_dates = dates_train_monthly

        def regression_n_opt_t(date, dates_train_daily, rolling_ana_data_used, opt_flag, target_risk):
            ''' regression (for factor weights and next period exp return)'''
            print (date)
            index_in_daily_dates = list(dates_train_daily).index(dates_train_daily[dates_train_daily<=date][-1])

            date_period = dates_train_daily[max(0,(index_in_daily_dates+1)-252*1)
                                        :max(252*1, (index_in_daily_dates+1))]
            data_t = rolling_ana_data_used[rolling_ana_data_used['date'].isin(date_period)]

            df_y = data_t['excess_ret']
            df_x = data_t[sel_features_adj].apply(lambda x: x*data_t['prev_d_exc_ret'], axis=0).ffill()
            x, y, sm_model, model_prefit, param_reg \
                = regression_sm(df_y, df_x)
            
            pred_t = pd.concat([data_t[['date','permno']], sm_model.predict(x).rename('pred_exc_ret')], axis=1)
            # pred_df = pd.concat([pred_df, pred_t], axis=0)
            # reg_param_df.loc[date,:] = param_reg
            param_reg_t = pd.DataFrame(param_reg.values, columns=[date], index=param_reg.index).T

            ''' optimization'''
            if rebal_freq == 'M':
                match_date = dates_train_daily[(dates_train_daily.month==date.month)
                                            &(dates_train_daily.year==date.year)][-1]
                
                exp_exc_rtn = pred_t.groupby('permno')[['date','pred_exc_ret']]\
                    .apply(lambda x: x.loc[x['date']==match_date, 'pred_exc_ret']).droplevel(1).replace(np.nan, 0)
            elif rebal_freq == 'W':
                match_date = dates_train_daily[(dates_train_daily<=date)][-1]
                
                exp_exc_rtn = pred_t.groupby('permno')[['date','pred_exc_ret']]\
                    .apply(lambda x: x.loc[x['date']==match_date, 'pred_exc_ret']).droplevel(1).replace(np.nan, 0)        
            else:
                exp_exc_rtn = pred_t.groupby('permno')[['date','pred_exc_ret']]\
                    .apply(lambda x: x.loc[x['date']==date, 'pred_exc_ret']).droplevel(1).replace(np.nan, 0)
            
            exp_exc_rtn_t = pd.DataFrame(exp_exc_rtn.values, columns=[date], index=exp_exc_rtn.index).T

            # prepare cov for optimizer 
            # IN: rationale: 
            # a) use current lookback period's data
            # b) based on JPM paper (1) use LedoitWolf() method to regulate the cov matrix, (2) check positive semi-definite conditions and fix it if needed
            return_data = data_t[['date','permno']+['prev_d_exc_ret']]\
                .groupby(['date','permno'])['prev_d_exc_ret'].last().unstack()[exp_exc_rtn.index].replace(np.nan, 0)
            cov_matrix = risk_models.fix_nonpositive_semidefinite(
                pd.DataFrame(LedoitWolf().fit(return_data).covariance_, 
                        index=return_data.columns, columns=return_data.columns)*n_multiple, fix_method='spectral') 
            # fix_method: {"spectral", "diag"}, defaults to "spectral"

            opt_w_ef = optimizer_ef(exp_exc_rtn*n_multiple, cov_matrix, opt_flag, target_risk, 
                        solver_name='CLARABEL', market_neutral=False)
            
            opt_w_t = pd.DataFrame(opt_w_ef, index=[date])

            return param_reg_t, exp_exc_rtn_t, opt_w_t

        # Run in parallel
        ## IN: note the results from the parallel processing has to be just one list of result at each t, which could be a tuple of different results at each time t
        print ('Start parallel running..')
        res_list = Parallel(n_jobs=-1)(delayed(
            regression_n_opt_t)(date, dates_train_daily, rolling_ana_data_used, opt_flag, target_risk)
            for date in rebal_dates
        )

        # Organize the results
        reg_param_df = pd.concat([res_t[0] for res_t in res_list])
        exp_exc_rtn_df = pd.concat([res_t[1] for res_t in res_list])
        opt_weight_df = pd.concat([res_t[2] for res_t in res_list])

        # put train output in the model object
        self.model_output_dic[f'train_results_{opt_flag}'] = {
            'reg_param_df':reg_param_df,
            'exp_exc_rtn_df':exp_exc_rtn_df, 
            'opt_weight_df':opt_weight_df
        }
        
        return  reg_param_df, exp_exc_rtn_df, opt_weight_df

    def test(self, checkpoint_name, ticker_list, opt_flag, last_win_only=False, target_risk=0.2, verbose=True):
        # Note: for this rollng window backtest method, we each rolling window is a newly trained model
        # therefore the test is OOS test with the same set up as train, which train model at each rebal
        # for MVP purpose, we can choose to only run last window at the end of data provided to show just the last model results without history performance

        rebal_freq = self.rebal_freq
        self.checkpoint_name = checkpoint_name
        if last_win_only is True:
            self.opt_tested.append(f'{opt_flag}_last_win')  #IN mod
        else:
            self.opt_tested.append(f'{opt_flag}')

        if (self.test_used is None) or (self.sel_features_adj is None):
            test_sp500 = self.input_data_obj.data_dic['test_sp500']
            test_mkt = self.input_data_obj.data_dic['test_mkt']
            test_used, sel_features_adj = \
                self.feature_engineer(test_sp500, test_mkt, ticker_list, check_visual=True)
            self.test_used = test_used
            self.sel_features_adj = sel_features_adj
        else:
            test_used = self.test_used
            sel_features_adj = self.sel_features_adj

        # train the model with rolling regression and opt
        # use dataframes to record rolling regression and optimization results
        # reg_param_df = pd.DataFrame(columns=['const']+sel_features_adj)
        # pred_df = pd.DataFrame()
        # opt_weight_df = pd.DataFrame()
        n_multiple = 252 # IN: number of period per ann

        rolling_ana_data_used = test_used
        dates_train_daily = rolling_ana_data_used['date'].unique()[20:] 
        # the first available date for features is the first date of the second month of all data
        dates_train_weekly = pd.Series(np.ones(len(dates_train_daily)), \
                                    index=pd.DatetimeIndex(dates_train_daily)).asfreq('W-WED').index
        dates_train_monthly = pd.Series(np.ones(len(dates_train_daily)), \
                                        index=pd.DatetimeIndex(dates_train_daily)).asfreq('ME').index
        if last_win_only is True: #IN_mod
            # print ('here')
            if rebal_freq == 'D':
                rebal_dates = dates_train_daily[-252:] # IN-mod
            elif rebal_freq == 'W':
                rebal_dates = dates_train_weekly[-52:] # IN-mod
            elif rebal_freq == 'M':
                rebal_dates = dates_train_monthly[-12:] # IN-mod
        else:
            # print ('there')
            if rebal_freq == 'D':
                rebal_dates = dates_train_daily
            elif rebal_freq == 'W':
                rebal_dates = dates_train_weekly
            elif rebal_freq == 'M':
                rebal_dates = dates_train_monthly

        def regression_n_opt_t(date, dates_train_daily, rolling_ana_data_used, opt_flag, target_risk):
            ''' regression (for factor weights and next period exp return)'''
            print (date)
            index_in_daily_dates = list(dates_train_daily).index(dates_train_daily[dates_train_daily<=date][-1])

            date_period = dates_train_daily[max(0,(index_in_daily_dates+1)-252*1)
                                        :max(252*1, (index_in_daily_dates+1))]
            data_t = rolling_ana_data_used[rolling_ana_data_used['date'].isin(date_period)]

            df_y = data_t['excess_ret']
            df_x = data_t[sel_features_adj].apply(lambda x: x*data_t['prev_d_exc_ret'], axis=0).ffill()
            x, y, sm_model, model_prefit, param_reg \
                = regression_sm(df_y, df_x)
            
            pred_t = pd.concat([data_t[['date','permno']], sm_model.predict(x).rename('pred_exc_ret')], axis=1)
            # pred_df = pd.concat([pred_df, pred_t], axis=0)
            # reg_param_df.loc[date,:] = param_reg
            param_reg_t = pd.DataFrame(param_reg.values, columns=[date], index=param_reg.index).T

            ''' optimization'''
            if rebal_freq == 'M':
                match_date = dates_train_daily[(dates_train_daily.month==date.month)
                                            &(dates_train_daily.year==date.year)][-1]
                
                exp_exc_rtn = pred_t.groupby('permno')[['date','pred_exc_ret']]\
                    .apply(lambda x: x.loc[x['date']==match_date, 'pred_exc_ret']).droplevel(1).replace(np.nan, 0)
            elif rebal_freq == 'W':
                match_date = dates_train_daily[(dates_train_daily<=date)][-1]
                
                exp_exc_rtn = pred_t.groupby('permno')[['date','pred_exc_ret']]\
                    .apply(lambda x: x.loc[x['date']==match_date, 'pred_exc_ret']).droplevel(1).replace(np.nan, 0)        
            else:
                exp_exc_rtn = pred_t.groupby('permno')[['date','pred_exc_ret']]\
                    .apply(lambda x: x.loc[x['date']==date, 'pred_exc_ret']).droplevel(1).replace(np.nan, 0)
            
            exp_exc_rtn_t = pd.DataFrame(exp_exc_rtn.values, columns=[date], index=exp_exc_rtn.index).T
            
            # prepare cov for optimizer 
            # IN: rationale: 
            # a) use current lookback period's data
            # b) based on JPM paper (1) use LedoitWolf() method to regulate the cov matrix, (2) check positive semi-definite conditions and fix it if needed
            return_data = data_t[['date','permno']+['prev_d_exc_ret']]\
                .groupby(['date','permno'])['prev_d_exc_ret'].last().unstack()[exp_exc_rtn.index].replace(np.nan, 0)
            cov_matrix = risk_models.fix_nonpositive_semidefinite(
                pd.DataFrame(LedoitWolf().fit(return_data).covariance_, 
                        index=return_data.columns, columns=return_data.columns)*n_multiple, fix_method='spectral') 
            # fix_method: {"spectral", "diag"}, defaults to "spectral"

            opt_w_ef = optimizer_ef(exp_exc_rtn*n_multiple, cov_matrix, opt_flag, target_risk, 
                        solver_name='CLARABEL', market_neutral=False)
            
            opt_w_t = pd.DataFrame(opt_w_ef, index=[date])

            return param_reg_t, exp_exc_rtn_t, opt_w_t

        # Run in parallel
        ## IN: note the results from the parallel processing has to be just one list of result at each t, which could be a tuple of different results at each time t
        print ('Start parallel running..')
        res_list = Parallel(n_jobs=-1)(delayed(
            regression_n_opt_t)(date, dates_train_daily, rolling_ana_data_used, opt_flag, target_risk)
            for date in rebal_dates
        )

        # Organize the results
        reg_param_df = pd.concat([res_t[0] for res_t in res_list])
        exp_exc_rtn_df = pd.concat([res_t[1] for res_t in res_list])
        opt_weight_df = pd.concat([res_t[2] for res_t in res_list])

        # put train output in the model object
        if last_win_only is True:  #IN_mod
            # print ('here2')
            self.model_output_dic[f'test_results_{opt_flag}_last_win'] = {
                'reg_param_df':reg_param_df,
                'exp_exc_rtn_df':exp_exc_rtn_df, 
                'opt_weight_df':opt_weight_df
            }
        else:
            # print ('there2')
            self.model_output_dic[f'test_results_{opt_flag}'] = {
                'reg_param_df':reg_param_df,
                'exp_exc_rtn_df':exp_exc_rtn_df, 
                'opt_weight_df':opt_weight_df
            }
        
        return  reg_param_df, exp_exc_rtn_df, opt_weight_df

    def eval(self, input_df_label, opt_flag, last_win_only, vol_scaler_flag, scaling_vol_tgt, plot_show=True): #IN mod
        # input_df_label should be 'train' or 'test' 
        # opt_flag should be: max_sharpe or target_risk
        # last_win_only is True or False (more for test label)
        
        if input_df_label == 'train':
            input_df = self.train_used
            mkt_df = self.input_data_obj.data_dic['train_mkt']
        elif input_df_label == 'test':
            input_df = self.test_used
            mkt_df = self.input_data_obj.data_dic['test_mkt']

        print(input_df)

        if self.rebal_freq == 'D':
            rebal_label = 'Daily'
        elif self.rebal_freq=='W':
            rebal_label = 'Weekly'
        elif self.rebal_freq=='M':
            rebal_label = 'Monthly'

        if last_win_only is True: #IN mod
            portf_name = f'{rebal_label}-Benchmark-{opt_flag}-{input_df_label}-last window'
            weight_df = self.model_output_dic[f'{input_df_label}_results_{opt_flag}_last_win']['opt_weight_df']
        else: #IN mod
            portf_name = f'{rebal_label}-Benchmark-{opt_flag}-{input_df_label}'
            weight_df = self.model_output_dic[f'{input_df_label}_results_{opt_flag}']['opt_weight_df']

        portf_rtn, portf_mkt_rtn, stats_df, scaler_df, fig_perf = \
                portfolio_performance(input_df, weight_df, portf_name, self.rebal_freq, mkt_df, \
                                    last_win_only, vol_scaler_flag, scaling_vol_tgt, plot_show) #IN mod
        
        # convert column names to ticker # IN mod2
        permno_ticker_dic = self.input_data_obj.data_dic['permno_ticker_dic']
        scaled_weight_df = weight_df.multiply(scaler_df, axis=0).rename(columns=permno_ticker_dic)

        # combine columns with the same name - given the same ticker may have different permno over time
        scaled_weight_df = scaled_weight_df.T.groupby(by=scaled_weight_df.columns).sum().T

        gc.collect()
        
        return portf_rtn, portf_mkt_rtn, stats_df, scaler_df, \
            fig_perf, scaled_weight_df
    
    def save_model_obj(self, directory, checkpoint_name):        
        '''save data'''
        # Save this data locally for later use
        if directory != self.dir:
            self.set_model_directory(directory) ## CHECK TO DO
        print ('\nSaving model object..')
        if checkpoint_name not in os.listdir(directory):
            os.mkdir(checkpoint_name)
        with open(f'./{checkpoint_name}/bm_model_obj_{self.rebal_freq}.pkl','wb') as f: ## CHECK TO DO
            pickle.dump(self, f)
    

def benchmark_run_test(data_directory, data_checkpoint_name, start_date, train_end_date,
                         model_directory, model_checkpoint_name,
                         ticker_list, rebal_freq, opt_flag, last_win_only=False, 
                         target_risk=0.2, force_retrain=True, force_take_new_data=True, verbose=True): #IN_mod

    data_obj = data_unit.data_main(data_directory, data_checkpoint_name, start_date, train_end_date, verbose=False)

    bm_model_obj = Benchmark_model_DL(data_obj, rebal_freq)
    print(bm_model_obj) 

    # if force_retrain is on, retrain, otherwise, check if model exists with given directory and checkpoint name #IN_mod
    # note: force retrain will not create a new model object, just delete the current target model, recreate that part in existing obj
    bm_model_obj.set_model_directory(model_directory) ## CHECK TO DO

    if bm_model_obj.check_checkpoint_model(model_checkpoint_name): #IN_mod
        bm_model_obj = bm_model_obj.load_model(model_checkpoint_name)
        print ('After loading model..')
        print(bm_model_obj)

        if last_win_only is True:
            if f'{opt_flag}_last_win' in bm_model_obj.opt_tested:
                if force_retrain is True:
                    print ('\nForce training is on:')
                    del bm_model_obj.model_output_dic[f'test_results_{opt_flag}_last_win']
                    bm_model_obj.opt_tested.remove(f'{opt_flag}_last_win')
                    bm_model_obj.save_model_obj(model_directory, model_checkpoint_name)
                    if force_take_new_data == True: # IN mod2
                        bm_model_obj.test_used = None
                        bm_model_obj.sel_features_adj=None
                        bm_model_obj.input_data_obj = data_obj
                    print ('\nThe model obj before retraining:')
                    print (bm_model_obj)
                    test_reg_param_df, test_exp_exc_rtn_df, test_opt_weight_df =\
                        bm_model_obj.test(model_checkpoint_name, ticker_list, opt_flag, last_win_only, target_risk, verbose) #IN_mod
                    print ('\nRetraining finished:')
                    print (bm_model_obj)
                    bm_model_obj.save_model_obj(model_directory, model_checkpoint_name)

                else:
                    test_reg_param_df = bm_model_obj.model_output_dic[f'test_results_{opt_flag}_last_win']['reg_param_df']
                    test_exp_exc_rtn_df = bm_model_obj.model_output_dic[f'test_results_{opt_flag}_last_win']['exp_exc_rtn_df']
                    test_opt_weight_df = bm_model_obj.model_output_dic[f'test_results_{opt_flag}_last_win']['opt_weight_df']
            else:
                test_reg_param_df, test_exp_exc_rtn_df, test_opt_weight_df =\
                    bm_model_obj.test(model_checkpoint_name, ticker_list, opt_flag, last_win_only, target_risk, verbose) #IN_mod
                bm_model_obj.save_model_obj(model_directory, model_checkpoint_name)
        else:
            if f'{opt_flag}' in bm_model_obj.opt_tested:
                if force_retrain:
                    print ('\nForce training is on:')
                    del bm_model_obj.model_output_dic[f'test_results_{opt_flag}']
                    bm_model_obj.opt_tested.remove(f'{opt_flag}')
                    bm_model_obj.save_model_obj(model_directory, model_checkpoint_name)
                    if force_take_new_data == True: #IN mod2
                        bm_model_obj.test_used = None
                        bm_model_obj.sel_features_adj=None
                        bm_model_obj.input_data_obj = data_obj
                    print ('\nThe model obj before retraining:')
                    print (bm_model_obj)
                    test_reg_param_df, test_exp_exc_rtn_df, test_opt_weight_df =\
                        bm_model_obj.test(model_checkpoint_name, ticker_list, opt_flag, last_win_only, target_risk, verbose) #IN_mod
                    print ('\nRetraining finished:')
                    print (bm_model_obj)
                    bm_model_obj.save_model_obj(model_directory, model_checkpoint_name)

                else:
                    test_reg_param_df = bm_model_obj.model_output_dic[f'test_results_{opt_flag}_last_win']['reg_param_df']
                    test_exp_exc_rtn_df = bm_model_obj.model_output_dic[f'test_results_{opt_flag}_last_win']['exp_exc_rtn_df']
                    test_opt_weight_df = bm_model_obj.model_output_dic[f'test_results_{opt_flag}_last_win']['opt_weight_df']
            else:
                test_reg_param_df, test_exp_exc_rtn_df, test_opt_weight_df =\
                    bm_model_obj.test(model_checkpoint_name, ticker_list, opt_flag, last_win_only, target_risk, verbose) #IN_mod
                bm_model_obj.save_model_obj(model_directory, model_checkpoint_name)
    else:
        test_reg_param_df, test_exp_exc_rtn_df, test_opt_weight_df =\
                bm_model_obj.test(model_checkpoint_name, ticker_list, opt_flag, last_win_only, target_risk, verbose) #IN_mod
        bm_model_obj.save_model_obj(model_directory, model_checkpoint_name)

    print(bm_model_obj)
    print('test', bm_model_obj.test_used)         

    return bm_model_obj, test_reg_param_df, test_exp_exc_rtn_df, test_opt_weight_df

