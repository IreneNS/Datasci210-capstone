import os
from io import BytesIO
import boto3
import logging
import importlib
from joblib import Parallel, delayed
import warnings
from datetime import datetime
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from pypfopt import (
    EfficientFrontier,
    risk_models,
    plotting,
)
import cvxpy as cp
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.covariance import LedoitWolf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import pdb
import gc

s3 = boto3.client('s3')

bucket_name = "capstone-general"
def read_s3_file(file):
    try:
        # Fetch the object from S3
        response = s3.get_object(Bucket=bucket_name, Key=file)
        
        # Read the file content as text
        file_content = response['Body'].read()
        
        return file_content
        
    except Exception as e:
        print(f"Error reading file from S3: {e}")
        return None

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
                 solver_name='CLARABEL', market_neutral=False, risk_free_rate=None):
    ef_ob = EfficientFrontier(ann_exp_exc_rtn, ann_cov_matrix)
    ef_ob._solver = solver_name # IN: 'CLARABEL', this solver works for max_sharpe choice, and likely better for 'target_risk' too
    # ef_ob._solver_options ={'ECOS':cp.ECOS}
    # ef_ob._solver = 'ECOS'

    if opt_flag == 'target_risk':
        ef_ob.efficient_risk(target_volatility=target_risk, market_neutral=market_neutral)
    elif opt_flag == 'max_sharpe':
        if risk_free_rate:
            # print ('here')
            # ef_ob.risk_free_rate = risk_free_rate # set risk free rate as zero; default: 0.02 in def max_sharpe(self, risk_free_rate=0.02)
            ef_ob.max_sharpe(risk_free_rate = risk_free_rate)
        else:
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

def mmd_cal(df, return_col_name):
    df = df.copy()
    df['cum_rtn']=(1+df[return_col_name]).cumprod()
    df['drawdown'] = (df['cum_rtn']-df['cum_rtn'].cummax())/df['cum_rtn'].cummax()
    df['max_drawdown'] =  df['drawdown'].cummin()
    return df['max_drawdown']

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

    if vol_scaler_flag:
        portf_rtn_0 = portf_rtn.copy()
        portf_rtn = (portf_rtn_0*(scaling_vol_tgt/np.sqrt(252))/(portf_rtn_0.rolling(60, min_periods=2).std())).fillna(0) #IN mod2
        scaler_df = portf_rtn.div(portf_rtn_0, axis=0)
        # scaled_mkt = mkt_df['return_sp']*scaling_vol_tgt/mkt_df['return_sp'].rolling(60).std()
        # unscaled_mkt = mkt_df['return_sp']
        unscaled_mkt = mkt_df['return_sp'].loc[portf_rtn.first_valid_index():]

    else:
        scaler_df = portf_rtn.div(portf_rtn, axis=0)
        # scaled_mkt = mkt_df['return_sp']*(portf_rtn.rolling(60).std()/mkt_df['return_sp'].rolling(60).std())
        # unscaled_mkt = mkt_df['return_sp']
        unscaled_mkt = mkt_df['return_sp'].loc[portf_rtn.first_valid_index():]
    
    fig1, ax1 = plt.subplots(1,2, figsize=(11,3.5))
    portf_mkt_rtn = pd.concat([portf_rtn.rename(portf_name), unscaled_mkt.rename('Unscaled Market')], axis=1)
    portf_mkt_rtn.cumsum().plot(ax=ax1[0])
    ax1[0].set_title(f'Cumulative Return Comparison')
    ax1[0].legend(loc='upper left')
    # plt.legend()
    # plt.show()

    if last_win_only is True:
        (portf_mkt_rtn.rolling(60).std()*np.sqrt(252)).plot(
            ax=ax1[1],
            title='Rolling Annual Vol Comparison')
        ax1[1].legend(bbox_to_anchor=(1,1))
    else:
        (portf_mkt_rtn.rolling(252).std()*np.sqrt(252)).plot(
        ax=ax1[1],
        title='Rolling Annual Vol Comparison')
        ax1[1].legend(bbox_to_anchor=(1,1))

    fig1.suptitle(f'{portf_name} vs (Unscaled) S&P500 Cumulative Return and Rolling Vol Comparison')
    plt.subplots_adjust(top=0.85, bottom=0.01, wspace=0.2)
    if plot_show is True: #IN mod
        plt.show()
    plt.close()

    stats_df = pd.DataFrame(columns=portf_mkt_rtn.columns)
    stats_df.loc['avg_rtn_ann',:] = portf_mkt_rtn.mean()*252
    stats_df.loc['vol_ann',:] = portf_mkt_rtn.std()*np.sqrt(252)
    stats_df.loc['sharpe_ann',:] = stats_df.loc['avg_rtn_ann',:]/stats_df.loc['vol_ann',:]
    stats_df.loc['max_drawdown',portf_name] = mmd_cal(portf_mkt_rtn, portf_name).iloc[-1]
    stats_df.loc['max_drawdown','Unscaled Market'] = mmd_cal(portf_mkt_rtn, 'Unscaled Market').iloc[-1]
    
    if plot_show is True: #IN mod
        print(stats_df)    

    return portf_rtn, portf_mkt_rtn, stats_df, scaler_df, fig1

def performance_comparison_all_recal(portf_rtn_df_l, mkt_rtn,\
                               rebal_freq, last_win_only=False, plot_show=True):
    # by the process the portf series should already have unique names except unscaled mkt
    portf_mkt_rtn_comb = pd.concat(portf_rtn_df_l + [mkt_rtn], axis=1)

    fig1, ax1 = plt.subplots(1,2, figsize=(11,3.5))
    
    # portf_mkt_rtn = pd.concat([portf_rtn.rename(portf_name), unscaled_mkt.rename('Unscaled Market')], axis=1)
    portf_mkt_rtn_comb.cumsum().plot(ax=ax1[0])
    ax1[0].set_title(f'Cumulative Return Comparison')
    ax1[0].legend(loc='upper left')

    # if last_win_only is True:
    if portf_mkt_rtn_comb.index[0]>=datetime.strptime('2021-01-01', '%Y-%m-%d'):
        (portf_mkt_rtn_comb.rolling(60).std()*np.sqrt(252)).plot(
            ax=ax1[1],
            title='Rolling Annual Vol Comparison')
        ax1[1].legend(bbox_to_anchor=(1,1))
    else:
        (portf_mkt_rtn_comb.rolling(252).std()*np.sqrt(252)).plot(
        ax=ax1[1],
        title='Rolling Annual Vol Comparison')
        ax1[1].legend(bbox_to_anchor=(1,1))

    if (len(portf_mkt_rtn_comb.loc['2021-01-01':,:])>0) and last_win_only is False: 
        ax1[0].axvline(x='2020-12-31', color='red', linestyle='--', linewidth=1.5, label='Train vs Test Division')
        ax1[1].axvline(x='2020-12-31', color='red', linestyle='--', linewidth=1.5, label='Train vs Test Division')
        ax1[0].legend(loc='upper left')
        ax1[1].legend(bbox_to_anchor=(1,1))
        
    fig1.suptitle(f'Proposed Portforlio ({rebal_freq}) vs (Unscaled) S&P500 Cumulative Return and Rolling Vol Comparison')
    plt.subplots_adjust(top=0.85, bottom=0.01, wspace=0.2)
    if plot_show is True: #IN mod
        plt.show()
    plt.close()

    # recalculate stats table
    stats_df_comb = pd.DataFrame(columns=portf_mkt_rtn_comb.columns)
    stats_df_comb.loc['avg_rtn_ann',:] = portf_mkt_rtn_comb.mean()*252
    stats_df_comb.loc['vol_ann',:] = portf_mkt_rtn_comb.std()*np.sqrt(252)
    stats_df_comb.loc['sharpe_ann',:] = stats_df_comb.loc['avg_rtn_ann',:]/stats_df_comb.loc['vol_ann',:]

    for col in portf_mkt_rtn_comb.columns:
        stats_df_comb.loc['max_drawdown',col] = mmd_cal(portf_mkt_rtn_comb, col).iloc[-1]

    # stats_df_comb = stats_df_comb.T.sort_values(by='sharpe_ann', ascending=False).T
    
    if plot_show is True:
        print(stats_df_comb)
    
    return portf_mkt_rtn_comb, stats_df_comb, fig1
    

def check_weights_plots(opt_weight_df):
    # check basic properties for optimization weights:
    np.round(opt_weight_df.sum(axis=1),2).plot(title='Total weights for the portfolio over time', figsize=(5,3))
    plt.show()

    opt_weight_df.count(axis=1).plot(title='Total number of stocks over time (not all have non zero weight)', figsize=(5,3))
    plt.show()

    opt_weight_df[opt_weight_df>0].count(axis=1).plot(title='Total number of stocks with non-zero weight over time', figsize=(5,3))
    plt.show()


def load_train_val_dataset(model_dir, version_no, is_s3=False):
    # full_dir = f'{model_dir}{model_checkpoint}/{rebal_freq}/' ## for sm ; folder: 'monthly', 'weekly', 'daily'

    if is_s3:
        train_file_key = f'{model_dir}train_dataset_dl_{version_no}.pkl'
        train_dataset = pickle.load(BytesIO(read_s3_file(train_file_key)))
        val_file_key = f'{model_dir}val_dataset_dl_{version_no}.pkl'
        val_dataset = pickle.load(BytesIO(read_s3_file(val_file_key)))
    else:
        with open(f'{model_dir}train_dataset_dl_{version_no}.pkl', 'rb') as f:
            train_dataset = pickle.load(f)
        with open(f'{model_dir}val_dataset_dl_{version_no}.pkl', 'rb') as f:
            val_dataset = pickle.load(f)

    return train_dataset, val_dataset


def load_test_dataset(model_dir, version_no, is_s3=False):
    # full_dir = f'{model_dir}{model_checkpoint}/{rebal_freq}/' ## for sm ; folder: 'monthly', 'weekly', 'daily'
    if is_s3:
        test_file_key = f'{model_dir}test_dataset_dl_{version_no}.pkl'
        test_dataset = pickle.load(BytesIO(read_s3_file(test_file_key)))
    else:
        with open(f'{model_dir}test_dataset_dl_{version_no}.pkl', 'rb') as f:
            test_dataset = pickle.load(f)
    return test_dataset