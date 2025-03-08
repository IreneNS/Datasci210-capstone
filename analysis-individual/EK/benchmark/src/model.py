from pathlib import Path
import pandas as pd
import numpy as np
import os
import sys
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
warnings.filterwarnings('default')
from sklearn.covariance import LedoitWolf
from joblib import Parallel, delayed
import time
import openpyxl
import src.ds210_ml_portf_data_retrieval

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
        print(os.listdir())
        print(checkpoint_name)
        if checkpoint_name not in os.listdir():
            return False
        elif f'bm_model_obj_{self.rebal_freq}.pkl' in os.listdir(os.path.join(self.dir, checkpoint_name)):
            print('found checkpoint')
            self.checkpoint_name = checkpoint_name
            print(self.checkpoint_name)
            return True
        else:
            return False

    # def load_model(self, checkpoint_name):
    #     if self.check_checkpoint_model(checkpoint_name):
    #         path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
    #                             f'dependencies/benchmark_model/{checkpoint_name}/bm_model_obj_{self.rebal_freq}.pkl')
    #         print(path)

    #         class CustomUnpickler(pickle.Unpickler):
    #             def find_class(self, module, name):
    #                 if name == 'Benchmark_model_DL':
    #                     from src.model import Benchmark_model_DL
    #                     return Benchmark_model_DL
    #                 return super().find_class(module, name)

    #         with open(path, 'rb') as f:
    #             unpickler = CustomUnpickler(f)  # Create the unpickler with the file
    #             bm_model_obj = unpickler.load()  # Then call load on the unpickler
    #         return bm_model_obj  # IN: or should it be just return bm_model_obj?
    #     else:
    #         raise KeyError ('Model result with given checkpoint name does not exist in given directory.')



    def load_model(self, checkpoint_name):
        if self.check_checkpoint_model(checkpoint_name):
            path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                            f'dependencies/benchmark_model/{checkpoint_name}/bm_model_obj_{self.rebal_freq}.pkl')
            print(path)
            
            # 1. First make sure your current directory structure is in sys.path
            current_dir = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.dirname(current_dir)
            
            if current_dir not in sys.path:
                sys.path.insert(0, current_dir)
            if parent_dir not in sys.path:
                sys.path.insert(0, parent_dir)
                
            # 2. Create and register the module manually
            import types
            import importlib.util
            
            # Try to create module from the actual file
            module_path = os.path.join(current_dir, "ds210_ml_portf_data_retrieval.py")
            if os.path.exists(module_path):
                spec = importlib.util.spec_from_file_location("ds210_ml_portf_data_retrieval", module_path)
                if spec:
                    module = importlib.util.module_from_spec(spec)
                    sys.modules["ds210_ml_portf_data_retrieval"] = module
                    spec.loader.exec_module(module)
                    print("Module loaded from:", module_path)
            else:
                # If file not found at expected location
                print(f"Module file not found at {module_path}")
                # Create empty module as fallback
                module = types.ModuleType("ds210_ml_portf_data_retrieval")
                sys.modules["ds210_ml_portf_data_retrieval"] = module
                
            # 3. Create a custom unpickler that can handle model mapping
            class CustomUnpickler(pickle.Unpickler):
                def find_class(self, module_name, class_name):
                    # Special handling for our module
                    if module_name == 'ds210_ml_portf_data_retrieval':
                        # If it's looking for classes from our troublesome module
                        if class_name in globals():
                            return globals()[class_name]
                        # Try to get it from the manually created module
                        elif hasattr(sys.modules["ds210_ml_portf_data_retrieval"], class_name):
                            return getattr(sys.modules["ds210_ml_portf_data_retrieval"], class_name)
                        # Last attempt - try the src version
                        try:
                            import src.ds210_ml_portf_data_retrieval as src_module
                            if hasattr(src_module, class_name):
                                return getattr(src_module, class_name)
                        except ImportError:
                            pass
                    # For Benchmark_model_DL specific case
                    elif class_name == 'Benchmark_model_DL':
                        # Return the current class
                        return self.__class__.__class__
                    
                    # Default behavior
                    return super().find_class(module_name, class_name)
                    
            try:
                with open(path, 'rb') as f:
                    unpickler = CustomUnpickler(f)
                    bm_model_obj = unpickler.load()
                return bm_model_obj
            except Exception as e:
                print(f"Unpickling error: {str(e)}")
                # Add debugging
                import traceback
                traceback.print_exc()
                raise





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
        print(rebal_dates)
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
        self.opt_tested.append(opt_flag)

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
        if last_win_only:
            if rebal_freq == 'D':
                rebal_dates = dates_train_daily[-1]
            elif rebal_freq == 'W':
                rebal_dates = dates_train_weekly[-1]
            elif rebal_freq == 'M':
                rebal_dates = dates_train_monthly[-1]
        else:
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
        print ('Start parallel running..', rebal_dates)
        res_list = Parallel(n_jobs=-1)(delayed(
            regression_n_opt_t)(date, dates_train_daily, rolling_ana_data_used, opt_flag, target_risk)
            for date in rebal_dates
        )

        # Organize the results
        reg_param_df = pd.concat([res_t[0] for res_t in res_list])
        exp_exc_rtn_df = pd.concat([res_t[1] for res_t in res_list])
        opt_weight_df = pd.concat([res_t[2] for res_t in res_list])

        # put train output in the model object
        if last_win_only:
            self.model_output_dic[f'test_results_{opt_flag}_last_win'] = {
                'reg_param_df':reg_param_df,
                'exp_exc_rtn_df':exp_exc_rtn_df, 
                'opt_weight_df':opt_weight_df
            }
        else:
            self.model_output_dic[f'test_results_{opt_flag}'] = {
                'reg_param_df':reg_param_df,
                'exp_exc_rtn_df':exp_exc_rtn_df, 
                'opt_weight_df':opt_weight_df
            }
        
        return  reg_param_df, exp_exc_rtn_df, opt_weight_df

    def eval(self, input_df_label, opt_flag, last_win_flag, vol_scaler_flag, scaling_vol_tgt):
        # input_df_label should be 'train' or 'test' 
        # opt_flag should be: max_sharpe or target_risk
        # last_win_flag is True or False (more for test label)
        
        if input_df_label == 'train':
            input_df = self.train_used
            mkt_df = self.input_data_obj.data_dic['train_mkt']
        elif input_df_label == 'test':
            input_df = self.test_used
            mkt_df = self.input_data_obj.data_dic['test_mkt']

        weight_df = self.model_output_dic[f'{input_df_label}_results_{opt_flag}']['opt_weight_df']

        if self.rebal_freq == 'D':
            rebal_label = 'Daily'
        elif self.rebal_freq=='W':
            rebal_label = 'Weekly'
        elif self.rebal_freq=='M':
            rebal_label = 'Monthly'

        if last_win_flag==True:
            portf_name = f'{rebal_label}-Benchmark-{opt_flag}-{input_df_label}-last window'
        else:
            portf_name = f'{rebal_label}-Benchmark-{opt_flag}-{input_df_label}'

        portf_rtn, portf_mkt_rtn, stats_df, scaler_df = \
            portfolio_performance(input_df, weight_df, portf_name, self.rebal_freq, mkt_df, \
                                  vol_scaler_flag, scaling_vol_tgt)
        
        return portf_rtn, portf_mkt_rtn, stats_df, scaler_df
    
    def save_model_obj(self, directory, checkpoint_name):        
        '''save data'''
        # Save this data locally for later use
        if directory != self.dir:
            self.set_model_directory(directory)
        print ('\nSaving model object..')
        if checkpoint_name not in os.listdir(directory):
            os.mkdir(checkpoint_name)
        with open(f'./{checkpoint_name}/bm_model_obj_{self.rebal_freq}.pkl','wb') as f:
            pickle.dump(self, f)

class Data_NN:
    '''this is the class to prepare and retreive NN (DL) used data
    the class will check the existence of checkpoint data first, 
    if not exist, then retrieve new version of data'''

    '''note: if you will create the data with new checkpoint name, you will
    need to put in WRDS user name, password, and "Y"'''

    def __init__(self, checkpoint_name='', start_date='2000-01-01', train_end_date='2020-12-31'):
        self.checkpoint_name = checkpoint_name
        self.data_dic = {}
        self.start_date = start_date
        self.train_end_date = train_end_date
         
    def __repr__(self):
        return f'\nThe checkpoint name of this data is {self.checkpoint_name}, \nand data_dic keys are {list(self.data_dic.keys())}'
    
    def set_directory(self, data_directory_path):
        #IN: note this directory_path should be where the data is at
        self.dir = data_directory_path
        os.chdir(data_directory_path)

    def check_checkpoint(self, checkpoint_name):
        print(os.listdir())
        if checkpoint_name in os.listdir():
            self.checkpoint_name = checkpoint_name
            return True
        else:
            return False 
        
    def retrieve_data(self, checkpoint_name):
        if self.check_checkpoint(checkpoint_name):
            with open(f'./{checkpoint_name}/data_useful_info_dic.pkl','rb') as f:
                data_useful_info_dic = pickle.load(f)
            with open(f'./{checkpoint_name}/sp500_used.pkl','rb') as f:
                sp500_used = pickle.load(f)
            # Data at mkt level
            with open(f'./{checkpoint_name}/mkt_daily.pkl','rb') as f:
                mkt_daily = pickle.load(f)

            # train/test split data
            with open(f'./{checkpoint_name}/train_sp500.pkl','rb') as f:
                train_sp500 = pickle.load(f)

            with open(f'./{checkpoint_name}/test_sp500.pkl','rb') as f:
                test_sp500 = pickle.load(f)

            with open(f'./{checkpoint_name}/train_mkt.pkl','rb') as f:
                train_mkt = pickle.load(f)
            
            with open(f'./{checkpoint_name}/test_mkt.pkl','rb') as f:
                test_mkt = pickle.load(f)
            
            with open(f'./{checkpoint_name}/ticker_permno_dic.pkl','rb') as f:
                ticker_permno_dic = pickle.load(f)
            
            with open(f'./{checkpoint_name}/permno_ticker_dic.pkl','rb') as f:
                permno_ticker_dic = pickle.load(f)

            self.data_dic= {'data_useful_info_dic': data_useful_info_dic,
                    'sp500_used': sp500_used,
                    'mkt_daily':mkt_daily,
                    'train_sp500': train_sp500,
                    'test_sp500': test_sp500,
                    'train_mkt': train_mkt,
                    'test_mkt': test_mkt,
                    'ticker_permno_dic': ticker_permno_dic,
                    'permno_ticker_dic': permno_ticker_dic}
        
        else:
            self.prepare_data(self.start_date, checkpoint_name)
            self.checkpoint_name = checkpoint_name

            with open(f'./{checkpoint_name}/data_useful_info_dic.pkl','rb') as f:
                data_useful_info_dic = pickle.load(f)
            with open(f'./{checkpoint_name}/sp500_used.pkl','rb') as f:
                sp500_used = pickle.load(f)
            # Data at mkt level
            with open(f'./{checkpoint_name}/mkt_daily.pkl','rb') as f:
                mkt_daily = pickle.load(f)

            # train/test split data
            with open(f'./{checkpoint_name}/train_sp500.pkl','rb') as f:
                train_sp500 = pickle.load(f)

            with open(f'./{checkpoint_name}/test_sp500.pkl','rb') as f:
                test_sp500 = pickle.load(f)

            with open(f'./{checkpoint_name}/train_mkt.pkl','rb') as f:
                train_mkt = pickle.load(f)
            
            with open(f'./{checkpoint_name}/test_mkt.pkl','rb') as f:
                test_mkt = pickle.load(f)

            with open(f'./{checkpoint_name}/ticker_permno_dic.pkl','rb') as f:
                ticker_permno_dic = pickle.load(f)
            
            with open(f'./{checkpoint_name}/permno_ticker_dic.pkl','rb') as f:
                permno_ticker_dic = pickle.load(f)

            self.data_dic= {'data_useful_info_dic': data_useful_info_dic,
                    'sp500_used': sp500_used,
                    'mkt_daily':mkt_daily,
                    'train_sp500': train_sp500,
                    'test_sp500': test_sp500,
                    'train_mkt': train_mkt,
                    'test_mkt': test_mkt,
                    'ticker_permno_dic': ticker_permno_dic,
                    'permno_ticker_dic': permno_ticker_dic}


    def prepare_data(self, start_date, checkpoint_name):
        '''establish connection'''
        wrds_db = wrds.Connection()

        '''Downloading the JKP data (Firm Characteristic Data) from WRDS'''
        print ('Downloading the JKP data (Firm Characteristic Data) from WRDS...')
        #IN: second version where the first five factors are based on paper
        char_chosen =['market_equity','be_me','at_gr1','ope_be','ret_12_1', # FF-5-factor
                    'ret_6_1','ret_3_1', # other mmt
                    'ret_60_12', 'ret_1_0'] # reversal

        cty_chosen = ['USA']

        sql_query= f"""
                SELECT id, eom, excntry, gvkey, permno, size_grp, me, ret_exc_lead1m, {','.join(map(str, char_chosen))}
                        FROM contrib.global_factor
                        WHERE common=1 and exch_main=1 and primary_sec=1 and obs_main=1 and
                        excntry in ({','.join("'"+str(x)+"'" for x in cty_chosen)}) and eom>=CAST('{start_date}' AS DATE)
                """
        char_data = wrds_db.raw_sql(sql_query)
        char_data = char_data.apply(lambda x: x.astype(float) if x.dtype=='Float64' else x, axis=0)
        #IN: added to make sure types are 'float64' instead of 'Float64'

        '''Download S&P500 constituents From CRSP and Compustat'''
        print ('Download S&P500 constituents From CRSP and Compustat..')
        '''Get S&P constituents in-index date range and respective return'''
        print ('Get S&P constituents in-index date range and respective return..')

        start_date_sp = start_date
        # IN: crsp.msp500list is monthly data, crsp.dsp500list is daily data; similarly crsp.msp is monthly data; crsp.dsp is daily data

        sp500_daily = wrds_db.raw_sql(f"""
                                select a.*, b.date, b.ret, b.vol
                                from crsp.dsp500list as a,
                                crsp.dsf as b
                                where a.permno=b.permno
                                and b.date >= a.start and b.date<= a.ending
                                and b.date>=CAST('{start_date_sp}' AS DATE)
                                order by date;
                                """, date_cols=['start', 'ending', 'date'])
        
        '''Get Identifier and Descriptive Data From CRSP and Compustat for merges'''
        print ('Get Identifier and Descriptive Data From CRSP and Compustat for merges..')
        # Add Other Descriptive Variables from CRSP

        mse = wrds_db.raw_sql("""
                                select comnam, ncusip, namedt, nameendt, 
                                permno, shrcd, exchcd, hsiccd, ticker
                                from crsp.msenames
                                """, date_cols=['namedt', 'nameendt'])

        # if nameendt is missing then set to today date
        mse['nameendt']=mse['nameendt'].fillna(pd.to_datetime('today'))

        # Merge with SP500 universe data
        sp500_crsp = pd.merge(sp500_daily, mse, how = 'left', on = 'permno')

        # Impose the date range restrictions
        sp500_crsp = sp500_crsp.loc[(sp500_crsp.date>=sp500_crsp.namedt) \
                                    & (sp500_crsp.date<=sp500_crsp.nameendt)]
        
        # Linking with Compustat through CCM
        ccm=wrds_db.raw_sql("""
                        select gvkey, liid as iid, lpermno as permno, linktype, linkprim, 
                        linkdt, linkenddt
                        from crsp.ccmxpf_linktable
                        where substr(linktype,1,1)='L'
                        and (linkprim ='C' or linkprim='P')
                        """, date_cols=['linkdt', 'linkenddt'])

        # if linkenddt is missing then set to today date
        ccm['linkenddt']=ccm['linkenddt'].fillna(pd.to_datetime('today'))

        # Merge the CCM data with S&P500 data
        # First just link by matching PERMNO
        sp500_crsp_ccm = pd.merge(sp500_crsp, ccm, how='left', on=['permno'])

        # Then set link date bounds
        sp500_crsp_ccm = sp500_crsp_ccm.loc[(sp500_crsp_ccm['date']>=sp500_crsp_ccm['linkdt'])\
                                &(sp500_crsp_ccm['date']<=sp500_crsp_ccm['linkenddt'])]
        sp500_crsp_ccm = sp500_crsp_ccm.apply(lambda x: x.astype(float) if x.dtype=='Float64' else x, axis=0)
        #IN: added to make sure types are 'float64' instead of 'Float64'

        '''Merge S&P500 Universe CRSP, Compustat Data with Firm Characteristic data'''
        print ('Merge S&P500 Universe CRSP, Compustat Data with Firm Characteristic data..')

        '''Prepare char_data to be merged'''
        print ('preparing char_data to be merged')
        # IN - get the index we need for char_data from sp500
        sp500_crsp_ccm_adj = sp500_crsp_ccm.copy().drop_duplicates(subset=['date','permno'])
        sp500_crsp_ccm_adj = sp500_crsp_ccm_adj.set_index(['date','permno'])
        daily_index = pd.date_range(start=sp500_crsp_ccm_adj.index.get_level_values('date').min(),
                                    end=sp500_crsp_ccm_adj.index.get_level_values('date').max(),
                                    freq='D')
        
        # IN: Use sp500_crsp_ccm_adj's extended index with daily frequency (note not BD) to reindex char_data
        char_data_adj = char_data.copy()[['permno','eom']+char_chosen+['ret_exc_lead1m']].dropna(subset=['permno'])
        char_data_adj['date'] = pd.to_datetime(char_data_adj['eom'])

        char_data_adj = char_data_adj.set_index(['date','permno'])\
            .reindex(pd.MultiIndex.from_product([daily_index, sp500_crsp_ccm_adj.index.get_level_values('permno').unique()],
                                                                    names=['date','permno']))\
                                                                    .groupby('permno').ffill().reset_index()
        
        '''Merge sp500 data with reindexed firm characteristic data'''
        print ('Merge sp500 data with prepared firm characteristic data...')    
        sp500_comb = pd.merge(sp500_crsp_ccm, char_data_adj, on=['date','permno'], how='left').sort_values(by=['permno','date']).reset_index()

        '''prepare data to be used for analysis'''
        print ('Prepare data to be used for later analysis..')

        #IN: important columns information:
        useful_columns = ['permno','date','comnam','ncusip','ticker','gvkey','hsiccd','ret','vol','ret_exc_lead1m']+char_chosen
        name_map = {'comnam':'company_name',
                    'hsiccd':'industry_code',
                    'ret':'return',
                    'vol':'volume',
                    'ret_exc_lead1m':'return_lead1m',
                    'be_me':'book_to_equity',
                    'at_gr1':'asset_growth',
                    'ope_be':'opt_to_book_eq'}

        time_columns = ['date']
        identifier_columns = [name_map[x] if x in name_map else x for x in ['permno','ncusip','ticker','gvkey']]
        categorical_columns = [name_map[x] if x in name_map else x for x in ['comnam','hsiccd']]
        numerical_columns = [name_map[x] if x in name_map else x for x in char_chosen + ['ret','vol','ret_exc_lead1m']]

        sp500_used = sp500_comb[useful_columns].rename(columns=name_map).drop_duplicates(subset=['permno','date'])

        '''initial check of data'''
        prelim_check_data(sp500_used, 'sp500_used', checkpoint_name=None)

        '''add selective daily metrics'''
        print ('Add selective daily metrics..')
        ## calculating the daily metrics
        ret_12m_d = sp500_used.set_index(['permno','date'])['return']\
            .groupby('permno').apply(lambda x: (x.rolling(21*12).sum()).droplevel(0)).rename('ret_12m_d').reset_index()

        ret_6m_d = sp500_used.set_index(['permno','date'])['return']\
            .groupby('permno').apply(lambda x: (x.rolling(21*6).sum()).droplevel(0)).rename('ret_6m_d').reset_index()

        ret_3m_d = sp500_used.set_index(['permno','date'])['return']\
            .groupby('permno').apply(lambda x: (x.rolling(21*3).sum()).droplevel(0)).rename('ret_3m_d').reset_index()

        ret_1m_d = sp500_used.set_index(['permno','date'])['return']\
            .groupby('permno').apply(lambda x: (x.rolling(21).sum()).droplevel(0)).rename('ret_1m_d').reset_index()

        ## adding them to sp500 data set
        print(f'sp500_used shape before adding daily metrics: {sp500_used.shape}')
        for data_d in [ret_12m_d, ret_6m_d, ret_3m_d, ret_1m_d]:
            sp500_used = pd.merge(sp500_used, data_d, on=['permno','date'], how='left')
        print(f'sp500_used shape after adding daily metrics: {sp500_used.shape}')

        ## update numerical columns to include the new daily metrics
        numerical_columns = numerical_columns + ['ret_12m_d','ret_6m_d','ret_3m_d','ret_1m_d']

        '''Further data cleaning'''
        print ('Further data cleaning..')
        print ('Clean outliers based on preliminary checks (on operating profit on book equity)..')

        print(f'sp500_used shape before outlier cleanng: {sp500_used.shape}')
        
        abnormal_threshold = 30 
        sp500_used.loc[sp500_used['opt_to_book_eq']>abnormal_threshold, 'opt_to_book_eq'] = np.nan
        temp_ffill = sp500_used.set_index(['permno','date'])['opt_to_book_eq'].groupby('permno').ffill().reset_index()
        sp500_used = pd.merge(sp500_used.drop(columns=['opt_to_book_eq']), temp_ffill, on=['permno','date'], how='left')

        print(f'sp500_used shape after outlier cleanng: {sp500_used.shape}')

        # Packing important inputs for later analysis for saving
        data_useful_info_dic={}

        data_useful_info_dic['useful_columns_raw'] = useful_columns
        data_useful_info_dic['name_map'] = name_map
        data_useful_info_dic['time_columns'] = time_columns
        data_useful_info_dic['identifier_columns'] = identifier_columns
        data_useful_info_dic['categorical_columns'] = categorical_columns
        data_useful_info_dic['numerical_columns'] = numerical_columns

        # plot checking resulting data
        print ('S&P data plot after processing..')
        block_distribution_plot(sp500_used, numerical_columns, 'Numerical Variables', n_per_row=4)
        block_time_series_plot(sp500_used, numerical_columns, 'Numerical Variables', n_per_row=4)

        # split train and test data
        print ('Split train and test data - S&P500..')
        train_sp500 = sp500_used[sp500_used['date']<=self.train_end_date]
        test_sp500 = sp500_used[sp500_used['date']>self.train_end_date]

        '''Retrieve market level info'''
        print ('Retrieve market level data..')
        print ('Retrieve S&P data..')
        # Download S&P 500 historical data
        sp500_mkt = yf.download('^GSPC', start='1990-01-01', end='2024-12-31')
        sp500_mkt = sp500_mkt.droplevel(1, axis=1)
        sp500_mkt.index = pd.to_datetime(sp500_mkt.index) #IN: to make sure it's datetime index

        # Calculate daily returns
        sp500_mkt['return'] = sp500_mkt['Close'].pct_change()
        sp500_mkt = mmd_cal(sp500_mkt)

        print('Retrieve Treasury data..')
        # Treasury data
        start_date_tsy = '2000-01-01'
        end_time_tsy = '2023-01-01'

        tickers = {
            "tsy10yr": "^TNX",
            "tsy5yr": "^FVX",
            "tsy2yr": "^IRX"  # Note: ^IRX is for 13-week T-bills, but you can replace it with the appropriate 2-year ticker if available
        }

        # Fetch historical yield data
        data = {}
        for bond, ticker in tickers.items():
            data[bond] = yf.download(ticker, start=start_date_tsy, end=end_time_tsy).droplevel(1, axis=1)["Close"]

        # Combine data into a single DataFrame
        yield_data = pd.DataFrame(data)
        yield_data.index = pd.to_datetime(yield_data.index) 

        print('Combine market level data')
        mkt_daily = pd.concat([sp500_mkt\
                       .rename(columns=dict(zip(sp500_mkt.columns,
                                        [f'{x}_sp' for x in sp500_mkt.columns]))), 
                                        yield_data/100], axis=1).loc[start_date_tsy:end_time_tsy,:]
        print ('Mkt data plot..')
        mkt_daily[['return_sp','tsy10yr','tsy2yr','tsy5yr']].plot(title='Market Data Plot')
        plt.show()

        # split market level data into train and test
        print ('Split train and test data - Market data..')
        train_mkt = mkt_daily.loc[:self.train_end_date,:]
        test_mkt = mkt_daily.loc[self.train_end_date:,:]

        '''make permno-ticker maps'''
        multi_index = train_sp500.set_index(['ticker','permno']).index.unique()
        ticker_permno_dic = {item[0]:item[1] for item in multi_index}
        permno_ticker_dic = {item[1]:item[0] for item in multi_index}

        '''save data'''
        # Save this data locally for later use
        print ('Saving data..')
        os.mkdir(checkpoint_name)
        with open(f'./{checkpoint_name}/char_data.pkl','wb') as f:
            pickle.dump(char_data, f)
        
        with open(f'./{checkpoint_name}/sp500_crsp.pkl','wb') as f:
            pickle.dump(sp500_crsp, f)

        with open(f'./{checkpoint_name}/sp500_crsp_ccm.pkl','wb') as f:
            pickle.dump(sp500_crsp_ccm, f)

        with open(f'./{checkpoint_name}/sp500_comb.pkl','wb') as f:
            pickle.dump(sp500_comb, f)

        with open(f'./{checkpoint_name}/sp500_used.pkl','wb') as f:
            pickle.dump(sp500_used, f)
        
        with open(f'./{checkpoint_name}/data_useful_info_dic.pkl','wb') as f:
            pickle.dump(data_useful_info_dic, f)

        with open(f'./{checkpoint_name}/mkt_daily.pkl','wb') as f:
            pickle.dump(mkt_daily, f)

        with open(f'./{checkpoint_name}/train_sp500.pkl','wb') as f:
            pickle.dump(train_sp500, f)

        with open(f'./{checkpoint_name}/test_sp500.pkl','wb') as f:
            pickle.dump(test_sp500, f)

        with open(f'./{checkpoint_name}/train_mkt.pkl','wb') as f:
            pickle.dump(train_mkt, f)

        with open(f'./{checkpoint_name}/test_mkt.pkl','wb') as f:
            pickle.dump(test_mkt, f)

        with open(f'./{checkpoint_name}/ticker_permno_dic.pkl','wb') as f:
            pickle.dump(ticker_permno_dic, f)

        with open(f'./{checkpoint_name}/permno_ticker_dic.pkl','wb') as f:
            pickle.dump(permno_ticker_dic, f)
            
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

def mmd_cal(df, return_col_name):
    df['cum_rtn']=(1+df[return_col_name]).cumprod()
    df['drawdown'] = (df['cum_rtn']-df['cum_rtn'].cummax())/df['cum_rtn'].cummax()
    df['max_drawdown'] =  df['drawdown'].cummin()
    return df['max_drawdown']

def portfolio_performance(input_df, weight_df, portf_name, rebal_freq, mkt_df, vol_scaler_flag=False, scaling_vol_tgt=0.3):
    # rebal_freq: 'D','M','W'
    if rebal_freq =='D':
        ff_n = 2
    elif rebal_freq == 'W':
        ff_n = 7
    elif rebal_freq == 'M':
        ff_n = 31
        
    return_df = input_df.set_index(['date','permno'])['return'].unstack()[weight_df.columns]
    weight_df=weight_df.asfreq('D').ffill(limit=ff_n).reindex(return_df.index).where(~(return_df.isnull()), 0)
    portf_rtn = (return_df*weight_df).sum(axis=1)

    if vol_scaler_flag:
        portf_rtn_0 = portf_rtn.copy()
        portf_rtn = portf_rtn_0*(scaling_vol_tgt/np.sqrt(252))/portf_rtn_0.rolling(60).std()
        scaler_df = portf_rtn.div(portf_rtn_0, axis=0)
        # scaled_mkt = mkt_df['return_sp']*scaling_vol_tgt/mkt_df['return_sp'].rolling(60).std()
        unscaled_mkt = mkt_df['return_sp']

    else:
        scaler_df = portf_rtn.div(portf_rtn, axis=0)
        # scaled_mkt = mkt_df['return_sp']*(portf_rtn.rolling(60).std()/mkt_df['return_sp'].rolling(60).std())
        unscaled_mkt = mkt_df['return_sp']
    
    fig1, ax1 = plt.subplots(1,2, figsize=(11,3.5))
    portf_mkt_rtn = pd.concat([portf_rtn.rename(portf_name), unscaled_mkt.rename('Unscaled Market')], axis=1)
    portf_mkt_rtn.cumsum().plot(ax=ax1[0])
    ax1[0].set_title(f'Cumulative Return Comparison')
    ax1[0].legend(loc='upper left')
    # plt.legend()
    # plt.show()

    (portf_mkt_rtn.rolling(252).std()*np.sqrt(252)).plot(
        ax=ax1[1],
        title='Rolling Annual Vol Comparison')
    ax1[1].legend(loc='upper left')

    fig1.suptitle(f'{portf_name} vs (Scaled) S&P500 Cumulative Return and Rolling Vol Comparison')
    plt.subplots_adjust(top=0.85, bottom=0.01, wspace=0.2)
    plt.show()
    
    stats_df = pd.DataFrame(columns=portf_mkt_rtn.columns)
    stats_df.loc['avg_rtn_ann',:] = portf_mkt_rtn.mean()*252
    stats_df.loc['vol_ann',:] = portf_mkt_rtn.std()*np.sqrt(252)
    stats_df.loc['sharpe_ann',:] = stats_df.loc['avg_rtn_ann',:]/stats_df.loc['vol_ann',:]
    stats_df.loc['max_drawdown',portf_name] = mmd_cal(portf_mkt_rtn, portf_name).iloc[-1]
    stats_df.loc['max_drawdown','Unscaled Market'] = mmd_cal(portf_mkt_rtn, 'Unscaled Market').iloc[-1]

    print(stats_df)    

    return portf_rtn, portf_mkt_rtn, stats_df, scaler_df

def check_weights_plots(opt_weight_df):
    # check basic properties for optimization weights:
    np.round(opt_weight_df.sum(axis=1),2).plot(title='Total weights for the portfolio over time', figsize=(5,3))
    plt.show()

    opt_weight_df.count(axis=1).plot(title='Total number of stocks over time (not all have non zero weight)', figsize=(5,3))
    plt.show()

    opt_weight_df[opt_weight_df>0].count(axis=1).plot(title='Total number of stocks with non-zero weight over time', figsize=(5,3))
    plt.show()


def data_main(data_directory, data_checkpoint_name, start_date, train_end_date, verbose=True):
    # data_directory = r'C:\Mine\U.S.-2019\NPB living - 2 - related\School-part time\Berkeley-202308\MIDS classes\210-Capstone\Project-related\code-IN/data-used'
    # data_checkpoint_name = 'data_checkpoint1'
    # start_date = '2000-01-01'

    data_obj = Data_NN(data_checkpoint_name, start_date, train_end_date)
    print(data_obj)

    data_obj.set_directory(data_directory)
    data_obj.retrieve_data(data_checkpoint_name)

    data_dic_used = data_obj.data_dic
    print(data_obj)

    # check key data
    if verbose:
        print('Check the key dataset:')
        prelim_check_data(data_dic_used['sp500_used'], 'sp500_used')

    return data_obj

def benchmark_run_test(data_directory, data_checkpoint_name, start_date, train_end_date,
                         model_directory, model_checkpoint_name,
                         ticker_list, rebal_freq, opt_flag, last_win_flag=False, target_risk=0.2, verbose=True):

    print(data_directory)
    print(data_checkpoint_name)

    data_obj = data_main(data_directory, data_checkpoint_name, start_date, train_end_date, verbose=False)

    bm_model_obj = Benchmark_model_DL(data_obj, rebal_freq)
    print('object exists: ', bm_model_obj) 

    # check if model exists with given directory and checkpoint name
    bm_model_obj.set_model_directory(model_directory)
    if bm_model_obj.check_checkpoint_model(model_checkpoint_name):
        bm_model_obj = bm_model_obj.load_model(model_checkpoint_name)    
        print ('After loading model..')
        print(bm_model_obj)  
        if (opt_flag in bm_model_obj.opt_tested) and last_win_flag:
            test_reg_param_df = bm_model_obj.model_output_dic[f'test_results_{opt_flag}_last_win']['reg_param_df']
            test_exp_exc_rtn_df = bm_model_obj.model_output_dic[f'test_results_{opt_flag}_last_win']['exp_exc_rtn_df']
            test_opt_weight_df = bm_model_obj.model_output_dic[f'test_results_{opt_flag}_last_win']['opt_weight_df']
        elif (opt_flag in bm_model_obj.opt_tested) and (last_win_flag==False):
            test_reg_param_df = bm_model_obj.model_output_dic[f'test_results_{opt_flag}']['reg_param_df']
            test_exp_exc_rtn_df = bm_model_obj.model_output_dic[f'test_results_{opt_flag}']['exp_exc_rtn_df']
            test_opt_weight_df = bm_model_obj.model_output_dic[f'test_results_{opt_flag}']['opt_weight_df']
        else: # IN: this is the case where the given opt method hasn't been tested, so we kept on testing the new opt case
            test_reg_param_df, test_exp_exc_rtn_df, test_opt_weight_df =\
                bm_model_obj.test(checkpoint_name=model_checkpoint_name, ticker_list=ticker_list, 
                                  opt_flag=opt_flag, last_win_only=False, target_risk=0.2, verbose=True)
            bm_model_obj.save_model_obj(model_directory, model_checkpoint_name)
            
    else:
        test_reg_param_df, test_exp_exc_rtn_df, test_opt_weight_df =\
            bm_model_obj.test(checkpoint_name=model_checkpoint_name, ticker_list=ticker_list, 
                              opt_flag=opt_flag, last_win_only=False, target_risk=0.2, verbose=True)
        
        # bm_model_obj.save_model_obj(model_directory, model_checkpoint_name)

    print(bm_model_obj)         

    return bm_model_obj, test_reg_param_df, test_exp_exc_rtn_df, test_opt_weight_df


