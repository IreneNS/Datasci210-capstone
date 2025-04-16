from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, validator,field_validator,ValidationError, confloat
import logging
from typing import Optional, List, Any, Dict
from fastapi import FastAPI, Query
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend
from fastapi_cache.decorator import cache
from contextlib import asynccontextmanager
import json
import hashlib
import io
import base64
import boto3
import gc
import pickle
from src.benchmark import *
from src.dlmodel import *

logger = logging.getLogger("uvicorn")
s3 = boto3.client('s3')

dynamodb = boto3.resource("dynamodb")
table_name = "capstone_model"
table = dynamodb.Table(table_name)
bucket_name = 'dsp-public-streamlit'
        
def get_s3_pickle(filepath:str):
    '''get and load pickles'''
    path = filepath
    pickle_buffer = BytesIO()
    s3.download_fileobj(bucket_name, path, pickle_buffer)
    pickle_buffer.seek(0)  # Reset buffer position
    return pickle.load(pickle_buffer)

def get_cached_result(key):
    """Check if the result exists in DynamoDB and return it if available."""
    logger.info(f'check table with key {key}')
    
    # Query DynamoDB
    response = table.get_item(Key={"request_name": key})
    
    # Check if the item exists
    if "Item" in response:
        logger.info('item found, returning')
        path = response["Item"]["result"]  # Return stored result
        # Extract bucket name and file key
        s3_parts = path.replace("s3://", "").split("/", 1)
        # bucket, file_key = s3_parts[0], s3_parts[1]

        # # Fetch object from S3
        # response = s3.get_object(Bucket=bucket, Key=file_key)

        # # Read and parse JSON
        # json_data = json.loads(response["Body"].read().decode("utf-8"))
        
        return path.replace("s3://", "https://").replace('streamlit', 'streamlit.s3.us-west-2.amazonaws.com')
    
    logger.info('no matching item found, running function')
    return None

def store_result(key, result):
    """Store calculation result in DynamoDB."""
    table.put_item(
        Item={
            "request_name": key,
            "result": result  # Store your computed result
        }
    )
    logger.info(f'result stored in db with key {key}')
    return result.replace("s3://", "https://").replace('streamlit', 'streamlit.s3.us-west-2.amazonaws.com')

def upload_json_to_s3(key, json_item):
    """Uploads JSON data to S3 and returns the file path."""
    logger.info('uploading results to s3')

    out_key = f'outputs/json/{key}.json'

    # Convert JSON data to string
    json_string = json.dumps(json_item)

    # Upload to S3
    s3.put_object(Bucket=bucket_name, Key=out_key, Body=json_string, ContentType="application/json")

    # Generate S3 URL
    s3_url = f"s3://{bucket_name}/{out_key}"
    logger.info(f's3 url: {s3_url}')

    return s3_url

def prepare_df(df):
    df = df.replace([np.inf, -np.inf], None).replace(np.nan, None)
    if not isinstance(df, pd.DataFrame):
        df = df.to_frame()
    df = df.reset_index()
    return df


@asynccontextmanager
async def lifespan_mechanism(app: FastAPI):
    """
    Lifespan mechanism to manage startup and shutdown tasks.
    - Startup: Initialize Redis Cache.
    - Shutdown: Close Redis Cache and any other tasks.
    """
    logger.info("Starting API...")

    # Yield to allow the FastAPI app to run
    yield

app = FastAPI(lifespan=lifespan_mechanism)

allow_tickers = {"top_100_ticker_l", "top_200_ticker_l", "top_300_ticker_l"}


class BenchmarkRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    ticker_list: Optional[list[str]] = []
    rebal_freq: str = "D"
    opt_flag: str = "max_sharpe"
    target_risk: confloat(ge=0.01, le=0.99) = 0.2
    last_win_only: bool = True

    @field_validator('ticker_list', mode='before')
    def check_ticker_list(cls, v):
        # Check if ticker_list is either empty or contains exactly one item
        if len(v) > 1:
            raise ValueError("ticker_list should contain exactly one element or be empty.")
        # If ticker_list is not empty, ensure each ticker starts with allowed prefixes
        if v:
            ticker = v[0]
            if not any(ticker.startswith(prefix) for prefix in allow_tickers):
                raise ValueError(f"Invalid ticker: {ticker}. Must be one of {allow_tickers}")
        return v

class ModelRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    ticker_list: Optional[list[str]] = []
    rebal_freq: str = "D"
    opt_flag: str = "max_sharpe"
    target_risk: confloat(ge=0.01, le=0.99) = 0.2
    last_win_only: bool = True
    @field_validator('ticker_list', mode='before')
    def check_ticker_list(cls, v):
        # Check if ticker_list is either empty or contains exactly one item
        if len(v) > 1:
            raise ValueError("ticker_list should contain exactly one element or be empty.")
        # If ticker_list is not empty, ensure each ticker starts with allowed prefixes
        if v:
            ticker = v[0]
            if not any(ticker.startswith(prefix) for prefix in allow_tickers):
                raise ValueError(f"Invalid ticker: {ticker}. Must be one of {allow_tickers}")
        return v
    

@app.get("/health")
async def health():
    '''
    health check
    '''
    return {"status": "healthy"}

@app.get("/")
async def health():
    '''
    health check
    '''
    return {"status": "healthy"}


@app.post("/benchmark")
async def benchmark_model(benchmark_request: BenchmarkRequest):
    '''
    run the model
    '''
    logger.info("start benchmark model")

    # make key for db
    if len(benchmark_request.ticker_list) > 0:
        key = f'bm_portf_{benchmark_request.ticker_list[0]}_{benchmark_request.target_risk}_{benchmark_request.last_win_only}'
        ticker_list = get_s3_pickle(f'data-used/{benchmark_request.ticker_list[0]}.pkl') #top_100_ticker_l, top_200_ticker_l, top_300_ticker_l, []
    else: 
        key = f'bm_portf_all_{benchmark_request.target_risk}_{benchmark_request.last_win_only}'
        ticker_list = []

    # check db
    db_result = get_cached_result(key)
    if db_result is not None:
        return db_result

    current_directory = os.path.dirname(os.path.abspath(__file__))
    data_directory = os.path.join(current_directory, 'dependencies/benchmark_model')
    model_directory = os.path.join(current_directory, 'dependencies/benchmark_model')
    data_checkpoint_name = 'data_checkpoint4'
    start_date = '2001-01-01'
    train_end_date = '2023-12-31'

    model_checkpoint_name = 'bm_model_checkpoint2'
    rebal_freq = benchmark_request.rebal_freq  # 'D','W','M'
    opt_flag = benchmark_request.opt_flag #'max_sharpe','target_risk'
    target_risk = benchmark_request.target_risk
    last_win_only = benchmark_request.last_win_only

    print(data_directory)

    fig_perf_train = ''
    fig_perf_test = ''

    # with open(f'{data_directory}/portf_rtn_comb.pkl', 'rb') as f:
    #     portf_rtn_comb = pickle.load(f)

    # # test model - benchmark model
    # bm_model_obj, train_reg_param_df, train_exp_exc_rtn_df, train_opt_weight_df=\
    #     benchmark_run_train(data_directory, data_checkpoint_name, start_date, train_end_date,
    #                         model_directory, model_checkpoint_name,
    #                         ticker_list, rebal_freq, opt_flag, target_risk=target_risk, 
    #                         force_retrain=True, force_take_new_data=True, verbose=True)

    # # evaluate the test model
    # portf_rtn_train, portf_mkt_rtn_train, stats_df_train, scaler_df_train, fig_perf_train, scaled_weight_df_train = \
    #     bm_model_obj.eval('train', opt_flag, last_win_only, vol_scaler_flag=True, scaling_vol_tgt=target_risk, plot_show=False)
    
    # portf_rtn_train = prepare_df(portf_rtn_train)
    # portf_mkt_rtn_train = prepare_df(portf_mkt_rtn_train)
    # stats_df_train = prepare_df(stats_df_train)
    # scaler_df_train = prepare_df(scaler_df_train)
    # scaled_weight_df_train = prepare_df(scaled_weight_df_train)

    # portf_rtn_train = portf_rtn_train.to_csv('portf_rtn_train.csv')
    # portf_mkt_rtn_train = portf_mkt_rtn_train.to_csv('portf_mkt_rtn_train.csv')   
    # stats_df_train = stats_df_train.to_csv('stats_df_train.csv')
    # scaler_df_train = scaler_df_train.to_csv('scaler_df_train.csv')
    # scaled_weight_df_train = scaled_weight_df_train .to_csv('scaled_weight_df_train.csv')

    # portf_rtn_train = pd.read_csv(f'{data_directory}/portf_rtn_train.csv').iloc[:,1:]
    # portf_mkt_rtn_train = pd.read_csv(f'{data_directory}/portf_mkt_rtn_train.csv').iloc[:,1:]
    # stats_df_train = pd.read_csv(f'{data_directory}/stats_df_train.csv').iloc[:,1:]
    # scaler_df_train = pd.read_csv(f'{data_directory}/scaler_df_train.csv').iloc[:,1:]
    # scaled_weight_df_train = pd.read_csv(f'{data_directory}/scaled_weight_df_train.csv').iloc[:,1:]

    # ## test model - benchmark model
    bm_model_obj, test_reg_param_df, test_exp_exc_rtn_df, test_opt_weight_df=\
        benchmark_run_test(data_directory, data_checkpoint_name, start_date, train_end_date,
                            model_directory, model_checkpoint_name,
                            ticker_list, rebal_freq, opt_flag, last_win_only=last_win_only, 
                            target_risk=target_risk, force_retrain=True,verbose=True)

    # evaluate the test model
    portf_rtn_test, portf_mkt_rtn_test, stats_df_test, scaler_df_test, fig_perf_test, scaled_weight_df_test = \
        bm_model_obj.eval('test', opt_flag, last_win_only, vol_scaler_flag=True, scaling_vol_tgt=target_risk, plot_show=False)


    # clean nans
    portf_rtn_test = prepare_df(portf_rtn_test)
    portf_mkt_rtn_test = prepare_df(portf_mkt_rtn_test)
    stats_df_test = prepare_df(stats_df_test)
    scaler_df_test = prepare_df(scaler_df_test)
    scaled_weight_df_test = prepare_df(scaled_weight_df_test)

    # df to dict for json
    return_data = {
        'portf_rtn_test': json.loads(portf_rtn_test.reset_index().to_json(orient="records")),
        'portf_mkt_rtn_test': json.loads(portf_mkt_rtn_test.reset_index().to_json(orient="records")),
        'stats_df_test': json.loads(stats_df_test.reset_index().to_json(orient="records")),
        'scaler_df_test': json.loads(scaler_df_test.reset_index().to_json(orient="records")), # reset index for date
        'scaled_weight_df_test':json.loads(scaled_weight_df_test.reset_index().to_json(orient="records")), # reset index for date
        'figure_name_test':fig_perf_test
    }
    
    # portf_rtn_test = pd.read_csv(f'{data_directory}/portf_rtn_test.csv').iloc[:,1:]
    # portf_mkt_rtn_test = pd.read_csv(f'{data_directory}/portf_mkt_rtn_test.csv').iloc[:,1:]
    # stats_df_test = pd.read_csv(f'{data_directory}/stats_df_test.csv').iloc[:,1:]
    # scaler_df_test = pd.read_csv(f'{data_directory}/scaler_df_test.csv').iloc[:,1:]
    # scaled_weight_df_test = pd.read_csv(f'{data_directory}/scaled_weight_df_test.csv').iloc[:,1:]

    # prepared train_val + test combine results:
    # portf_rtn_comb = pd.concat([portf_rtn_train, portf_rtn_test])
    # print(portf_mkt_rtn_train.columns)
    # portf_mkt_rtn_train.columns = ['date','Daily-Benchmark', 'Unscaled Market']
    # portf_mkt_rtn_test.columns = ['date','Daily-Benchmark', 'Unscaled Market']
    # portf_mkt_rtn_comb = pd.concat([portf_mkt_rtn_train.iloc[:, 1:], portf_mkt_rtn_test.iloc[:, 1:]])
    
    # portf_name = 'Daily-Benchmark'

    # stats_df_comb = pd.DataFrame(columns=['Daily-Benchmark', 'Unscaled Market'])
    # stats_df_comb.loc['avg_rtn_ann',:] = portf_mkt_rtn_comb.mean()*252
    # stats_df_comb.loc['vol_ann',:] = portf_mkt_rtn_comb.std()*np.sqrt(252)
    # stats_df_comb.loc['sharpe_ann',:] = stats_df_comb.loc['avg_rtn_ann',:]/stats_df_comb.loc['vol_ann',:]
    # stats_df_comb.loc['max_drawdown', portf_name] = mmd_cal(portf_mkt_rtn_comb, portf_name).iloc[-1]
    # stats_df_comb.loc['max_drawdown','Unscaled Market'] = mmd_cal(portf_mkt_rtn_comb, 'Unscaled Market').iloc[-1]

    # scaler_df_comb = pd.concat([scaler_df_train, scaler_df_test]).ffill()
    # scaled_weight_df_comb = pd.concat([scaled_weight_df_train, scaled_weight_df_test])

    

    # # df to dict for json
    # return_data = {
    #     'portf_rtn_comb': json.loads(portf_rtn_comb.reset_index().to_json(orient="records")),
    #     'portf_mkt_rtn_comb': json.loads(portf_mkt_rtn_comb.reset_index().to_json(orient="records")),
    #     'stats_df_comb': json.loads(stats_df_comb.reset_index().to_json(orient="records")),
    #     'scaler_df_comb': json.loads(scaler_df_comb.reset_index().to_json(orient="records")), # reset index for date
    #     'scaled_weight_df_comb':json.loads(scaled_weight_df_comb.reset_index().to_json(orient="records")), # reset index for date
    #     'figure_name_train':fig_perf_train,
    #     'figure_name_test':fig_perf_test
    # }

    # fill db
    s3_path = upload_json_to_s3(key, return_data)
    link = store_result(key, s3_path)

    return link


@app.post("/model")
async def dl_model(model_request: ModelRequest):
    '''
    run the dl model
    '''

    # make key
    if len(model_request.ticker_list) > 0:
        key = f'dl_portf_{model_request.ticker_list[0]}_{model_request.target_risk}_{model_request.last_win_only}'
        ticker_list = get_s3_pickle(f'data-used/{model_request.ticker_list[0]}.pkl') #top_100_ticker_l, top_200_ticker_l, top_300_ticker_l, []
    else: 
        key = f'dl_portf_all_{model_request.target_risk}_{model_request.last_win_only}'
        ticker_list = []

    # check db
    db_result = get_cached_result(key)
    if db_result is not None:
        return db_result

    last_win_only= model_request.last_win_only 
    period = 'test' # train_val, train_val_test, test
    scaling_vol_tgt = model_request.target_risk

    # run model
    portf_rtn, portf_mkt_rtn, stats_df, scaler_df, fig_perf, scaled_weight_df = \
        run_dl_for_interface(period, last_win_only, ticker_list, scaling_vol_tgt, verbose=True)

    # clean nans
    portf_rtn = prepare_df(portf_rtn)
    portf_mkt_rtn = prepare_df(portf_mkt_rtn)
    stats_df = prepare_df(stats_df)
    scaler_df = prepare_df(scaler_df)
    scaled_weight_df = prepare_df(scaled_weight_df)
 
    # df to json
    return_data = {
        'portf_rtn': json.loads(portf_rtn.reset_index().to_json(orient="records")),
        'portf_mkt_rtn': json.loads(portf_mkt_rtn.reset_index().to_json(orient="records")),
        'stats_df': json.loads(stats_df.reset_index().to_json(orient="records")),
        'scaler_df': json.loads(scaler_df.reset_index().to_json(orient="records")), # index is date
        'scaled_weight_df': json.loads(scaled_weight_df.reset_index().to_json(orient="records")), # index is date
        'figure_name': fig_perf  # Make sure this is a string, not a raw figure
    }

    # update db
    s3_path = upload_json_to_s3(key, return_data)
    link = store_result(key, s3_path)

    return link