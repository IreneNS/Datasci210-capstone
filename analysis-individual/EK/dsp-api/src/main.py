from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, validator, ValidationError, confloat
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

    @validator('ticker_list', pre=True)
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
    @validator('ticker_list', pre=True)
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
    tickers = ''

    if len(benchmark_request.ticker_list) > 0:
        tickers=benchmark_request.ticker_list[0]
    else:
        tickers=''

    key = f'benchmark_portf_{tickers}_{benchmark_request.target_risk}_{benchmark_request.last_win_only}'

    db_result = get_cached_result(key)
    if db_result is not None:
        return db_result

    current_directory = os.path.dirname(os.path.abspath(__file__))
    data_directory = os.path.join(current_directory, 'dependencies/benchmark_model')
    model_directory = os.path.join(current_directory, 'dependencies/benchmark_model')
    # data_directory = r'./dependencies/'
    data_checkpoint_name = 'data_checkpoint4'
    start_date = '2000-01-01'
    train_end_date = '2023-12-31'

    model_checkpoint_name = 'bm_model_checkpoint2'
    ticker_list = benchmark_request.ticker_list
    rebal_freq = benchmark_request.rebal_freq  # 'D','W','M'
    opt_flag = benchmark_request.opt_flag #'max_sharpe','target_risk'
    target_risk = benchmark_request.target_risk
    last_win_only = benchmark_request.last_win_only
    force_retrain=False

    ## test model - benchmark model
    bm_model_obj, test_reg_param_df, test_exp_exc_rtn_df, test_opt_weight_df=\
        benchmark_run_test(data_directory, data_checkpoint_name, start_date, train_end_date,
                            model_directory, model_checkpoint_name,
                            ticker_list, rebal_freq, opt_flag, last_win_only=last_win_only, 
                            target_risk=target_risk, force_retrain=True,verbose=True)

    # evaluate the test model
    portf_rtn_test, portf_mkt_rtn_test, stats_df_test, scaler_df_test, fig_perf_test, scaled_weight_df_test = \
        bm_model_obj.eval('test', opt_flag, last_win_only, vol_scaler_flag=False, scaling_vol_tgt=0.2, plot_show=False)

    # test_reg_param_df = test_reg_param_df.replace([np.inf, -np.inf], None).replace(np.nan, None)
    # test_exp_exc_rtn_df = test_exp_exc_rtn_df.replace([np.inf, -np.inf], None).replace(np.nan, None)
    # test_opt_weight_df = test_opt_weight_df.replace([np.inf, -np.inf], None).replace(np.nan, None)

    portf_rtn_test = portf_rtn_test.replace([np.inf, -np.inf], None).replace(np.nan, None)
    portf_mkt_rtn_test = portf_mkt_rtn_test.replace([np.inf, -np.inf], None).replace(np.nan, None)
    stats_df_test = stats_df_test.replace([np.inf, -np.inf], None).replace(np.nan, None)
    scaler_df_test = scaler_df_test.replace([np.inf, -np.inf], None).replace(np.nan, None)
    scaled_weight_df_test = scaled_weight_df_test.replace([np.inf, -np.inf], None).replace(np.nan, None)
    if not isinstance(portf_rtn_test, pd.DataFrame):
        portf_rtn_test = portf_rtn_test.to_frame()
    if not isinstance(portf_mkt_rtn_test, pd.DataFrame):
        portf_mkt_rtn_test = portf_mkt_rtn_test.to_frame()
    if not isinstance(stats_df_test, pd.DataFrame):
        stats_df_test = stats_df_test.to_frame()
    if not isinstance(scaler_df_test, pd.DataFrame):
        stats_df_test = scaler_df_test.to_frame()
    if not isinstance(scaled_weight_df_test, pd.DataFrame):
        stats_df_test = scaled_weight_df_test.to_frame()

    # Convert DataFrame to dict for JSONResponse
    return_data = {
        # 'test_reg_param_df': test_reg_param_df.to_dict(orient='records'),
        'portf_rtn_test': json.loads(portf_rtn_test.to_json(orient="records")),
        'portf_mkt_rtn_test': json.loads(portf_mkt_rtn_test.to_json(orient="records")),
        'stats_df_test': json.loads(stats_df_test.to_json(orient="records")),
        'scaler_df_test': json.loads(scaler_df_test.to_json(orient="records")),
        'scaled_weight_df_test':json.loads(scaled_weight_df_test.to_json(orient="records")),
        'figure_name':fig_perf_test
    }

    s3_path = upload_json_to_s3(key, return_data)
    link = store_result(key, s3_path)

    return link


@app.post("/model")
async def dl_model(model_request: ModelRequest):
    '''
    run the dl model
    '''
    key = f'dl_portf_{model_request.ticker_list[0]}_{model_request.target_risk}_{model_request.last_win_only}'

    db_result = get_cached_result(key)
    if db_result is not None:
        return db_result


    last_win_only= model_request.last_win_only # change later
    if len(model_request.ticker_list) > 0:
        ticker_list = get_s3_pickle(f'data-used/{model_request.ticker_list[0]}.pkl') #top_100_ticker_l, top_200_ticker_l, top_300_ticker_l, []
    else: 
        ticker_list = []
    period = 'test' # train_val, train_val_test, test
    scaling_vol_tgt = model_request.target_risk

    portf_rtn, portf_mkt_rtn, stats_df, scaler_df, fig_perf, scaled_weight_df = \
        run_dl_for_interface(period, last_win_only, ticker_list, scaling_vol_tgt, verbose=True)

    portf_rtn = portf_rtn.replace([np.inf, -np.inf], None).replace(np.nan, None)
    portf_mkt_rtn = portf_mkt_rtn.replace([np.inf, -np.inf], None).replace(np.nan, None)
    stats_df = stats_df.replace([np.inf, -np.inf], None).replace(np.nan, None)
    scaler_df = scaler_df.replace([np.inf, -np.inf], None).replace(np.nan, None)
    scaled_weight_df = scaled_weight_df.replace([np.inf, -np.inf], None).replace(np.nan, None)

    if not isinstance(portf_rtn, pd.DataFrame):
        portf_rtn = portf_rtn.to_frame()
    if not isinstance(portf_mkt_rtn, pd.DataFrame):
        portf_mkt_rtn = portf_mkt_rtn.to_frame()
    if not isinstance(stats_df, pd.DataFrame):
        stats_df = stats_df.to_frame()
    if not isinstance(scaler_df, pd.DataFrame):
        scaler_df = scaler_df.to_frame()
    if not isinstance(scaled_weight_df, pd.DataFrame):
        scaled_weight_df = scaled_weight_df.to_frame()

  
    # # Convert DataFrame to dict for JSONResponse
    # return_data = {
    #     # 'test_reg_param_df': test_reg_param_df.to_dict(orient='records'),
    #     'portf_rtn': portf_rtn.to_json(orient="records"),
    #     'portf_mkt_rtn': portf_mkt_rtn.to_json(orient="records"),
    #     'stats_df': stats_df.to_json(orient="records"),
    #     'scaler_df': scaler_df.to_json(orient="records"),
    #     'scaled_weight_df':scaled_weight_df.to_json(orient="records"),
    #     'figure_name':fig_perf
    # }

    return_data = {
        'portf_rtn': json.loads(portf_rtn.to_json(orient="records")),
        'portf_mkt_rtn': json.loads(portf_mkt_rtn.to_json(orient="records")),
        'stats_df': json.loads(stats_df.to_json(orient="records")),
        'scaler_df': json.loads(scaler_df.to_json(orient="records")),
        'scaled_weight_df': json.loads(scaled_weight_df.to_json(orient="records")),
        'figure_name': fig_perf  # Make sure this is a string, not a raw figure
    }

    s3_path = upload_json_to_s3(key, return_data)
    link = store_result(key, s3_path)

    return link