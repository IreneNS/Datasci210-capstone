from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict
import logging
from typing import Optional, List, Any, Dict
from fastapi import FastAPI, Query
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend
from fastapi_cache.decorator import cache
from redis import asyncio
from contextlib import asynccontextmanager
import json
import hashlib
import io
import base64
from src.benchmark import *

logger = logging.getLogger("uvicorn")

@asynccontextmanager
async def lifespan_mechanism(app: FastAPI):
    """
    Lifespan mechanism to manage startup and shutdown tasks.
    - Startup: Initialize Redis Cache.
    - Shutdown: Close Redis Cache and any other tasks.
    """
    logger.info("Starting API...")
    
    # Load the Redis Cache URL from the environment, fallback to local Redis if not set
    HOST_URL = os.getenv("REDIS_URL", "redis://127.0.0.1/0")
    redis = asyncio.from_url(HOST_URL, encoding="utf8", decode_responses=True)

    test_result = await redis.get("test")
    logger.info(f"Redis connection test: {test_result}")

    # Initialize FastAPI Cache with Redis
    FastAPICache.init(RedisBackend(redis), prefix="benchmark")
    logger.info('redis initiated')

    # Yield to allow the FastAPI app to run
    yield

app = FastAPI(lifespan=lifespan_mechanism)

class BenchmarkRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    ticker_list: Optional[list[str]] = []
    rebal_freq: str = "D"
    opt_flag: str = "max_sharpe"
    target_risk: float = 0.2
    

@app.get("/health")
async def health():
    '''
    health check
    '''
    return {"status": "healthy"}

# Custom cache key builder that works with Pydantic models
def custom_key_builder(
    func,
    namespace: str = "",
    http_request: Request = None,
    response: Any = None,
    *args,
    **kwargs
):
    # Extract the Pydantic model from kwargs
    benchmark_request = kwargs.get('benchmark_request')
    if benchmark_request:
        # We found it directly in kwargs
        logger.info(f"Found ModelRequest directly in kwargs: {benchmark_request}")
        model_dict = benchmark_request.model_dump()
        model_str = json.dumps(model_dict, sort_keys=True)
        params_hash = hashlib.md5(model_str.encode()).hexdigest()
        cache_key = f"benchmark:{func.__name__}:{params_hash}"
        logger.info(f"Generated cache key: {cache_key}")
        return cache_key
    else:
        # Search deeply through all values
        logger.info("ModelRequest not found directly, searching nested...")
        for k, v in kwargs.items():
            logger.info(f"Examining kwarg: {k}, type: {type(v)}")
            # Check if any value is our model
            if hasattr(v, 'model_dump') and hasattr(v, 'model_config'):
                logger.info(f"Found potential Pydantic model in {k}")
                try:
                    model_dict = v.model_dump()
                    model_str = json.dumps(model_dict, sort_keys=True)
                    params_hash = hashlib.md5(model_str.encode()).hexdigest()
                    cache_key = f"benchmark:{func.__name__}:{params_hash}"
                    logger.info(f"Generated cache key: {cache_key}")
                    return cache_key
                except Exception as e:
                    logger.error(f"Error processing model: {e}")
                    return f"benchmark:{func.__name__}:default"
    # for arg in kwargs.values():
    #     if isinstance(arg, BenchmarkRequest):
    #         benchmark_request = arg
    #         break

    # Create a cache key based on the request parameters
    # if benchmark_request:
    #     # Convert the model to a consistent string representation for caching
    #     model_dict = benchmark_request.model_dump()
    #     model_str = json.dumps(model_dict, sort_keys=True)
    #     params_hash = hashlib.md5(model_str.encode()).hexdigest()
    #     # Sort keys to ensure consistent order
    #     cache_key = f"benchmark:{func.__name__}:{params_hash}"
    #     logger.info(f"Generated cache key: {cache_key}")

    #     return cache_key
    
    # Fallback to a simple function name-based key
    return f"benchmark:{func.__name__}:default"


@app.post("/benchmark-test")
@cache(expire=3600, key_builder=custom_key_builder)
async def benchmark(http_request: Request, benchmark_request: BenchmarkRequest):
    '''
    run the model
    '''
    logger.info("Cache MISS - executing function")

    current_directory = os.path.dirname(os.path.abspath(__file__))
    data_directory = os.path.join(current_directory, 'dependencies/benchmark_model')
    data_checkpoint_name = 'data_checkpoint1'
    start_date = '2000-01-01'
    train_end_date = '2023-12-31'

    model_directory = os.path.join(current_directory, 'dependencies/benchmark_model')
    model_checkpoint_name = 'bm_model_checkpoint1'
    ticker_list = benchmark_request.ticker_list
    rebal_freq = benchmark_request.rebal_freq  # 'D','W','M'
    opt_flag = benchmark_request.opt_flag #'max_sharpe','target_risk'
    target_risk = benchmark_request.target_risk
    last_win_only=True
    force_retrain=False

    ## test model - benchmark model
    bm_model_obj, test_reg_param_df, test_exp_exc_rtn_df, test_opt_weight_df=\
        benchmark_run_test(data_directory, data_checkpoint_name, start_date, train_end_date,
                            model_directory, model_checkpoint_name,
                            ticker_list, rebal_freq, opt_flag, last_win_only=True, 
                            target_risk=target_risk, force_retrain=False,verbose=True)

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

    # Convert DataFrame to dict for JSONResponse
    return_data = {
        # 'test_reg_param_df': test_reg_param_df.to_dict(orient='records'),
        'portf_rtn_test': portf_rtn_test.to_frame().to_dict(orient='records'),
        'portf_mkt_rtn_test': portf_mkt_rtn_test.to_dict(orient='records'),
        'stats_df_test': stats_df_test.to_dict(orient="records"),
        'scaler_df_test': scaler_df_test.to_frame().to_dict(orient="records")
    }

    
    return JSONResponse(content=return_data)


@app.post("/benchmark")
async def benchmark_model(benchmark_request: BenchmarkRequest):
    '''
    run the model
    '''
    logger.info("Cache MISS - executing function")

    current_directory = os.path.dirname(os.path.abspath(__file__))
    data_directory = 'https://dsp-public-streamlit.s3.us-west-2.amazonaws.com/data-used/' ## CHECK TO DO
    data_checkpoint_name = 'data_checkpoint4'
    start_date = '2000-01-01'
    train_end_date = '2023-12-31'

    model_directory = 'https://dsp-public-streamlit.s3.us-west-2.amazonaws.com/NN-related/benchmark_model/' ## CHECK TO DO
    model_checkpoint_name = 'bm_model_checkpoint2'
    ticker_list = benchmark_request.ticker_list
    rebal_freq = benchmark_request.rebal_freq  # 'D','W','M'
    opt_flag = benchmark_request.opt_flag #'max_sharpe','target_risk'
    target_risk = benchmark_request.target_risk
    last_win_only=True
    force_retrain=False

    ## test model - benchmark model
    bm_model_obj, test_reg_param_df, test_exp_exc_rtn_df, test_opt_weight_df=\
        benchmark_run_test(data_directory, data_checkpoint_name, start_date, train_end_date,
                            model_directory, model_checkpoint_name,
                            ticker_list, rebal_freq, opt_flag, last_win_only=True, 
                            target_risk=target_risk, force_retrain=False,verbose=True)

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
        'portf_rtn_test': portf_rtn_test.to_json(orient="records"),
        'portf_mkt_rtn_test': portf_mkt_rtn_test.to_json(orient="records"),
        'stats_df_test': stats_df_test.to_json(orient="records"),
        'scaler_df_test': scaler_df_test.to_json(orient="records"),
        'scaled_weight_df_test':scaled_weight_df_test.to_json(orient="records")
    }

    
    return return_data

