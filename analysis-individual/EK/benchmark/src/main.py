from fastapi import FastAPI
from fastapi.responses import JSONResponse
from src.model import *
from src.model import Benchmark_model_DL, benchmark_run_test


app = FastAPI()

@app.get("/health")
async def health():
    '''
    health check
    '''
    return {"status": "healthy", "location":os.path.abspath(__file__)}

@app.get("/benchmark")
async def model():
    '''
    run the model
    '''
    # Add a debug print statement
    print("Benchmark_model_DL class:", Benchmark_model_DL)
    print("benchmark_run_test function:", benchmark_run_test)

    current_directory = os.path.dirname(os.path.abspath(__file__))
    data_directory = os.path.join(current_directory, 'dependencies/benchmark_model')
    model_directory = os.path.join(current_directory, 'dependencies/benchmark_model')
    # data_directory = r'./dependencies/'
    data_checkpoint_name = 'data_checkpoint1'
    start_date = '2000-01-01'
    train_end_date = '2023-12-31'

    # model_directory = r'./dependencies/'
    model_checkpoint_name = 'bm_model_checkpoint1'
    ticker_list = []
    rebal_freq = 'M' # 'D','W','M'
    opt_flag = 'target_risk' #'max_sharpe','target_risk'

    ## get data
    # data_obj = data_main(data_directory, data_checkpoint_name, start_date, train_end_date, verbose=False)

    ## test model - benchmark model
    bm_model_obj, test_reg_param_df, test_exp_exc_rtn_df, test_opt_weight_df=\
        benchmark_run_test(data_directory, data_checkpoint_name, start_date, train_end_date,
                            model_directory, model_checkpoint_name,
                            ticker_list, rebal_freq, opt_flag, target_risk=0.2, verbose=True)

    ## evaluate the test model
    # portf_rtn_test, portf_mkt_rtn_test, stats_df_test, scaler_df_test = \
    #     bm_model_obj.eval('test',opt_flag, last_win_flag=False, vol_scaler_flag=False, scaling_vol_tgt=0.3)
    
    test_opt_weight_df = test_opt_weight_df.replace([np.inf, -np.inf], None).fillna(None)
    # Convert DataFrame to dict for JSONResponse
    test_opt_weight_df_dict = test_opt_weight_df.to_dict(orient="records")

    return JSONResponse(content=test_opt_weight_df_dict)