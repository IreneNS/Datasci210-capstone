# !pip install --upgrade wrds --quiet
# !pip install  --upgrade openpyxl -- quiet
# !pip install yfinance --quiet
# !pip install PyPortfolioOpt --quiet
# !pip install cvxopt --quiet
# !pip install cvxpy --quiet
# !pip install openpyxl --quiet
# !pip install tensorboard -- quiet
# !pip install optuna --quiet

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP  ## ELAINE
import smdistributed.dataparallel.torch.torch_smddp  ## ELAINE
dist.init_process_group(backend="smddp")  ## ELAINE
from torch.cuda.amp import GradScaler, autocast  ## ELAINE
import os
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
import importlib
import warnings
# warnings.filterwarnings("ignore")
warnings.filterwarnings('default')
from sklearn.covariance import LedoitWolf
import argparse
import json
import ast
import s3fs
import logging

# setups ## ELAINE
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
fs = s3fs.S3FileSystem()

os.environ['NCCL_IGNORE_DISABLED_P2P'] = '1'  ## ELAINE (don't even think this works)
torch.cuda.set_per_process_memory_fraction(0.8)  ## ELAINE
torch.cuda.empty_cache()  ## ELAINE

class DL_dataset(Dataset):
    def __init__(self, initial_data, rebal_freq, window_days, feature_cols, tgt_cols):
        self.data = []
        self.dates = []  # the dates corresponding to the sample (x,y)
        self.permnos = [] # the list of permno_id whose returns are predicted for each rebal date
        self.rebal_freq = rebal_freq # IN: 'D', 'W', 'M'
        if (initial_data['date'].unique()[0].year == 2000)&\
                initial_data['date'].unique()[0].month == 1:
            self.dates_daily = initial_data['date'].unique()[20:]
            # the first available date for features is the first date of the second month of all data
        else:
            self.dates_daily = initial_data['date'].unique()
        self.dates_weekly = pd.Series(np.ones(len(self.dates_daily)), \
                               index=pd.DatetimeIndex(self.dates_daily)).asfreq('W-WED').index
        self.dates_monthly = pd.Series(np.ones(len(self.dates_daily)), \
                                index=pd.DatetimeIndex(self.dates_daily)).asfreq('ME').index
        self.window_days = window_days
        self.feature_cols = feature_cols
        self.tgt_cols = tgt_cols

        # create rolling windows group by date across different stocks
        if self.rebal_freq == 'D':
            self.rebal_dates = self.dates_daily
        elif self.rebal_freq == 'W':
            self.rebal_dates = self.dates_weekly
        elif self.rebal_freq == 'M':
            self.rebal_dates = self.dates_monthly

        initial_data = initial_data.sort_values(['date','permno'])
        #just to make sure it's in the right order but it should already been

        for i, date in enumerate(self.rebal_dates):
            print (date)
            ''' create rolling samples'''

            index_in_daily_dates = list(self.dates_daily).index(self.dates_daily[self.dates_daily<=date][-1])

            date_period = self.dates_daily[max(0,(index_in_daily_dates+1)-self.window_days*1)
                                        :max(self.window_days*1, (index_in_daily_dates+1))]
            # extract rolling window data
            data_t = initial_data[initial_data['date'].isin(date_period)][self.feature_cols+['date','permno','prev_d_exc_ret']]
            # to match the inputs of BM, may not be necessary
            data_t[sel_features_adj] = data_t[sel_features_adj].apply(lambda x: x*data_t['prev_d_exc_ret'], axis=0).ffill()
            #IN: use this method to line up x and y stocks; short rolling window ok to ffill
            data_t = data_t.drop(columns=['prev_d_exc_ret'])

            ## prepare the Y data
            # df_x = data_t[sel_features_adj].apply(lambda x: x*data_t['prev_d_exc_ret'], axis=0).ffill()
            if rebal_freq == 'M':
                match_date = self.dates_daily[(self.dates_daily.month==date.month)
                                            &(self.dates_daily.year==date.year)][-1]
            elif rebal_freq == 'W':
                match_date = self.dates_daily[(self.dates_daily<=date)][-1]
            else:
                match_date = date

            target_data = initial_data[initial_data['date']==match_date][['permno']+self.tgt_cols].sort_values('permno')
            # regulate the number of stocks per day as the same for later modeling purpose (e.g. transformer need inputs are of same dimension)
            if len(target_data)<500:
                pad_df = pd.DataFrame(np.ones((500-len(target_data),len(target_data.columns))),
                                      columns = target_data.columns,
                                        index = list(500 + np.arange(500-len(target_data))))
                target_data = pd.concat([target_data, pad_df])
            else:
                target_data = target_data.iloc[:500,:] # if target_data has >=500 rows, only take the first 500 rows

            y = target_data[self.tgt_cols].values # shape: (num_stocks,)
            permnos_t = target_data['permno'].tolist()

            # convert y to torch tensor
            y = torch.tensor(y, dtype=torch.float32).nan_to_num()
            # print('/t')
            # print (y)

            ## prepare the X data
            # pivot so that each stock has its own row, with window_days sequences as columns (values are the multiple columns)
            data_t_adj = data_t[data_t['permno'].isin(permnos_t)]
            pivoted_data_t = data_t_adj.pivot(index='permno', columns='date')

            # reshape the pivoted_data
            X = pivoted_data_t.values.reshape(pivoted_data_t.shape[0], len(date_period), -1)
            # using len(date_period) instead of just the window_days, just in case certain period have odd days at the start or end period

            # to make sure X have the same shapes for later processing, padding to 500, 252 to dimension 0 and 1 (latter may not be needed)
            if pivoted_data_t.shape[0]<500:
                n_pad_0 = 500-pivoted_data_t.shape[0]
                X = np.pad(X, ((0, n_pad_0), (0, 0), (0, 0)), mode='constant', constant_values=0)
            # if pivoted_data_t.shape[1]<252:
            #     n_pad_1 = 252 - pivoted_data_t.shape[1]
            #     X = np.pad(X, ((0, 0), (0, n_pad_1), (0, 0)), mode='constant', constant_values=0)

            X = torch.tensor(X, dtype=torch.float32).nan_to_num() 
            # this by default change nan to 0, and pos/neg inf to large finite float
            # drop all stocks whose features are nan at the last day of this window
            # (i.e. most likely have component change within the window, using the last day stock as target in this window, matching the predicted target)
            # after this operation, each X should almost always close to 500 stocks, i.e. first dimension: ~500 (note: it is still not always 500)
            # X = X[~torch.isnan(X[:,-1,:]).all(dim=1)]

            # print('/t')
            # print (X)

            # append the current sample
            self.data.append((X,y))
            self.dates.append(torch.tensor(match_date.timestamp()))    
            # convert datetime into torch.tensor, otherwise model can't batch
            self.permnos.append(torch.tensor(permnos_t))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # return self.data[idx]   #(X,y) for a specific day across all stocks
        return {
            'features': self.data[idx][0],
            'target': self.data[idx][1],
            'date': self.dates[idx],
            'permnos': self.permnos[idx]
        }

    def check_sample_dimension_match(self):
        mismatch_l = []
        for i in range(len(self.data)):
            if self.data[i][0].shape[0]!=self.data[i][1].shape[0]:
                mismatch_l.append((i, (self.data[i][0].shape, self.data[i][1].shape)))
        return mismatch_l
        
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        assert embed_dim % num_heads == 0 #embed_dim must be divisible by num_heads

        self.num_heads = num_heads
        self.head_dim = embed_dim//num_heads

        self.query = nn.Linear(embed_dim,embed_dim)
        self.key = nn.Linear(embed_dim,embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.scale = nn.Linear(embed_dim,embed_dim)
        # final projection layer
        self.fc_out = nn.Linear(embed_dim,embed_dim)

        self.scaler = torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))

    def forward(self, x):
        # x: [batch_size, stock_num, embed_dim]
        batch_size, stock_num, embed_dim = x.shape

        # compute Q,K,V
        Q = self.query(x).reshape(batch_size, stock_num, self.num_heads, self.head_dim).transpose(1,2) #transpose(1,2) is to swap dimension 1 and 2
        K = self.key(x).reshape(batch_size, stock_num, self.num_heads, self.head_dim).transpose(1,2)
        V = self.value(x).reshape(batch_size, stock_num, self.num_heads, self.head_dim).transpose(1,2)

        # Scaled dot-product attention
        attn_weights = torch.softmax((Q@K.transpose(-2,-1))/self.scaler, dim=-1)
        # [batch_size, head_num, stock_num, stock_num], each head will do it seperately so what participate in the doc product are just the last two dimension
        # dim=1, menas we apply the softmax operation across the row (for each sample in batch); in this case will be the same as dim=-1
        attn_output = attn_weights @ V
        # weighted sum across stocks for each feature;
        # each row of attn_weights modified each col of V; [batch_size, head_num, stock_num, embed_dim_per_head]

        # merge heads back
        attn_output = attn_output.transpose(1,2).reshape(batch_size, stock_num, embed_dim)

        # final linear transformation
        output = self.fc_out(attn_output)

        return output


class ReturnPredictionModel_SelfAttention(nn.Module):
    def __init__(self, input_dim=9, lstm_dim=64, lstm_layers=10,
                 num_heads=8, hidden_dims_l=[128,32], dropout=0.3, output_dim=1):
        super(ReturnPredictionModel_SelfAttention, self).__init__()

        # set up LSTM layer to extract temporal feature for each stock
        self.lstm = nn.LSTM(input_dim, lstm_dim,
                            num_layers=lstm_layers,
                            batch_first=True,
                            bidirectional=True)
        # batch_first: if the first dimension is batch size
        # bidirectional=True means process in both directions, then the final cell state has two embeddings

        # set up sel-attention layer
        self.attention = MultiHeadSelfAttention(lstm_dim*2, num_heads) # bidirectional=True, thus *2

        # set sequantial non linear layer
        # self.hidden_layers = nn.ModuleList()
        hidden_layers_l = []
        last_hidden_layer_dim = lstm_dim*2 # bidirectional=True, thus *2
        for i in range(len(hidden_dims_l)):
            hidden_seq = [nn.Linear(last_hidden_layer_dim, hidden_dims_l[i]),
                nn.ReLU(),
                nn.Dropout(dropout)]
            last_hidden_layer_dim = hidden_dims_l[i]
            hidden_layers_l.extend(hidden_seq)
        self.hidden_layers = nn.Sequential(*hidden_layers_l)

        # fianl prediction layer
        self.fc = nn.Linear(hidden_dims_l[-1], output_dim)

    def forward(self, x):
        # x: shape: [batch_size, stock_num, seq_len, input_feat_dim]

        batch_size, stock_num, seq_len, input_feat_dim = x.shape

        # reshape the input to [batch_size*stock_num, seq_len, input_feat_dm] for LSTM
        x = x.view(batch_size*stock_num, seq_len, input_feat_dim)

        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)
        # h_n shape: num_layers*num_directions, batch_size*stock_num, hidden_dim
        # note: h_n[-1] is the same as h_n[-1,:,:] here
        # note: c_n[-1] is more representation with more memory of the sequence
        # h_c[-1] is more affected by the 'current time stamp'
        # final_lstm_rep shoudl be of shape: batch_size, stock_num, hidden_dim
        # if to use didirectional=False:
        # final_lstm_rep = c_n[-1].view(batch_size, stock_num, -1) # reshape to desired shape for following process
        
        # if to use bidirectional=True, then need to concate the last two layers
        c_last_forward, c_last_backward = c_n[-2], c_n[-1]
        c_last_comb = torch.cat((c_last_forward, c_last_backward), dim=1) # shape: batch_size*stock_num, hidden_dim
        final_lstm_rep = c_last_comb.view(batch_size, stock_num, -1) # shape: batch_size, stock_num, 2*hidden_dim
        
        # self-attention
        attention_output = self.attention(final_lstm_rep)
        # note: the input of attention is of [batch_size, stock_num, hidden_dim]
        # the output of attention is of [batch_size, stock_num,  hidden_dim]
        # note that by default, the dim in those ml layers are typically referring to the column dim,
        # because by the basic understanding of nn, batch_size does not come in play,
        # it just means how many samples are processe in the same way at the same time
        # and the row dimension is typically referring to the item dimension, representing the item to be processed

        # non-linear layers
        hidden_layer_output = self.hidden_layers(attention_output)
        
        # last layer for prediction
        output = self.fc(hidden_layer_output) # output: [batch_size, stock_num, output_dim (here is 1)]

        return output


class ReturnPredictionModel_Transformer(nn.Module):
    def __init__(self, input_dim=9, lstm_dim=64, lstm_layers=5, transformer_layers=3,
                 num_heads=4, attn_dropout=0.1, forward_dim=5, hidden_dims_l=[128,32], dropout=0.3, output_dim=1):
        super(ReturnPredictionModel_Transformer, self).__init__()

        # set up LSTM layer to extract temporal feature for each stock
        self.lstm = nn.LSTM(input_dim, lstm_dim,
                            num_layers=lstm_layers,
                            batch_first=True,
                            bidirectional=True)
        # batch_first: if the first dimension is batch size
        # bidirectional=True means process in both directions

        # set up transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model = lstm_dim*2, # lstm output dim; # bidirectional=True, thus *2
            nhead = num_heads, # number of attention heads
            dim_feedforward = forward_dim, # feedforeward layer inside each transformer layer
            dropout = attn_dropout,
            batch_first = True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer,
                                                num_layers = transformer_layers)

        # set sequantial non linear layer
        hidden_layers_l = []
        last_hidden_layer_dim = lstm_dim*2 # bidirectional=True, thus *2
        for i in range(len(hidden_dims_l)):
            hidden_seq = [nn.Linear(last_hidden_layer_dim, hidden_dims_l[i]),
                nn.ReLU(),
                nn.Dropout(dropout)]
            last_hidden_layer_dim = hidden_dims_l[i]
            hidden_layers_l.extend(hidden_seq)
        self.hidden_layers = nn.Sequential(*hidden_layers_l)

        # final prediction layer
        self.fc = nn.Linear(hidden_dims_l[-1], output_dim)

    def forward(self, x):
        # x: shape: [batch_size, stock_num, seq_len, input_feat_dim]

        batch_size, stock_num, seq_len, input_feat_dim = x.shape

        # reshape the input to [batch_size*stock_num, seq_len, input_feat_dim] for LSTM
        x = x.view(batch_size*stock_num, seq_len, input_feat_dim)

        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)
        # h_t shape: num_layers*num_directions, batch_size*stock_num, hidden_dim
        # note: h_n[-1] is the same as h_n[-1,:,:] here
        # note: c_n[-1] is more representation with more memory of the sequence
        # h_n[-1] is more affected by the 'current time stamp'; could potentially try use concat h_n and c_n by column
        # final_lstm_rep shoudl be of shape: batch_size, stock_num, hidden_dim
        # final_lstm_rep = c_n[-1].view(batch_size, stock_num, -1) # reshape to desired shape for following process
        
        # if to use bidirectional=True, then need to concate the last two layers
        c_last_forward, c_last_backward = c_n[-2], c_n[-1]
        c_last_comb = torch.cat((c_last_forward, c_last_backward), dim=1) # shape: batch_size*stock_num, hidden_dim
        final_lstm_rep = c_last_comb.view(batch_size, stock_num, -1) # shape: batch_size, stock_num, 2*hidden_dim
        
        # transformer
        transformer_output = self.transformer(final_lstm_rep)
        # note: the input of transformer is of [batch_size, stock_num, hidden_dim]
        # the output of transformer is of [batch_size, stock_num,  output_dim (1 here for return)]
        # note that by default, the dim in those ml layers are typically referring to the column dim,
        # because by the basic understanding of nn, batch_size does not come in play,
        # it just means how many samples are processe in the same way at the same time
        # and the row dimension is typically referring to the item dimension, representing the item to be processed
        # such as in lstm, we want process the temporal wise item so this dimension was arranged as the time
        # note we want process stock wise item and change the feature space from hidden_dim to output_dim

        # non-linear layers
        hidden_layer_output = self.hidden_layers(transformer_output)
        
        # last layer for prediction
        output = self.fc(hidden_layer_output) # output == input dim: [batch_size, stock_num, 1]

        return output

class ReturnPredictionModel_Transformer_squar(nn.Module):
    def __init__(self, input_dim=9, embed_dim=36, window_size=128,
                 transformer_layers_seq=2, num_heads_seq=4, attn_dropout_seq=0.1, forward_dim_seq=5, 
                 transformer_layers_cs=3, num_heads_cs=4, attn_dropout_cs=0.1, forward_dim_cs=5, 
                 hidden_dims_l=[252,128], dropout=0.3, output_dim=1):
        super(ReturnPredictionModel_Transformer, self).__init__()

        # set up LSTM layer to extract temporal feature for each stock
        # self.lstm = nn.LSTM(input_dim, lstm_dim,
        #                     num_layers = lstm_layers,
        #                     batch_first=True,
        #                     bidirectional=True)
        # # batch_first: if the first dimension is batch size
        # # bidirectional=True means process in both directions

        # projection layer/encoding layer
        self.input_projection = nn.Linear(input_dim, embed_dim)
        
        # set up transformer encoder layer for seq
        encoder_layer_seq = nn.TransformerEncoderLayer(
            d_model=embed_dim, # lstm output dim
            nhead=num_heads_seq, # number of attention heads
            dim_feedforward=forward_dim_seq, # feedforeward layer inside each transformer layer
            dropout=attn_dropout_seq,
            batch_first=True
        )
        self.transformer_seq = nn.TransformerEncoder(encoder_layer_seq,
                                                    num_layers=transformer_layers_seq)        
        
        # set up transformer encoder layer for cs
        encoder_layer_cs = nn.TransformerEncoderLayer(
            d_model=embed_dim*window_size,  # lstm output dim
            nhead=num_heads_cs,  # number of attention heads
            dim_feedforward=forward_dim_cs,  # feedforeward layer inside each transformer layer
            dropout=attn_dropout_cs,
            batch_first=True
        )
        self.transformer_cs = nn.TransformerEncoder(encoder_layer_cs,
                                                    num_layers=transformer_layers_cs)

        # set sequantial non linear layer
        hidden_layers_l = []
        last_hidden_layer_dim = embed_dim*window_size
        for i in range(len(hidden_dims_l)):
            hidden_seq = [nn.Linear(last_hidden_layer_dim, hidden_dims_l[i]),
                nn.ReLU(),
                nn.Dropout(dropout)]
            last_hidden_layer_dim = hidden_dims_l[i]
            hidden_layers_l.extend(hidden_seq)
        self.hidden_layers = nn.Sequential(*hidden_layers_l)

        # final prediction layer
        self.fc = nn.Linear(hidden_dims_l[-1], output_dim)

    def forward(self, x):
        # x: shape: [batch_size, stock_num, seq_len, input_feat_dim]

        batch_size, stock_num, seq_len, input_feat_dim = x.shape

        # reshape the input to [batch_size*stock_num, seq_len, input_feat_dim] for LSTM
        x = x.view(batch_size*stock_num, seq_len, input_feat_dim)

        # # LSTM
        # lstm_out, (h_n, c_n) = self.lstm(x)
        # # h_t shape: num_layers*num_directions, batch_size*stock_num, hidden_dim
        # # note: h_n[-1] is the same as h_n[-1,:,:] here
        # # note: c_n[-1] is more representation with more memory of the sequence
        # # h_n[-1] is more affected by the 'current time stamp'; could potentially try use concat h_n and c_n by column
        # # final_lstm_rep shoudl be of shape: batch_size, stock_num, hidden_dim
        # # final_lstm_rep = c_n[-1].view(batch_size, stock_num, -1) # reshape to desired shape for following process

        # # if to use bidirectional=True, then need to concate the last two layers
        # c_last_forward, c_last_backward = c_n[-2], c_n[-1]
        # c_last_comb = torch.cat((c_last_forward, c_last_backward), dim=1) # shape: batch_size*stock_num, hidden_dim
        # final_lstm_rep = c_last_comb.view(batch_size, stock_num, -1) # shape: batch_size, stock_num, 2*hidden_dim

        # project the shape
        x = self.input_projection(x) 

        # transformer_seq
        transformer_output_seq = self.transformer_seq(x)

        # transformer_cs
        transformer_output_cs = self.transformer(transformer_output_seq)
        # note: the input of transformer is of [batch_size, stock_num, hidden_dim]
        # the output of transformer is of [batch_size, stock_num,  output_dim (1 here for return)]
        # note that by default, the dim in those ml layers are typically referring to the column dim,
        # because by the basic understanding of nn, batch_size does not come in play,
        # it just means how many samples are processe in the same way at the same time
        # and the row dimension is typically referring to the item dimension, representing the item to be processed
        # such as in lstm, we want process the temporal wise item so this dimension was arranged as the time
        # note we want process stock wise item and change the feature space from hidden_dim to output_dim

        # non-linear layers
        hidden_layer_output = self.hidden_layers(transformer_output_cs)

        # last layer for prediction
        output = self.fc(hidden_layer_output)  # output == input dim: [batch_size, stock_num, 1]

        return output


def set_seed(seed=42):
    random.seed(seed)  # Python's random module
    np.random.seed(seed)  # NumPy
    torch.manual_seed(seed)  # PyTorch CPU
    torch.cuda.manual_seed(seed)  # PyTorch GPU
    torch.cuda.manual_seed_all(seed)  # If using multi-GPU
    torch.backends.cudnn.deterministic = True  # Ensures deterministic behavior
    torch.backends.cudnn.benchmark = False  # Ensures reproducibility


def remove_seed():
    # random.seed(None)  # Removes fixed seed from Python’s random module
    np.random.seed(None)  # Removes fixed seed from NumPy
    torch.seed()  # Resets PyTorch’s seed to a random one
    torch.backends.cudnn.deterministic = False  # Allows CuDNN to use non-deterministic algorithms
    torch.backends.cudnn.benchmark = True  # Enables optimizations that may introduce randomness


def clear_memory(model_dl, delete_model=False):
    if delete_model:
        del model_dl
    gc.collect()
    torch.cuda.empty_cache()


def save_model(model, train_loss_l, val_loss_l, model_dir, model_filename):
    # torch.save(model_dl.state_dict(), dl_model_directory+model_filename)
    # full_dir = f'{model_dir}{model_checkpoint}/{rebal_freq}/'
    # model_item = model.module.state_dict() if is_distributed else model.state_dict()

    module = model.module if hasattr(model, 'module') else model ## ELAINE
    model_item = module.state_dict()

    model_path = model_dir + 'models/model.pth'  ## ELAINE
    with fs.open(model_path, 'wb') as f:
        torch.save(model_item, f)

    # save the train and val loss results
    with fs.open(f'{model_dir}train_loss_l_{model_filename}.pkl', 'wb') as f:
        pickle.dump(train_loss_l, f)
    with fs.open(f'{model_dir}val_loss_l_{model_filename}.pkl', 'wb') as f:
        pickle.dump(val_loss_l, f)


def load_train_val_dataset(model_dir):
    # full_dir = f'{model_dir}{model_checkpoint}/{rebal_freq}/' ## for sm ; folder: 'monthly', 'weekly', 'daily'

    with fs.open(f'{model_dir}train_dataset_dl.pkl', 'rb') as f:
        train_dataset = pickle.load(f)
    with fs.open(f'{model_dir}val_dataset_dl.pkl', 'rb') as f:
        val_dataset = pickle.load(f)

    return train_dataset, val_dataset


def load_test_dataset(model_dir):
    # full_dir = f'{model_dir}{model_checkpoint}/{rebal_freq}/' ## for sm ; folder: 'monthly', 'weekly', 'daily'

    with fs.open(f'{model_dir}test_dataset_dl.pkl', 'rb') as f:
        test_dataset = pickle.load(f)

    return test_dataset


## ELAINE
def cuda_logger(local_rank: int, message: str):
    if local_rank == 0:
        logger.info(message)


def train_dl_mode(args):
    is_distributed = False #len(args.hosts)>1 and args.backend is not None ## ELAINE
    use_cuda = args.num_gpus >0
    device = torch.device("cuda" if use_cuda else "cpu")
    logger.info(f'device: {device}')

    # if set up for parallelism using DDP (optional) ## ELAINE
    # Get local rank for device assignment
    local_rank = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
    logger.info(f"cuda:{local_rank}")

    # set the seed for random seeding
    torch.cuda.manual_seed_all(args.seed)  ## ELAINE
    if use_cuda:
        torch.cuda.manual_seed(args.seed)

    torch.cuda.empty_cache()
    # getting data
    batch_size = args.batch_size
    train_dataset, val_dataset = load_train_val_dataset(args.model_dir)
    # train_sampler = torch.utils.data.DistributedSampler(train_dataset) if is_distributed else None
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=False, 
                                  pin_memory=True,  # Speeds up data transfer to GPU  ## ELAINE
                                  num_workers=0,    # Parallel data loading  ## ELAINE
                                  prefetch_factor=2)  # Prefetch batches to reduce waiting  ## ELAINE
    # IN: make sure do not shuffle for time series data
    # val_sampler = torch.utils.data.DistributedSampler(val_dataset) if is_distributed else None
    val_dataloader = DataLoader(val_dataset, batch_size, shuffle=False)
    cuda_logger(local_rank, 'val dataloader step passed')
    # IN: make sure do not shuffle for time series data
    cuda_logger(local_rank, 'data loaded')

    # set up model
    hidden_dims_l = ast.literal_eval(args.hidden_dims_l)

    model_args = {
        'input_dim': args.input_dim,
        'lstm_dim': args.lstm_dim,
        'lstm_layers': args.lstm_layers,
        'transformer_layers': args.transformer_layers,
        'num_heads': args.num_heads,
        'attn_dropout': args.attn_dropout,
        'forward_dim': args.forward_dim,
        'hidden_dims_l': hidden_dims_l,
        'dropout': args.dropout,
        'output_dim': args.output_dim
    }

    model = ReturnPredictionModel_Transformer(**model_args).to(device)
    # Wrap the model with DDP  ## ELAINE
    model = DDP(model, device_ids=[local_rank])

    logger.info(f'device {local_rank}: model prepared')

    # critier
    criterion = nn.L1Loss()
    # set optimizr
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # prepare to train
    train_loss_l = []
    val_loss_l = []
    best_loss = np.inf
    patience_counter = 0
    best_model_state = None
    accumulation_steps = 4  ## ELAINE - gradient accumulation (memory)
    scaler = GradScaler()  ## ELAINE - AMP for memory management

    cuda_logger(local_rank, 'training start')
    for epoch in range(args.num_epochs):
        cuda_logger(local_rank, f'EPOCH {epoch} START')
        model.train()
        train_loss_ep = 0
        val_loss_ep = 0

        for batch_idx, batch_item in enumerate(train_dataloader):
            x_batch, y_batch = \
                batch_item['features'].to(device), \
                batch_item['target'].to(device)

            # Clear memory after data transfer  ## ELAINE
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()

            # Use mixed precision for forward pass  ## ELAINE
            with torch.cuda.amp.autocast():
                pred = model(x_batch)
                loss = criterion(pred, y_batch) / accumulation_steps

            # Scale loss and backprop  ## ELAINE
            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            train_loss_ep += loss.item() * accumulation_steps  ## ELAINE

            # Optimizer step with accumulation  ## ELAINE
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_dataloader):
                if scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
                # Free memory after optimization
                torch.cuda.empty_cache()

        avg_train_loss_ep = train_loss_ep/len(train_dataloader)
        print(f'Device: {local_rank}; Epoch {epoch+1}/{args.num_epochs}, Train loss: {avg_train_loss_ep:.3f}')
        train_loss_l.append(avg_train_loss_ep)
        cuda_logger(local_rank, f'training end for epoch {epoch}')

        cuda_logger(local_rank, f'validation start for epoch {epoch}')
        # log validation loss
        model.eval()  # set the model to evaluation mode
        with torch.no_grad():
            for batch_idx, batch_item in enumerate(val_dataloader):
                x_batch, y_batch = \
                    batch_item['features'].to(device), \
                    batch_item['target'].to(device)
                pred = model(x_batch)
                loss = criterion(pred, y_batch)
                val_loss_ep += loss.item()
        avg_val_loss_ep = val_loss_ep/len(val_dataloader)
        val_loss_l.append(avg_val_loss_ep)

        print(f'Epoch {epoch+1}/{args.num_epochs}, Val loss: {avg_val_loss_ep:.3f}')
        cuda_logger(local_rank, f'validation end for epoch {epoch}')

        # check if stop early
        if avg_val_loss_ep < best_loss:
            best_loss = avg_val_loss_ep 
            patience_counter = 0  # reset patience counter once improved
            best_model_state = model.state_dict()  # mark best model
        else:
            patience_counter +=1

        if patience_counter >= args.patience:
            print('Early stopping triggered')
            break  # stop training if validatio loss failed to keep reducing after > patience epochs

    # load best model weights
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # wait for all devices to finish
    dist.barrier()  ## ELAINE
    # save model 
    if local_rank == 0:
        save_model(model, train_loss_l, val_loss_l, args.model_dir, args.model_filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    args_list = [
        ('--model_dir', {'type': str, 'default': 's3://capstone-general/NN-related/dl_model/',
            'help':'put in model directory before model checkpoint name and rebal freq'}),
        ('--model_filename',{'type':str, 'default':'model_dl_gen_sm_0', 'help':'put in wanted model file name'}),
        ('--batch_size', {'type':int, 'default': 48, 'help':'put in batch size for loader'}),
        ('--learning_rate', {'type':float, 'default':0.0002, 'help':'put in learning rate'}),
        ('--input_dim', {'type':int, 'default':9, 'help':'input_dim'}),
        ('--lstm_dim', {'type':int, 'default':64, 'help':'lstm_dim'}),
        ('--lstm_layers', {'type':int, 'default':5, 'help':'lstm_layers'}),
        ('--transformer_layers', {'type':int, 'default':3, 'help':'transformer_layers'}),
        ('--num_heads', {'type':int, 'default':4, 'help':'num_heads'}),
        ('--attn_dropout', {'type':float, 'default':0.1, 'help':'attn_dropout'}),
        ('--forward_dim', {'type':int, 'default':64, 'help':'forward_dim'}),
        ('--hidden_dims_l', {'type':str, 'default':'[128,32]', 'help':'hidden_dims_l'}),
        ('--dropout', {'type':float, 'default':0.3, 'help':'dropout'}),
        ('--output_dim', {'type':int, 'default':1, 'help':'output_dim'}),
        ('--num_epochs', {'type':int, 'default':30, 'help':'num_epochs'}),
        ('--patience', {'type':int, 'default':5, 'help':'patience'}),
        ('--seed', {'type':int, 'default':42, 'help':'seed'})
]

    for arg, kwargs in args_list:
        parser.add_argument(arg, **kwargs)

    # other environment args
    # parser.add_argument(
    #         "--backend",
    #         type=str,
    #         default=None,
    #         help="backend for distributed training (tcp, gloo on cpu and gloo, nccl on gpu)",
    #     )

    # Container environment
    parser.add_argument("--hosts", type=list, default=json.loads(os.environ["SM_HOSTS"]))
    parser.add_argument("--current-host", type=str, default=os.environ["SM_CURRENT_HOST"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    # parser.add_argument("--data-dir", type=str, default=os.environ["SM_CHANNEL_TRAINING"])
    parser.add_argument("--num-gpus", type=int, default=os.environ["SM_NUM_GPUS"])

    # excecute the train
    train_dl_mode(parser.parse_args())


