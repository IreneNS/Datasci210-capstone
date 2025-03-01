# imports
# any of these may need to be added to requirements.txt. See file
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from sklearn.metrics import accuracy_score
import argparse
import pandas as pd
import numpy as np
import os
from transformers import BertTokenizer, BertModel, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_from_disk
from tqdm.auto import tqdm
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# this is just for suppressing a transformer warning
loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
for logger in loggers:
    if "transformers" in logger.name.lower():
        logger.setLevel(logging.ERROR)

#### COPIED FROM COLAB ####
# classes and functions
class CustomDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # Get a row of data
        data_row = self.dataframe.iloc[idx]
        for column in data_row.index:
            if isinstance(data_row[column], pd.Timestamp):
                data_row[column] = data_row[column].timestamp()
        # Return the row as a dictionary or a tuple, depending on your needs
        return data_row.to_dict()

class ProgressTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.training_bar = None

    def train(self, *args, **kwargs):
        num_training_steps = (
            len(self.train_dataset) // (self.args.per_device_train_batch_size * self.args.gradient_accumulation_steps)
        ) * self.args.num_train_epochs
        self.training_bar = tqdm(total=num_training_steps, desc="Training Progress")
        return super().train(*args, **kwargs)

    def log(self, logs):#, start_time=None):
        super().log(logs)#, start_time)
        if 'step' in logs:
            self.training_bar.n = logs['step']
            self.training_bar.refresh()


class MarketDataset(Dataset):
    def __init__(self, df, feature_mean, feature_std):
        """Initialize dataset with scaled features."""
        self.features = (np.array(df["Summary"].tolist()) - feature_mean) / feature_std
        self.features = self.features.tolist()
        self.labels = df["binary_return"].astype(int).tolist()

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return {
            "inputs": torch.tensor(self.features[idx], dtype=torch.float32),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }

class EmbeddingClassifier(nn.Module):
    def __init__(self, embedding_dim=768, hidden_dim=256, num_labels=2):
        """Initialize MLP with an extra hidden layer."""
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),  # Extra layer
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_labels)
        )

    def forward(self, inputs, labels=None):
        logits = self.network(inputs)
        if torch.any(torch.isnan(logits)):
            print("Logits contain NaN values!")
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, logits
        return logits
        

def convert_to_binary_labels(df):
    """Converts next-day returns into binary labels: 1 if positive, 0 if negative/zero."""
    df["binary_return"] = (df["return_sp_lag"] > 0).astype(int)
    return df
    

def compute_metrics(eval_pred):
    """Calculate accuracy for evaluation."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, predictions)}


#### SAGEMAKER REQUIRED: MAIN FUNCTION ####
## The training job will run this. ##
def main():

    # initiate argument parser
    # this will accept our hyperparameters, environment variables, data paths
    parser = argparse.ArgumentParser()

    # TRIPLE-CHECK the data type restrictions.
    # note the default values. comparing to these is also a good way to check if your arguments were successfully passed. 
    parser.add_argument("--pretrained_model", type=str, default=os.environ['BASE_MODEL']) 
    parser.add_argument("--train", type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument("--val", type=str, default=os.environ['SM_CHANNEL_VAL'])
    parser.add_argument("--output_dir", type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    args, _ = parser.parse_known_args()
    logger.info("Parsed arguments: %s", args)  

    
    #### COPIED FROM COLAB ####
    # begin model relevant objects
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def extract_embeddings_batch(texts, batch_size=128, max_length=512):
        """
        Converts a batch of financial news texts into FinBERT embeddings (768-dimensional CLS tokens),
        optimized for A100 GPU with larger batch sizes and mixed precision.
        """
        # Use dynamic batching to handle variable input sizes efficiently
        num_texts = len(texts)
        print('text length: ', num_texts)
        embeddings = []
        tokenizer = BertTokenizer.from_pretrained(args.pretrained_model)
        embedding_model = BertModel.from_pretrained(args.pretrained_model).to(device)
        for param in embedding_model.parameters():
            param.requires_grad = False  # Freeze all layers
    
        # Process in batches, optimized for A100's high memory and parallel processing
        for i in range(0, num_texts, batch_size):
            batch_texts = texts[i:i + batch_size]
            # Tokenize with padding to max_length, return PyTorch tensors
            inputs = tokenizer(
                batch_texts,
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt",
                return_attention_mask=True  # Ensure attention mask is included for efficiency
            ).to(device)  # Move to GPU without specifying dtype here

            with torch.no_grad():
                # Use FP16 for embedding_model outputs via autocast for A100 optimization
                with torch.cuda.amp.autocast():
                    outputs = embedding_model(**inputs)
                    batch_embeddings = outputs.last_hidden_state[:, 0, :].half()  # Convert to FP16 on GPU
                batch_embeddings = batch_embeddings.cpu().numpy()  # Move to CPU for NumPy

            embeddings.extend(batch_embeddings)
    
        return np.array(embeddings, dtype=np.float16)  # Use FP16 for NumPy array to save memory

    def aggregate_embeddings(df, save_path=None, batch_size=128, max_length=512):
        """
        Groups news articles by day and averages FinBERT embeddings into a single vector
        """
        unique_dates = df["Date"].unique()
        daily_embeddings = []
    
        # Add tqdm progress bar for embedding aggregation
        for date in tqdm(unique_dates[:3], desc='aggregating embeddings'):
            texts = df[df["Date"] == date]["Summary"].tolist()
            if not texts:  # Skip empty days
                continue
    
            # Process texts in batches optimized for A100
            embeddings = extract_embeddings_batch(texts, batch_size=batch_size, max_length=max_length)
            avg_embedding = np.mean(embeddings, axis=0, dtype=np.float16)  # Use FP16 for memory efficiency
            daily_embeddings.append([date, avg_embedding])
    
        # Convert to DataFrame
        daily_embeddings_df = pd.DataFrame(daily_embeddings, columns=["Date", "Summary"])
        result = daily_embeddings_df.merge(df[["Date", "binary_return"]].drop_duplicates(), on="Date")
    
        if save_path:
            result.to_parquet(save_path, compression='snappy')  # Use efficient compression
            print(f"✅ Aggregated embeddings saved to {save_path}")
    
        return result

    
    #### SAGEMAKER REQUIRED: Data Loading ####
    # keep in mind args.train (the passed train path) is to the directory and not a file
    train_df = pd.read_parquet(os.path.join(args.train, 'train.parquet'))
    val_df = pd.read_parquet(os.path.join(args.val, 'val.parquet'))
    train_df = aggregate_embeddings(convert_to_binary_labels(train_df))
    val_df = aggregate_embeddings(convert_to_binary_labels(val_df))

    
    # new function due to sagemaker picky errors? write as needed depending on logs
    def check_data(df):
        if df.isnull().values.any():
            print("Data contains NaN values!")

    check_data(train_df)
    check_data(val_df)

    
    #### COPIED FROM COLAB ####
    train_features = np.array(train_df["Summary"].tolist())
    feature_mean = train_features.mean(axis=0)
    feature_std = train_features.std(axis=0)

    train_dataset = MarketDataset(train_df, feature_mean, feature_std)
    val_dataset = MarketDataset(val_df, feature_mean, feature_std)


    classifier_model = EmbeddingClassifier(embedding_dim=768).to(device)

    #### SAGEMAKER REQUIRED // COPIED FROM COLAB: TRAINING SETUP ####
    # pass arguments as appropriate/desired.
    training_args = TrainingArguments(
        output_dir=args.output_dir, # !CHANGE! replace with arg
        per_device_train_batch_size=32,
        num_train_epochs=args.epochs,
        save_strategy="epoch",
        # logging_dir="./logs",
        evaluation_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",  # Use accuracy to pick best model
        greater_is_better=True,
        save_total_limit=1,
        report_to="none",
        fp16=True if torch.cuda.is_available() else False,
        weight_decay=0.01, # !CHANGE! can be HP arg
        logging_steps=10, 
        gradient_accumulation_steps=4, # !CHANGE! can be HP arg
        dataloader_num_workers=4, # !CHANGE! can be HP arg
        optim="adamw_torch",
        lr_scheduler_type="cosine", # !CHANGE!can be HP arg
        learning_rate=0.01,  # !CHANGE! can be HP arg
        warmup_ratio=0.1, # !CHANGE! can be HP arg
        max_grad_norm=1.0, # !CHANGE! can be HP arg
)

    trainer = ProgressTrainer(
        model=classifier_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )


    trainer.train()
    
    # need to save model from inside script, this is to put model.tar.gz in specified directory
    trainer.save_model(args.output_dir)

    # bonus: want to store another file (loss record, epoch logs, notes, etc)? you can save them normally. it's best to put them all in similar places
    # whateverlog.to_csv(os.path.join(args.output_dir, 'whatever.csv'))

if __name__ == "__main__":
    # to run
    main()