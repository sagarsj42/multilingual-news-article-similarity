import os
os.chdir('/scratch/sagarsj42')
os.environ['TORCH_HOME'] = '/scratch/sagarsj42/torch-cache'
os.environ['TRANSFORMERS_CACHE'] = '/scratch/sagarsj42/transformers'

from functools import partial

import pandas as pd

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import torchmetrics

import transformers
from transformers import XLMRobertaTokenizer, XLMRobertaModel
# from transformers import get_linear_schedule_with_warmup

print('loaded packages')

DEVICE = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
N_GPUS = torch.cuda.device_count()

print('Using device:', DEVICE)
print('# GPUs:', N_GPUS)

EXP_NAME = 'mlns-xlmr-regressor-relu'
TRAIN_BATCH_SIZE = 4
DEV_BATCH_SIZE = 8
ACCUMULATE_GRAD = 8
N_EPOCHS = 10

class SimilarityRegressor(nn.Module):
    def __init__(self, encoder, embed_size=768, hidden_size=256):
        super(SimilarityRegressor, self).__init__()

        self.encoder = encoder
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        
        self.linear1 = nn.Linear(self.embed_size, self.hidden_size)
        self.activation1 = nn.LeakyReLU(negative_slope=0.1)
        self.dropout1 = nn.Dropout(p=0.2)
        self.linear2 = nn.Linear(2*self.hidden_size, self.hidden_size//2)
        self.dropout2 = nn.Dropout(p=0.2)
        self.activation2 = nn.LeakyReLU(negative_slope=0.1)
        self.linear3 = nn.Linear(self.hidden_size//2, 1)
        self.activation3 = nn.ReLU()

    def common_compute(self, x):
        x = self.encoder(**x).pooler_output
        x = self.linear1(x)
        x = self.activation1(x)
        x = self.dropout1(x)

        return x
    
    def forward(self, x1, x2):
        x1 = self.common_compute(x1)
        x2 = self.common_compute(x2)
        x = torch.cat([torch.abs(x1 - x2), (x1 + x2)], dim=-1)
        x = self.linear2(x)
        x = self.activation2(x)
        x = self.dropout2(x)
        x = self.linear3(x)
        x = self.activation3(x)

        return x

class LitSimilarityRegressor(pl.LightningModule):
    def __init__(self, encoder, scheduler_args, embed_size=768, hidden_size=256):
        super(LitSimilarityRegressor, self).__init__()
        self.model = SimilarityRegressor(encoder, embed_size=embed_size, hidden_size=hidden_size)
        self.scheduler_args = scheduler_args

        self.train_loss = torchmetrics.MeanMetric(compute_on_step=True)
        self.dev_loss = torchmetrics.MeanMetric(compute_on_step=False)

        self.train_mape = torchmetrics.MeanAbsolutePercentageError(compute_on_step=True)
        self.dev_mape = torchmetrics.MeanAbsolutePercentageError(compute_on_step=False)

        self.train_pcc = torchmetrics.PearsonCorrCoef(compute_on_step=True)
        self.dev_pcc = torchmetrics.PearsonCorrCoef(compute_on_step=False)

    def forward(self, x1, x2):
        return self.model(x1, x2)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=1e-5, betas=(0.9, 0.99), eps=1e-8, weight_decay=0.01)
        scheduler = transformers.get_linear_schedule_with_warmup(optimizer, **self.scheduler_args)

        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        x1, x2, scores = batch
        output = self(x1, x2)
        loss = F.mse_loss(input=output, target=scores)

        return {'loss': loss, 'preds': output.clamp(min=1.0, max=4.0), 'target': scores}

    def validation_step(self, batch, batch_idx):
        x1, x2, scores = batch
        output = self(x1, x2)
        loss = F.mse_loss(input=output, target=scores)

        return {'loss': loss, 'preds': output.clamp(min=1.0, max=4.0), 'target': scores}

    def predict_step(self, batch, batch_idx):
        x1, x2, _ = batch
        output = self(x1, x2).clamp(min=1, max=4)

        return {'preds': output}

    def training_step_end(self, outs):
        loss = outs['loss']
        preds = outs['preds']
        target = outs['target']

        self.log('train/step/loss', self.train_loss(loss))
        self.log('train/step/mape', self.train_mape(preds, target))
        self.log('train/step/pcc', self.train_pcc(preds, target))

    def validation_step_end(self, outs):
        loss = outs['loss']
        preds = outs['preds']
        target = outs['target']

        self.dev_loss(loss)
        self.dev_mape(preds, target)
        self.dev_pcc(preds, target)

    def training_epoch_end(self, outs):
        self.log('train/epoch/loss', self.train_loss)
        self.log('train/epoch/mape', self.train_mape)
        self.log('train/epoch/pcc', self.train_pcc)

    def validation_epoch_end(self, outs):
        self.log('dev/loss', self.dev_loss)
        self.log('dev/mape', self.dev_mape)
        self.log('dev/pcc', self.dev_pcc)

class MultilingualNewsSimDataset(Dataset):
    def __init__(self, df):
        super(MultilingualNewsSimDataset, self).__init__()
        self.df = df

    def __getitem__(self, idx):
        return self.df.iloc[idx][['pair_id', 'text_1', 'text_2', 'score']].to_dict()
        
    def __len__(self):
        return self.df.shape[0]

def collate_fn(batch, tokenizer):
    texts_1, texts_2, scores = list(), list(), list()
    for sample in batch:
        text1 = str(sample['text_1']).lower().strip()
        text2 = str(sample['text_2']).lower().strip()
        
        score = torch.tensor([sample['score']])
        texts_1.append(text1)
        texts_2.append(text2)
        scores.append(score)

    texts_1 = tokenizer(texts_1, truncation=True, padding=True, return_tensors='pt')
    texts_2 = tokenizer(texts_2, truncation=True, padding=True, return_tensors='pt')
    scores = torch.cat(scores, dim=0).unsqueeze(1)

    return texts_1, texts_2, scores

class MLNSDataModule(pl.LightningDataModule):
    def __init__(self, train_dataset, dev_dataset, test_dataset, train_batch_size, dev_batch_size, collate_fn, tokenizer):
        super(MLNSDataModule, self).__init__()
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset
        self.train_batch_size = train_batch_size
        self.dev_batch_size = dev_batch_size
        self.collate_fn = collate_fn
        self.tokenizer = tokenizer

    def train_dataloader(self):
        collate_partial = partial(self.collate_fn, tokenizer=self.tokenizer)
        return DataLoader(self.train_dataset, shuffle=True, batch_size=self.train_batch_size, collate_fn=collate_partial)

    def val_dataloader(self):
        collate_partial = partial(self.collate_fn, tokenizer=self.tokenizer)
        return DataLoader(self.dev_dataset, shuffle=False, batch_size=self.dev_batch_size, collate_fn=collate_partial)

    def test_dataloader(self):
        collate_partial = partial(self.collate_fn, tokenizer=self.tokenizer)
        return DataLoader(self.test_dataset, shuffle=False, batch_size=self.dev_batch_size, collate_fn=collate_partial)

    def predict_dataloader(self):
        return self.test_dataloader()

tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
encoder = XLMRobertaModel.from_pretrained('xlm-roberta-base')

train_df = pd.read_csv(os.path.join(os.getcwd(), 'semeval8-train-sss.csv'), index_col=0)
dev_df = pd.read_csv('semeval8-dev.csv', index_col=0)
test_df = pd.read_csv('semeval-2022-task-8-eval-df.csv', index_col=0)

train_dataset = MultilingualNewsSimDataset(train_df)
dev_dataset = MultilingualNewsSimDataset(dev_df)
test_dataset = MultilingualNewsSimDataset(test_df)

print('train/dev/test data sizes', len(train_dataset), len(dev_dataset), len(test_dataset))

data_module = MLNSDataModule(train_dataset, dev_dataset, test_dataset, TRAIN_BATCH_SIZE, DEV_BATCH_SIZE, collate_fn=collate_fn, tokenizer=tokenizer)

num_training_steps = (len(train_dataset) * N_EPOCHS) / (TRAIN_BATCH_SIZE * ACCUMULATE_GRAD)
scheduler_args = {'num_training_steps': num_training_steps, 'num_warmup_steps': num_training_steps // 10}

model = LitSimilarityRegressor(encoder, embed_size=768, hidden_size=512, scheduler_args=scheduler_args)

logger = pl.loggers.WandbLogger(save_dir=EXP_NAME, project=EXP_NAME, log_model=False)

checkpoint_callback = ModelCheckpoint(
    dirpath=os.path.join(EXP_NAME, 'lightning-checkpoints'),
    filename='{epoch}-{step}',
    # monitor='dev/pcc',
    # mode='max',
    save_top_k=-1,
    verbose=True,
    save_last=True,
    save_weights_only=False,
    every_n_epochs=1
)

lr_monitor_callback = LearningRateMonitor(logging_interval='step')

trainer = pl.Trainer(
    max_epochs=N_EPOCHS,
    accumulate_grad_batches=ACCUMULATE_GRAD,
    accelerator='gpu',
    gpus=1,
    # overfit_batches=10,
    check_val_every_n_epoch=1, val_check_interval=0.25,
    log_every_n_steps=2, enable_progress_bar=True,
    gradient_clip_val=0.25, track_grad_norm=2,
    enable_checkpointing=True,
    callbacks=[checkpoint_callback, lr_monitor_callback],
    logger=logger,
    enable_model_summary=True
)

trainer.fit(model, datamodule=data_module)

test_pred = trainer.predict(datamodule=data_module)

all_outputs = list()
for batch_outputs in test_pred:
    all_outputs.append(batch_outputs['preds'])
all_outputs = torch.cat(all_outputs, dim=0)

pd.DataFrame(list(zip(test_df.pair_id, all_outputs.squeeze(1).tolist())), columns=['pair_id', 'Overall']).to_csv(EXP_NAME + '-test-results.csv')
