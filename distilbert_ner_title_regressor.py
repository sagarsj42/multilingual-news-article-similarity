#!/usr/bin/env python
# coding: utf-8

# In[2]:


# get_ipython().system('mkdir -p /scratch/sagarsj42/torch-cache')
# get_ipython().system('mkdir -p /scratch/sagarsj42/transformers')
import os
os.chdir('/scratch/sagarsj42')
os.environ['TORCH_HOME'] = '/scratch/sagarsj42/torch-cache'
os.environ['TRANSFORMERS_CACHE'] = '/scratch/sagarsj42/transformers'


# In[18]:


import json
from functools import partial

import pandas as pd

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torchmetrics

from transformers import AutoTokenizer, AutoModel

# In[6]:


# get_ipython().system('scp sagarsj42@ada:/share1/sagarsj42/semeval8-train-sss.csv .')
# get_ipython().system('scp sagarsj42@ada:/share1/sagarsj42/semeval8-dev.csv .')
# get_ipython().system('scp sagarsj42@ada:/share1/sagarsj42/semeval-2022-task-8-eval-df.csv .')
# get_ipython().system('scp sagarsj42@ada:/share1/sagarsj42/semeval8-train-ner.json .')
# get_ipython().system('scp sagarsj42@ada:/share1/sagarsj42/semeval8-dev-ner.json .')
# get_ipython().system('scp sagarsj42@ada:/share1/sagarsj42/semeval8-test-ner.json .')


# In[7]:


EXP_NAME = 'mlns-ner-title-distilbert-regressor'
TRAIN_BATCH_SIZE = 2
DEV_BATCH_SIZE = 8
ACCUMULATE_GRAD = 8


# In[8]:


class SimilarityRegressor(nn.Module):
    def __init__(self, encoder, embed_size=768, hidden_size=256):
        super(SimilarityRegressor, self).__init__()

        self.encoder = encoder
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        
        self.linear1 = nn.Linear(self.embed_size, self.hidden_size)
        self.activation1 = nn.LeakyReLU(negative_slope=0.1)
        self.dropout1 = nn.Dropout(p=0.2)
        # self.linear2 = nn.Linear(2*self.hidden_size, self.hidden_size//3)
        # self.dropout2 = nn.Dropout(p=0.2)
        # self.activation2 = nn.LeakyReLU(negative_slope=0.1)
        self.linear3 = nn.Linear(2*self.hidden_size, 1)
        self.activation3 = nn.Sigmoid()

    def common_compute(self, x):
        x = self.encoder(**x)[0][:, 0]
        x = self.linear1(x)
        x = self.activation1(x)
        x = self.dropout1(x)

        return x
    
    def forward(self, x1, x2):
        x1 = self.common_compute(x1)
        x2 = self.common_compute(x2)
        x = torch.cat([torch.abs(x1 - x2), (x1 + x2)], dim=-1)
        # x = self.linear2(x)
        # x = self.activation2(x)
        # x = self.dropout2(x)
        x = self.linear3(x)
        x = 3*self.activation3(x) + 1

        return x


# In[9]:


class LitSimilarityRegressor(pl.LightningModule):
    def __init__(self, encoder, embed_size=768, hidden_size=256):
        super(LitSimilarityRegressor, self).__init__()
        self.model = SimilarityRegressor(encoder, embed_size=embed_size, hidden_size=hidden_size)

        self.train_loss = torchmetrics.MeanMetric(compute_on_step=True)
        self.dev_loss = torchmetrics.MeanMetric(compute_on_step=False)

        self.train_mape = torchmetrics.MeanAbsolutePercentageError(compute_on_step=True)
        self.dev_mape = torchmetrics.MeanAbsolutePercentageError(compute_on_step=False)

        self.train_pcc = torchmetrics.PearsonCorrCoef(compute_on_step=True)
        self.dev_pcc = torchmetrics.PearsonCorrCoef(compute_on_step=False)

    def forward(self, x1, x2):
        return self.model(x1, x2)

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=5e-6, betas=(0.9, 0.99), eps=1e-8, weight_decay=0.01)

    def training_step(self, batch, batch_idx):
        x1, x2, scores = batch
        output = self(x1, x2)
        loss = F.mse_loss(input=output, target=scores)

        return {'loss': loss, 'preds': output, 'target': scores}

    def validation_step(self, batch, batch_idx):
        x1, x2, scores = batch
        output = self(x1, x2)
        loss = F.mse_loss(input=output, target=scores)

        return {'loss': loss, 'preds': output, 'target': scores}

    def predict_step(self, batch, batch_idx):
        x1, x2, _ = batch
        output = self(x1, x2)

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


# In[12]:


class MultilingualNewsSimDataset(Dataset):
    def __init__(self, df, ner):
        super(MultilingualNewsSimDataset, self).__init__()
        self.df = df
        self.ner = ner

    def __getitem__(self, idx):
        pair_id = self.df.iloc[idx]['pair_id']
        ners = self.ner[pair_id]
        sample_dict =  self.df.iloc[idx][['pair_id', 'title_1', 'meta_description_1', 'meta_keywords_1', 'tags_1', 
                                          'title_2', 'meta_description_2', 'meta_keywords_2', 'tags_2', 
                                          'score']].to_dict()
        sample_dict['ner_1'] = ners[0]
        sample_dict['ner_2'] = ners[1]
        
        return sample_dict
        
    def __len__(self):
        return self.df.shape[0]


# In[13]:


def collate_fn(batch, tokenizer):
    texts_1, texts_2, scores = list(), list(), list()
    for sample in batch:
        text1 = str(sample['ner_1']).strip() +                 str(sample['title_1']).lower().strip() + str(sample['meta_description_1']).lower().strip() +             ' '.join(sample['meta_keywords_1']).lower().strip() + ' '.join(sample['tags_1']).lower().strip()
        text2 = str(sample['ner_2']).strip() +                 str(sample['title_2']).lower().strip() + str(sample['meta_description_2']).lower().strip() +             ' '.join(sample['meta_keywords_2']).lower().strip() + ' '.join(sample['tags_2']).lower().strip()
        
        score = torch.tensor([sample['score']])
        texts_1.append(text1)
        texts_2.append(text2)
        scores.append(score)

    texts_1 = tokenizer(texts_1, truncation=True, padding=True, return_tensors='pt')
    texts_2 = tokenizer(texts_2, truncation=True, padding=True, return_tensors='pt')
    scores = torch.cat(scores, dim=0).unsqueeze(1)

    return texts_1, texts_2, scores


# In[14]:


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


# In[15]:


tokenizer = AutoTokenizer.from_pretrained("distilbert-base-multilingual-cased")
encoder = AutoModel.from_pretrained("distilbert-base-multilingual-cased")


# In[16]:


train_df = pd.read_csv(os.path.join(os.getcwd(), 'semeval8-train-sss.csv'), index_col=0)
dev_df = pd.read_csv('semeval8-dev.csv', index_col=0)
test_df = pd.read_csv('semeval-2022-task-8-eval-df.csv', index_col=0)

train_df.shape, dev_df.shape, test_df.shape


# In[20]:


with open('semeval8-train-ner.json', 'r') as f:
    train_ner = json.load(f)
with open('semeval8-dev-ner.json', 'r') as f:
    dev_ner = json.load(f)
with open('semeval8-test-ner.json', 'r') as f:
    test_ner = json.load(f)
    
len(list(train_ner.keys())), len(list(dev_ner.keys())), len(list(test_ner.keys()))


# In[22]:


train_dataset = MultilingualNewsSimDataset(train_df, train_ner)
dev_dataset = MultilingualNewsSimDataset(dev_df, dev_ner)
test_dataset = MultilingualNewsSimDataset(test_df, test_ner)

len(train_dataset), len(dev_dataset), len(test_dataset)


# In[23]:


train_dataset[4]


# In[25]:


collate_partial = partial(collate_fn, tokenizer=tokenizer)
dataloader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, collate_fn=collate_partial)
next(iter(dataloader))


# In[26]:


data_module = MLNSDataModule(train_dataset, dev_dataset, test_dataset, TRAIN_BATCH_SIZE, DEV_BATCH_SIZE, collate_fn=collate_fn, tokenizer=tokenizer)
data_module


# In[27]:


model = LitSimilarityRegressor(encoder, embed_size=768, hidden_size=256)
model


# In[28]:


logger = pl.loggers.WandbLogger(save_dir=EXP_NAME, project=EXP_NAME, log_model=False)
logger


# In[29]:


checkpoint_callback = ModelCheckpoint(
    dirpath=os.path.join(EXP_NAME, 'lightning-checkpoints'),
    filename='{epoch}-{step}',
    monitor='dev/pcc',
    mode='max',
    save_top_k=5,
    verbose=True,
    save_last=True,
    save_weights_only=False,
    every_n_epochs=1
)


# In[30]:


trainer = pl.Trainer(
    max_epochs=10,
    accumulate_grad_batches=ACCUMULATE_GRAD,
    accelerator='gpu',
    gpus=1,
    # strategy='ddp',
    # overfit_batches=10,
    check_val_every_n_epoch=1, val_check_interval=0.25,
    log_every_n_steps=2, enable_progress_bar=True,
    gradient_clip_val=0.25, track_grad_norm=2,
    enable_checkpointing=True,
    callbacks=[checkpoint_callback],
    logger=logger,
    enable_model_summary=True
)

trainer


# In[31]:


# trainer.fit(model, datamodule=data_module)


# In[32]:

model = LitSimilarityRegressor.load_from_checkpoint(checkpoint_path='mlns-ner-title-distilbert-regressor/lightning-checkpoints/last.ckpt', 
    encoder=encoder, embed_size=768, hidden_size=256)

test_pred = trainer.predict(model, datamodule=data_module)
print(len(test_pred))


# In[33]:


all_outputs = list()
for batch_outputs in test_pred:
    all_outputs.append(batch_outputs['preds'])
all_outputs = torch.cat(all_outputs, dim=0)

all_outputs.shape

pd.DataFrame(list(zip(test_df.pair_id, all_outputs.squeeze(1).tolist())), columns=['pair_id', 'Overall']).to_csv(EXP_NAME + '-test-results.csv')

# In[ ]:

collate_partial = partial(collate_fn, tokenizer=tokenizer)
train_dataloader = DataLoader(train_dataset, batch_size=DEV_BATCH_SIZE, shuffle=False, collate_fn=collate_partial)
print(len(train_dataloader))

train_pred = trainer.predict(model, dataloaders=[train_dataloader])

all_outputs = list()
for batch_outputs in train_pred:
    all_outputs.append(batch_outputs['preds'])
all_outputs = torch.cat(all_outputs, dim=0)

pd.DataFrame(list(zip(train_df.pair_id, all_outputs.squeeze(1).tolist())), columns=['pair_id', 'Overall']).to_csv(EXP_NAME + '-train-results.csv')

dev_dataloader = DataLoader(dev_dataset, batch_size=DEV_BATCH_SIZE, shuffle=False, collate_fn=collate_partial)

dev_pred = trainer.predict(model, dataloaders=[dev_dataloader])

all_outputs = list()
for batch_outputs in dev_pred:
    all_outputs.append(batch_outputs['preds'])
all_outputs = torch.cat(all_outputs, dim=0)

pd.DataFrame(list(zip(dev_df.pair_id, all_outputs.squeeze(1).tolist())), columns=['pair_id', 'Overall']).to_csv(EXP_NAME + '-dev-results.csv')

# %%
