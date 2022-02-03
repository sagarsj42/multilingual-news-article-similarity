#!/usr/bin/env python
# coding: utf-8

# In[1]:
# from IPython import get_ipython

# get_ipython().system('mkdir -p /scratch/dhaval.taunk/torch-cache')
# get_ipython().system('mkdir -p /scratch/dhaval.taunk/transformers')
import os
# os.mkdir('/scratch')
# os.mkdir('/scratch/dhaval.taunk')
# os.mkdir(' /scratch/dhaval.taunk/torch-cache')
# os.mkdir('/scratch/dhaval.taunk/transformers')
os.chdir('/scratch/sagarsj42')
os.environ['TORCH_HOME'] = '/scratch/sagarsj42/torch-cache'
os.environ['TRANSFORMERS_CACHE'] = '/scratch/sagarsj42/transformers'
os.environ['TOKENIZERS_PARALLELISM'] = 'true'


# In[2]:


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

from transformers import XLMRobertaTokenizer, XLMRobertaModel
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModel


# In[3]:


# !echo 'yes' | scp dhaval.taunk@ada:/share1/dhaval.taunk/semeval8-train-sss.csv .
# !echo 'yes' | scp dhaval.taunk@ada:/share1/dhaval.taunk/semeval8-dev.csv .


# In[4]:


DEVICE = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
N_GPUS = torch.cuda.device_count()



# In[6]:


EXP_NAME = 'mlns-distilbert-regressor_data_augmentation'
TRAIN_BATCH_SIZE = 4
DEV_BATCH_SIZE = 8
ACCUMULATE_GRAD = 8


# In[7]:


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
        self.activation2 = nn.ReLU()
        self.linear3 = nn.Linear(self.hidden_size//2, 1)
        self.activation3 = nn.Sigmoid()

    def common_compute(self, x):
        # print(self.encoder(**x)[0].shape)
        x = self.encoder(**x)[0][:, 0]
        x = self.linear1(x)
        x = self.activation1(x)
        x = self.dropout1(x)

        return x
    
    def forward(self, x1, x2):
        x1 = self.common_compute(x1)
        x2 = self.common_compute(x2)
        x = torch.cat([(x1 + x2)/2.0, torch.abs(x1 - x2)], dim=-1)
        x = self.linear2(x)
        x = self.activation2(x)
        x = self.dropout2(x)
        x = self.linear3(x)
        x = 5*self.activation3(x)

        return x


# In[8]:


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
        return AdamW(self.parameters(), lr=5e-6, betas=(0.9, 0.99), eps=1e-8, weight_decay=0.02)

    def training_step(self, batch, batch_idx):
        x1, x2, scores = batch
        output = self(x1, x2)
        loss = F.mse_loss(input=output, target=scores)

        return {'loss': loss, 'preds': output, 'target': scores}

    def validation_step(self, batch, batch_idx):
        x1, x2, scores = batch
        output = self(x1, x2)
        loss = F.mse_loss(input=output, target=scores)

        return {'loss': loss, 'preds': output.clamp(min=1.0, max=4.0), 'target': scores}

    def predict_step(self, batch, batch_idx):
        x1, x2, _ = batch
        output = self(x1, x2)

        return {'preds': output.clamp(min=1.0, max=4.0)}

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


# In[9]:


class MultilingualNewsSimDataset(Dataset):
    def __init__(self, df):
        super(MultilingualNewsSimDataset, self).__init__()
        self.df = df

    def __getitem__(self, idx):
        return self.df.iloc[idx][['pair_id', 'meta_description_1', 'title_1', 'text_1', 'meta_description_2', 'title_2', 'text_2', 'score']].to_dict()
        
    def __len__(self):
        return self.df.shape[0]


# In[10]:


def collate_fn(batch, tokenizer):
    texts_1, texts_2, scores = list(), list(), list()
    for sample in batch:
        text1 = str(sample['title_1']).lower().strip()+str(sample['meta_description_1']).lower().strip()+str(sample['text_1']).lower().strip()
        text2 = str(sample['title_2']).lower().strip()+str(sample['meta_description_2']).lower().strip()+str(sample['text_2']).lower().strip()

        score = torch.tensor([sample['score']])
        texts_1.append(text1)
        texts_2.append(text2)
        scores.append(score)

    texts_1 = tokenizer(texts_1, truncation=True, padding=True, return_tensors='pt')
    texts_2 = tokenizer(texts_2, truncation=True, padding=True, return_tensors='pt')
    scores = torch.cat(scores, dim=0).unsqueeze(1)

    return texts_1, texts_2, scores


# In[11]:


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


# In[12]:


# tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
# encoder = XLMRobertaModel.from_pretrained('xlm-roberta-base')

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-multilingual-cased")

encoder = AutoModel.from_pretrained("distilbert-base-multilingual-cased")

encoder


# In[13]:


train_df = pd.read_csv('semeval8-train-sss.csv', index_col=0)
dev_df = pd.read_csv('semeval8-dev.csv', index_col=0)
test_df = pd.read_csv('semeval-2022-task-8-eval-df.csv', index_col=0)

train_df.shape, dev_df.shape, test_df.shape


# In[14]:


import re
import random
def split_text(text):
    result = re.split('(?<=\.)\s|,\s*|;\s*', text)
    
    random.shuffle(result)
    
    return ' '.join(result)


# In[15]:


train_sample1 = train_df.sample(frac=1)
train_sample2 = train_df.sample(frac=1)
train_sample3 = train_df.sample(frac=1)

train_sample1['text_1'] = train_sample1['text_1'].fillna('').apply(split_text)
train_sample1['text_2'] = train_sample1['text_2'].fillna('').apply(split_text)

train_sample2['text_1'] = train_sample2['text_1'].fillna('').apply(split_text)
train_sample2['text_2'] = train_sample2['text_2'].fillna('').apply(split_text)

train_sample3['text_1'] = train_sample3['text_1'].fillna('').apply(split_text)
train_sample3['text_2'] = train_sample3['text_2'].fillna('').apply(split_text)

train_df = pd.concat([train_df, train_sample1, train_sample2, train_sample3]).sample(frac=1)

train_df.shape, dev_df.shape, test_df.shape


# In[16]:


train_dataset = MultilingualNewsSimDataset(train_df)
dev_dataset = MultilingualNewsSimDataset(dev_df)
test_dataset = MultilingualNewsSimDataset(test_df)

len(train_dataset), len(dev_dataset), len(test_dataset)


# In[17]:


# collate_partial = partial(collate_fn, tokenizer=tokenizer)
# dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate_partial)
# batch = next(iter(dataloader))
# batch


# In[18]:


# model.device


# In[19]:


# x1, x2, _ = batch
# outputs = model(x1.to(DEVICE), x2.to(DEVICE))
# outputs.shape


# In[20]:


# outputs


# In[21]:


data_module = MLNSDataModule(train_dataset, dev_dataset, test_dataset, TRAIN_BATCH_SIZE, DEV_BATCH_SIZE, collate_fn=collate_fn, tokenizer=tokenizer)
data_module


# In[22]:


# model = LitSimilarityRegressor(encoder, embed_size=768, hidden_size=256)
chkpt_path = '/scratch/sagarsj42/mlns-distilbert-regressor_data_augmentation/lightning-checkpoints/last.ckpt'
model = LitSimilarityRegressor.load_from_checkpoint(checkpoint_path=chkpt_path, encoder=encoder, embed_size=768, hidden_size=256)

# model = LitSimilarityRegressor.load_from_checkpoint(encoder, )
model


# In[23]:


logger = pl.loggers.WandbLogger(save_dir=EXP_NAME, project=EXP_NAME, log_model=False)
logger


# In[24]:


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


# In[28]:


trainer = pl.Trainer(
    max_epochs=5,
    accumulate_grad_batches=ACCUMULATE_GRAD,
    # accelerator='gpu',
    gpus=1,
    #accelerator='ddp',
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


# In[27]:


#trainer.fit(model, datamodule=data_module)


# In[26]:


# get_ipython().system(" ls '/scratch/dhaval.taunk/mlns-distilbert-regressor_data_augmentation1/lightning-checkpoints/'")


# In[31]:


chkpt_path = '/scratch/sagarsj42/mlns-distilbert-regressor_data_augmentation/lightning-checkpoints/last.ckpt'
test_pred = trainer.predict(model, datamodule=data_module, ckpt_path=chkpt_path)
len(test_pred)


# In[32]:


all_outputs = list()
for batch_outputs in test_pred:
    all_outputs.append(batch_outputs['preds'])
all_outputs = torch.cat(all_outputs, dim=0)

all_outputs.shape


# In[33]:


pd.DataFrame(list(zip(test_df.pair_id, all_outputs.squeeze(1).tolist())), columns=['pair_id', 'Overall']).to_csv(EXP_NAME + 'test-results.csv')


# In[34]:


# get_ipython().system('ls')


# In[35]:


# get_ipython().system('zip results_10_epochs_distbert_data_augmentation1.zip test-results.csv')


# In[1]:


# get_ipython().system(' ls')


# In[ ]:




