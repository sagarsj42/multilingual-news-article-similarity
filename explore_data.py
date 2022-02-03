#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('mkdir -p /scratch/sagarsj42/torch-cache')
get_ipython().system('mkdir -p /scratch/sagarsj42/transformers')
import os
os.chdir('/scratch/sagarsj42')
os.environ['TORCH_HOME'] = '/scratch/sagarsj42/torch-cache'
os.environ['TRANSFORMERS_CACHE'] = '/scratch/sagarsj42/transformers'


# In[2]:


import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from transformers import XLMRobertaTokenizer


# In[3]:


get_ipython().system('scp sagarsj42@ada:/share1/sagarsj42/semeval-2022-task-8-train-df.csv .')
get_ipython().system('scp sagarsj42@ada:/share1/sagarsj42/semeval-2022-task-8-eval-df.csv .')


# In[4]:


train_df = pd.read_csv('semeval-2022-task-8-train-df.csv', index_col=0)
train_df.info()


# In[5]:


train_df.head()


# In[6]:


def get_id_texts(df):
    id1 = df['id_1'].tolist()
    id2 = df['id_2'].tolist()
    text1 = df['text_1'].tolist()
    text2 = df['text_2'].tolist()
    id_text = dict()

    for idx, text in zip(id1, text1):
        id_text[idx] = text
    for idx, text in zip(id2, text2):
        id_text[idx] = text

    return id_text


# In[7]:


id_text = get_id_texts(train_df)
len(id_text)


# In[8]:


list(id_text.keys())[:10]


# In[9]:


text = id_text[1484189120]
text


# In[10]:


len(text)


# In[11]:


tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
tokenizer


# In[12]:


text_tok = tokenizer.tokenize(text)
len(text_tok), text_tok


# In[13]:


text_tok_enc = tokenizer.encode(text)
len(text_tok_enc), text_tok_enc


# In[14]:


text_char_len = list()
text_tok_len = list()

for text in id_text.values():
    text = str(text)
    text_char_len.append(len(text))
    text_tok_len.append(len(tokenizer.encode(text)))

text_char_len = np.array(text_char_len)
text_tok_len = np.array(text_tok_len)

len(text_char_len), len(text_tok_len)


# In[15]:


text_char_len.min(), text_char_len.max(), np.median(text_char_len), text_char_len.mean(), text_char_len.std()


# In[16]:


plt.figure(figsize=(9, 6))
sns.histplot(text_char_len, binrange=(0, 20000))
plt.title('Histogram of # characters per text')
plt.show()


# In[17]:


text_tok_len.min(), text_tok_len.max(), np.median(text_tok_len), text_tok_len.mean(), text_tok_len.std()


# In[18]:


plt.figure(figsize=(9, 6))
sns.histplot(text_tok_len, binrange=(0, 10000))
plt.title('Histogram of # tokens per text')
plt.show()


# In[19]:


sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1)
sss


# In[20]:


y_sss = list(map(round, train_df['score']))
len(y_sss)


# In[21]:


for train_index, dev_index in sss.split(train_df, y_sss):
    trains_df = train_df.iloc[train_index]
    dev_df = train_df.iloc[dev_index]

trains_df.shape, dev_df.shape


# In[22]:


trains_df.to_csv('semeval8-train-sss.csv')
dev_df.to_csv('semeval8-dev.csv')


# In[ ]:




