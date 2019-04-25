
# coding: utf-8

# In[1]:


from gensim.models import KeyedVectors
import pandas as pd
import numpy as np

from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import matplotlib
from matplotlib.font_manager import FontProperties


# In[2]:


words = 500000

filename = "/datasets/Chinese/Tencent_AILab_ChineseEmbedding/Tencent_AILab_ChineseEmbedding.txt"
wv_from_text = KeyedVectors.load_word2vec_format(filename, binary=False, limit=words)


# In[3]:


wv_from_text.most_similar('工作')


# In[4]:


def get_hsk_list(level):
    hsk_list = pd.read_csv("HSK_wordlists/hsk{}_all.txt".format(level), delimiter='\t', header=None)
    hsk_list.columns = ['character', 'traditional', 'pinyin1', 'pinyin2', 'traduction']
    hsk_list['vector'] = hsk_list.character.map(lambda c : wv_from_text[c])
    hsk_list['level'] = level
    return hsk_list


# In[9]:


data = pd.concat([get_hsk_list(1), get_hsk_list(2)]).reset_index(drop=True)


# In[10]:


level1_mask = data.level == 1
level2_mask = data.level == 2


# In[11]:


X = np.array(data.vector.tolist())

pcaModel = PCA(n_components=2)
X_decomposed = pcaModel.fit_transform(X)


# In[12]:


x = X_decomposed[:, 0]
y = X_decomposed[:, 1]

ChineseFont1 = FontProperties(fname = '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc')

fig, ax = plt.subplots()
ax.scatter(x[level1_mask], y[level1_mask], c='red', label="hsk 1")
ax.scatter(x[level2_mask], y[level2_mask], c='blue', label="hsk 2")
#ax.scatter(x[level3_mask], y[level3_mask], c='green', label="hsk 3")

for i, char in enumerate(data.character.values):
    ax.annotate(char, (x[i], y[i]), fontproperties = ChineseFont1)

fig.set_figheight(13)
fig.set_figwidth(13)
plt.legend()
plt.show()
