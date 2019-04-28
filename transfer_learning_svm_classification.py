
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

from __future__ import absolute_import, division, print_function, unicode_literals

import matplotlib.font_manager as mfm
import matplotlib.pyplot as plt

font_path = "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"
prop = mfm.FontProperties(fname=font_path)

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, preprocessing, applications

# Helper libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print(tf.__version__)
from PIL import Image

import os
from pycasia import CASIA


# In[2]:


def get_hsk_list(level):
    hsk_list = pd.read_csv("../HSK_wordlists/hsk{}_all.txt".format(level), delimiter='\t', header=None)
    hsk_list.columns = ['character', 'traditional', 'pinyin1', 'pinyin2', 'traduction']
    hsk_list['level'] = level
    return hsk_list

hsk_list = pd.concat([get_hsk_list(1)]).reset_index(drop=True)
hsk_list = hsk_list[hsk_list.character.str.len() == 1]
hsk_set = set(hsk_list.character)

casia = CASIA.CASIA()
casia.get_dataset("HWDB1.1trn_gnt_P1")
casia.character_sets = ["HWDB1.1trn_gnt_P1", "HWDB1.1trn_gnt_P1", "HWDB1.1tst_gnt"]
casia.filter_characters(hsk_set)

hsk_dataset = [[image, character] for image, character in casia.load_character_images()]


# ## Integer label

# In[3]:


hsk_df = pd.DataFrame(hsk_dataset, columns = ["image", "character"])
classes = hsk_df.character.unique().shape[0]
hsk_df = hsk_df.reset_index(drop=True)
print(classes)

character2label = {char:i for i,char in enumerate(hsk_df.character.unique())}
label2character = {i:char for (char, i) in character2label.items()}

hsk_df['label'] = hsk_df.character.map(character2label.get)


# ## Split dataset and normalize images

# In[4]:


from sklearn.model_selection import train_test_split

def image2vector(pillow_im_array):
    return [np.asarray(im.resize((32,32))) for im in pillow_im_array]

def reshape_images(image_set):
    image_set = np.concatenate(image_set).reshape(-1, 32,32, 1)
    return np.tile(image_set, (1,3))

hsk_df.loc[:, "image"] = image2vector(hsk_df.image.values)
images = reshape_images(hsk_df.image.values)
labels = hsk_df.label.values

train_images, val_images, train_labels, val_labels = train_test_split(images, labels, test_size=0.1)


# ## Normalize images

# In[5]:


# These are default values for imagenet, dataset used to train VGG-16
# https://discuss.pytorch.org/t/how-to-preprocess-input-for-pre-trained-networks/683
mean =  [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

def normalize(data):
    return (data - mean) / std

train_images = train_images / 255.0
val_images = val_images / 255.0

train_images = normalize(train_images)
val_images = normalize(val_images)


# ## Visualize

# In[6]:


def visualize_images(image_sample, label_sample):
    plt.figure(figsize=(10,10))
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(image_sample[i].reshape(32,32, 3)[:, :, 0], cmap=plt.cm.binary)
        plt.xlabel(label2character[label_sample[i]], fontproperties=prop)
    plt.show()
    
visualize_images(train_images, train_labels)


# ## Transfer learning

# In[7]:


model = applications.VGG16(include_top=False, weights='imagenet')

train_features = model.predict(train_images)
val_features = model.predict(val_images)

train_features.shape


# # Visualize features

# In[9]:


class_0_mask = np.where(train_labels == 0)[0][:4]
class_1_mask = np.where(train_labels == 1)[0][:4]

def visualize_features(mask):
    for i in range(4):
        plt.subplot(2,2,i+1)
        plt.plot(np.arange(512), train_features[mask[i], 0,0])
        plt.xlabel(label2character[train_labels[mask[i]]], fontproperties=prop)
    plt.show()
    
visualize_features(class_0_mask)
visualize_features(class_1_mask)


# ## Classify

# In[11]:


from sklearn.svm import LinearSVC

svm_model = LinearSVC()
svm_model.fit(train_features.reshape(-1, 512), train_labels)
accuracy = svm_model.score(val_features.reshape(-1, 512), val_labels)
print("Accuracy : {}".format(accuracy))


# ## Which characters are most difficult to classify?

# In[12]:


prediction = svm_model.predict(val_features.reshape(-1, 512))


# In[13]:


confusion_matrix = np.zeros((classes, classes))

for pred, label in zip(prediction, val_labels):
    confusion_matrix[pred, label] += 1


# In[14]:


plt.figure(figsize=(20,20))
plt.xticks(np.arange(classes),list(label2character.values()),fontproperties=prop)
plt.yticks(np.arange(classes),list(label2character.values()),fontproperties=prop)
plt.imshow(confusion_matrix)


# In[15]:


confusion_no_diag = np.tril(confusion_matrix, k=-1) + np.triu(confusion_matrix, k=1)
raveled_conf_no_diag = np.ravel(confusion_no_diag)
top_n = 6
sorted_ravel_conf = np.argpartition(raveled_conf_no_diag, -top_n)[-top_n:]
for idx in sorted_ravel_conf:
    pred, label = np.unravel_index(idx, (82,82))
    print("Predicted : ",label2character[pred])
    print("True label : ", label2character[label])
    print("Times misclassified : ", confusion_matrix[pred, label])


# ## Type 1 error
# Characters, that are often predicted to incorrect character class.

# In[18]:


top_n = 5
type_1_error_counts = confusion_no_diag.sum(axis=1)
for char_idx in np.argpartition(type_1_error_counts, -top_n)[-top_n:]:
    print("Character {} is classified to other classes {} times.".format(
        label2character[char_idx], type_1_error_counts[char_idx]
    ))


# ## Type 2 error
# Characters, that often get incorrect predictions.

# In[17]:


top_n = 5
type_1_error_counts = confusion_no_diag.sum(axis=0)
for char_idx in np.argpartition(type_1_error_counts, -top_n)[-top_n:]:
    print("Some other class is classified to {} {} times.".format(
        label2character[char_idx], type_1_error_counts[char_idx]
    ))

