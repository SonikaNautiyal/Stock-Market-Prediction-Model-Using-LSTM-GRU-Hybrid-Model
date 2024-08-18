#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().system('pip install keras-tuner')


# In[3]:


from tensorflow import keras
from tensorflow.keras import layers


# In[9]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns


# In[10]:


df=pd.read_csv('C:/Users/nauti/OneDrive/Desktop/programs/SP_dataset.csv')


# In[11]:


df.describe


# In[12]:


df.head()


# In[13]:


df.tail()


# In[14]:


df = df.set_index("Date") 
print(df.shape)
print(df.columns)


# In[15]:


df.head(5)


# In[16]:


#plotting the every index plot with respect to date
for i in  df.columns:
  df[[i]].plot()
  plt.title("S&P")
  plt.show()


# In[17]:


# Comulative Return
plt.figure(figsize=(20,20))
dr = df.cumsum()
dr.plot()
plt.title('S&P Cumulative Returns')


# In[18]:


#ploting the line plot to see the trend in data set
plt.figure(figsize=(20,5))
sns.lineplot(data =df,)


# In[19]:


#drawing the heat map 
plt.figure(figsize=(10,10))
sns.heatmap(df[:100],   robust=False,
                annot=None, fmt='.2g', annot_kws=None, linewidths=0, linecolor='white', cbar=True, 
                 square=False, xticklabels='auto', yticklabels='auto', mask=None, ax=None)


# In[20]:


# histogram
plt.figure(figsize=(10,10))
plt.hist(df[:10],
            bottom=None,
            histtype='bar',
            align='mid', orientation='vertical', rwidth=None)


# In[21]:


#LSTM is very sensitive neural network so we have to normalize the data set in the same range
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
df=scaler.fit_transform(df)
#class sklearn.preprocessing.MinMaxScaler(feature_range=(0, 1), *, copy=True)
#caler=MinMaxScaler(feature_range=(0,1))
#df1=scaler.fit_transform(np.array(df).reshape(-1,1))


# In[22]:


df.shape


# In[23]:


#splitting the dataset into test train 
train=df[0:3000]
test =df[3500:]
#validate=df[2500:2999]


# In[24]:


print(test.shape)
print(train.shape)


# In[25]:


plt.figure(figsize=(10,10))
plt.hist(train,
            bottom=None,
            histtype='bar',
            align='mid', orientation='vertical', rwidth=None)


# In[26]:


plt.figure(figsize=(10,10))
plt.hist(test,
            bottom=None,
            histtype='bar',
            align='mid', orientation='vertical', rwidth=None)


# In[27]:


#scaled_train_samples=scaler.fit_transform(train)
#scaled_test_samples=scaler.fit_transform(test)


# In[28]:


import numpy as np
def create_dataset(dataset,time_stamp =1):
  X, Y = [], []
  for i in range(len(dataset)-time_stamp-1):
    a= dataset[i:(i+time_stamp),0]
    X.append(a)
    Y.append(df[i+time_stamp,0])
  return np.array(X),np.array(Y)


# In[29]:


time_stamp=100
x_train, y_train=create_dataset(train,time_stamp)
x_test, y_test = create_dataset(test, time_stamp)


# In[30]:


print(x_train.shape)
print(x_test.shape)
print(y_train)
print(y_test)


# In[31]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM


# In[32]:


print(x_test)
print(x_train)
print(x_train.shape)


# In[33]:


x_train =x_train.reshape(x_train.shape[0],x_train.shape[1] , 1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1] , 1)


# In[34]:


print(x_train.shape)
print(y_train.shape)


# In[35]:


print(x_test.shape)
print(y_test.shape)


# In[37]:


model=Sequential()
model.add(LSTM(100,return_sequences=True,input_shape=(100,1)))
model.add(LSTM(100,return_sequences=True))
model.add(LSTM(50,return_sequences=False))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam')


# In[38]:


model.summary()


# In[39]:


history=model.fit(
    x_train,y_train,
    validation_split=0.1,
    shuffle=False, 
    epochs=50,batch_size=16,verbose=1)


# In[40]:


plt.plot(history.history['loss'],label='train')
plt.plot(history.history['val_loss'],label='validation')
plt.legend()


# In[41]:


### Lets Do the prediction and check performance metrics
train_predict=model.predict(x_train)
test_predict=model.predict(x_test)


# In[42]:


test_predict.shape


# In[43]:


#calcualtion of RMSE
import math
from sklearn.metrics import mean_squared_error, precision_score,recall_score,f1_score
math.sqrt(mean_squared_error(y_train,train_predict))
from sklearn.metrics import confusion_matrix
x=confusion_matrix=(x_test, model.predict(x_test))


# In[44]:


print(x)


# In[45]:


math.sqrt(mean_squared_error(y_test,test_predict))


# In[46]:


# Visualising the results
plt.figure(figsize=(14,5))
plt.plot(y_train, color = 'red', label = 'Real S&P stock price')
plt.plot(train_predict, color = 'blue', label = 'Predicted S&P stock price')
plt.title('S&P stock price')
plt.xlabel('Time')
plt.ylabel('S&P Stock Price')
plt.legend()
plt.show()


# In[48]:


model=Sequential()
model.add(tf.keras.layers.GRU(100,return_sequences=True,input_shape=(100,1)))
model.add(tf.keras.layers.GRU(50,return_sequences=False))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam')


# In[1]:


history=model.fit(
    x_train,y_train,
    validation_split=0.1,
    shuffle=False, 
    epochs=50,batch_size=16,verbose=1)


# In[50]:


plt.plot(history.history['loss'],label='train')
plt.plot(history.history['val_loss'],label='validation')
plt.legend()


# In[51]:


train_predict.shape


# In[52]:


### Lets Do the prediction and check performance metrics
train_predict=model.predict(x_train)
test_predict=model.predict(x_test)


# In[53]:


math.sqrt(mean_squared_error(y_train,train_predict))


# In[54]:


# Visualising the results
plt.figure(figsize=(14,5))
plt.plot(y_train, color = 'red', label = 'Real S&P stock price')
plt.plot(train_predict, color = 'blue', label = 'Predicted S&P stock price')
plt.title('S&P stock price')
plt.xlabel('Time')
plt.ylabel('S&P Stock Price')
plt.legend()
plt.show()


# In[56]:


model=Sequential()
model.add(LSTM(100,return_sequences=True,input_shape=(100,1)))
model.add(layer = tf.keras.layers.Dropout(.2,))
model.add(tf.keras.layers.GRU(50,return_sequences=False))
model.add(tf.keras.layers.Dropout(.2,))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer = tf.keras.optimizers.Adam(learning_rate=0.001))


# In[57]:


history=model.fit(
    x_train,y_train,
    validation_split=0.1,
    shuffle=False, 
    epochs=20,batch_size=16,verbose=1,)


# In[58]:


plt.plot(history.history['loss'],label='train')
plt.plot(history.history['val_loss'],label='validation')
plt.legend()


# In[59]:


### Lets Do the prediction and check performance metrics
train_predict=model.predict(x_train)
test_predict=model.predict(x_test)


# In[60]:


print(train_predict.shape)
print(test_predict.shape)


# In[61]:


math.sqrt(mean_squared_error(y_train,train_predict))


# In[62]:


# Visualising the results
plt.figure(figsize=(14,5))
plt.plot(y_train, color = 'red', label = 'Real S&P stock price')
plt.plot(train_predict, color = 'blue', label = 'Predicted S&P stock price')
plt.title('S&P stock price')
plt.xlabel('Time')
plt.ylabel('S&P Stock Price')
plt.legend()
plt.show()


# In[ ]:





# In[ ]:




