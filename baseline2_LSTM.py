
# coding: utf-8

# In[1]:


#reference: https://www.kaggle.com/benjibb/lstm-stock-prediction-20170507?scriptVersionId=1139231
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import datetime
import math, time
import itertools
from sklearn import preprocessing
import datetime
from operator import itemgetter
from sklearn.metrics import mean_squared_error
from math import sqrt
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
from keras.models import load_model
import keras
import h5py
import requests
import os


# In[2]:


def load_data(stock, seq_len):
    amount_of_features = len(stock.columns) # 5
    data = stock.as_matrix() 
    sequence_length = seq_len + 1 # index starting from 0
    result = []
    
    for index in range(len(data) - sequence_length): # maxmimum date = lastest date - sequence length
        result.append(data[index: index + sequence_length]) # index : index + 22days
    
    result = np.array(result)
    print(result.shape)
    row = round(0.9 * result.shape[0]) # 90% split
    train = result[:int(row), :] # 90% date, all features 
    
    x_train = train[:, :-1] 
    y_train = train[:, -1][:,-1]
    print(x_train.shape)
    print(y_train.shape)
    
    x_test = result[int(row):, :-1]
    y_test = result[int(row):, -1][:,-1]
    print(x_test.shape)
    print(y_test.shape)

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], amount_of_features))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], amount_of_features))  

    return [x_train, y_train, x_test, y_test]


# In[3]:


def hit_ratio(realvalue,predction):
    ratio = 0.0
    
    return ratio


# In[4]:


df = pd.read_csv("prices-split-adjusted.csv", index_col = 0)
df["adj close"] = df.close # Moving close to the last column
df.drop(['close'], 1, inplace=True) # Moving close to the last column
df.head()
# print set(df.symbol)


# In[5]:


from sklearn import preprocessing
def normalize_data(df):
    min_max_scaler = preprocessing.MinMaxScaler()
    df['open'] = min_max_scaler.fit_transform(df.open.values.reshape(-1,1))
    df['high'] = min_max_scaler.fit_transform(df.high.values.reshape(-1,1))
    df['low'] = min_max_scaler.fit_transform(df.low.values.reshape(-1,1))
    df['volume'] = min_max_scaler.fit_transform(df.volume.values.reshape(-1,1))
    df['adj close'] = min_max_scaler.fit_transform(df['adj close'].values.reshape(-1,1))
    return df
df = normalize_data(df)
df.head()


# In[6]:


symbols = list(set(df.symbol))
stock_interest = 'AAPL'
len(symbols)
df3 = df[df.symbol == stock_interest]
df3.drop(['symbol'],1,inplace=True)
print (df3.head())
df4 = df[df.symbol == 'GOOG']
df4.drop(['symbol'],1,inplace=True)
print (df4.head())


# In[7]:


symbols = list(set(df.symbol))
stock_interest = 'GOOG'
len(symbols)
df2 = df[df.symbol == stock_interest]
df2.drop(['symbol'],1,inplace=True)
print (df2.head())
df = df[df.symbol == stock_interest]
df.drop(['symbol'],1,inplace=True)
print (df.head())


# In[8]:


# df3 = normalize_data(df3)
# df3.head()
# df4 = normalize_data(df4)
# df4.head()


# In[9]:


print (df.shape)
print (df2.shape)
df = normalize_data(df)
df.head()
df2 = normalize_data(df2)
df2.head()


# In[10]:


def build_model(layers):
    d = 0.3
    model = Sequential()
    
    model.add(LSTM(256, input_shape=(layers[1], layers[0]), return_sequences=True))
    model.add(Dropout(d))
        
    model.add(LSTM(256, input_shape=(layers[1], layers[0]), return_sequences=False))
    model.add(Dropout(d))
        
    model.add(Dense(32,kernel_initializer="uniform",activation='relu'))        
    model.add(Dense(1,kernel_initializer="uniform",activation='linear'))
    
    # adam = keras.optimizers.Adam(decay=0.2)
        
    start = time.time()
    model.compile(loss='mse',optimizer='adam', metrics=['accuracy'])
    print("Compilation Time : ", time.time() - start)
    return model


# In[12]:


window = 22
X_train, y_train, X_test, y_test = load_data(df2, window)
print(X_train.shape)
print(y_train.shape)
# print (X_train[0], y_train[0])


# In[15]:


model = build_model([5,window,1])
model.fit(X_train,y_train,batch_size=512,epochs=90,validation_split=0.1,verbose=1)


# In[16]:


# print(X_test[-1])
diff=[]
ratio=[]
p = model.predict(X_test)
print (p.shape)
# for each data index in test data
for u in range(len(y_test)):
    # pr = prediction day u
    pr = p[u][0]
    # (y_test day u / pr) - 1
    ratio.append((y_test[u]/pr)-1)
    diff.append(abs(y_test[u]- pr))
    # print(u, y_test[u], pr, (y_test[u]/pr)-1, abs(y_test[u]- pr))
    # Last day prediction
    # print(p[-1]) 


# In[17]:


df = pd.read_csv("prices-split-adjusted.csv", index_col = 0)
df["adj close"] = df.close # Moving close to the last column
df.drop(['close'], 1, inplace=True) # Moving close to the last column
df = df[df.symbol == stock_interest]
df.drop(['symbol'],1,inplace=True)

# Bug fixed at here, please update the denormalize function to this one
def denormalize(df, normalized_value): 
    df = df['adj close'].values.reshape(-1,1)
    normalized_value = normalized_value.reshape(-1,1)
    
    #return df.shape, p.shape
    min_max_scaler = preprocessing.MinMaxScaler()
    a = min_max_scaler.fit_transform(df)
    new = min_max_scaler.inverse_transform(normalized_value)
    return new

newp = denormalize(df, p)
newy_test = denormalize(df, y_test)


# In[18]:


def model_score(model, X_train, y_train, X_test, y_test):
    trainScore = model.evaluate(X_train, y_train, verbose=0)
    print('Train Score: %.5f MSE (%.2f RMSE)' % (trainScore[0], math.sqrt(trainScore[0])))

    testScore = model.evaluate(X_test, y_test, verbose=0)
    print('Test Score: %.5f MSE (%.2f RMSE)' % (testScore[0], math.sqrt(testScore[0])))
    return trainScore[0], testScore[0]


model_score(model, X_train, y_train, X_test, y_test)


# In[19]:


def trend_score(actual, pred):
    trendScore = 0
    for i in range(len(actual)-1):
        if ((actual[i+1]-actual[i])*(pred[i+1]-actual[i])>0):
            trendScore=trendScore+1
    return trendScore/(len(actual)-1)

trScore = trend_score(newy_test, newp)    
print(trScore)


# In[22]:


import matplotlib.pyplot as plt2

plt2.plot(newp,color='red', label='Prediction')
plt2.plot(newy_test,color='blue', label='Actual')
plt2.legend(loc='best')
plt2.show()


# In[24]:


print (stock_interest)


# In[26]:


import statsmodels.api as sm
import statsmodels.api as sm
res = sm.tsa.arma_order_select_ic(y_train, ic=['aic', 'bic'], trend='nc')
print (res.aic_min_order)
print (res.bic_min_order)


# In[28]:


print (y_train.shape)
print (y_test.shape)
y_all = list(y_train).append(list(y_test))


# In[29]:


# print list(y_train)
y_all = np.zeros(1565+174)
y_all[:1565] = y_train
y_all[1565:] = y_test
plt2.plot(y_all)
plt2.show()


# In[30]:


from statsmodels.tsa.arima_model import ARMA
my_order = res.aic_min_order
arma_model = ARMA(y_train,(3,2),freq = 'Q').fit()
arma_y = arma_model.predict(start = 1565,end = 1565+174,dynamic=True)

# arma_y = arma_model.predict(start = 4,end = 1700)
plt2.figure(2)
plt2.plot(arma_y,color='blue', label='ARMA Prediction')
plt2.plot(y_train,color='red', label='Actual')
plt2.legend(loc='best')
plt2.show()


# In[31]:


# arma_model.fit()
# arma_y = arma_model.predict(start = 1,end = 1700,exog = y_train)
arma_pred = denormalize(df, arma_y)
# import matplotlib.pyplot as plt2

plt2.plot(newp,color='red', label='LSTM Prediction')
plt2.plot(newy_test,color='blue', label='Actual')
plt2.plot(arma_pred,color='green', label='ARMA Prediction')
plt2.legend(loc='best')
plt2.show()


# In[32]:


print (len(y_test))
print (len(arma_y))
print (arma_y[:10])
print (y_train[-10:-1])
print (y_train[:10])
print (len(y_train))
plt2.figure(2)
plt2.plot(arma_y,color='blue', label='ARMA Prediction')
plt2.plot(y_train[:157],color='red', label='Actual')
plt2.legend(loc='best')
plt2.show()
                

