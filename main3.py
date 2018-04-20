
# coding: utf-8

# In[62]:


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
from keras.layers import Input
from keras.models import Model
import keras
import h5py
import requests
import os


# In[63]:


def normalize_data(df):
    min_max_scaler = preprocessing.MinMaxScaler()
    df['open'] = min_max_scaler.fit_transform(df.open.values.reshape(-1,1))
    df['high'] = min_max_scaler.fit_transform(df.high.values.reshape(-1,1))
    df['low'] = min_max_scaler.fit_transform(df.low.values.reshape(-1,1))
    df['volume'] = min_max_scaler.fit_transform(df.volume.values.reshape(-1,1))
    df['adj close'] = min_max_scaler.fit_transform(df['adj close'].values.reshape(-1,1))
    return df

# function for build the training data and test data
def load_data(stock, seq_len):
    amount_of_features = len(stock.columns) # 5
    data = stock.as_matrix() 
    sequence_length = seq_len + 1 # index starting from 0
    result = []
    
    for index in range(len(data) - sequence_length): # maxmimum date = lastest date - sequence length
        result.append(data[index: index + sequence_length]) # index : index + 22days
    
    result = np.array(result)
#     print(result.shape)
    row = round(0.9 * result.shape[0]) # 90% split
    train = result[:int(row), :] # 90% date, all features 
    
    x_train = train[:, :-1] 
    y_train = train[:, -1][:,-1]
#     print(x_train.shape)
#     print(y_train.shape)
    
    x_test = result[int(row):, :-1]
    y_test = result[int(row):, -1][:,-1]
#     print(x_test.shape)
#     print(y_test.shape)

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], amount_of_features))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], amount_of_features))  

    return [x_train, y_train, x_test, y_test]


# In[64]:


df = pd.read_csv("prices-split-adjusted.csv", index_col = 0)
df["adj close"] = df.close # Moving close to the last column
df.drop(['close'], 1, inplace=True) # Moving close to the last column
df.head()


# In[65]:


# df = normalize_data(df)
# df.head()


# In[66]:


symbols = list(set(df.symbol))
# print (symbols)


# In[67]:


science_list = ['GOOG','AAPL','YHOO','AMZN']
encoding_dim = 10 
# for symbol in science_list:
#     thisdf = df[df.symbol == symbol]
#     input_img = Input(shape=(22,))
#     encoded = Dense(encoding_dim, activation='relu')(input_img)
#     decoded = Dense(784, activation='sigmoid')(encoded)
#     autoencoder = Model(input_img, decoded)
#     encoder = Model(input_img, encoded)
stock_interest = 'AAPL'
df3 = df[df.symbol == stock_interest]
df3.drop(['symbol'],1,inplace=True)
df3.head()


# In[68]:


window = 22
X_train, y_train, X_test, y_test = load_data(df3, window)
print (X_train.shape)


# In[69]:


print (X_train.shape)
X_train = X_train.reshape(1565,22*5)


# In[70]:


print (X_test.shape)
X_test = X_test.reshape(174,22*5)


# In[71]:


#Test the performance of autoencoder on the time series
# build the model
input_img = Input(shape=(22*5,))
encoding_dim = 10 
encoded = Dense(encoding_dim, activation='relu')(input_img)
decoded = Dense(22*5, activation='linear')(encoded)
autoencoder = Model(input_img, decoded)
encoder = Model(input_img, encoded)
encoded_input = Input(shape=(encoding_dim,))
decoder_layer = autoencoder.layers[-1]
decoder = Model(encoded_input, decoder_layer(encoded_input))
autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')


# In[72]:


print (X_train)
earlyStopping=keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, verbose=0, mode='auto')
autoencoder.fit(X_train, X_train,
                epochs=150,
                batch_size=256,
                callbacks=[earlyStopping],
                shuffle=True,
                validation_data=(X_test, X_test))


# In[73]:


# thisx = X_test
print (X_test.shape)
thisx = np.array([X_test[1,:]])
print (thisx.shape)
encoded_imgs = encoder.predict(X_test)
decoded_imgs = decoder.predict(encoded_imgs)
print(encoded_imgs.shape)
print(decoded_imgs.shape)


# In[74]:



# print (thisx)
# print (decoded_imgs[0])


# In[75]:


plt.plot(thisx[0],label = ' original')
plt.plot(decoded_imgs[0],label = ' prediction')
plt.legend()
plt.show()


# In[76]:


#TODO
# science_list = ['GOOG','AAPL','YHOO','AMZN']
# science_list = ['GOOG','AAPL','YHOO','AMZN', 'MSFT', 'IBM', 'GOOGL', 'FB']
science_list = ['GOOG','AAPL','YHOO','AMZN', 'MSFT', 'IBM', 'GOOGL', 'FB', 'EBAY', 'NVDA', 'ORCL', 'INTC']
encoder_list = []
decoder_list = []
window = 22
for symbol in science_list:
    # prepare the data
    df3 = df[df.symbol == symbol]
    df3.drop(['symbol'],1,inplace=True)
    df3 = normalize_data(df3)
    df3.head()
    X_train, y_train, X_test, y_test = load_data(df3, window)
    X_train = X_train.reshape(X_train.shape[0],22*5)
    X_test = X_test.reshape(X_test.shape[0],22*5)
    
    
    
    # train an autoencoder 
    input_img = Input(shape=(22*5,))
    encoding_dim = 10 
    encoded = Dense(encoding_dim, activation='relu')(input_img)
    decoded = Dense(22*5, activation='linear')(encoded)
    autoencoder = Model(input_img, decoded)
    encoder = Model(input_img, encoded)
    encoded_input = Input(shape=(encoding_dim,))
    decoder_layer = autoencoder.layers[-1]
    decoder = Model(encoded_input, decoder_layer(encoded_input))
    autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')
    earlyStopping=keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, verbose=0, mode='auto')
    autoencoder.fit(X_train, X_train,
                epochs=150,
                batch_size=256,
                callbacks=[earlyStopping],
                shuffle=True,
                validation_data=(X_test, X_test))
    
#     autoencoder_list.append(None) # append the trained autoencoder
    encoder_list.append(encoder)
    decoder_list.append(decoder)

    
# generate the predicted values for all stocks and put them in one matrix
# train a LSTM using those values
# test the results


# In[77]:


# train the main model
# for stock in list
# get the data and transform the data
#  bulid the train data
# put into keras

print (encoded_imgs[4])


# In[78]:


def generate_data(df,symbol_list,encoder_list,deoder_list,window,feature_len):
    data = []
    testdata = []
    label = []
    testlabel = []
    num_symbol = len(symbol_list)
    first = 0
    for symbol in symbol_list:
        df3 = df[df.symbol == symbol]
        df3.drop(['symbol'],1,inplace=True)
        df3 = normalize_data(df3)
        X_train, y_train, X_test, y_test = load_data(df3, window)
        
        X_train = X_train.reshape(X_train.shape[0],22*5)
        num_train = X_train.shape[0]
        X_test = X_test.reshape(X_test.shape[0],22*5)
        num_test = X_test.shape[0]

        
        thisdata = np.zeros([num_train,num_symbol * feature_len])
        thistest = np.zeros([num_test,num_symbol * feature_len])
#         print (num_test)
        i = 0
        for encoder in encoder_list:
            
            encoded_imgs = encoder.predict(X_train)
            thisdata[:,i*feature_len:(i+1)*feature_len] = encoded_imgs
            encoded_imgs = encoder.predict(X_test)
            thistest[:,i*feature_len:(i+1)*feature_len] = encoded_imgs
            i += 1
        if first == 0:
            data = thisdata
            first =1
            testdata = thistest
            label = y_train
            testlabel = y_test
#             print (testdata.shape)
        else:
            data = np.append(data,thisdata,axis=0)
            testdata = np.append(testdata,thistest,axis=0)
            label = np.append(label,y_train)
            testlabel = np.append(testlabel,y_test)
#             print (testdata.shape)
    return data,testdata,label,testlabel


def generate_data_decoded(df,symbol_list,encoder_list,deoder_list,window,feature_len):
    data = []
    testdata = []
    label = []
    testlabel = []
    num_symbol = len(symbol_list)
    first = 0
    for symbol in symbol_list:
        df3 = df[df.symbol == symbol]
        df3.drop(['symbol'],1,inplace=True)
        df3 = normalize_data(df3)
        X_train, y_train, X_test, y_test = load_data(df3, window)
        
        X_train = X_train.reshape(X_train.shape[0],22*5)
        num_train = X_train.shape[0]
        X_test = X_test.reshape(X_test.shape[0],22*5)
        num_test = X_test.shape[0]

        
        thisdata = np.zeros([num_train,window,num_symbol * feature_len])
        thistest = np.zeros([num_test,window,num_symbol * feature_len])
#         print (num_test)
        i = 0
        for encoder in encoder_list:
            
            encoded_imgs = encoder.predict(X_train)
            decoded_imgs = decoder.predict(encoded_imgs)
            decoded_imgs = decoded_imgs.reshape(num_train,window,feature_len)
#             print (decoded_imgs.shape)
            
            thisdata[:,:,i*feature_len:(i+1)*feature_len] = decoded_imgs
            encoded_imgs = encoder.predict(X_test)
            decoded_imgs = decoder.predict(encoded_imgs)
            decoded_imgs = decoded_imgs.reshape(num_test,window,feature_len)
            thistest[:,:,i*feature_len:(i+1)*feature_len] = decoded_imgs
            i += 1
        if first == 0:
            data = thisdata
            first =1
            testdata = thistest
            label = y_train
            testlabel = y_test
#             print (testdata.shape)
        else:
            data = np.append(data,thisdata,axis=0)
            testdata = np.append(testdata,thistest,axis=0)
            label = np.append(label,y_train)
            testlabel = np.append(testlabel,y_test)
#             print (testdata.shape)
    return data,testdata,label,testlabel
# xx = generate_data(df,science_list[0:3],encoder_list[0:3],decoder_list[0:3],window,10)
# xx2 = generate_data_decoded(df,science_list[0:3],encoder_list[0:3],decoder_list[0:3],window,5)    


# In[ ]:


xx = generate_data(df,science_list[0:3],encoder_list[0:3],decoder_list[0:3],window,10)
xx2 = generate_data_decoded(df,science_list[0:3],encoder_list[0:3],decoder_list[0:3],window,5)  


# In[111]:


xx_s8 = generate_data(df,science_list[0:7],encoder_list[0:7],decoder_list[0:7],window,10)
xx2_s8 = generate_data_decoded(df,science_list[0:7],encoder_list[0:7],decoder_list[0:7],window,5)  


# In[80]:


xx_s12 = generate_data(df,science_list,encoder_list,decoder_list,window,10)
xx2_s12 = generate_data_decoded(df,science_list,encoder_list,decoder_list,window,5)  


# In[81]:


print (np.array(xx).shape)
print (xx[0].shape)
print (xx[1].shape)
print (xx[2].shape)
print (xx[3].shape)
# xx = xx.reshape(1565,10*4)


# In[119]:


def build_model1(layers):
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
def build_model2(layers):
    d = 0.3
    model = Sequential()
    print (layers)
#     model.add(LSTM(256, input_dim=layers[0], return_sequences=True))
#     model.add(Dropout(d))        
#     model.add(LSTM(256, input_dim=layers[0], return_sequences=False))
#     model.add(Dropout(d))
    model.add(Dense(32,input_dim = 40*3, kernel_initializer="uniform",activation='relu')) 
    model.add(Dropout(d))
#     model.add(Dense(32, kernel_initializer="uniform",activation='relu')) 
#     model.add(Dropout(d))
    model.add(Dense(1,kernel_initializer="uniform",activation='linear'))
    # adam = keras.optimizers.Adam(decay=0.2)
    start = time.time()
    model.compile(loss='mse',optimizer='adam', metrics=['accuracy'])
    print("Compilation Time : ", time.time() - start)
    return model


# In[83]:


model = build_model2([40,1])
model2 = build_model1([20,window,1])


# In[31]:


# X_train = xx[0].reshape(xx[0].shape[0],)
model.fit(xx[0],xx[2],batch_size=512,epochs=90,validation_split=0.1,verbose=1)
model2.fit(xx2[0],xx2[2],batch_size=512,epochs=90,validation_split=0.1,verbose=1)


# In[115]:


predictions = []
data1 = [xx, xx_s8, xx_s12]
# for i in range(3):
i=1
model = build_model2([40*(i+1),1])
model.fit(data1[i][0],data1[i][2],batch_size=512,epochs=90,validation_split=0.1,verbose=1)
predictions.append(model.predict(data1[i][1]))


# In[120]:


i=2
model = build_model2([40*(i+1),1])
model.fit(data1[i][0],data1[i][2],batch_size=512,epochs=90,validation_split=0.1,verbose=1)
predictions.append(model.predict(data1[i][1]))


# In[112]:


predictions2 = []
data2 = [xx2, xx2_s8, xx2_s12]
# for i in range(3):
i=1
model2 = build_model1([20*(i+1),window,1])  
model2.fit(data2[i][0],data2[i][2],batch_size=512,epochs=90,validation_split=0.1,verbose=1)
predictions2.append(model2.predict(data2[i][1]))


# In[118]:


i=2
model2 = build_model1([20*(i+1),window,1])  
model2.fit(data2[i][0],data2[i][2],batch_size=512,epochs=90,validation_split=0.1,verbose=1)
predictions2.append(model2.predict(data2[i][1]))


# In[59]:


p = model.predict(xx[1])
p2 = model2.predict(xx2[1])


# In[96]:


plt.plot(predictions2[175:348],label = 'Prediction')
plt.plot(xx[3][175:348],label = 'Observation')
# plt.plot(p2[175:348],label = 'Prediction 2')
plt.legend(loc='lower right')
plt.show()


# In[ ]:


p_s8 = model.predict(xx_s8[1])
p2_s8 = model2.predict(xx2_s8[1])


# In[61]:


# print (len(p))
# print (len(p2))


# In[35]:


# TODO:
# 1. re-normalize
# 2. compare the loss
# 3. try with different symbols
# 4. play with build model two
df = pd.read_csv("prices-split-adjusted.csv", index_col = 0)
df["adj close"] = df.close # Moving close to the last column
df.drop(['close'], 1, inplace=True) # Moving close to the last column
df = df[df.symbol == stock_interest]
# df.drop(['symbol'],1,inplace=True)

def denormalize(df, normalized_value): 
    df = df['adj close'].values.reshape(-1,1)
    normalized_value = normalized_value.reshape(-1,1)
    
    #return df.shape, p.shape
    min_max_scaler = preprocessing.MinMaxScaler()
    a = min_max_scaler.fit_transform(df)
    new = min_max_scaler.inverse_transform(normalized_value)
    return new


# In[36]:


# get the prediction from lstm
def build_model_baseline(layers):
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


# In[131]:


# stock_interest = 'GOOG'
# df2 = df[df.symbol == stock_interest]
# df2.drop(['symbol'],1,inplace=True)
# df2 = normalize_data(df2)
# window = 22
# X_train, y_train, X_test, y_test = load_data(df2, window)
# model = build_model_baseline([5,window,1])
# model.fit(X_train,y_train,batch_size=512,epochs=90,validation_split=0.1,verbose=1)
# plstm = model.predict(X_test)


# In[132]:


# i = 0
# plt.plot(p[i*175:(i+1)*175-1],label = 'Prediction')
# plt.plot(xx[3][i*175:(i+1)*175-1],label = 'Observation')
# plt.plot(p2[i*175:(i+1)*175-1],label = 'Prediction 2')
# plt.plot(plstm,label='LSTM')
# plt.legend(loc='best')
# plt.show()


# In[38]:


stock_interest = 'AAPL'
df2 = df[df.symbol == stock_interest]
df2.drop(['symbol'],1,inplace=True)
df2 = normalize_data(df2)
window = 22
X_train, y_train, X_test, y_test = load_data(df2, window)
model = build_model_baseline([5,window,1])
model.fit(X_train,y_train,batch_size=512,epochs=90,validation_split=0.1,verbose=1)
plstm2 = model.predict(X_test)


# In[123]:


# print (y_test)
# print (plstm2)

i = 1
plt.plot(predictions[0][i*175:(i+1)*175-2],label = 'Prediction')
plt.plot(xx[3][i*175:(i+1)*175-2],label = 'Observation')
plt.plot(predictions2[0][i*175:(i+1)*175-2],label = 'Prediction 2')
plt.plot(plstm2,label='LSTM')
plt.legend(loc='best')
plt.show()


# In[124]:


i = 1
plt.plot(predictions[1][i*175:(i+1)*175-2],label = 'Prediction')
plt.plot(xx[3][i*175:(i+1)*175-2],label = 'Observation')
plt.plot(predictions2[1][i*175:(i+1)*175-2],label = 'Prediction 2')
plt.plot(plstm2,label='LSTM')
plt.legend(loc='best')
plt.show()


# In[100]:


print(predictions2[0].shape)


# In[ ]:


# pay attention to scale


# In[133]:


# TODO:
# 1. re-normalize
# 2. compare the loss
# 3. try with different symbols
# 4. play with build model two
# df = pd.read_csv("data/prices-split-adjusted.csv", index_col = 0)
# df["adj close"] = df.close # Moving close to the last column
# df.drop(['close'], 1, inplace=True) # Moving close to the last column
df3 = df[df.symbol == 'AAPL']
df3.drop(['symbol'],1,inplace=True)

def denormalize(df, normalized_value): 
    df = df['adj close'].values.reshape(-1,1)
    normalized_value = normalized_value.reshape(-1,1)
    
    #return df.shape, p.shape
    min_max_scaler = preprocessing.MinMaxScaler()
    a = min_max_scaler.fit_transform(df)
    new = min_max_scaler.inverse_transform(normalized_value)
    return new


# In[50]:


newp = denormalize(df3, p[i*175:(i+1)*175-1])
newxx = denormalize(df, xx[3][i*175:(i+1)*175-1])
newp2 = denormalize(df3, p2[i*175:(i+1)*175-1])
newplsm2 = denormalize(df3, plstm2)
obser = denormalize(df3,y_test)


# In[53]:


plt.plot(newp[0:173],label = 'Prediction')
# plt.plot(newxx,label = 'Observation')
plt.plot(obser[0:173],label = 'Observation')
plt.plot(newp2[0:173],label = 'Prediction 2')
plt.plot(newplsm2[0:173],label='LSTM')
plt.legend(loc='best')
plt.show()


# In[151]:


i = 1
newp_s8 = denormalize(df3, predictions[0][i*174:(i+1)*174])
newxx = denormalize(df3, xx[3][i*174:(i+1)*174])
newp2_s8 = denormalize(df3, predictions2[0][i*174:(i+1)*174])
newplsm2 = denormalize(df3, plstm2)
# obser = denormalize(df3,y_test)


# In[152]:


# print(newxx[:10])
# print(xx[3][i*175:i*175+10])
# print(predictions[0][i*175:i*175+10])


# In[153]:



# obser = denormalize(df3,y_test)
plt.plot(newp_s8[0:173],label = 'Prediction')
# plt.plot(newxx,label = 'Observation')
plt.plot(newxx[0:173],label = 'Observation')
plt.plot(newp2_s8[0:173],label = 'Prediction 2')
plt.plot(newplsm2[0:173],label='LSTM')
plt.legend(loc='best')
plt.show()


# In[159]:


i = 1
newp_s12 = denormalize(df3, predictions[1][i*174:(i+1)*174])
newxx = denormalize(df3, xx[3][i*174:(i+1)*174])
newp2_s12 = denormalize(df3, predictions2[1][i*174:(i+1)*174])
newplsm2 = denormalize(df3, plstm2)
plt.plot(newp_s12[0:173],label = 'Prediction')
# plt.plot(newxx,label = 'Observation')
plt.plot(newxx[0:173],label = 'Observation')
plt.plot(newp2_s12[0:173],label = 'Prediction 2')
plt.plot(newplsm2[0:173],label='LSTM')
plt.legend(loc='best')
plt.show()


# In[160]:


import sklearn.metrics
# mse_t1 = []
# mse_t2 = []
# mse_lstm = []
mse_t1.append(sklearn.metrics.mean_squared_error(newp_s12,newxx))
mse_t2.append(sklearn.metrics.mean_squared_error(newp2_s12,newxx))
mse_lstm.append(sklearn.metrics.mean_squared_error(newplsm2,newxx))
print(mse_t1)
print(mse_t2)
print(mse_lstm)


# In[54]:


def trend_score(actual, pred):
    trendScore = 0
    for i in range(len(actual)-1):
        if ((actual[i+1]-actual[i])*(pred[i+1]-actual[i])>0):
            trendScore=trendScore+1
    return trendScore/(len(actual)-1)


# In[56]:


hitp1 = trend_score(obser, newp)
hitp2 = trend_score(obser, newp2) 
hitlstm = trend_score(obser, newplsm2) 
print(hitp1)
print(hitp2)
print(hitlstm)


# In[155]:


hitp1 = trend_score(newxx, newp)
hitp2 = trend_score(newxx, newp2) 
hitlstm = trend_score(newxx, newplsm2) 
print(hitp1)
print(hitp2)
print(hitlstm)


# In[156]:


hitp1_s8 = trend_score(newxx, newp_s8)
hitp2_s8 = trend_score(newxx, newp2_s8) 
hitlstm = trend_score(newxx, newplsm2) 
print(hitp1_s8)
print(hitp2_s8)
print(hitlstm)


# In[161]:


hitp1_s12 = trend_score(newxx, newp_s12)
hitp2_s12 = trend_score(newxx, newp2_s12) 
hitlstm = trend_score(newxx, newplsm2) 
print(hitp1_s12)
print(hitp2_s12)
print(hitlstm)

