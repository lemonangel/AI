
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 13:04:57 2017

@author: 
"""

import numpy
import matplotlib.pyplot as plt
from pandas import read_csv
import math
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# fix random seed for reproducibility
numpy.random.seed(7)
# load the dataset
#Load variate lws(wind speed column 10),dewp (colum 6, the dew point)
pd.dataframe = read_csv('PRSA_data_2010.1.1-2014.12.31.csv', index_col=0, usecols=[0,5,6], engine='python', skipfooter=3)
df = pd.dataframe
import scipy as sp
import numpy as np
import scipy as sp


#sp.sum(sp.isnan(dataset))
dataset = df.values
dataset = df.dropna(axis=0)
#dataset = df.dropna(how = 'all')

dataset = dataset.astype('float32')

# normalize the dataset

# split into train and test sets
# need x_scaler and y_scaler, otherwise invert_transform will be failed.
split = round(0.67*len(dataset))

scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

TrainX = dataset[:split,1]
XTest = dataset[split:,1]

TrainY = dataset[:split,0]
YTest = dataset[split:,0]

ori_dataset_y = dataset[:,0]
#dataset_y = y_scaler.fit_transform(ori_dataset_y)



split = round(0.67*len(dataset))
TrainX, TrainY = dataset[:split,1],  dataset[:split,0]

#TrainX, TrainY = create_dataset(train, look_back)
XTest, YTest = dataset[split:,1],  dataset[split:,0]
# reshape input to be [samples, time steps, features]
#TrainX = numpy.reshape(TrainX, (TrainX.shape[0], TrainX.shape[2]))
#testX = numpy.reshape(testX, (testX.shape[0], testX.shape[1], 1))

#simple MLP model
def  create_model():
    model = Sequential()
    model.add(Dense(32, input_dim=1, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam',metrics=['accuracy'])
    return model
#model.add(LSTM(32, input_shape=(look_back, 1)))
#model.add(Dense(1))
model = create_model()
history = model.fit(TrainX, TrainY, validation_data=(XTest, YTest),epochs=10, batch_size=1)
# make predictions
np.random.seed(7)
seed=7

trainPredict = model.predict(TrainX)
trainPredict_extended = np.zeros((len(trainPredict),2))
trainPredict_extended[:,0:1] = trainPredict
trainPredict = scaler.inverse_transform(trainPredict_extended)[:,0]


testPredict = model.predict(XTest)
testPredict_extended = np.zeros((len(testPredict),2))
testPredict_extended[:,0:1] = testPredict
testPredict = scaler.inverse_transform(testPredict_extended)[:,0]

TrainY_extended = np.zeros((len(TrainY),2))
TrainY_extended[:,0] = TrainY
TrainY = scaler.inverse_transform(TrainY_extended)[:,0]

YTest_extended = np.zeros((len(YTest),2))
YTest_extended[:,0] = YTest
YTest = scaler.inverse_transform(YTest_extended)[:,0]

# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(TrainY, trainPredict))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(YTest, testPredict))
print('Test Score: %.2f RMSE' % (testScore))
# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, 0] = numpy.nan
trainPredict = np.reshape(trainPredict, (len(trainPredict,)))
trainPredictPlot[0:len(trainPredict),0] = trainPredict
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, 0] = numpy.nan
testPredict = np.reshape(testPredict, (len(testPredict,)))
testPredictPlot[len(trainPredict):len(dataset), 0] = testPredict
# plot baseline and predictions
dataset = scaler.inverse_transform(dataset)
"""
line_r = plt.plot(dataset[:,0], color='r', label = 'Actual data')
line_g = plt.plot(trainPredictPlot, color='g',label = 'Train data')
line_b = plt.plot(testPredictPlot, color='blue', label = 'Predict data')
"""
line_r = plt.plot(dataset[:,0], color='r')
line_g = plt.plot(trainPredictPlot, color='g')
line_b = plt.plot(testPredictPlot, color='b')
#plt.xlim([32000,35000])
plt.legend(['Actual data', 'Train data', 'Predict data'],loc = 'upper right')