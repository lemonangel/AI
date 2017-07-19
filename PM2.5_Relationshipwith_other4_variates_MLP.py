# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 14:16:45 2017

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
pd.dataframe = read_csv('PRSA_data_2010.1.1-2014.12.31.csv', index_col=0, usecols=[0,5,6,7,8,10], engine='python', skipfooter=3)
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

TrainX = dataset[:split,1:5]
XTest = dataset[split:,1:5]
#TrainX = x_scaler.fit_transform(TrainX)
#XTest = x_scaler.fit_transform(XTest)
TrainY = dataset[:split,0]
YTest = dataset[split:,0]
#TrainY= y_scaler.fit_transform(TrainY)
#YTest = y_scaler.fit_transform(YTest)

ori_dataset_y = dataset[:,0]
#dataset_y = y_scaler.fit_transform(ori_dataset_y)


# split into train and test sets
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,1:5], dataset[train_size:len(dataset),1:5]
# reshape into X=t and Y=t+1
look_back = 0
split = round(0.67*len(dataset))
TrainX, TrainY = dataset[:split,1:5],  dataset[:split,0]

#TrainX, TrainY = create_dataset(train, look_back)
XTest, YTest = dataset[split:,1:5],  dataset[split:,0]
# reshape input to be [samples, time steps, features]
#TrainX = numpy.reshape(TrainX, (TrainX.shape[0], TrainX.shape[2]))
#testX = numpy.reshape(testX, (testX.shape[0], testX.shape[1], 1))

#normal CNN model
def  create_model():
    model = Sequential()
    model.add(Dense(32, input_dim=4, activation='relu'))
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
trainPredict_extended = np.zeros((len(trainPredict),5))
trainPredict_extended[:,0:1] = trainPredict
trainPredict = scaler.inverse_transform(trainPredict_extended)[:,0]


testPredict = model.predict(XTest)
testPredict_extended = np.zeros((len(testPredict),5))
testPredict_extended[:,0:1] = testPredict
testPredict = scaler.inverse_transform(testPredict_extended)[:,0]

TrainY_extended = np.zeros((len(TrainY),5))
TrainY_extended[:,0] = TrainY
TrainY = scaler.inverse_transform(TrainY_extended)[:,0]

YTest_extended = np.zeros((len(YTest),5))
YTest_extended[:,0] = YTest
YTest = scaler.inverse_transform(YTest_extended)[:,0]

# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(TrainY, trainPredict))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(YTest, testPredict))
print('Test Score: %.2f RMSE' % (testScore))
# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredict = np.reshape(trainPredict, (len(trainPredict,)))
trainPredictPlot[0:len(trainPredict),0] = trainPredict
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredict = np.reshape(testPredict, (len(testPredict,)))
testPredictPlot[len(trainPredict):len(dataset), 0] = testPredict
# plot baseline and predictions
dataset = scaler.inverse_transform(dataset)

plt.plot(dataset[:,0], color='r', label = 'Actual data')
plt.plot(trainPredictPlot, color='g',label = 'Train data')
plt.plot(testPredictPlot, color='blue', label = 'Predict data')
#plt.xlim([32000,35000])
plt.legend(['Actual data', 'Train data', 'Predict data'],loc = 'upper right')
#plt.legend(handles = [line_r,line_g,line_b], loc = 'upper_right')
#plt.legend(handles = [line_r,line_g,line_b],['Actual data', 'Train data', 'Predict data'],loc = 'upper right')

plt.show()

plt.plot(history.history['loss'])
plt.xlabel('epoch')

plt.ylabel('loss')
plt.show()
print(model.summary())
