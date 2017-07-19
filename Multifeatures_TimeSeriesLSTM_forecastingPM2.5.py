# -*- coding: utf-8 -*-
"""
Spyder Editor 
Readme:

This is using multi-features time series to forecast PM2.5 value.
The input:
   PM2.5,	Dew point,	Temperature,	Pressure,	Wind speed
Output：
   PM2.5.
You can change the number of inputs but you need to do a little change in the code.
(1) Change the numpy reshape of the Test and Train dataset to the number of features you would like to do
(2) Change the input dim for the first layer of the model.
(3) When you want to plot the output, tune the array size before inverse transform of the value.

by Jul-11/2017.
"""
# LSTM for international airline passengers problem with time step regression framing
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
# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), :]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return numpy.array(dataX), numpy.array(dataY)

# fix random seed for reproducibility
numpy.random.seed(7)
# load the dataset
#pd.dataframe = read_csv('PRSA_data_2010.1.1-2014.12.31.csv', usecols=[5], engine='python', skipfooter=3)
#pd.dataframe = read_csv('PRSA_data_2010.1.1-2014.12.31.csv', index_col=0, usecols=[0,5,6,7,8,10], engine='python', skipfooter=3)
pd.dataframe = read_csv('PRSA_data_2010.1.1-2014.12.31.csv', index_col=0, usecols=[0,5,6,7,8,10], engine='python', skipfooter=3)
df = pd.dataframe
df = pd.dataframe
import scipy as sp
import numpy as np
import scipy as sp
#sp.sum(sp.isnan(dataset))
dataset = df.values
dataset = df.dropna(axis=0)
sp.sum(sp.isnan(dataset))
dataset = dataset.astype('float32')
# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)
# split into train and test sets
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
# reshape into X=t and Y=t+1
look_back = 3
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], trainX.shape[1], 5))
testX = numpy.reshape(testX, (testX.shape[0], testX.shape[1], 5))
# create and fit the LSTM network
model = Sequential()
model.add(LSTM(32, input_shape=(look_back, 5)))
model.add(Dense(8))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
history=model.fit(trainX, trainY, epochs=10, batch_size=1, verbose=2)
# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)


# invert predictions
trainPredict_extended = np.zeros((len(trainPredict),5))
trainPredict_extended[:,0:1] = trainPredict
trainPredict = scaler.inverse_transform(trainPredict_extended)[:,0]

testPredict_extended = np.zeros((len(testPredict),5))
testPredict_extended[:,0:1] = testPredict
testPredict = scaler.inverse_transform(testPredict_extended)[:,0]

trainY_extended = np.zeros((len(trainY),5))
trainY_extended[:,0] = trainY
trainY = scaler.inverse_transform(trainY_extended)[:,0]

testY_extended = np.zeros((len(testY),5))
testY_extended[:,0] = testY
testY = scaler.inverse_transform(testY_extended)[:,0]
#dataset = scaler.inverse_transform(dataset)



# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY, trainPredict))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY, testPredict))
print('Test Score: %.2f RMSE' % (testScore))
# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, 0] = trainPredict
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, 0] = testPredict
# plot baseline and predictions


plt.plot(scaler.inverse_transform(dataset)[:,0], color='r')
plt.plot(trainPredictPlot, color='g')
plt.plot(testPredictPlot,color='b')
plt.legend(['Actual data', 'Train data', 'Predict data'])
#plt.xlim([16000,18000])

#plt.legend('sinx',0)
plt.show()
plt.plot(history.history['loss'])
plt.xlabel('epoch')

plt.ylabel('loss')
plt.show()
print(model.summary())