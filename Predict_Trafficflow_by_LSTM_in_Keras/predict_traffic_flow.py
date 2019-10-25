
# import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math
import read_traffic_flow
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import pandas as pd
import os
import matplotlib as mpl
os.getcwd()

# parameters settings
files_dir = "C:\\python\\projects\\giraffe\\Predict_Trafficflow_by_LSTM_in_Keras"
with_weekends = True
epoch_num = 2

# data and DATEs...
data_train, data_test, DATE_data_train, DATE_data_test = \
    read_traffic_flow.read_traffic_flow(files_dir, with_weekends)

data_train_original = data_train.reshape(data_train.shape[0], -1).astype('float32')
data_test = data_test.reshape(data_test.shape[0], -1).astype('float32')

'''
plt.figure(1)
plt.plot(data_train)
# plt.figure(2)
# plt.plot(data_validate)
plt.figure(3)
plt.plot(data_test)
plt.show()
'''

# plot time-series with dates
TS = pd.DataFrame({'date_time': DATE_data_train, 'flow': data_train})
TS = TS.sort_values(by='date_time')
print(TS.info())
print(type(TS))
print(TS.columns)

# field datetime is a "type" object...but we want this as index (for the date_time structure)
# remove seconds
TS['datetime'] = pd.to_datetime(TS['date_time'], format='%m/%d/%Y %H:%M')
print(type(TS))
TS = TS.set_index('date_time')
TS.drop(['datetime'], axis=1, inplace=True)
print(TS.columns)

'''
# plot Time Series
TS.plot(figsize=(20,8), linewidth=4, fontsize=8, rot=45)
plt.xlabel('datetime', fontsize=20)
plt.show()
'''

# create 2 arrays of data shifted by 1 time-stamp (5 minutes)
# this is the so-called windowed cross-sectional conversion with a time window of 5 miuntes
def create_dataset(dataset, timestamps, look_back = 1):
    data_x, data_y, time_x, time_y = [], [], [], []
    for i in range(len(dataset)-look_back-1):
        data_x.append(dataset[i:(i+look_back)])
        time_x.append(timestamps[i:(i+look_back)])
        data_y.append(dataset[i + look_back])
        time_y.append(timestamps[i + look_back])
    return np.array(data_x), np.array(data_y), \
           np.array(time_x), np.array(time_y)


# fix random seed for reproducibility
np.random.seed(7)
# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
data_train = scaler.fit_transform(data_train_original)
data_test = scaler.fit_transform(data_test)

# RESHUFFLE the training data (mix the data)
# np.random.shuffle(data_train)

look_back = 1  # it is the same thing of the lag
# train_x and train_y are shifted by 1 time stamp that is 5 minutes
train_x, train_y, time_train_x, time_train_y = create_dataset(data_train, DATE_data_train, look_back)
test_x, test_y, time_test_x, time_test_y = create_dataset(data_test, DATE_data_test, look_back)

train_x = train_x.reshape(train_x.shape[0], 1, train_x.shape[1])
test_x = test_x.reshape(test_x.shape[0], 1, test_x.shape[1])


# create and fit the LSTM network
model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back))) # 4 neurons, 1 layers
# model.add(LSTM(256, input_shape=(1, look_back))) # 4 neurons by 64 layers
model.add(Dense(1))  # add only one layer
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(train_x, train_y,
          epochs=epoch_num, batch_size=1)  # ,verbose = 2

# m.fit(temp, y_train, batch_size=config["batch"],
#       epochs=config["epochs"],
#       validation_split=0.05)

# make predictions with the obtained model
trainPredict = model.predict(train_x)  # close to train_y
testPredict = model.predict(test_x)    # close to test_y

# invert predictions (go back to real data....from normalized)
trainPredict = scaler.inverse_transform(trainPredict)
train_y = scaler.inverse_transform([train_y.ravel()])
testPredict = scaler.inverse_transform(testPredict)
test_y = scaler.inverse_transform([test_y.ravel()])
train_y = train_y.T
test_y = test_y.T
# print(trainPredict.shape, train_y.shape, testPredict.shape, test_y.shape)
# print(train_y.shape, trainPredict.shape)
# print("PAUSE"); input()
trainScore = math.sqrt(mean_absolute_error(train_y, trainPredict))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_absolute_error(test_y, testPredict))
print('Test Score: %.2f RMSE' % (testScore))

# new time-series with length = train data + test data (make an empty array)
l = data_train.shape[0] + data_test.shape[0]
# shift train predictions for plotting
trainPredictPlot = np.empty((l, 1))
trainPredictPlot[:, :] = np.nan
# populate the first part of the time series with the train-predicted data
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

# shift test predictions for plotting (this is the new forecasted part of the time-series
testPredictPlot = np.empty((l, 1))
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+look_back*3:l-1, :] = testPredict

test_y = test_y + 1

# differences between predicted values and test values
loss_test = np.sum(np.abs(testPredict - test_y) / test_y) / test_y.shape[0]

#####################################################################
################## plot train and test data #########################

'''
# plot baseline and predictions (TOTAL data)
plt.plot(scaler.inverse_transform(
    np.vstack((data_train, data_test))), 'yellow')
'''

all_DATES = np.concatenate((DATE_data_train, DATE_data_test))
# all_DATES = all_DATES.reshape(all_DATES.shape[0],-1)

data_total = scaler.inverse_transform(np.vstack((data_train, data_test))).round(0).T
data_total = data_total.ravel()
data_trainPrediction = trainPredictPlot.round(0).T.ravel()
data_testPredict = testPredictPlot.round(0).T.ravel()


'''
TS_total = pd.DataFrame({'date_time': all_DATES, 'flow': data_total})
TS_total = TS_total.sort_values(by = 'date_time')
# remove seconds
TS_total['datetime'] = pd.to_datetime(TS_total['date_time'], format='%m/%d/%Y %H:%M')
TS_total = TS_total.set_index('date_time')
TS_total.drop(['datetime'], axis=1, inplace=True)
# plot Time Series
TS_total.plot(figsize=(20,8), linewidth=5, fontsize=8, rot=45, color='yellow')
plt.xlabel('datetime', fontsize=20)
plt.show()

###### plot predictions ####################################

plt.plot(trainPredictPlot, 'red')


# with DATEs...
len(trainPredictPlot) # 7863 all dates
data_trainPrediction = trainPredictPlot.round(0).T.ravel()
TS_data_trainPrediction = pd.DataFrame({'date_time': all_DATES, 'flow': data_trainPrediction})
TS_data_trainPrediction = TS_data_trainPrediction.sort_values(by = 'date_time')
# remove seconds
TS_data_trainPrediction['datetime'] = pd.to_datetime(TS_data_trainPrediction['date_time'], format='%m/%d/%Y %H:%M')
TS_data_trainPrediction = TS_data_trainPrediction.set_index('date_time')
TS_data_trainPrediction.drop(['datetime'], axis=1, inplace=True)
# plot Time Series
TS_data_trainPrediction.plot(figsize=(20,8), linewidth=3, fontsize=8, rot=45, color='red')
plt.xlabel('datetime', fontsize=20)
plt.show()


plt.plot(testPredictPlot, 'blue')

# with DATEs
len(testPredictPlot) # 7863, all dates
data_testPredict = testPredictPlot.round(0).T.ravel()
TS_data_testPredict = pd.DataFrame({'date_time': all_DATES, 'flow': data_testPredict})
TS_data_testPredict = TS_data_testPredict.sort_values(by = 'date_time')
# remove seconds
TS_data_testPredict['datetime'] = pd.to_datetime(TS_data_testPredict['date_time'], format='%m/%d/%Y %H:%M')
TS_data_testPredict = TS_data_testPredict.set_index('date_time')
TS_data_testPredict.drop(['datetime'], axis=1, inplace=True)
# plot Time Series
TS_data_testPredict.plot(figsize=(20,8), linewidth=5, fontsize=8, rot=45, color='blue')
plt.xlabel('datetime', fontsize=20)
plt.show()
'''


# make a unique dataframe with train+test=total data, train predict and test predict
TS_forecast = pd.DataFrame({'date_time': all_DATES,
                           # 'total_flow': data_total,
                            'train_predictions': data_trainPrediction,
                            'test_predictions': data_testPredict})
TS_forecast = TS_forecast.sort_values(by='date_time')
# remove seconds and transform into a time-series
TS_forecast['datetime'] = pd.to_datetime(TS_forecast['date_time'], format='%m/%d/%Y %H:%M')
TS_forecast = TS_forecast.set_index('date_time')
TS_forecast.drop(['datetime'], axis=1, inplace=True)
# TS_forecast.plot(figsize=(20,8), linewidth=3, fontsize=8, rot=45)


# TS_forecast.total_flow.plot(figsize=(20,8), linewidth=2, fontsize=9, rot=45,color = "yellow")
TS_forecast.train_predictions.plot(figsize=(20,8), linewidth=2, fontsize=9, rot=45, color = "red")
TS_forecast.test_predictions.plot(figsize=(20,8), linewidth=2, fontsize=9, rot=45, color = "blue")
plt.legend(loc='best')
plt.xlabel(' ', fontsize=20)
plt.show()


'''
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates

fig, ax = plt.subplots()
ax.plot(TS_forecast.loc['2017-01':'2017-02', 'Consumption'], marker='o', linestyle='-')
ax.set_ylabel('Daily Consumption (GWh)')
ax.set_title('Jan-Feb 2017 Electricity Consumption')
# Set x-axis major ticks to weekly interval, on Mondays
ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MONDAY))
# Format x-tick labels as 3-letter month name and day number
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'));

fig, ax = plt.subplots(figsize=(20,8))
ax.plot(TS_forecast.index.values,
        TS_forecast['train_predictions'].values, color='purple')
# ax.set(xlabel="Date", ylabel="Precipitation (Inches)",
#        title="Daily Precipitation \nBoulder, Colorado 2013")

# Format the x axis
ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
# ax.xaxis.set_major_formatter(DateFormatter("%m/%d/%Y %H:%M"))

plt.show()
'''



plt.title("yellow: total; red: trainPred;" +
          " blue: testPred\n" +
          "If take weekends into account = {}\n".format(with_weekends) +
          "After {} epochs".format(epoch_num))
# print("shape of total:", scaler.inverse_transform(
#     np.vstack((data_train, data_test))).shape)
# print("trainPredict.shape:", trainPredict.shape)
# print("testPredict.shape:", testPredict.shape)
print("Loss of data_test = {}".format(loss_test))
loss_lst = np.abs(testPredict - test_y).ravel()
# print("loss_lst:", loss_lst)
# for i in range(len(loss_lst)):
#     if loss_lst[i] > 100:
#         print('{} = |{} - {}|'.format(loss_lst[i],
#               testPredict.ravel().tolist()[i], test_y.ravel().tolist()[i]))
with open('./loss.txt', 'w') as fout:
    fout.writelines('\n'.join(loss_lst.astype(str).tolist()))


######################################################################
######################################################################

# add new dates....to the time-series
max(all_DATES)
d = '2017-9-28 01:15'
# make a one day length time-series
new_DATES = pd.date_range(d, periods=288, freq='5min')
## !!! use the symbol # % to remove the 0 leading the month and the hour
new_DATES = new_DATES.strftime("%#m/%d/%Y %#H:%M")
new_DATES = np.array(new_DATES)
updated_DATES = np.concatenate((all_DATES, new_DATES))

# build a new testPredictPlot AS one NEW DAY
L = len(updated_DATES)
new_testPredictPlot = np.empty((L, 1))
new_testPredictPlot[:, :] = np.nan
new_testPredictPlot[288+len(trainPredict)+look_back*3:L-1, :] = testPredict

# create a trainPredictPlot but with a longer time-series of NAs
new_trainPredictPlot = np.empty((L, 1))
new_trainPredictPlot[:, :] = np.nan
# populate the first part of the time series with the train-predicted data
new_trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

data_trainPrediction = new_trainPredictPlot.round(0).T.ravel()
data_testPredict = new_testPredictPlot.round(0).T.ravel()

# make a unique dataframe with train+test=total data, train predict and test predict
TS_forecast = pd.DataFrame({'date_time': updated_DATES,
                            #'total_flow': data_total,
                            'train_predictions': data_trainPrediction,
                            'test_predictions': data_testPredict})
TS_forecast = TS_forecast.sort_values(by='date_time')
# remove seconds and transform into a time-series
TS_forecast['datetime'] = pd.to_datetime(TS_forecast['date_time'], format='%m/%d/%Y %H:%M')
TS_forecast = TS_forecast.set_index('date_time')
TS_forecast.drop(['datetime'], axis=1, inplace=True)
# TS_forecast.plot(figsize=(20,8), linewidth=3, fontsize=8, rot=45)


TS_forecast.train_predictions.plot(figsize=(20,8), linewidth=2, fontsize=9, rot=45, color = "red")
TS_forecast.test_predictions.plot(figsize=(20,8), linewidth=2, fontsize=9, rot=45, color = "blue")
plt.legend(loc='best')
plt.xlabel(' ', fontsize=20)
plt.show()