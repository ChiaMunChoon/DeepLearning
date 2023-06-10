#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from IPython.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))


# In[ ]:


import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt #For data visualizations
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import math
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential 
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import SimpleRNN
from keras.callbacks import EarlyStopping


# In[ ]:


# Data Exploration
dataset = pd.read_csv('Sunspots.csv',index_col="Date",parse_dates=True)

dataset['Monthly Mean Total Sunspot Number'].plot(figsize=(20,6))


# In[ ]:


# Subsetting the feature
df_set =  dataset.iloc[:,1:2].values
df_set


# In[ ]:


# Spliting data into train and testing
train , test = train_test_split(df_set,test_size=.2,shuffle=False)
print('Train: %s, Test: %s ' % (train.shape, test.shape))


# In[ ]:


# Normalizing the feature using MinMaxScaler to between 0 - 1
scaler = MinMaxScaler(feature_range = (0,1)) 
train_scaler = scaler.fit_transform(train)


# In[ ]:


# Reshaping input data into array of (samples, timesteps, feature)  
def prep_data(data, timesteps):
    x, y = list(), list()
    for i in range(len(data)):
        end_ix = i + timesteps
        if end_ix > len(data) - 1:
            break
        seq_x, seq_y = data[i:end_ix], data[end_ix]
        x.append(seq_x)
        y.append(seq_y)
    return np.array(x), np.array(y)


# In[ ]:


timesteps = 12
feature = 1
# split into samples
x_train, y_train = prep_data(train_scaler, timesteps)


# In[ ]:


x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],feature)


# In[ ]:


print(x_train.shape)
print(y_train.shape)


# In[ ]:


# Model Selection

# Initialize sequential
model = Sequential() 
    
#adding first SimpleRNN as input layer
model.add(SimpleRNN(units = 70, input_shape = (x_train.shape[1], 1))) 

#adding the output layer
model.add(Dense(units=1))

# Compiling model
model.compile(optimizer='adam', loss='mean_squared_error', metrics = ['RootMeanSquaredError'])


# In[ ]:


# Define early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=10)

# Model Evaluation
history = model.fit(x_train, y_train, epochs=50, batch_size=100, validation_split=0.2, callbacks=[early_stopping])


# In[ ]:


dataset_total = dataset.iloc[:,1:2]
inputs = dataset_total[len(dataset_total) - len(test) - timesteps :].values
inputs = inputs.reshape(-1, 1)
inputs = scaler.transform(inputs)


# In[ ]:


x_test, y_test = prep_data(inputs, timesteps)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], feature)


# In[ ]:


predicted = model.predict(x_test)


# In[ ]:


# RMSE 
test_rmse = np.sqrt(np.mean((predicted - y_test) ** 2))
print("Test RMSE:", test_rmse)


# In[ ]:


train_predicted = model.predict(x_train)


# In[ ]:


# RMSE 
train_rmse = np.sqrt(np.mean((train_predicted - y_train) ** 2))
print("Train RMSE:", train_rmse)


# In[ ]:


predicted= scaler.inverse_transform(predicted)
y_test = scaler.inverse_transform(y_test)


# In[ ]:


train_predicted = scaler.inverse_transform(train_predicted)
y_train = scaler.inverse_transform(y_train)


# In[ ]:


plt.plot(y_test, label = "Real Sunspots Values")
plt.plot(predicted, label = "Predicted Sunspots Values")
plt.title("Testing Sunspots Prediction")
plt.xlabel("Time")
plt.ylabel("Sunspots Values")
plt.legend()
plt.show()


# In[ ]:


predicted


# In[ ]:


y_test 


# In[ ]:


plt.plot(y_train, label = "Real Sunspots Values")
plt.plot(train_predicted, label = "Predicted Sunspots Values")
plt.title("Training Sunspots Prediction")
plt.xlabel("Time")
plt.ylabel("Sunspots Values")
plt.legend()
plt.show()


# In[ ]:


y_test 


# In[ ]:


# Plot the learning curve
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Learning Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[ ]:


# Plot the learning curve
plt.plot(history.history['root_mean_squared_error'], label='Training Loss')
plt.plot(history.history['val_root_mean_squared_error'], label='Validation Loss')
plt.title('Learning Curve')
plt.xlabel('Epochs')
plt.ylabel('RMSE')
plt.legend()
plt.show()


# In[ ]:


# Model Selection

# Initialize sequential
model = Sequential() 
    
#adding first SimpleRNN as input layer and dropout to avoid overfitting
model.add(SimpleRNN(units = 50, return_sequences=True, input_shape = (X_train.shape[1], 1))) 
model.add(Dropout(0.2))
    
#adding second SimpleRNN and dropout to avoid overfitting
model.add(SimpleRNN(units = 50)) 
#model.add(Dropout(0.2))
model.add(Dense(units=1, activation='tanh'))
#adding the output layer
model.add(Dense(units=1))


# In[ ]:


# Model Selection

# Initialize sequential
model = Sequential() 
    
#adding first SimpleRNN as input layer and dropout to avoid overfitting
model.add(SimpleRNN(units = 50, return_sequences=True, input_shape = (X_train.shape[1], 1))) 
model.add(Dropout(0.2))
    
#adding second SimpleRNN and dropout to avoid overfitting
model.add(SimpleRNN(units = 50)) 
model.add(Dropout(0.2))

#adding the output layer
model.add(Dense(units=1))

# Compiling model
model.compile(optimizer='adam', loss='mean_squared_error', metrics = ['RootMeanSquaredError'])

