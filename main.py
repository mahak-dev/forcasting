import streamlit as st
import requests
from streamlit_lottie import st_lottie
import pandas_datareader as pdr
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
from PIL import Image

import math
from sklearn.metrics import mean_squared_error
from numpy import array
import matplotlib.pyplot as plt
from keras.models import load_model

## st.set_page_config(layout = "wide")
# image and background
st.set_page_config(page_title="Stock Analysis",page_icon=":alien:", layout="wide")

def add_bg():
        st.markdown(
                f"""<style>
                .stApp{{
                        background-image: url("https://media0.giphy.com/media/JtBZm3Getg3dqxK0zP/giphy.gif?cid=ecf05e47iipdz3idcgc5j6dotyqxro978ul3lq0aiyy84esw&rid=giphy.gif&ct=g");
                        background-attachment: fixed;
                        background-size: cover;
                }}</style>
                """,unsafe_allow_html=True
        )
        

def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

##Lotties
lottie_Hello = load_lottieurl("https://lottie.host/b6fee2da-e123-47aa-8275-10edc05e75ae/pewE2ydK08.json")
lottie_dataset = load_lottieurl("https://assets8.lottiefiles.com/packages/lf20_8rs5Fb08t9.json")
lottie_contact = load_lottieurl("https://assets1.lottiefiles.com/packages/lf20_u25cckyh.json")
##section 1 
with st.container():
    left_column, right_column = st.columns(2)
    with left_column:
        st.subheader("Hello World! :wave:")
        st.title("Stock Price Prediction Model")
        st.write("A stock market prediction is an attempt to forecast the future value of an individual stock, a particular sector or the market, or the market as a whole. These forecasts generally use fundamental analysis of a company or economy, or technical analysis of charts, or a combination of the two.")
        
        
    with right_column:
        st_lottie(lottie_Hello, height=400, key="stock")


st.write("---")
with st.container():
        companies = Image.open('companies.jpeg')
        
        st.image(companies, width=1200)
        

        
st.write("---")
# use local CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
         
local_css("style/style.css")


tickers = Image.open('tickers.jpeg')
st.image(tickers, width=1000)
st.subheader("Choose Tickers")
user_input = st.text_input('Enter Stock Ticker', 'TSLA')

df = pdr.DataReader(user_input,'yahoo' )


n_years = st.slider("Years of predictions:", 1, 4)
period = n_years * 365

data_load_state = st.text("Load Data...")
data_load_state.text("Loading data...done!")

# Describing Data 
with st.container():
        st.subheader('Dataset that we are goning to use')
        left, right = st.columns(2)
        with left:
                st.write(df.describe())
        with right:
                st_lottie(lottie_dataset, height=400, key="dataset")


st.write("---")
# Visualizations
st.subheader('Closing price v/s time chart')
fig = plt.figure(figsize=(10,4))
plt.plot(df.Close)
st.pyplot(fig)



df1 = df.reset_index()['Close']
##Spliting dataset into train and test split
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range = (0,1))
df1 = scaler.fit_transform(np.array(df1).reshape(-1,1))


training_size=int(len(df1)*0.65)
test_size=len(df1)-training_size
train_data,test_data=df1[0:training_size,:],df1[training_size:len(df1),:1]


import numpy
def create_dataset(dataset, time_step=1):
        dataX, dataY = [], []
        for i in range(len(dataset)-time_step-1):
                a = dataset[i:(i+time_step), 0]
                dataX.append(a)
                dataY.append(dataset[i + time_step, 0])
        return numpy.array(dataX), numpy.array(dataY)


time_step = 100
X_train, y_train = create_dataset(train_data, time_step)
X_test, ytest = create_dataset(test_data, time_step)

##X_test, ytest = create_dataset(test_data, time_step)
X_train = X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1] , 1)


## load model
model = load_model('keras_model.h5')






## testing part
train_predict=model.predict(X_train)
test_predict=model.predict(X_test)


train_predict=scaler.inverse_transform(train_predict)
test_predict=scaler.inverse_transform(test_predict)

math.sqrt(mean_squared_error(y_train,train_predict))
    
### Test data RMSE
math.sqrt(mean_squared_error(ytest,test_predict))


### plotting 
## shift train prediction for plotting
look_back=100
trainPredictPlot=numpy.empty_like(df1)
trainPredictPlot[:, :]=np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
## shift test prediction for plotting
testPredictPlot=numpy.empty_like(df1)
testPredictPlot[:, :]=numpy.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(df1)-1, :]=test_predict
## plot base line and prediction 



plt.show()

st.write("---")

st.subheader('Final Graph with Comparision')
fig = plt.figure(figsize=(10,4))
plt.plot(scaler.inverse_transform(df1))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
st.pyplot(fig)



testdatalen = len(test_data)-100
x_input = test_data[testdatalen:].reshape(1,-1)
temp_input=list(x_input)
temp_input=temp_input[0].tolist()


## demostrate prediction for next 10 days



lst_output=[]
n_steps = 100
i=0
while(i<30):

    if(len(temp_input)>100):
        # print temp input
        x_input = np.array(temp_input[1:])
        print("{} day input {}".format(i, x_input))
        x_input = x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        #print x_input
        yhat = model.predict(x_input, verbose=0)
        print("{} day output {}".format(i, yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input = temp_input[1:]
        # print temp_input
        lst_output.extend(yhat.tolist())
        i = i+1
    else:
        x_input = x_input.reshape((1, n_steps, 1))
        yhat = model.predict(x_input, verbose=0)
        print(yhat[0])
        temp_input.extend(yhat[0].tolist())
        print(len(temp_input))
        lst_output.extend(yhat.tolist())
        i=i+1
           
day_new=np.arange(0,101)
day_pred=np.arange(101,131)

dflen = len(df1) -101

st.write("---")

## Prediction of Next 10 days Graph red lined
st.subheader('Prediction Graph of Future 10 Days (red line)')
plt.figure(figsize = (10,4))
fig2 = plt.figure(figsize = (10,4))
plt.plot(day_new,scaler.inverse_transform(df1[dflen:]), label = 'Previous Price')
plt.plot(day_pred,scaler.inverse_transform(lst_output),'r', label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()



st.pyplot(fig2)

st.write("---")

## 10 days Prediction graph
st.subheader('Prediction of Future 10 Days connected Graph.')
df3 = df1.tolist()
df3.extend(lst_output)
fig3 =plt.figure(figsize = (10,4))
plt.plot(df3[dflen:],'g', label = 'future trend')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
st.pyplot(fig3)

st.write("---")

df3=scaler.inverse_transform(df3).tolist()

st.subheader('Graph for Original Latest price.')
fig4 = plt.figure(figsize=(12,4))

plt.plot(df3,'b', label = 'Stock Price Analysis')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
#st.pyplot(fig4)
#fig4.layout(title_text="Time Series Data", xaxis_rangeslider_visible = True)
st.plotly_chart(fig4)

st.write("---")

# plotly graph demo
def plot_raw_data():
        fig5 = go.Figure()
        fig5.add_trace(go.Scatter( y=df['Open'], name = 'stock_open'))
        fig5.add_trace(go.Scatter( y=df['Close'], name = 'stock_close'))
        fig5.layout.update(xaxis_rangeslider_visible = True,width=1200, height=600)
        st.plotly_chart(fig5)

st.subheader("Time Series Graph")
plot_raw_data()

with st.container():
        
        mentor = Image.open('mentor1.jpeg')
        st.image(mentor, width=1200)
        
        
                

with st.container():
    st.write("---")
    st.header("Get In Touch With Me!")
    
    # Docuemenation 
    contact_form = """
    <div class="container">
  <form target="_blank" action="https://formsubmit.co/guptamahak740@gmail.com" method="POST">
    <div class="form-group">
      <div class="form-row">
        <div class="col">
          <input type="text" name="name" class="form-control" placeholder="Full Name" required>
        </div>
        <div class="col">
          <input type="email" name="email" class="form-control" placeholder="Email Address" required>
        </div>
      </div>
    </div>
    <div class="form-group">
      <textarea placeholder="Your Message" class="form-control" name="message" rows="10" required></textarea>
    </div>
    <button type="submit" class="btn btn-lg btn-dark btn-block">Submit Form</button>
  </form>
</div>
    """
    
    left_column, right_column = st.columns(2)
    with left_column:
        st.markdown(contact_form, unsafe_allow_html=True)
    with right_column:
        st_lottie(lottie_contact, height=400, key="contact")
        
        

st.write("---")

with st.container():
        companies = Image.open('Thankyou.png')
        
        st.image(companies, width=1200)
