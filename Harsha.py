import streamlit as st
import pickle
import pandas as pd
import io
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import datetime

# Load the data
df = pd.read_excel("RainfallandWaterLevel.xlsx")



# Define the features and target
X = df[['POONDI', 'CHOLAVARAM', 'REDHILLS', 'CHEMBARAMBAKKAM']].values
y = df['Date'].values

# Fit the model
model = LinearRegression()
model.fit(X, y)


Model = pickle.load(open('Model.pkl', 'rb'))


# Convert date column to datetime object
df['Date'] = pd.to_datetime(df['Date'])

# Set the date column as index
df.set_index('Date', inplace=True)

# Resample the data to monthly frequency
data = df.resample('M').mean()

# Split the data into train and test sets
train_data = df.iloc[:len(data)-12]
test_data = df.iloc[len(data)-12:]

# Define ARIMA model
model = ARIMA(train_data, order=(1, 1, 1)) # Example order=(p, d, q) values


# Fit the ARIMA model
model_fit = model.fit()



# Define the function for making predictions
def predict_water_level(pondi, cholavaram, redhills, chembarambakkam):
    water_level = model.predict([[pondi, cholavaram, redhills, chembarambakkam]])
    return water_level[0]

def main():


    menu = ['Home', 'Chennai Water Level','Predict']
    choice = st.sidebar.selectbox('Select an option', menu)

    if choice == 'Home':
        st.header('Welcome to the Rainfall and Water Level Prediction app!')
        st.write('This app uses a trained model to predict the water level for a given date on the historical data.')
        
                
    elif choice == 'Predict':
        st.title('Chennai Water Level Prediction')
        st.write('Water level forecast using ARIMA.')
        date_input = st.date_input('Select a Date for Water Level Prediction:', value=pd.to_datetime('2023-05-01'))
        date_input = pd.to_datetime(date_input).to_period('M')
        if date_input in test_data.index:
            forecasted_water_level = model_fit.forecast(steps=1).loc[date_input]
            st.write(f'**Water Level Prediction for {date_input}:** {forecasted_water_level[0]:.2f} meters')
        else:
            st.warning('Please select a valid date within the test data range.')
       
    elif choice == "Chennai Water Level":
        # Create the Streamlit app
        st.title('Chennai Water Level Prediction')

        # Add the inputs for each reservoir
        pondi_level = st.slider('Pondi Reservoir Level', 0, 5000, 1000)
        cholavaram_level = st.slider('Cholavaram Reservoir Level', 0, 5000, 1000)
        redhills_level = st.slider('Redhills Reservoir Level', 0, 5000, 1000)
        chembarambakkam_level = st.slider('Chembarambakkam Reservoir Level', 0, 5000, 1000)

        # Make the prediction and display the result
        water_level = predict_water_level(pondi_level, cholavaram_level, redhills_level, chembarambakkam_level)
        st.write('The predicted water level for Chennai is:', water_level)

if __name__ == '__main__':
    main()
