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

df['Date'] = pd.to_datetime(df['Date'])

a = df['Date'].map(pd.Timestamp.to_julian_date).values.reshape(-1, 1)
b = df['Total_Water_Level'].values

model = LinearRegression()
model.fit(a,b)


# Define the features and target
X = df[['POONDI', 'CHOLAVARAM', 'REDHILLS', 'CHEMBARAMBAKKAM']].values
y = df['Date'].values

# Fit the model
model = LinearRegression()
model.fit(X, y)


Model = pickle.load(open('Model.pkl', 'rb'))


def predict_date(date):
    d = np.array([date])
    d = pd.to_datetime(d, infer_datetime_format=True)
    val = Model.predict(d.values.reshape(-1, 1))
    return val[0]

def predict():

    st.title("Total Water Level Prediction")
    date_str = st.text_input("Enter date (yyyy-mm-dd):", "")
    date = datetime.strptime(date_str, "%Y-%m-%d") 

    # Call the machine learning model to predict the total water level
    total_water_level = model.predict([[year, month, day]])

    # Display the predicted total water level
    st.success(f"Predicted total water level for {date_str}: {total_water_level[0]}")



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
        st.title('Water Level Prediction')
        st.write('Enter a date to predict water level:')
        date_input = st.date_input('Date', value=pd.to_datetime('2023-04-11'), min_value=df['Date'].min(), max_value=df['Date'].max())
        predicted_water_level = model.predict([[date_input.to_julian_date()]])[0]
        st.write(f'Predicted Water Level: {predicted_water_level}')

       
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
