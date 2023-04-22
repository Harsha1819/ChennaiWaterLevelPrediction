import streamlit as st
import pickle
import pandas as pd
import io
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
from sklearn.linear_model import LinearRegression

# Load the data
df = pd.read_excel("RainfallandWaterLevel.xlsx")



# Define the features and target
X = df[['POONDI', 'CHOLAVARAM', 'REDHILLS', 'CHEMBARAMBAKKAM']].values
y = df['Date'].values

# Fit the model
model = LinearRegression()
model.fit(X, y)


Model = pickle.load(open('Model.pkl', 'rb'))

# Define the function for making predictions
def predict_water_level(pondi, cholavaram, redhills, chembarambakkam):
    water_level = model.predict([[pondi, cholavaram, redhills, chembarambakkam]])
    return water_level[0]

def main():


    menu = ['Home', 'Chennai Water Level']
    choice = st.sidebar.selectbox('Select an option', menu)

    if choice == 'Home':
        st.header('Welcome to the Rainfall and Water Level Prediction app!')
        st.write('This app uses a trained model to predict the water level for a given date on the historical data.')
        
       
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
        st.write('The predicted water level for Chennai is : ', water_level )

if __name__ == '__main__':
    main()
